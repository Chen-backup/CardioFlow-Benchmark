from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

class ChannelLayerNorm(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.norm(state.transpose(1, 2)).transpose(1, 2)

class DilatedCycleMemoryBlock(nn.Module):

    def __init__(self, width: int, dilation: int, dropout: float, kernel_size: int=5) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('Cycle-memory kernels must have odd length')
        self.norm = ChannelLayerNorm(width)
        self.temporal = nn.Conv1d(width, width, kernel_size, padding=dilation * (kernel_size // 2), dilation=dilation, groups=width, bias=False)
        self.gate = nn.Conv1d(width, 2 * width, 1)
        self.out = nn.Conv1d(width, width, 1)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out.bias)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, state: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        update = self.temporal(self.norm(state) * support)
        content, gate = self.gate(update).chunk(2, dim=1)
        update = self.out(F.silu(content) * torch.sigmoid(gate))
        scale = torch.sigmoid(self.residual_scale).to(dtype=state.dtype)
        return (state + scale * self.dropout(update)) * support

@dataclass
class CycleStateOutput:
    trajectory: torch.Tensor
    record_state: torch.Tensor
    cycle_memory: torch.Tensor
    projection_energy: torch.Tensor

@dataclass
class PathologyEvidenceOutput:
    logits: torch.Tensor
    attention: torch.Tensor
    attentive_mean: torch.Tensor
    attentive_std: torch.Tensor

class PathologyEvidenceQueryHead(nn.Module):

    def __init__(self, width: int, num_classes: int, *, queries_per_class: int=4) -> None:
        super().__init__()
        if width < 1 or num_classes < 1 or queries_per_class < 1:
            raise ValueError('width, num_classes, and queries_per_class must be positive')
        self.width = width
        self.num_classes = num_classes
        self.queries_per_class = queries_per_class
        self.queries = nn.Parameter(torch.empty(num_classes, queries_per_class, width))
        nn.init.orthogonal_(self.queries.flatten(0, 1))
        evidence_channels = 2 * queries_per_class * width
        self.score_weight = nn.Parameter(torch.empty(num_classes, evidence_channels))
        self.score_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.uniform_(self.score_weight, -evidence_channels ** (-0.5), evidence_channels ** (-0.5))

    @staticmethod
    def _resize_support(observation_support: torch.Tensor | None, trajectory: torch.Tensor) -> torch.Tensor:
        if observation_support is None:
            return trajectory.new_ones((trajectory.shape[0], 1, trajectory.shape[-1]))
        support = observation_support
        if support.ndim == 2:
            support = support.unsqueeze(1)
        if support.ndim != 3 or support.shape[:2] != (trajectory.shape[0], 1):
            raise ValueError('observation support must have shape [batch, 1, time]')
        support = F.interpolate(support.to(device=trajectory.device, dtype=trajectory.dtype), size=trajectory.shape[-1], mode='nearest').clamp(0.0, 1.0).detach()
        if torch.any(support.sum(dim=-1) <= 0):
            raise ValueError('Every record must have at least one observed time point')
        return support

    def forward(self, trajectory: torch.Tensor, observation_support: torch.Tensor | None=None) -> PathologyEvidenceOutput:
        if trajectory.ndim != 3 or trajectory.shape[1] != self.width:
            raise ValueError(f'Expected trajectory [batch, {self.width}, time], got {tuple(trajectory.shape)}')
        support = self._resize_support(observation_support, trajectory)
        sequence = trajectory.transpose(1, 2)
        content = F.layer_norm(sequence.float(), (self.width,))
        queries = F.normalize(self.queries.float(), dim=-1) * self.width ** 0.5
        attention_logits = torch.einsum('btd,cqd->bcqt', content, queries)
        attention_logits = attention_logits / self.width ** 0.5
        valid = support[:, :, None, :] >= 0.5
        attention = torch.softmax(attention_logits.masked_fill(~valid, torch.finfo(attention_logits.dtype).min), dim=-1)
        attention = attention * support.float().unsqueeze(2)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-08)
        values = sequence.float()
        attentive_mean = torch.einsum('bcqt,btd->bcqd', attention, values)
        second_moment = torch.einsum('bcqt,btd->bcqd', attention, values.square())
        variance = second_moment - attentive_mean.square()
        attentive_std = (variance.clamp_min(0.0) + 1e-06).sqrt()
        evidence = torch.cat((attentive_mean, attentive_std), dim=-1).flatten(2)
        logits = (evidence * self.score_weight.float().unsqueeze(0)).sum(dim=-1)
        logits = logits + self.score_bias.float().unsqueeze(0)
        return PathologyEvidenceOutput(logits.to(dtype=trajectory.dtype), attention.to(dtype=trajectory.dtype), attentive_mean.to(dtype=trajectory.dtype), attentive_std.to(dtype=trajectory.dtype))

class CardiacCycleStateMixer(nn.Module):

    def __init__(self, cardiac_channels: int, *, width: int=192, cycle_slots: int=8, dilations: tuple[int, ...]=(1, 4, 16, 64), heads: int=4, dropout: float=0.15, energy_window: int=9) -> None:
        super().__init__()
        if width < 1 or width % heads:
            raise ValueError('Mixer width must be a positive multiple of attention heads')
        if cycle_slots < 1:
            raise ValueError('At least one cardiac-cycle memory slot is required')
        if energy_window < 1 or energy_window % 2 != 1:
            raise ValueError('Projection-energy window must be a positive odd integer')
        self.cardiac_channels = cardiac_channels
        self.width = width
        self.cycle_slots = cycle_slots
        self.energy_window = energy_window
        self.state_projection = nn.Conv1d(cardiac_channels, width, 1, bias=False)
        self.input_norm = ChannelLayerNorm(width)
        self.input_bias = nn.Parameter(torch.zeros(1, width, 1))
        self.input_dropout = nn.Dropout(dropout)
        self.temporal_memory = nn.ModuleList((DilatedCycleMemoryBlock(width, dilation, dropout) for dilation in dilations))
        self.memory_slots = nn.Parameter(torch.empty(cycle_slots, width))
        nn.init.orthogonal_(self.memory_slots)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(width)
        self.slot_ffn_norm = nn.LayerNorm(width)
        self.slot_ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
        self.trajectory_norm = nn.LayerNorm(width)
        self.memory_dropout = nn.Dropout(dropout)
        self.broadcast_scale = nn.Parameter(torch.tensor(0.1))

    @property
    def record_channels(self) -> int:
        return (4 + self.cycle_slots) * self.width

    @staticmethod
    def _resize_support(support: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
        if support is None:
            return state.new_ones((state.shape[0], 1, state.shape[-1]))
        if support.ndim == 2:
            support = support.unsqueeze(1)
        if support.ndim != 3 or support.shape[1] != 1:
            raise ValueError('observation support must have shape [batch, 1, time]')
        return F.interpolate(support.to(device=state.device, dtype=state.dtype), size=state.shape[-1], mode='nearest').clamp(0.0, 1.0).detach()

    def _check_state(self, cardiac_state: torch.Tensor) -> None:
        if cardiac_state.ndim != 3 or cardiac_state.shape[1] != self.cardiac_channels:
            raise ValueError(f'Expected complete cardiac state [batch, {self.cardiac_channels}, time], got {tuple(cardiac_state.shape)}')

    def _projection_energy_envelope(self, projected: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        radius = self.energy_window // 2
        squared = projected.float().square() * support.float()
        local_energy = F.avg_pool1d(squared, kernel_size=self.energy_window, stride=1, padding=radius)
        local_support = F.avg_pool1d(support.float(), kernel_size=self.energy_window, stride=1, padding=radius).clamp_min(1.0 / self.energy_window)
        envelope = (local_energy / local_support).clamp_min(1e-08).sqrt()
        return envelope.to(dtype=projected.dtype) * support

    @staticmethod
    def _record_summary(trajectory: torch.Tensor, memory: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        state = trajectory.float()
        mask = support.float()
        count = mask.sum(dim=-1).clamp_min(1.0)
        mean = (state * mask).sum(dim=-1) / count
        variance = ((state - mean.unsqueeze(-1)).square() * mask).sum(dim=-1) / count
        deviation = (variance.clamp_min(0.0) + 1e-06).sqrt()
        temperature = 0.5
        valid = mask >= 0.5
        scaled = (state / temperature).masked_fill(~valid, -10000.0)
        soft_peak = temperature * (torch.logsumexp(scaled, dim=-1) - count.log())
        if state.shape[-1] > 1:
            transition_mask = mask[..., 1:] * mask[..., :-1]
            transition_count = transition_mask.sum(dim=-1).clamp_min(1.0)
            variation = ((state[..., 1:] - state[..., :-1]).abs() * transition_mask).sum(dim=-1) / transition_count
        else:
            variation = torch.zeros_like(mean)
        summary = torch.cat((mean, deviation, soft_peak, variation, memory.float().flatten(1)), dim=1)
        return summary.to(dtype=trajectory.dtype)

    def forward(self, cardiac_state: torch.Tensor, observation_support: torch.Tensor | None=None) -> CycleStateOutput:
        self._check_state(cardiac_state)
        support = self._resize_support(observation_support, cardiac_state)
        projected = self.state_projection(cardiac_state * support)
        projection_energy = self._projection_energy_envelope(projected, support)
        trajectory = F.silu(self.input_norm(projected) + self.input_bias)
        trajectory = trajectory * (1.0 + torch.log1p(projection_energy))
        trajectory = self.input_dropout(trajectory) * support
        for block in self.temporal_memory:
            trajectory = block(trajectory, support)
        sequence = trajectory.transpose(1, 2)
        valid = support[:, 0] >= 0.5
        slots = self.memory_slots.unsqueeze(0).expand(sequence.shape[0], -1, -1)
        gathered, _ = self.cross_attention(slots, sequence, sequence, key_padding_mask=~valid, need_weights=False)
        memory = self.slot_norm(slots + self.memory_dropout(gathered))
        memory = memory + self.memory_dropout(self.slot_ffn(self.slot_ffn_norm(memory)))
        recalled, _ = self.cross_attention(self.trajectory_norm(sequence), memory, memory, need_weights=False)
        scale = torch.tanh(self.broadcast_scale).to(dtype=sequence.dtype)
        trajectory = (sequence + scale * self.memory_dropout(recalled)).transpose(1, 2) * support
        record_state = self._record_summary(trajectory, memory, support)
        return CycleStateOutput(trajectory, record_state, memory, projection_energy)
