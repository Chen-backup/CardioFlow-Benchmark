from __future__ import annotations
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cycle_state import CardiacCycleStateMixer

class ConvNormAct(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int=1, groups: int=1):
        padding = kernel // 2
        super().__init__(nn.Conv1d(in_channels, out_channels, kernel, stride=stride, padding=padding, groups=groups, bias=False), nn.BatchNorm1d(out_channels), nn.SiLU())

class BoundedGroupRecruitmentCalibrator(nn.Module):

    def __init__(self, population_width: int, *, minimum_gain: float=0.5, maximum_gain: float=4.0, initial_gains: tuple[float, float, float]=(1.0, 2.0, 3.0)) -> None:
        super().__init__()
        if population_width < 1:
            raise ValueError('population_width must be positive')
        if not minimum_gain < maximum_gain:
            raise ValueError('minimum_gain must be smaller than maximum_gain')
        initial = torch.as_tensor(initial_gains, dtype=torch.float32)
        if initial.shape != (3,):
            raise ValueError('initial_gains must contain exactly three values')
        if torch.any(initial <= minimum_gain) or torch.any(initial >= maximum_gain):
            raise ValueError('initial gains must lie strictly inside the gain bounds')
        fraction = (initial - minimum_gain) / (maximum_gain - minimum_gain)
        self.population_width = int(population_width)
        self.minimum_gain = float(minimum_gain)
        self.maximum_gain = float(maximum_gain)
        self.raw_gain = nn.Parameter(torch.logit(fraction))

    def bounded_gain(self) -> torch.Tensor:
        span = self.maximum_gain - self.minimum_gain
        return self.minimum_gain + span * torch.sigmoid(self.raw_gain)

    def forward(self, source_probability: torch.Tensor) -> torch.Tensor:
        if source_probability.ndim != 3:
            raise ValueError('source_probability must have shape [batch, 3H, time]')
        expected_channels = 3 * self.population_width
        if source_probability.shape[1] != expected_channels:
            raise ValueError(f'Expected {expected_channels} source probabilities, got {source_probability.shape[1]}')
        probability = source_probability.clamp(0.0, 1.0)
        gain = self.bounded_gain().to(probability.dtype).repeat_interleave(self.population_width).view(1, expected_channels, 1)
        epsilon = torch.finfo(probability.dtype).eps
        safe_probability = probability.clamp(max=1.0 - epsilon)
        calibrated = -torch.expm1(gain * torch.log1p(-safe_probability))
        return torch.where(probability >= 1.0, torch.ones_like(calibrated), calibrated)

class AdaptiveIAF(nn.Module):

    def __init__(self, sources: int=3, slope: float=8.0, mode: str='staircase', max_events: int=96, integration_step: float=0.5, threshold_spread: float=0.0, population_width: int | None=None, phase_tiling: bool=False, phase_tile_size: int | None=None, adaptive_threshold: bool=True):
        super().__init__()
        self.sources = sources
        self.slope = slope
        self.mode = mode
        self.max_events = max_events
        self.integration_step = integration_step
        self.phase_tiling = phase_tiling
        self.adaptive_threshold = adaptive_threshold
        self.register_buffer('annealing_hardness', torch.tensor(0.0))
        self.history_window = max(32, int(round(32.0 / integration_step)))
        self.logit_adapt_decay = nn.Parameter(torch.full((sources,), 3.0))
        threshold_initial = torch.full((sources,), 3.3)
        if threshold_spread > 0.0:
            if population_width is None or sources % population_width != 0:
                raise ValueError('Threshold spread requires complete anatomical populations')
            rank = torch.linspace(-1.0, 1.0, population_width)
            threshold_initial += float(threshold_spread) * rank.repeat(sources // population_width)
        self.base_threshold = nn.Parameter(threshold_initial)
        self.adapt_gain = nn.Parameter(torch.full((sources,), -2.5))
        if phase_tiling:
            if population_width is None or sources % population_width != 0:
                raise ValueError('Phase tiling requires complete anatomical populations')
            tile_size = phase_tile_size or population_width
            if population_width % tile_size != 0:
                raise ValueError('Phase tile size must divide each population')
            phase = (torch.arange(tile_size, dtype=torch.float32) + 0.5) / tile_size
            within_population = phase.repeat(population_width // tile_size)
            initial_phase = within_population.repeat(sources // population_width)
        else:
            initial_phase = torch.zeros(sources)
        self.register_buffer('initial_phase', initial_phase, persistent=phase_tiling)

    @staticmethod
    def _history(x: torch.Tensor, decay: torch.Tensor, window: int=64) -> torch.Tensor:
        channels = x.shape[1]
        positions = torch.arange(window, device=x.device, dtype=x.dtype)
        kernel = decay.to(x.dtype).view(channels, 1) ** positions.view(1, window)
        shifted = F.pad(x, (1, 0))[..., :-1]
        return F.conv1d(F.pad(shifted, (window - 1, 0)), kernel.flip(-1).unsqueeze(1), groups=channels)

    def set_training_progress(self, progress: float) -> None:
        if not 0.0 <= progress <= 1.0:
            raise ValueError('training progress must lie in [0, 1]')
        phase = min(1.0, 2.0 * float(progress))
        hardness = 0.5 - 0.5 * math.cos(math.pi * phase)
        self.annealing_hardness.fill_(hardness)

    def forward(self, drive: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, sources, _ = drive.shape
        if sources != self.sources:
            raise ValueError(f'Expected {self.sources} source drives, got {sources}')
        adapt_decay = torch.sigmoid(self.logit_adapt_decay)
        decay_exponent = min(1.0, 2.0 * self.integration_step)
        adapt_decay = adapt_decay.pow(decay_exponent)
        threshold = (F.softplus(self.base_threshold) + 0.1).view(1, -1, 1)
        gain = F.softplus(self.adapt_gain).view(1, -1, 1)
        recent_drive = self._history(drive, adapt_decay, window=self.history_window)
        normalizer = (adapt_decay.view(1, -1, 1) ** torch.arange(self.history_window, device=drive.device, dtype=drive.dtype).view(1, 1, -1)).sum(dim=-1, keepdim=True)
        adaptation = gain * recent_drive / normalizer.clamp_min(1.0)
        if not self.adaptive_threshold:
            adaptation = torch.zeros_like(adaptation)
        effective_threshold = threshold * (1.0 + adaptation)
        increment = (self.integration_step * drive / effective_threshold).clamp(0.0, 0.95)
        charge = torch.cumsum(increment, dim=-1)
        initial_phase = self.initial_phase.to(drive.dtype).view(1, -1, 1)
        tiled_charge = charge + initial_phase
        count = torch.floor(tiled_charge)
        previous_charge = F.pad(charge, (1, 0))[..., :-1] + initial_phase
        previous_count = torch.floor(previous_charge)
        hard = (count > previous_count).to(drive.dtype)
        if self.mode == 'direct':
            spike = increment
            probability = increment
        elif self.mode in {'straight_through', 'population', 'annealed'}:
            previous_phase = torch.remainder(previous_charge, 1.0)
            probability = torch.sigmoid(self.slope * (previous_phase + increment - 1.0))
            straight_through = hard + probability - probability.detach()
            if self.mode == 'straight_through':
                spike = straight_through
            elif self.mode == 'population':
                spike = probability
            else:
                hardness = self.annealing_hardness.to(probability.dtype)
                spike = probability.lerp(straight_through, hardness) if self.training else straight_through
        elif self.mode == 'staircase':
            levels = torch.arange(1, self.max_events + 1, device=drive.device, dtype=drive.dtype).view(1, 1, 1, -1)
            smooth_count = torch.sigmoid(self.slope * (tiled_charge.unsqueeze(-1) - levels)).sum(dim=-1)
            previous_smooth = F.pad(smooth_count, (1, 0))[..., :-1]
            spike = (smooth_count - previous_smooth).clamp_min(0.0)
            probability = spike
        else:
            raise ValueError(f'Unknown IAF mode: {self.mode}')
        phase = torch.remainder(tiled_charge, 1.0)
        voltage = phase * effective_threshold
        return (spike, {'probability': probability, 'drive': drive, 'hard_spike': hard, 'voltage': voltage, 'phase': phase, 'charge_phase': torch.remainder(charge, 1.0), 'initial_phase': initial_phase, 'adaptation': adaptation})

class PostIAFRecoveryControl(nn.Module):

    def __init__(self, population_width: int, kernel_size: int=5) -> None:
        super().__init__()
        channels = 6 * population_width
        self.kernel_size = kernel_size
        self.project = nn.Sequential(nn.Conv1d(9 * population_width, channels, 1, bias=False), nn.GroupNorm(6, channels), nn.SiLU())
        self.memory = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=False)
        self.mix = nn.Conv1d(channels, channels, 1)
        nn.init.normal_(self.mix.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.mix.bias)

    def forward(self, spikes: torch.Tensor, phase: torch.Tensor, adaptation: torch.Tensor) -> torch.Tensor:
        state = self.project(torch.cat((spikes, phase, adaptation), dim=1))
        state = F.pad(state, (self.kernel_size - 1, 0))
        return self.mix(F.silu(self.memory(state)))

class CableCoupledIAF(nn.Module):

    def __init__(self, sources: int, population_width: int, slope: float=8.0, mode: str='straight_through', integration_step: float=1.0, threshold_spread: float=0.0, phase_tiling: bool=False, phase_tile_size: int | None=None, adaptive_threshold: bool=True) -> None:
        super().__init__()
        if sources % population_width != 0:
            raise ValueError('IAF sources must contain complete anatomical populations')
        self.population_width = population_width
        self.populations = sources // population_width
        self.cell = AdaptiveIAF(sources, slope=slope, mode=mode, integration_step=integration_step, threshold_spread=threshold_spread, population_width=population_width, phase_tiling=phase_tiling, phase_tile_size=phase_tile_size, adaptive_threshold=adaptive_threshold)
        self.conductance_logit = nn.Parameter(torch.full((self.populations,), -2.2))

    @staticmethod
    def _sealed_laplacian(voltage: torch.Tensor) -> torch.Tensor:
        left = torch.cat((voltage[:, :, :1], voltage[:, :, :-1]), dim=2)
        right = torch.cat((voltage[:, :, 1:], voltage[:, :, -1:]), dim=2)
        return left + right - 2.0 * voltage

    def forward(self, drive: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, predicted = self.cell(drive)
        batch, _, steps = drive.shape
        voltage = predicted['phase'].view(batch, self.populations, self.population_width, steps)
        laplacian = self._sealed_laplacian(voltage)
        conductance = (0.05 * torch.sigmoid(self.conductance_logit)).view(1, self.populations, 1, 1)
        gap_current = conductance * laplacian
        corrected_drive = (drive.view(batch, self.populations, self.population_width, steps) + gap_current).clamp(0.0, 1.0).reshape_as(drive)
        spikes, corrected = self.cell(corrected_drive)
        corrected['uncoupled_drive'] = drive
        corrected['uncoupled_phase'] = predicted['phase']
        corrected['gap_current'] = gap_current.reshape_as(drive)
        corrected['gap_conductance'] = conductance.reshape(1, self.populations, 1)
        return (spikes, corrected)

class CausalDelay(nn.Module):

    def __init__(self, channels: int, max_delay: int, initial_delays: list[int], evidence_conditioned: bool=False):
        super().__init__()
        if len(initial_delays) != channels:
            raise ValueError('One initial delay is required per channel')
        self.max_delay = max_delay
        logits = torch.full((channels, max_delay), -3.0)
        for channel, delay in enumerate(initial_delays):
            logits[channel, min(max(delay, 0), max_delay - 1)] = 3.0
        self.logits = nn.Parameter(logits)
        if evidence_conditioned:
            self.evidence_mix_logit = nn.Parameter(torch.full((channels,), -1.4))
            self.evidence_temperature = nn.Parameter(torch.full((channels,), 2.0))
        else:
            self.register_parameter('evidence_mix_logit', None)
            self.register_parameter('evidence_temperature', None)

    def forward(self, x: torch.Tensor, downstream_evidence: torch.Tensor | None=None, return_expected: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        static_weights = torch.softmax(self.logits, dim=-1)
        if downstream_evidence is None:
            kernel = static_weights.unsqueeze(1)
            padded = F.pad(x, (kernel.shape[-1] - 1, 0))
            delayed = F.conv1d(padded, kernel.flip(-1), groups=x.shape[1])
            weights = static_weights.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            if downstream_evidence.shape != x.shape:
                raise ValueError('Delay evidence must match its upstream population')
            if self.evidence_mix_logit is None or self.evidence_temperature is None:
                raise ValueError('This causal delay was not configured for downstream evidence')
            bank = torch.stack([F.pad(x, (delay, 0))[..., :x.shape[-1]] for delay in range(self.max_delay)], dim=2)
            upstream_centered = bank - bank.mean(dim=-1, keepdim=True)
            downstream_centered = downstream_evidence - downstream_evidence.mean(dim=-1, keepdim=True)
            numerator = (upstream_centered * downstream_centered.unsqueeze(2)).mean(dim=-1)
            upstream_scale = (upstream_centered.square().mean(dim=-1) + 0.0001).sqrt()
            downstream_scale = (downstream_centered.square().mean(dim=-1) + 0.0001).sqrt().unsqueeze(-1)
            denominator = upstream_scale * downstream_scale
            correlation = (numerator / denominator).clamp(-1.0, 1.0)
            temperature = F.softplus(self.evidence_temperature).view(1, -1, 1)
            evidence_weights = torch.softmax(temperature * correlation, dim=-1)
            mixture = torch.sigmoid(self.evidence_mix_logit).view(1, -1, 1)
            weights = (1.0 - mixture) * static_weights.unsqueeze(0) + mixture * evidence_weights
            delayed = (bank * weights.unsqueeze(-1)).sum(dim=2)
        if not return_expected:
            return delayed
        positions = torch.arange(self.max_delay, device=x.device, dtype=x.dtype)
        return (delayed, (weights * positions).sum(dim=-1))

    def expected_delay(self) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=-1)
        positions = torch.arange(weights.shape[-1], device=weights.device, dtype=weights.dtype)
        return (weights * positions).sum(dim=-1)

class IdentityDelay(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.register_buffer("zero_delay", torch.zeros(self.channels))

    def forward(self, x: torch.Tensor, downstream_evidence: torch.Tensor | None=None, return_expected: bool=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != self.channels:
            raise ValueError(f'Expected {self.channels} channels, got {x.shape[1]}')
        if not return_expected:
            return x
        return (x, x.new_zeros((x.shape[0], self.channels)))

    def expected_delay(self) -> torch.Tensor:
        return self.zero_delay

class RefractoryGate(nn.Module):

    def __init__(self, channels: int, initial_decay: float=0.86, slope: float=10.0, window: int=32):
        super().__init__()
        initial_logit = torch.logit(torch.tensor(initial_decay))
        self.logit_decay = nn.Parameter(torch.full((channels,), initial_logit))
        self.threshold = nn.Parameter(torch.full((channels,), -0.2))
        self.slope = slope
        self.window = window

    def forward(self, incoming: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.logit_decay)
        threshold = torch.sigmoid(self.threshold).view(1, -1, 1)
        occupancy_open = AdaptiveIAF._history(incoming, decay, window=self.window).clamp(0.0, 1.0)
        gate_open = torch.sigmoid(self.slope * (1.0 - occupancy_open - threshold))
        proposal = incoming * gate_open
        occupancy = AdaptiveIAF._history(proposal, decay, window=self.window).clamp(0.0, 1.0)
        availability = 1.0 - occupancy
        gate = torch.sigmoid(self.slope * (availability - threshold))
        return (incoming * gate, availability, gate)

class ConductionVelocityRestitution(nn.Module):

    def __init__(self, groups: int, population: int, time_scale: int=1, maximum_extra_delay: int=3) -> None:
        super().__init__()
        self.groups = groups
        self.population = population
        self.maximum_extra_delay = maximum_extra_delay * time_scale
        initial_decay = 0.9 ** (1.0 / time_scale)
        self.logit_decay = nn.Parameter(torch.full((groups,), float(torch.logit(torch.tensor(initial_decay)))))
        initial_half_activity = 3.0 / max(1, population)
        self.raw_half_activity = nn.Parameter(torch.full((groups,), float(torch.log(torch.expm1(torch.tensor(initial_half_activity))))))
        self.raw_sharpness = nn.Parameter(torch.full((groups,), 0.54132485))
        self.window = 32 * time_scale

    def forward(self, wavefront: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, steps = wavefront.shape
        if channels != self.groups * self.population:
            raise ValueError(f'Expected {self.groups * self.population} restitution channels, got {channels}')
        cells = wavefront.view(batch, self.groups, self.population, steps)
        activity = cells.mean(dim=2)
        decay = torch.sigmoid(self.logit_decay)
        recent = AdaptiveIAF._history(activity, decay, window=self.window)
        half_activity = F.softplus(self.raw_half_activity).view(1, -1, 1)
        severity = recent / (recent + half_activity + 1e-06)
        delays = torch.arange(self.maximum_extra_delay + 1, device=wavefront.device, dtype=wavefront.dtype).view(1, 1, -1, 1)
        center = self.maximum_extra_delay * severity.unsqueeze(2)
        sharpness = F.softplus(self.raw_sharpness).view(1, -1, 1, 1)
        weights = torch.softmax(-sharpness * (delays - center).square(), dim=2)
        transported = torch.zeros_like(cells)
        for extra_delay in range(self.maximum_extra_delay + 1):
            component = cells * weights[:, :, extra_delay].unsqueeze(2)
            if extra_delay:
                component = F.pad(component, (extra_delay, 0))[..., :steps]
            transported = transported + component
        return (transported.reshape(batch, channels, steps), severity)

class RenewalPhaseResponse(nn.Module):

    def __init__(self, channels: int, time_scale: int=1) -> None:
        super().__init__()
        initial_decay = 0.9 ** (1.0 / time_scale)
        initial_logit = float(torch.logit(torch.tensor(initial_decay)))
        self.logit_decay = nn.Parameter(torch.full((channels,), initial_logit))
        initial_period = 40.0 * time_scale
        self.raw_period = nn.Parameter(torch.full((channels,), initial_period - 2.0))
        self.window = 64 * time_scale

    def _record_period(self, evidence: torch.Tensor, population_period: torch.Tensor) -> torch.Tensor:
        pooled = evidence.mean(dim=1)
        steps = pooled.shape[-1]
        minimum = min(20 * max(1, self.window // 64), max(1, steps - 1))
        maximum = min(self.window, max(minimum, steps - 1))
        lags = torch.arange(minimum, maximum + 1, device=evidence.device, dtype=torch.long)
        fft_length = 1 << (2 * steps - 1).bit_length()
        spectrum = torch.fft.rfft(pooled.float(), n=fft_length)
        autocorrelation = torch.fft.irfft(spectrum.conj() * spectrum, n=fft_length)[..., :steps]
        correlation = autocorrelation.index_select(-1, lags) / autocorrelation[..., :1].clamp_min(1e-05)
        prior_period = population_period.mean().to(evidence.dtype)
        prior_scale = (0.35 * prior_period).clamp_min(4.0)
        lag_values = lags.to(evidence.dtype)
        score = 20.0 * correlation.to(evidence.dtype) - 0.5 * ((lag_values - prior_period) / prior_scale).square()
        estimated = (torch.softmax(score, dim=-1) * lag_values).sum(dim=-1).view(-1, 1, 1)
        event_count = pooled.sum(dim=-1, keepdim=True).unsqueeze(-1)
        reliability = torch.sigmoid(event_count - 3.0)
        return (1.0 - reliability) * prior_period + reliability * estimated

    def forward(self, events: torch.Tensor, period_evidence: torch.Tensor | None=None, period_override: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.logit_decay).clamp(0.75, 0.999)
        trace = AdaptiveIAF._history(events, decay, window=self.window)
        lower = decay.view(1, -1, 1).pow(self.window)
        age = torch.log(trace.clamp_min(lower).clamp_max(1.0)) / torch.log(decay.view(1, -1, 1))
        age = age.clamp(0.0, float(self.window))
        population_period = F.softplus(self.raw_period) + 2.0
        if period_override is None:
            record_period = self._record_period(events if period_evidence is None else period_evidence, population_period)
        else:
            record_period = period_override
        relative = population_period / population_period.mean().clamp_min(1.0)
        period = record_period * relative.view(1, -1, 1)
        phase_response = (1.0 - age / period).clamp(0.0, 1.0)
        prematurity = events * phase_response
        return (prematurity, age / period, record_period)

class PopulationRenewalPhaseResponse(RenewalPhaseResponse):

    def __init__(self, channels: int, population_size: int, time_scale: int=1) -> None:
        if channels % population_size:
            raise ValueError('Renewal channels must contain complete populations')
        self.population_size = population_size
        self.groups = channels // population_size
        super().__init__(self.groups, time_scale=time_scale)
        initial = min(0.9, max(0.01, 3.5 / population_size))
        self.register_buffer('consensus_threshold', torch.full((self.groups,), initial))

    def consensus_event(self, events: torch.Tensor) -> torch.Tensor:
        batch, channels, steps = events.shape
        if channels != self.groups * self.population_size:
            raise ValueError(f'Expected {self.groups * self.population_size} renewal channels, got {channels}')
        recruitment = events.view(batch, self.groups, self.population_size, steps).mean(dim=2)
        threshold = self.consensus_threshold.view(1, -1, 1)
        hard = (recruitment > threshold).to(events.dtype)
        previous = F.pad(hard, (1, 0))[..., :-1]
        return hard * (1.0 - previous)

    def forward(self, events: torch.Tensor, period_evidence: torch.Tensor | None=None, period_override: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del period_evidence
        consensus = self.consensus_event(events).detach()
        decay = torch.sigmoid(self.logit_decay).clamp(0.75, 0.999)
        trace = AdaptiveIAF._history(consensus, decay, window=self.window)
        lower = decay.view(1, -1, 1).pow(self.window)
        age = torch.log(trace.clamp_min(lower).clamp_max(1.0)) / torch.log(decay.view(1, -1, 1))
        age = age.clamp(0.0, float(self.window))
        population_period = F.softplus(self.raw_period) + 2.0
        if period_override is None:
            record_period = self._record_period(consensus, population_period)
        else:
            record_period = period_override
        relative = population_period / population_period.mean().clamp_min(1.0)
        period = record_period * relative.view(1, -1, 1)
        group_response = (1.0 - age / period).clamp(0.0, 1.0)
        response = group_response.repeat_interleave(self.population_size, dim=1)
        phase = (age / period).repeat_interleave(self.population_size, dim=1)
        return (events * response, phase, record_period)

class RepolarizationEcho(nn.Module):

    def __init__(self, channels: int=2, bases: int=3, time_scale: int=1):
        super().__init__()
        self.channels = channels
        self.bases = bases
        self.window = 32 * time_scale
        fast = torch.tensor([0.7, 0.82, 0.9]).pow(1.0 / time_scale).view(1, bases).expand(channels, -1)
        slow = torch.tensor([0.9, 0.96, 0.985]).pow(1.0 / time_scale).view(1, bases).expand(channels, -1)
        self.fast_logit = nn.Parameter(torch.logit(fast))
        self.slow_logit = nn.Parameter(torch.logit(slow))

    def forward(self, activation: torch.Tensor, mixture_logits: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        fast = torch.sigmoid(self.fast_logit)
        slow = torch.maximum(torch.sigmoid(self.slow_logit), fast + 0.02).clamp_max(0.999)
        plateau_traces, recovery_traces = ([], [])
        for basis in range(self.bases):
            fast_trace = AdaptiveIAF._history(activation, fast[:, basis], window=self.window)
            slow_trace = AdaptiveIAF._history(activation, slow[:, basis], window=self.window)
            plateau_traces.append(slow_trace)
            recovery_traces.append((slow_trace - fast_trace).clamp_min(0.0))
        plateaus = torch.stack(plateau_traces, dim=2)
        recoveries = torch.stack(recovery_traces, dim=2)
        if mixture_logits is None:
            weights = activation.new_full((activation.shape[0], self.channels, self.bases), 1.0 / self.bases)
        else:
            compact = mixture_logits.mean(dim=-1) if mixture_logits.ndim == 3 else mixture_logits
            weights = torch.softmax(compact.view(activation.shape[0], self.channels, self.bases), dim=-1)
        weights = weights.unsqueeze(-1)
        return ((plateaus * weights).sum(dim=2), (recoveries * weights).sum(dim=2))

class CardiacFitzHughNagumo(nn.Module):

    def __init__(self, population: int, time_scale: int=1) -> None:
        super().__init__()
        self.population = population
        self.time_scale = time_scale
        self.raw_excitability = nn.Parameter(torch.full((2,), -0.91629076))
        self.raw_recovery_rate = nn.Parameter(torch.full((2,), -0.84729786))
        self.raw_recovery_gain = nn.Parameter(torch.zeros(2))
        self.raw_input_gain = nn.Parameter(torch.zeros(2))
        self.raw_voltage_decay = nn.Parameter(torch.full((2,), float(torch.logit(torch.tensor(0.72)))))
        self.raw_reaction_gain = nn.Parameter(torch.full((2,), -1.5))
        self.window = 64 * time_scale
        self.picard_iterations = 3

    @staticmethod
    def _causal_integral(drive: torch.Tensor, decay: torch.Tensor, window: int) -> torch.Tensor:
        channels = drive.shape[1]
        positions = torch.arange(window, device=drive.device, dtype=drive.dtype)
        kernel = decay.to(drive.dtype).view(channels, 1) ** positions.view(1, window)
        return F.conv1d(F.pad(drive, (window - 1, 0)), kernel.flip(-1).unsqueeze(1), groups=channels)

    def forward(self, activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, steps = activation.shape
        if channels != 2 * self.population:
            raise ValueError(f'Expected {2 * self.population} myocardial channels, got {channels}')
        excitability = (0.05 + 0.35 * torch.sigmoid(self.raw_excitability)).repeat_interleave(self.population).view(1, -1, 1)
        recovery_decay = (0.9 + 0.09 * torch.sigmoid(self.raw_recovery_rate)).pow(1.0 / self.time_scale).repeat_interleave(self.population)
        recovery_gain = (0.02 + 0.08 * torch.sigmoid(self.raw_recovery_gain)).repeat_interleave(self.population).view(1, -1, 1)
        input_gain = (0.5 + torch.sigmoid(self.raw_input_gain)).repeat_interleave(self.population).view(1, -1, 1)
        voltage_decay = torch.sigmoid(self.raw_voltage_decay).pow(1.0 / self.time_scale).repeat_interleave(self.population)
        reaction_gain = F.softplus(self.raw_reaction_gain).repeat_interleave(self.population).view(1, -1, 1)
        current = input_gain * activation
        voltage = self._causal_integral(current, voltage_decay, self.window).clamp(-0.5, 1.5)
        recovery = recovery_gain * self._causal_integral(voltage.clamp_min(0.0), recovery_decay, self.window)
        for _ in range(self.picard_iterations - 1):
            reaction = voltage * (1.0 - voltage) * (voltage - excitability)
            voltage = self._causal_integral(current + reaction_gain * reaction - recovery, voltage_decay, self.window).clamp(-0.5, 1.5)
            recovery = recovery_gain * self._causal_integral(voltage.clamp_min(0.0), recovery_decay, self.window)
        return (voltage.clamp_min(0.0), recovery.clamp_min(0.0))

class ElectrotonicTissueResponse(nn.Module):

    def __init__(self, population: int, time_scale: int=1):
        super().__init__()
        fast_group = torch.tensor([0.35, 0.25, 0.2, 0.2, 0.3, 0.3]).pow(1.0 / time_scale)
        slow_group = torch.tensor([0.78, 0.6, 0.58, 0.58, 0.72, 0.72]).pow(1.0 / time_scale)
        fast = fast_group.repeat_interleave(population)
        slow = slow_group.repeat_interleave(population)
        self.fast_logit = nn.Parameter(torch.logit(fast))
        self.slow_logit = nn.Parameter(torch.logit(slow))
        self.log_amplitude = nn.Parameter(torch.full((6 * population,), 0.54132485))
        self.window = 32 * time_scale
        self.population = population

    def forward(self, graph: torch.Tensor) -> torch.Tensor:
        event_channels = 6 * self.population
        events = graph[:, :event_channels]
        fast = torch.sigmoid(self.fast_logit)
        slow = torch.maximum(torch.sigmoid(self.slow_logit), fast + 0.02).clamp_max(0.995)
        fast_trace = AdaptiveIAF._history(events, fast, window=self.window)
        slow_trace = AdaptiveIAF._history(events, slow, window=self.window)
        amplitude = F.softplus(self.log_amplitude).view(1, -1, 1)
        depolarization = events + amplitude * (slow_trace - fast_trace).clamp_min(0.0)
        return torch.cat((depolarization, graph[:, event_channels:]), dim=1)

class ChargeConservingTissueResponse(nn.Module):

    def __init__(self, population: int, time_scale: int=1) -> None:
        super().__init__()
        window = 17 * time_scale
        lag = torch.arange(window, dtype=torch.float32) / time_scale
        kernels = [torch.nn.functional.one_hot(torch.tensor(0), num_classes=window).to(torch.float32)]
        for rise, decay in ((0.5, 2.0), (1.0, 5.0), (2.0, 10.0)):
            alpha = torch.exp(-lag / decay) - torch.exp(-lag / rise)
            alpha[0] = 0.0
            kernels.append(alpha / alpha.sum().clamp_min(1e-06))
        self.register_buffer('kernel_bank', torch.stack(kernels))
        initial = torch.zeros(6, len(kernels))
        initial[:, 0] = 2.0
        self.mixture_logits = nn.Parameter(initial)
        self.population = population

    def response_kernels(self) -> torch.Tensor:
        return torch.softmax(self.mixture_logits, dim=-1) @ self.kernel_bank

    def forward(self, graph: torch.Tensor) -> torch.Tensor:
        event_channels = 6 * self.population
        events = graph[:, :event_channels]
        responses = []
        for kernel in self.kernel_bank:
            weight = kernel.flip(0).view(1, 1, -1).expand(event_channels, 1, -1)
            responses.append(F.conv1d(F.pad(events, (kernel.numel() - 1, 0)), weight, groups=event_channels))
        bank = torch.stack(responses, dim=2)
        mixture = torch.softmax(self.mixture_logits, dim=-1).repeat_interleave(self.population, dim=0).view(1, event_channels, -1, 1)
        membrane = (bank * mixture).sum(dim=2)
        return torch.cat((membrane, graph[:, event_channels:]), dim=1)

@dataclass
class ConductionState:
    graph: torch.Tensor
    av_gate: torch.Tensor
    bundle_gate: torch.Tensor
    myocardial_gate: torch.Tensor
    expected_delays: torch.Tensor
    av_incoming: torch.Tensor
    bundle_incoming: torch.Tensor
    myocardial_incoming: torch.Tensor
    record_av_delay: torch.Tensor
    right_atrial: torch.Tensor
    left_atrial: torch.Tensor
    av_availability: torch.Tensor
    bundle_availability: torch.Tensor
    myocardial_availability: torch.Tensor
    atrial_prematurity: torch.Tensor
    atrial_cycle_phase: torch.Tensor
    myocardial_prematurity: torch.Tensor
    myocardial_cycle_phase: torch.Tensor
    ventricular_prematurity: torch.Tensor
    record_intrinsic_period: torch.Tensor
    av_velocity_restitution: torch.Tensor
    bundle_velocity_restitution: torch.Tensor
    myocardial_velocity_restitution: torch.Tensor

class PositivePopulationMix(nn.Module):

    def __init__(self, input_population: int, output_population: int):
        super().__init__()
        logits = torch.full((output_population, input_population), -4.0)
        for output_index in range(output_population):
            logits[output_index, output_index % input_population] = 4.0
        self.logits = nn.Parameter(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.logits, dim=-1)
        return torch.einsum('oi,bit->bot', weights, x)

class RefractoryConductionGraph(nn.Module):

    def __init__(self, max_delay: int=12, population_size: int=1, time_scale: int=1, observed_av_delay: bool=False, biatrial_conduction: bool=False, biatrial_av_convergence: bool=False, renewal_phase_response: bool=False, population_renewal_clock: bool=False, conduction_velocity_restitution: bool=False, velocity_max_extra_delay: int=3, fitzhugh_nagumo_tissue: bool=False, use_conduction_delays: bool=True, use_refractory_gating: bool=True, use_ventricular_dynamics: bool=True):
        super().__init__()
        self.population_size = population_size
        self.observed_av_delay = observed_av_delay
        self.biatrial_conduction = biatrial_conduction
        self.biatrial_av_convergence = biatrial_av_convergence
        self.use_refractory_gating = use_refractory_gating
        self.use_ventricular_dynamics = use_ventricular_dynamics
        if biatrial_av_convergence and (not biatrial_conduction):
            raise ValueError('Biatrial AV convergence requires biatrial conduction')
        self.renewal_phase_response = renewal_phase_response
        self.population_renewal_clock = population_renewal_clock
        self.conduction_velocity_restitution = conduction_velocity_restitution
        if population_renewal_clock and (not renewal_phase_response):
            raise ValueError('Population renewal clock requires renewal dynamics')
        history_window = 32 * time_scale
        delay_class = CausalDelay if use_conduction_delays else IdentityDelay
        if use_conduction_delays:
            self.av_delay = delay_class(population_size, max_delay, [3 * time_scale] * population_size, evidence_conditioned=observed_av_delay)
            self.bundle_delay = delay_class(2 * population_size, max_delay, [2 * time_scale] * (2 * population_size))
            self.myocardial_delay = delay_class(2 * population_size, max_delay, [time_scale] * (2 * population_size))
        else:
            self.av_delay = delay_class(population_size)
            self.bundle_delay = delay_class(2 * population_size)
            self.myocardial_delay = delay_class(2 * population_size)
        if conduction_velocity_restitution:
            self.av_velocity = ConductionVelocityRestitution(1, population_size, time_scale=time_scale, maximum_extra_delay=velocity_max_extra_delay)
            self.bundle_velocity = ConductionVelocityRestitution(2, population_size, time_scale=time_scale, maximum_extra_delay=velocity_max_extra_delay)
            self.myocardial_velocity = ConductionVelocityRestitution(2, population_size, time_scale=time_scale, maximum_extra_delay=velocity_max_extra_delay)
        else:
            self.av_velocity = None
            self.bundle_velocity = None
            self.myocardial_velocity = None
        self.av_gate = RefractoryGate(population_size, initial_decay=0.82 ** (1.0 / time_scale), window=history_window)
        self.bundle_gate = RefractoryGate(2 * population_size, initial_decay=0.88 ** (1.0 / time_scale), window=history_window)
        self.myocardial_gate = RefractoryGate(2 * population_size, initial_decay=0.9 ** (1.0 / time_scale), window=history_window)
        self.fitzhugh_nagumo_tissue = fitzhugh_nagumo_tissue
        self.repolarization = CardiacFitzHughNagumo(population_size, time_scale=time_scale) if fitzhugh_nagumo_tissue else RepolarizationEcho(2 * population_size, time_scale=time_scale)
        self.junction_to_his = PositivePopulationMix(population_size, population_size)
        self.his_to_bundle = PositivePopulationMix(population_size, 2 * population_size)
        self.bundle_to_myocardium = PositivePopulationMix(2 * population_size, 2 * population_size)
        self.ectopic_to_myocardium = PositivePopulationMix(population_size, 2 * population_size)
        self.ectopic_route = nn.Parameter(torch.zeros(2 * population_size))
        if renewal_phase_response:
            renewal_class = PopulationRenewalPhaseResponse if population_renewal_clock else RenewalPhaseResponse
            renewal_kwargs = {'population_size': population_size} if population_renewal_clock else {}
            self.atrial_renewal = renewal_class(population_size, time_scale=time_scale, **renewal_kwargs)
            self.myocardial_renewal = renewal_class(2 * population_size, time_scale=time_scale, **renewal_kwargs)
            self.premature_av_delay = 3 * time_scale
            self.apd_restitution_gain = nn.Parameter(torch.tensor(0.5))
        else:
            self.atrial_renewal = None
            self.myocardial_renewal = None
            self.premature_av_delay = 0
            self.register_parameter('apd_restitution_gain', None)
        if biatrial_conduction:
            self.right_to_left = PositivePopulationMix(population_size, population_size)
            self.interatrial_delay = CausalDelay(population_size, max_delay, [3 * time_scale] * population_size) if use_conduction_delays else IdentityDelay(population_size)
        else:
            self.right_to_left = None
            self.interatrial_delay = None

    def forward(self, sources: torch.Tensor, recovery_control: torch.Tensor | None=None, source_probability: torch.Tensor | None=None) -> ConductionState:
        h = self.population_size
        if sources.shape[1] != 3 * h:
            raise ValueError(f'Expected {3 * h} population source channels, got {sources.shape[1]}')
        atrial, junctional, ventricular = (sources[:, :h], sources[:, h:2 * h], sources[:, 2 * h:])
        if self.interatrial_delay is not None:
            left_atrial = self.interatrial_delay(self.right_to_left(atrial))
            atrial_to_av = 0.5 * (atrial + left_atrial) if self.biatrial_av_convergence else atrial
            atrial_tissue = 0.5 * (atrial + left_atrial)
        else:
            left_atrial = atrial
            atrial_to_av = atrial
            atrial_tissue = atrial
        if self.av_velocity is not None:
            atrial_to_av, av_velocity_restitution = self.av_velocity(atrial_to_av)
        else:
            av_velocity_restitution = atrial_to_av.new_zeros(atrial_to_av.shape[0], 1, atrial_to_av.shape[-1])
        if self.observed_av_delay:
            if source_probability is None:
                raise ValueError('Observed AV delay requires IAF population evidence')
            junctional_evidence = source_probability[:, h:2 * h]
            av_incoming, record_av_delay = self.av_delay(atrial_to_av, junctional_evidence, return_expected=True)
        else:
            av_incoming = self.av_delay(atrial_to_av)
            record_av_delay = self.av_delay.expected_delay().unsqueeze(0).expand(sources.shape[0], -1)
        if self.atrial_renewal is not None:
            atrial_period_evidence = source_probability[:, :h] if source_probability is not None else atrial_to_av
            atrial_prematurity, atrial_cycle_phase, record_intrinsic_period = self.atrial_renewal(atrial_to_av, period_evidence=atrial_period_evidence)
            if self.observed_av_delay:
                premature_at_av = self.av_delay(atrial_prematurity, junctional_evidence)
            else:
                premature_at_av = self.av_delay(atrial_prematurity)
            slowed_premature = F.pad(premature_at_av, (self.premature_av_delay, 0))[..., :premature_at_av.shape[-1]]
            av_incoming = (av_incoming - premature_at_av + slowed_premature).clamp(0.0, 1.0)
        else:
            atrial_prematurity = torch.zeros_like(atrial_to_av)
            atrial_cycle_phase = torch.zeros_like(atrial_to_av)
            record_intrinsic_period = atrial_to_av.new_full((atrial_to_av.shape[0], 1, 1), 40.0 * max(1, self.av_gate.window // 32))
        if self.use_refractory_gating:
            av_passed, av_availability, av_gate = self.av_gate(av_incoming)
        else:
            av_passed, av_availability, av_gate = av_incoming, torch.ones_like(av_incoming), torch.ones_like(av_incoming)
        his = (av_passed + 0.6 * self.junction_to_his(junctional)).clamp(0.0, 1.0)
        bundle_source = self.his_to_bundle(his)
        if self.bundle_velocity is not None:
            bundle_source, bundle_velocity_restitution = self.bundle_velocity(bundle_source)
        else:
            bundle_velocity_restitution = bundle_source.new_zeros(bundle_source.shape[0], 2, bundle_source.shape[-1])
        bundle_incoming = self.bundle_delay(bundle_source)
        if self.use_refractory_gating:
            branches, bundle_availability, bundle_gate = self.bundle_gate(bundle_incoming)
        else:
            branches, bundle_availability, bundle_gate = bundle_incoming, torch.ones_like(bundle_incoming), torch.ones_like(bundle_incoming)
        myocardial_source = self.bundle_to_myocardium(branches)
        if self.myocardial_velocity is not None:
            myocardial_source, myocardial_velocity_restitution = self.myocardial_velocity(myocardial_source)
        else:
            myocardial_velocity_restitution = myocardial_source.new_zeros(myocardial_source.shape[0], 2, myocardial_source.shape[-1])
        normal_myocardial = self.myocardial_delay(myocardial_source)
        if self.use_ventricular_dynamics:
            ectopic = torch.sigmoid(self.ectopic_route).view(1, 2 * h, 1) * self.ectopic_to_myocardium(ventricular)
        else:
            ectopic = torch.zeros_like(normal_myocardial)
        myocardial_incoming = (normal_myocardial + ectopic).clamp(0.0, 1.0)
        if self.myocardial_renewal is not None:
            myocardial_prematurity, myocardial_cycle_phase, _ = self.myocardial_renewal(myocardial_incoming, period_override=record_intrinsic_period)
            ectopic_fraction = ectopic / myocardial_incoming.clamp_min(0.0001)
            ventricular_prematurity = myocardial_prematurity * ectopic_fraction.clamp(0.0, 1.0)
        else:
            myocardial_prematurity = torch.zeros_like(myocardial_incoming)
            myocardial_cycle_phase = torch.zeros_like(myocardial_incoming)
            ventricular_prematurity = torch.zeros_like(myocardial_incoming)
        if self.use_refractory_gating:
            myocardium, myocardial_availability, myocardial_gate = self.myocardial_gate(myocardial_incoming)
        else:
            myocardium, myocardial_availability, myocardial_gate = myocardial_incoming, torch.ones_like(myocardial_incoming), torch.ones_like(myocardial_incoming)
        if self.apd_restitution_gain is not None:
            restitution_direction = myocardial_prematurity.new_tensor([1.0, 0.0, -1.0]).view(1, 1, 3, 1)
            recovery_control = recovery_control.view(recovery_control.shape[0], 2 * h, 3, recovery_control.shape[-1])
            recovery_control = recovery_control + F.softplus(self.apd_restitution_gain) * myocardial_prematurity.unsqueeze(2) * restitution_direction
            recovery_control = recovery_control.flatten(1, 2)
        if self.fitzhugh_nagumo_tissue:
            action_potential, recovery_echo = self.repolarization(myocardium)
        else:
            action_potential, recovery_echo = self.repolarization(myocardium, recovery_control)
        graph = torch.cat((atrial_tissue, his, branches, myocardium, action_potential, recovery_echo), dim=1)
        delays = torch.cat((self.av_delay.expected_delay(), self.bundle_delay.expected_delay(), self.myocardial_delay.expected_delay()))
        return ConductionState(graph=graph, av_gate=av_gate, bundle_gate=bundle_gate, myocardial_gate=myocardial_gate, expected_delays=delays, av_incoming=av_incoming, bundle_incoming=bundle_incoming, myocardial_incoming=myocardial_incoming, record_av_delay=record_av_delay, right_atrial=atrial, left_atrial=left_atrial, av_availability=av_availability, bundle_availability=bundle_availability, myocardial_availability=myocardial_availability, atrial_prematurity=atrial_prematurity, atrial_cycle_phase=atrial_cycle_phase, myocardial_prematurity=myocardial_prematurity, myocardial_cycle_phase=myocardial_cycle_phase, ventricular_prematurity=ventricular_prematurity, record_intrinsic_period=record_intrinsic_period, av_velocity_restitution=av_velocity_restitution, bundle_velocity_restitution=bundle_velocity_restitution, myocardial_velocity_restitution=myocardial_velocity_restitution)

class ChannelNorm1d(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError('ChannelNorm1d expects [batch, channels, time]')
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class WavefrontInferenceBlock(nn.Module):

    def __init__(self, channels: int, *, dilation: int, expansion: int=4, dropout: float=0.0, residual_scale: float=0.1) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError('channels must be positive')
        if dilation < 1:
            raise ValueError('dilation must be positive')
        if expansion < 1:
            raise ValueError('expansion must be positive')
        hidden_channels = expansion * channels
        self.dilation = int(dilation)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=9, padding=4 * dilation, dilation=dilation, groups=channels, bias=True)
        self.channel_norm = ChannelNorm1d(channels)
        self.expand = nn.Conv1d(channels, hidden_channels, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Conv1d(hidden_channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.full((1, channels, 1), float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(x)
        update = self.channel_norm(update)
        update = self.expand(update)
        update = self.activation(update)
        update = self.dropout(update)
        update = self.project(update)
        return x + self.residual_scale.to(update.dtype) * update

def _fixed_lead_geometry(limb_inverse_mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    limb_forward = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [-0.5, -0.5], [1.0, -0.5], [-0.5, 1.0]])
    lead_forward = torch.zeros(12, 8)
    lead_forward[:6, :2] = limb_forward
    lead_forward[6:, 2:] = torch.eye(6)
    independent_projection = torch.zeros(8, 12)
    if limb_inverse_mode == 'pseudoinverse':
        independent_projection[:] = torch.linalg.pinv(lead_forward)
    elif limb_inverse_mode == 'direct':
        independent_projection[0, 0] = 1.0
        independent_projection[1, 1] = 1.0
        independent_projection[2:, 6:] = torch.eye(6)
    else:
        raise ValueError(f'Unknown limb inverse mode: {limb_inverse_mode}')
    inverse_xyz = torch.tensor([[0.156, -0.01, -0.172, -0.074, 0.122, 0.231, 0.239, 0.194], [-0.227, 0.887, 0.057, -0.019, -0.106, -0.022, 0.041, 0.048], [0.022, 0.102, -0.229, -0.31, -0.246, -0.063, 0.055, 0.108]])
    lead_basis = torch.cat((independent_projection, inverse_xyz @ independent_projection), dim=0)
    return (lead_basis, lead_forward, inverse_xyz)

def _project_lead_coordinates(x: torch.Tensor, observed_lead_mask: torch.Tensor | None, *, lead_basis: torch.Tensor, lead_forward: torch.Tensor, inverse_xyz: torch.Tensor) -> torch.Tensor:
    static_coordinates = torch.einsum('oi,bit->bot', lead_basis.to(dtype=x.dtype), x)
    if observed_lead_mask is None:
        return static_coordinates
    lead_mask = observed_lead_mask
    if lead_mask.ndim == 2:
        lead_mask = lead_mask.unsqueeze(-1)
    if lead_mask.shape != (x.shape[0], x.shape[1], 1):
        raise ValueError('observed_lead_mask must have shape [batch, leads, 1]')
    observed = (lead_mask[..., 0] >= 0.5).detach()
    observed_count = observed.sum(dim=1)
    if torch.any(observed_count < 1):
        raise ValueError('Every ECG must contain at least one observed lead')
    partial = observed_count < x.shape[1]
    if not torch.any(partial):
        return static_coordinates
    geometry_dtype = torch.float32
    forward = lead_forward.to(device=x.device, dtype=geometry_dtype)
    signal = x.to(dtype=geometry_dtype)
    observed_float = observed.to(dtype=geometry_dtype)
    masked_signal = signal * observed_float.unsqueeze(-1)
    selected_lead = observed_float.argmax(dim=1)
    selected_forward = forward.index_select(0, selected_lead)
    selected_signal = masked_signal.gather(1, selected_lead.view(-1, 1, 1).expand(-1, 1, x.shape[-1]))
    single_independent = selected_forward.unsqueeze(-1) * selected_signal / selected_forward.square().sum(dim=1, keepdim=True).unsqueeze(-1)
    independent = static_coordinates[:, :8].to(dtype=geometry_dtype)
    single = observed_count == 1
    independent = torch.where(single.view(-1, 1, 1), single_independent, independent)
    uncommon_subset = partial & ~single
    if torch.any(uncommon_subset):
        subset_forward = forward.unsqueeze(0) * observed_float[uncommon_subset].unsqueeze(-1)
        subset_independent = torch.einsum('bij,bjt->bit', torch.linalg.pinv(subset_forward), masked_signal[uncommon_subset])
        replacement = independent.clone()
        replacement[uncommon_subset] = subset_independent
        independent = replacement
    xyz = torch.einsum('oi,bit->bot', inverse_xyz.to(device=x.device, dtype=geometry_dtype), independent)
    partial_coordinates = torch.cat((independent, xyz), dim=1).to(dtype=x.dtype)
    return torch.where(partial.view(-1, 1, 1), partial_coordinates, static_coordinates)

class MultiScaleCardiacDrive(nn.Module):

    def __init__(self, population: int, dropout: float, stride: int=2, isoelectric_centering: bool=False, limb_inverse_mode: str='pseudoinverse', polarity_invariant_drive: bool=False):
        super().__init__()
        self.isoelectric_centering = isoelectric_centering
        self.polarity_invariant_drive = polarity_invariant_drive
        lead_basis, lead_forward, inverse_xyz = _fixed_lead_geometry(limb_inverse_mode)
        self.register_buffer('lead_basis', lead_basis)
        self.register_buffer('lead_forward', lead_forward)
        self.register_buffer('inverse_xyz', inverse_xyz)
        wavefront_channels = 66 if polarity_invariant_drive else 33
        self.wavefront = ConvNormAct(wavefront_channels, population, 15, stride=stride)
        self.wavefront_inference = nn.Sequential(*(WavefrontInferenceBlock(population, dilation=dilation, expansion=4, dropout=dropout, residual_scale=0.1) for dilation in (1, 2, 4, 8)))
        self.filters = nn.ModuleList([nn.Conv1d(population, population, kernel, padding=kernel // 2, groups=population, bias=False) for kernel in (5, 15, 31)])
        self.fuse = nn.Sequential(nn.Conv1d(3 * population, population, 1, bias=False), nn.BatchNorm1d(population), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, observed_lead_mask: torch.Tensor | None=None) -> torch.Tensor:
        coordinates = _project_lead_coordinates(x, observed_lead_mask, lead_basis=self.lead_basis, lead_forward=self.lead_forward, inverse_xyz=self.inverse_xyz)
        if self.isoelectric_centering:
            coordinates = coordinates - coordinates.median(dim=-1, keepdim=True).values
        slope = F.pad(coordinates[..., 1:] - coordinates[..., :-1], (1, 0))
        curvature = F.pad(slope[..., 1:] - slope[..., :-1], (1, 0))
        wavefront = torch.cat((coordinates, slope, curvature), dim=1)
        if self.polarity_invariant_drive:
            wavefront = torch.cat((wavefront, wavefront.abs()), dim=1)
        x = self.wavefront_inference(self.wavefront(wavefront))
        return self.fuse(torch.cat([branch(x) for branch in self.filters], dim=1))

class StationaryWaveletCardiacDrive(nn.Module):

    def __init__(self, population: int, dropout: float, stride: int=2, isoelectric_centering: bool=False, limb_inverse_mode: str='pseudoinverse') -> None:
        super().__init__()
        self.isoelectric_centering = isoelectric_centering
        lead_basis, lead_forward, inverse_xyz = _fixed_lead_geometry(limb_inverse_mode)
        self.register_buffer('lead_basis', lead_basis)
        self.register_buffer('lead_forward', lead_forward)
        self.register_buffer('inverse_xyz', inverse_xyz)
        self.register_buffer('lowpass', torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0)
        self.wavefront = ConvNormAct(5 * 11, population, 9, stride=stride)
        self.filters = nn.ModuleList([nn.Conv1d(population, population, kernel, padding=kernel // 2, groups=population, bias=False) for kernel in (5, 15, 31)])
        self.fuse = nn.Sequential(nn.Conv1d(3 * population, population, 1, bias=False), nn.BatchNorm1d(population), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, observed_lead_mask: torch.Tensor | None=None) -> torch.Tensor:
        coordinates = _project_lead_coordinates(x, observed_lead_mask, lead_basis=self.lead_basis, lead_forward=self.lead_forward, inverse_xyz=self.inverse_xyz)
        if self.isoelectric_centering:
            coordinates = coordinates - coordinates.median(dim=-1, keepdim=True).values
        approximation = coordinates
        details = []
        for dilation in (1, 2, 4, 8):
            kernel = self.lowpass.to(x.dtype).view(1, 1, -1).expand(11, 1, -1)
            smooth = F.conv1d(approximation, kernel, padding=2 * dilation, dilation=dilation, groups=11)
            details.append(approximation - smooth)
            approximation = smooth
        wavelet = torch.cat((*details, approximation), dim=1)
        encoded = self.wavefront(wavelet)
        return self.fuse(torch.cat([branch(encoded) for branch in self.filters], dim=1))

class AnatomicalSourceCurrent(nn.Module):

    def __init__(self, population: int, time_scale: int=1) -> None:
        super().__init__()
        base_kernels = (9, 3, 5)
        kernels = tuple((2 * int(round((kernel - 1) * time_scale / 2.0)) + 1 for kernel in base_kernels))
        self.integrators = nn.ModuleList([nn.Conv1d(population, population, kernel, padding=kernel // 2, groups=population, bias=False) for kernel in kernels])
        self.projections = nn.ModuleList([nn.Conv1d(population, population, 1) for _ in kernels])
        initial_rates = (0.05, 0.01, 0.005)
        with torch.no_grad():
            for integrator in self.integrators:
                integrator.weight.zero_()
                integrator.weight[:, 0, integrator.kernel_size[0] // 2] = 1.0
            for projection, rate in zip(self.projections, initial_rates):
                nn.init.normal_(projection.weight, mean=0.0, std=0.01)
                projection.bias.fill_(torch.logit(torch.tensor(rate)))

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return torch.cat([projection(F.silu(integrator(encoded))) for integrator, projection in zip(self.integrators, self.projections)], dim=1)

class ConservedAnatomicalSourceRouter(nn.Module):

    def __init__(self, population: int, contextual: bool=False) -> None:
        super().__init__()
        self.population = population
        self.contextual = contextual
        self.integrator = nn.Conv1d(population, population, 5, padding=2, groups=population, bias=False)
        self.magnitude = nn.Conv1d(population, population, 1, groups=1 if contextual else population)
        self.route = nn.Conv1d(population, 3 * population, 1, groups=1 if contextual else population)
        with torch.no_grad():
            self.integrator.weight.zero_()
            self.integrator.weight[:, 0, 2] = 1.0
            nn.init.normal_(self.magnitude.weight, mean=0.0, std=0.01)
            self.magnitude.bias.fill_(torch.logit(torch.tensor(0.065)))
            nn.init.normal_(self.route.weight, mean=0.0, std=0.01)
            prior = torch.tensor([0.77, 0.15, 0.08]).log()
            self.route.bias.copy_(prior.repeat(population))

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        batch, channels, steps = encoded.shape
        if channels != self.population:
            raise ValueError(f'Expected {self.population} encoded source channels, got {channels}')
        local = F.silu(self.integrator(encoded))
        magnitude = torch.sigmoid(self.magnitude(local))
        route = torch.softmax(self.route(local).view(batch, self.population, 3, steps), dim=2)
        source = magnitude.unsqueeze(2) * route
        return source.permute(0, 2, 1, 3).reshape(batch, 3 * self.population, steps)

class TransmembraneCurrentSource(nn.Module):

    def __init__(self, population: int, stages: int=10, fractional: bool=False, time_scale: int=1, kirchhoff: bool=False) -> None:
        super().__init__()
        if stages != 10:
            raise ValueError('The current-source graph is defined for ten cardiac stages')
        self.population = population
        self.stages = stages
        self.fractional = fractional
        self.kirchhoff = kirchhoff
        self.memory_steps = 31 * time_scale
        edges = ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9))
        laplacian = torch.zeros(stages, stages)
        incidence = torch.zeros(len(edges), stages)
        for edge_index, (left, right) in enumerate(edges):
            incidence[edge_index, left] = 1.0
            incidence[edge_index, right] = -1.0
            laplacian[left, left] += 1.0
            laplacian[right, right] += 1.0
            laplacian[left, right] -= 1.0
            laplacian[right, left] -= 1.0
        self.register_buffer('anatomical_laplacian', laplacian)
        self.register_buffer('anatomical_incidence', incidence)
        unit_softplus = torch.log(torch.expm1(torch.tensor(1.0)))
        self.log_capacitance = nn.Parameter(unit_softplus.repeat(stages))
        if kirchhoff:
            self.register_parameter('log_conductivity', None)
            self.log_edge_conductance = nn.Parameter(unit_softplus.repeat(len(edges)))
        else:
            self.log_conductivity = nn.Parameter(unit_softplus.repeat(stages))
            self.register_parameter('log_edge_conductance', None)
        if fractional:
            initial_unit = torch.tensor((0.75 - 0.05) / 0.94)
            self.raw_fractional_order = nn.Parameter(torch.logit(initial_unit).repeat(stages))
        else:
            self.register_parameter('raw_fractional_order', None)

    def fractional_orders(self) -> torch.Tensor:
        if self.raw_fractional_order is None:
            return self.log_capacitance.new_ones(self.stages)
        return 0.05 + 0.94 * torch.sigmoid(self.raw_fractional_order)

    def _temporal_current(self, membrane: torch.Tensor) -> torch.Tensor:
        if not self.fractional:
            return F.pad(membrane[..., 1:] - membrane[..., :-1], (1, 0))
        order = self.fractional_orders()
        coefficients = [torch.ones_like(order)]
        for lag in range(1, self.memory_steps):
            coefficients.append(coefficients[-1] * (1.0 - (order + 1.0) / float(lag)))
        weights = torch.stack(coefficients, dim=-1)
        kernels = weights.repeat_interleave(self.population, dim=0).unsqueeze(1).flip(-1)
        centered = membrane - membrane[..., :1]
        batch, _, _, steps = centered.shape
        flat = centered.reshape(batch, self.stages * self.population, steps)
        filtered = F.conv1d(F.pad(flat, (self.memory_steps - 1, 0)), kernels.to(membrane.dtype), groups=self.stages * self.population)
        return filtered.view_as(membrane)

    def _axial_current(self, membrane: torch.Tensor) -> torch.Tensor:
        if self.kirchhoff:
            edge_conductance = F.softplus(self.log_edge_conductance)
            incidence = self.anatomical_incidence.to(membrane.dtype)
            laplacian = incidence.T @ (edge_conductance.unsqueeze(1) * incidence)
        else:
            laplacian = self.anatomical_laplacian.to(membrane.dtype)
        return torch.einsum('ij,bjht->biht', laplacian, membrane)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch, channels, steps = state.shape
        if channels != self.stages * self.population:
            raise ValueError(f'Expected {self.stages * self.population} cardiac channels, got {channels}')
        membrane = state.view(batch, self.stages, self.population, steps)
        temporal = self._temporal_current(membrane)
        axial = self._axial_current(membrane)
        capacitance = F.softplus(self.log_capacitance).view(1, -1, 1, 1)
        if self.kirchhoff:
            current = capacitance * temporal + axial
        else:
            conductivity = F.softplus(self.log_conductivity).view(1, -1, 1, 1)
            current = capacitance * temporal + conductivity * axial
        return current.reshape(batch, channels, steps)

class AnatomicalMultipoleProjection(nn.Module):

    def __init__(self, population: int, stages: int=10):
        super().__init__()
        index = torch.arange(population, dtype=torch.float32)
        golden_angle = torch.pi * (3.0 - 5.0 ** 0.5)
        z = 1.0 - 2.0 * (index + 0.5) / population
        radius = (1.0 - z.square()).sqrt()
        sphere = torch.stack((radius * torch.cos(golden_angle * index), radius * torch.sin(golden_angle * index), z), dim=-1)
        orientations = torch.stack([torch.roll(sphere, shifts=stage * max(1, population // stages), dims=0) for stage in range(stages)])
        self.raw_orientation = nn.Parameter(orientations)
        self.log_amplitude = nn.Parameter(torch.zeros(stages, population))
        self.population = population
        self.stages = stages

    def project(self, state: torch.Tensor, stage_indices: tuple[int, ...] | list[int]) -> torch.Tensor:
        batch, channels, steps = state.shape
        groups = len(stage_indices)
        if channels != groups * self.population:
            raise ValueError(f'Expected {groups * self.population} channels for {groups} multipole groups, got {channels}')
        if any((index < 0 or index >= self.stages for index in stage_indices)):
            raise ValueError('Multipole stage index is out of range')
        activation = state.view(batch, groups, self.population, steps)
        index = torch.as_tensor(stage_indices, device=state.device, dtype=torch.long)
        orientation = F.normalize(self.raw_orientation.index_select(0, index), dim=-1)
        amplitude = F.softplus(self.log_amplitude.index_select(0, index)).unsqueeze(-1)
        dipole_basis = orientation
        x, y, z = orientation.unbind(dim=-1)
        quadrupole_basis = torch.stack((x.square() - y.square(), 2.0 * z.square() - x.square() - y.square(), 2.0 * x * y, 2.0 * x * z, 2.0 * y * z), dim=-1)
        quadrupole_basis = F.normalize(quadrupole_basis, dim=-1)
        moment_basis = torch.cat((dipole_basis, quadrupole_basis), dim=-1)
        moments = torch.einsum('bspt,spm->bsmt', activation, moment_basis * amplitude)
        return (moments / self.population ** 0.5).reshape(batch, 8 * groups, steps)

    def forward(self, graph: torch.Tensor) -> torch.Tensor:
        return self.project(graph, tuple(range(self.stages)))

class FreeSpatialMultipoleProjection(nn.Module):

    def __init__(self, population: int, stages: int=10):
        super().__init__()
        self.population = int(population)
        self.stages = int(stages)
        self.projector = nn.Conv1d(self.stages * self.population, 8 * self.stages, 1, bias=False)
        nn.init.normal_(self.projector.weight, mean=0.0, std=1.0 / max(1, self.population) ** 0.5)

    def project(self, state: torch.Tensor, stage_indices: tuple[int, ...]) -> torch.Tensor:
        if tuple(stage_indices) == tuple(range(self.stages)):
            return self.forward(state)
        batch, _, steps = state.shape
        stage_count = len(stage_indices)
        states = state.view(batch, self.stages, self.population, steps)
        selected = states[:, list(stage_indices)].reshape(batch, stage_count * self.population, steps)
        weight = self.projector.weight.view(8 * self.stages, self.stages, self.population, 1)
        selected_weight = weight.view(self.stages, 8, self.stages, self.population, 1)[list(stage_indices), :, list(stage_indices)]
        selected_weight = selected_weight.reshape(8 * stage_count, self.population, 1)
        return F.conv1d(selected, selected_weight, bias=None, groups=stage_count)

    def forward(self, graph: torch.Tensor) -> torch.Tensor:
        return self.projector(graph)

class MultipoleLeadField(nn.Module):

    def __init__(self, stages: int=10, response_length: int=63, shared_response_weights: bool=False):
        super().__init__()
        channels = 8 * stages
        self.response_length = response_length
        self.shared_response_weights = shared_response_weights
        if shared_response_weights:
            self.response = nn.Parameter(torch.zeros(1, 1, response_length))
        else:
            self.response = nn.Conv1d(channels, channels, response_length, groups=channels, bias=False)
        self.lead_field = nn.Conv1d(channels, 12, 1, bias=False)
        with torch.no_grad():
            if shared_response_weights:
                self.response.zero_()
                self.response[:, :, -1] = 1.0
            else:
                self.response.weight.zero_()
                self.response.weight[:, 0, -1] = 1.0
        inverse_xyz = torch.tensor([[0.156, -0.01, -0.172, -0.074, 0.122, 0.231, 0.239, 0.194], [-0.227, 0.887, 0.057, -0.019, -0.106, -0.022, 0.041, 0.048], [0.022, 0.102, -0.229, -0.31, -0.246, -0.063, 0.055, 0.108]])
        _, _, right_vectors = torch.linalg.svd(inverse_xyz, full_matrices=True)
        inverse_multipole = torch.cat((inverse_xyz, right_vectors[3:]), dim=0)
        independent_forward = torch.linalg.pinv(inverse_multipole)
        forward = torch.zeros(12, 8)
        forward[[0, 1, 6, 7, 8, 9, 10, 11]] = independent_forward
        forward[2] = forward[1] - forward[0]
        forward[3] = -(forward[0] + forward[1]) / 2.0
        forward[4] = forward[0] - forward[1] / 2.0
        forward[5] = forward[1] - forward[0] / 2.0
        with torch.no_grad():
            self.lead_field.weight.copy_(forward.repeat(1, stages).unsqueeze(-1) / stages)

    def filter_response(self, multipoles: torch.Tensor) -> torch.Tensor:
        padded = F.pad(multipoles, (self.response_length - 1, 0))
        if self.shared_response_weights:
            channels = multipoles.shape[1]
            weight = self.response.to(dtype=multipoles.dtype).expand(channels, -1, -1)
            return F.conv1d(padded, weight, groups=channels)
        return self.response(padded)

    def project_leads(self, response_state: torch.Tensor, output_length: int) -> torch.Tensor:
        leads = self.lead_field(response_state)
        return F.interpolate(leads, size=output_length, mode='linear', align_corners=False)

    def observed_components(self, response_state: torch.Tensor, observed_lead_mask: torch.Tensor | None) -> torch.Tensor:
        if response_state.ndim != 3 or response_state.shape[1] != self.lead_field.in_channels:
            raise ValueError('response_state must have shape [batch, multipoles, time]')
        batch = response_state.shape[0]
        if observed_lead_mask is None:
            lead_weight = response_state.new_ones((batch, 12))
        else:
            lead_weight = observed_lead_mask
            if lead_weight.ndim == 3:
                if lead_weight.shape[-1] != 1:
                    raise ValueError('observed_lead_mask must have shape [batch, 12, 1]')
                lead_weight = lead_weight[..., 0]
            if lead_weight.shape != (batch, 12):
                raise ValueError('observed_lead_mask must have shape [batch, 12, 1]')
            lead_weight = lead_weight.to(device=response_state.device, dtype=response_state.dtype)
        lead_weight = lead_weight / lead_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        field = self.lead_field.weight[..., 0].to(dtype=response_state.dtype)
        observed_field = torch.einsum('bl,lc->bc', lead_weight, field)
        return response_state * observed_field.unsqueeze(-1)

    def forward(self, multipoles: torch.Tensor, output_length: int) -> torch.Tensor:
        response_state = self.filter_response(multipoles)
        return self.project_leads(response_state, output_length)

class StateEvidenceReadout(nn.Module):

    def __init__(self, state_channels: int, num_classes: int, kernel_sizes: tuple[int, ...]=(15, 31, 31), dilations: tuple[int, ...]=(1, 2, 8), dendritic_compartments: int=1):
        super().__init__()
        if len(kernel_sizes) != len(dilations):
            raise ValueError('Each evidence kernel requires one dilation')
        if dendritic_compartments < 1:
            raise ValueError('At least one dendritic compartment is required')
        self.num_classes = num_classes
        self.dendritic_compartments = dendritic_compartments
        self.evidence = nn.ModuleList([nn.Conv1d(state_channels, num_classes * dendritic_compartments, kernel, padding=dilation * (kernel // 2), dilation=dilation, bias=True) for kernel, dilation in zip(kernel_sizes, dilations)])
        self.scale_logits = nn.Parameter(torch.zeros(num_classes, len(kernel_sizes)))
        self.pool_logits = nn.Parameter(torch.tensor([0.0, -1.0, 1.0]).repeat(num_classes, 1))
        self.log_temperature = nn.Parameter(torch.full((num_classes,), -1.5))
        if dendritic_compartments > 1:
            self.dendritic_threshold = nn.Parameter(torch.full((num_classes, dendritic_compartments), -1.0))
            self.dendritic_mix_logits = nn.Parameter(torch.zeros(num_classes, dendritic_compartments))
        else:
            self.register_parameter('dendritic_threshold', None)
            self.register_parameter('dendritic_mix_logits', None)

    def _integrate_dendrites(self, potential: torch.Tensor) -> torch.Tensor:
        if self.dendritic_compartments == 1:
            return potential
        batch, _, steps = potential.shape
        potential = potential.view(batch, self.num_classes, self.dendritic_compartments, steps)
        threshold = self.dendritic_threshold.view(1, self.num_classes, self.dendritic_compartments, 1)
        conductance = torch.sigmoid(2.0 * (potential.abs() - threshold))
        local_spike = potential * conductance
        mixture = torch.softmax(self.dendritic_mix_logits, dim=-1).view(1, self.num_classes, self.dendritic_compartments, 1)
        return (local_spike * mixture).sum(dim=2)

    def forward(self, state: torch.Tensor, observation_support: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if observation_support is not None:
            observation_support = F.interpolate(observation_support, size=state.shape[-1], mode='linear', align_corners=False).clamp(0.0, 1.0)
            state = state * observation_support
        scale_trajectories = torch.stack([self._integrate_dendrites(evidence(state)) for evidence in self.evidence], dim=2)
        weights = torch.softmax(self.scale_logits, dim=-1).view(1, self.scale_logits.shape[0], -1, 1)
        trajectory = (scale_trajectories * weights).sum(dim=2)
        temperature = F.softplus(self.log_temperature).view(1, -1).clamp_min(0.05)
        if observation_support is None:
            mean = trajectory.mean(dim=-1)
            variance = (trajectory.float() - mean.float().unsqueeze(-1)).square().mean(dim=-1)
            deviation = (variance.clamp_min(0.0) + 1e-06).sqrt().to(trajectory.dtype)
            transient = temperature * (torch.logsumexp(trajectory / temperature.unsqueeze(-1), dim=-1) - torch.log(trajectory.new_tensor(float(trajectory.shape[-1]))))
        else:
            count = observation_support.sum(dim=-1).clamp_min(1.0)
            mean = (trajectory * observation_support).sum(dim=-1) / count
            variance = ((trajectory - mean.unsqueeze(-1)).square() * observation_support).sum(dim=-1) / (count - 1.0).clamp_min(1.0)
            deviation = (variance.float().clamp_min(0.0) + 1e-06).sqrt().to(trajectory.dtype)
            valid = observation_support >= 0.5
            scaled = trajectory / temperature.unsqueeze(-1)
            scaled = scaled.masked_fill(~valid, -10000.0)
            valid_count = valid.sum(dim=-1).clamp_min(1).to(trajectory.dtype)
            transient = temperature * (torch.logsumexp(scaled, dim=-1) - valid_count.log())
        summaries = torch.stack((mean, deviation, transient), dim=-1)
        logits = (summaries * torch.softmax(self.pool_logits, dim=-1).unsqueeze(0)).sum(dim=-1)
        return (logits, trajectory)

class MultiCompartmentTempotronReadout(nn.Module):

    def __init__(self, state_channels: int, num_classes: int, compartments: int=4, synapse_kernel: int=7, multi_spike: bool=False, biphasic_psp: bool=False, response_mode_pooling: bool=False) -> None:
        super().__init__()
        if compartments < 1:
            raise ValueError('Tempotron requires at least one dendritic compartment')
        self.num_classes = num_classes
        self.compartments = compartments
        self.synapse_kernel = synapse_kernel
        self.multi_spike = multi_spike
        self.biphasic_psp = biphasic_psp
        self.response_mode_pooling = response_mode_pooling
        channels = num_classes * compartments
        self.synapse = nn.Sequential(nn.Conv1d(state_channels, channels, synapse_kernel, padding=0, bias=False), nn.BatchNorm1d(channels))
        if biphasic_psp:
            slow_decays = torch.tensor([0.85, 0.94, 0.98]).view(1, 3).expand(channels, -1)
            fast_decays = torch.tensor([0.5, 0.75, 0.9]).view(1, 3).expand(channels, -1)
            self.slow_decay_logits = nn.Parameter(torch.logit(slow_decays))
            self.fast_fraction_logits = nn.Parameter(torch.logit(fast_decays / slow_decays))
            self.register_parameter('decay_logits', None)
        else:
            initial_decays = torch.tensor([0.65, 0.9, 0.98]).view(1, 3).expand(channels, -1)
            self.decay_logits = nn.Parameter(torch.logit(initial_decays))
            self.register_parameter('slow_decay_logits', None)
            self.register_parameter('fast_fraction_logits', None)
        self.scale_logits = nn.Parameter(torch.zeros(num_classes, compartments, 3))
        self.dendritic_threshold = nn.Parameter(torch.full((num_classes, compartments), -1.0))
        self.dendritic_mix_logits = nn.Parameter(torch.zeros(num_classes, compartments))
        self.log_temperature = nn.Parameter(torch.full((num_classes,), -1.5))
        self.somatic_threshold = nn.Parameter(torch.zeros(num_classes))
        if response_mode_pooling:
            self.response_mode_logits = nn.Parameter(torch.tensor([0.0, -1.0, 1.0]).repeat(num_classes, 1))
        else:
            self.register_parameter('response_mode_logits', None)
        if multi_spike:
            self.output_drive_threshold = nn.Parameter(torch.ones(num_classes))
            self.output_iaf = AdaptiveIAF(num_classes, slope=8.0, mode='population')
            self.log_count_gain = nn.Parameter(torch.full((num_classes,), -1.0))
        else:
            self.register_parameter('output_drive_threshold', None)
            self.output_iaf = None
            self.register_parameter('log_count_gain', None)
        self.windows = (32, 64, 128)

    def forward(self, state: torch.Tensor, observation_support: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if observation_support is None:
            support = state.new_ones((state.shape[0], 1, state.shape[-1]))
        else:
            support = F.interpolate(observation_support, size=state.shape[-1], mode='linear', align_corners=False).clamp(0.0, 1.0)
            state = state * support
        current = self.synapse(F.pad(state, (self.synapse_kernel - 1, 0)))
        traces = []
        if self.biphasic_psp:
            slow_decays = torch.sigmoid(self.slow_decay_logits)
            fast_decays = slow_decays * torch.sigmoid(self.fast_fraction_logits)
            for basis, window in enumerate(self.windows):
                slow = slow_decays[:, basis]
                fast = fast_decays[:, basis]
                slow_trace = AdaptiveIAF._history(current, slow, window=window)
                fast_trace = AdaptiveIAF._history(current, fast, window=window)
                positions = torch.arange(window, device=state.device, dtype=state.dtype)
                kernel = slow.to(state.dtype).unsqueeze(-1) ** positions - fast.to(state.dtype).unsqueeze(-1) ** positions
                peak_response = kernel.amax(dim=-1).clamp_min(0.001)
                traces.append((slow_trace - fast_trace) / peak_response.view(1, -1, 1))
        else:
            decays = torch.sigmoid(self.decay_logits)
            for basis, window in enumerate(self.windows):
                decay = decays[:, basis]
                trace = AdaptiveIAF._history(current, decay, window=window)
                positions = torch.arange(window, device=state.device, dtype=state.dtype)
                normalizer = (decay.to(state.dtype).unsqueeze(-1) ** positions).sum(dim=-1)
                traces.append(trace / normalizer.view(1, -1, 1).clamp_min(1.0))
        trace_bank = torch.stack(traces, dim=2).view(state.shape[0], self.num_classes, self.compartments, 3, state.shape[-1])
        scale_weight = torch.softmax(self.scale_logits, dim=-1).view(1, self.num_classes, self.compartments, 3, 1)
        dendritic_voltage = (trace_bank * scale_weight).sum(dim=3)
        threshold = self.dendritic_threshold.view(1, self.num_classes, self.compartments, 1)
        conductance = torch.sigmoid(2.0 * (dendritic_voltage.abs() - threshold))
        local_voltage = dendritic_voltage * conductance
        dendritic_mix = torch.softmax(self.dendritic_mix_logits, dim=-1).view(1, self.num_classes, self.compartments, 1)
        trajectory = (local_voltage * dendritic_mix).sum(dim=2)
        temperature = F.softplus(self.log_temperature).view(1, -1).clamp_min(0.05)
        valid = support >= 0.5
        scaled = (trajectory / temperature.unsqueeze(-1)).masked_fill(~valid, -10000.0)
        valid_count = valid.sum(dim=-1).clamp_min(1).to(trajectory.dtype)
        peak = temperature * (torch.logsumexp(scaled, dim=-1) - valid_count.log())
        if self.response_mode_logits is not None:
            count = support.sum(dim=-1).clamp_min(1.0)
            mean = (trajectory * support).sum(dim=-1) / count
            variance = ((trajectory - mean.unsqueeze(-1)).square() * support).sum(dim=-1) / (count - 1.0).clamp_min(1.0)
            deviation = (variance.float().clamp_min(0.0) + 1e-06).sqrt().to(trajectory.dtype)
            response_modes = torch.stack((mean, deviation, peak), dim=-1)
            peak = (response_modes * torch.softmax(self.response_mode_logits, dim=-1).unsqueeze(0)).sum(dim=-1)
        if self.output_iaf is not None:
            output_drive = torch.sigmoid(4.0 * (trajectory - self.output_drive_threshold.view(1, -1, 1)))
            _, output_state = self.output_iaf(output_drive)
            output_count = (output_state['probability'] * support).sum(dim=-1)
            count_evidence = torch.log1p(output_count)
            peak = peak + F.softplus(self.log_count_gain).view(1, -1) * count_evidence
        logits = peak - self.somatic_threshold.view(1, -1)
        return (logits, trajectory)

class CardiacPhaseFoldedReadout(StateEvidenceReadout):

    def __init__(self, state_channels: int, num_classes: int, kernel_sizes: tuple[int, ...]=(15, 31, 31), dilations: tuple[int, ...]=(1, 2, 8), phase_count: int=5) -> None:
        super().__init__(state_channels, num_classes, kernel_sizes=kernel_sizes, dilations=dilations)
        self.phase_count = phase_count
        initial_phase = torch.full((num_classes, phase_count), -2.0)
        initial_phase[:, -1] = 2.0
        self.phase_logits = nn.Parameter(initial_phase)
        self.phase_summary_logits = nn.Parameter(torch.tensor([0.0, -1.0, 1.0]).repeat(num_classes, phase_count, 1))
        self.rate_weight = nn.Parameter(torch.zeros(num_classes, phase_count))

    def forward(self, state: torch.Tensor, phase_anchors: torch.Tensor, observation_support: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if phase_anchors.shape[1] != self.phase_count - 1:
            raise ValueError(f'Expected {self.phase_count - 1} cardiac phase anchors, got {phase_anchors.shape[1]}')
        if observation_support is None:
            support = state.new_ones((state.shape[0], 1, state.shape[-1]))
        else:
            support = F.interpolate(observation_support, size=state.shape[-1], mode='linear', align_corners=False).clamp(0.0, 1.0)
            state = state * support
        anatomical_anchors = F.interpolate(phase_anchors, size=state.shape[-1], mode='linear', align_corners=False).clamp_min(0.0) * support
        anatomical_anchors = F.avg_pool1d(anatomical_anchors, 7, stride=1, padding=3) * support
        anchors = torch.cat((anatomical_anchors, support), dim=1).detach()
        mass = anchors.sum(dim=-1, keepdim=True)
        phase_weight = anchors / mass.clamp_min(0.0001)
        scale_trajectories = torch.stack([self._integrate_dendrites(evidence(state)) for evidence in self.evidence], dim=2)
        scale_weight = torch.softmax(self.scale_logits, dim=-1).view(1, self.num_classes, -1, 1)
        trajectory = (scale_trajectories * scale_weight).sum(dim=2)
        phase_weight = phase_weight.unsqueeze(1)
        phase_mean = (trajectory.unsqueeze(2) * phase_weight).sum(dim=-1)
        phase_variance = ((trajectory.unsqueeze(2) - phase_mean.unsqueeze(-1)).square() * phase_weight).sum(dim=-1)
        phase_deviation = (phase_variance.clamp_min(0.0) + 1e-06).sqrt()
        temperature = F.softplus(self.log_temperature).view(1, -1, 1).clamp_min(0.05)
        scaled = trajectory.unsqueeze(2) / temperature.unsqueeze(-1)
        phase_transient = temperature * torch.logsumexp(scaled + phase_weight.clamp_min(1e-08).log(), dim=-1)
        summaries = torch.stack((phase_mean, phase_deviation, phase_transient), dim=-1)
        summary_weight = torch.softmax(self.phase_summary_logits, dim=-1).unsqueeze(0)
        phase_evidence = (summaries * summary_weight).sum(dim=-1)
        phase_mix = torch.softmax(self.phase_logits, dim=-1).unsqueeze(0)
        logits = (phase_evidence * phase_mix).sum(dim=-1)
        duration = support.sum(dim=-1).clamp_min(1.0)
        density = mass.squeeze(-1) / duration
        logits = logits + (density.unsqueeze(1) * self.rate_weight.unsqueeze(0)).sum(dim=-1)
        return (logits, trajectory)

class EtiologyBoundStateReadout(nn.Module):
    VALID_ROLES = ('rhythm', 'conduction', 'ectopic', 'structural', 'recovery', 'global')

    def __init__(self, role_channels: dict[str, int], class_roles: tuple[str, ...] | list[str], kernel_sizes: tuple[int, ...], dilations: tuple[int, ...]):
        super().__init__()
        self.class_roles = tuple(class_roles)
        unknown = sorted(set(self.class_roles).difference(self.VALID_ROLES))
        if unknown:
            raise ValueError(f'Unknown physiological class roles: {unknown}')
        self.role_indices: dict[str, tuple[int, ...]] = {role: tuple((index for index, value in enumerate(self.class_roles) if value == role)) for role in self.VALID_ROLES if role in self.class_roles}
        self.readouts = nn.ModuleDict({role: StateEvidenceReadout(role_channels[role], len(indices), kernel_sizes=kernel_sizes, dilations=dilations) for role, indices in self.role_indices.items()})

    def forward(self, states: dict[str, torch.Tensor], observation_support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sample = next(iter(states.values()))
        logits = sample.new_zeros((sample.shape[0], len(self.class_roles)))
        trajectory = sample.new_zeros((sample.shape[0], len(self.class_roles), sample.shape[-1]))
        for role, indices in self.role_indices.items():
            role_logits, role_trajectory = self.readouts[role](states[role], observation_support)
            index = torch.tensor(indices, device=sample.device)
            logits = logits.index_copy(1, index, role_logits)
            trajectory = trajectory.index_copy(1, index, role_trajectory)
        return (logits, trajectory)

class SoftEtiologyStateReadout(nn.Module):
    VALID_ROLES = ('rhythm', 'conduction', 'ectopic', 'structural', 'recovery')

    def __init__(self, role_channels: dict[str, int], class_roles: tuple[str, ...] | list[str], kernel_sizes: tuple[int, ...], dilations: tuple[int, ...], prior_strength: float=2.0):
        super().__init__()
        self.class_roles = tuple(class_roles)
        unknown = sorted(set(self.class_roles).difference(self.VALID_ROLES))
        if unknown:
            raise ValueError(f'Unknown physiological class roles: {unknown}')
        self.readouts = nn.ModuleDict({role: StateEvidenceReadout(role_channels[role], len(self.class_roles), kernel_sizes=kernel_sizes, dilations=dilations) for role in self.VALID_ROLES})
        route_logits = torch.zeros(len(self.class_roles), len(self.VALID_ROLES))
        role_to_index = {role: index for index, role in enumerate(self.VALID_ROLES)}
        for class_index, role in enumerate(self.class_roles):
            route_logits[class_index, role_to_index[role]] = float(prior_strength)
        self.route_logits = nn.Parameter(route_logits)

    def forward(self, states: dict[str, torch.Tensor], observation_support: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        role_outputs = [self.readouts[role](states[role], observation_support) for role in self.VALID_ROLES]
        role_logits = torch.stack([output[0] for output in role_outputs], dim=2)
        role_trajectories = torch.stack([output[1] for output in role_outputs], dim=2)
        route = torch.softmax(self.route_logits, dim=-1)
        logits = (role_logits * route.unsqueeze(0)).sum(dim=2)
        trajectory = (role_trajectories * route.view(1, *route.shape, 1)).sum(dim=2)
        return (logits, trajectory)

class PhysiologicalEvidenceFilter(nn.Sequential):

    def __init__(self, input_channels: int, width: int, kernel: int, output_channels: int=1):
        super().__init__(nn.Conv1d(input_channels, width, 1, bias=False), nn.BatchNorm1d(width), nn.SiLU(), nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width, bias=False), nn.BatchNorm1d(width), nn.SiLU(), nn.Conv1d(width, output_channels, 1))

class _WaveStateEncoder(nn.Module):

    def __init__(self, input_channels: int, width: int, support_kernel: int) -> None:
        super().__init__()
        self.width = width
        self.project = nn.Conv1d(input_channels, width, 1, bias=False)
        self.project_norm = nn.LayerNorm(width)
        self.support = nn.Conv1d(width, width, support_kernel, padding=support_kernel // 2, groups=width, bias=False)
        self.support_norm = nn.LayerNorm(width)
        self.mix = nn.Conv1d(width, width, 1)

    @staticmethod
    def _channel_norm(state: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(state.transpose(1, 2)).transpose(1, 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = F.silu(self._channel_norm(self.project(state), self.project_norm))
        supported = self.support(state)
        supported = F.silu(self._channel_norm(supported, self.support_norm))
        return state + self.mix(supported)

class _WaveGrammarBlock(nn.Module):

    def __init__(self, width: int, dilation: int, dropout: float, kernel_size: int=7) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError('Wave-grammar kernels must be positive and odd')
        self.dilation = dilation
        self.norm = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2 * dilation, dilation=dilation, groups=width, bias=False)
        self.gate = nn.Conv1d(width, 2 * width, 1)
        self.project = nn.Conv1d(width, width, 1)
        nn.init.normal_(self.project.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.project.bias)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        update = self.norm(state.transpose(1, 2)).transpose(1, 2)
        update = self.temporal(update)
        content, gate = self.gate(update).chunk(2, dim=1)
        update = self.project(F.silu(content) * torch.sigmoid(gate))
        return state + torch.sigmoid(self.residual_scale) * self.dropout(update)

class _ECRPyramidStage(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, stride: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Conv1d(input_channels, output_channels, 7, stride=stride, padding=3, bias=False)
        self.down_norm = nn.LayerNorm(output_channels)
        self.temporal = nn.Conv1d(output_channels, output_channels, 7, padding=3 * dilation, dilation=dilation, groups=output_channels, bias=False)
        self.temporal_norm = nn.LayerNorm(output_channels)
        self.expand = nn.Conv1d(output_channels, 2 * output_channels, 1)
        self.project = nn.Conv1d(2 * output_channels, output_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity() if stride == 1 and input_channels == output_channels else nn.Conv1d(input_channels, output_channels, 1, stride=stride, bias=False)

    @staticmethod
    def _channel_norm(state: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(state.transpose(1, 2)).transpose(1, 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        residual = self.skip(state)
        state = self._channel_norm(self.down(state), self.down_norm)
        state = self.temporal(F.silu(state))
        state = self._channel_norm(state, self.temporal_norm)
        state = self.project(self.dropout(F.silu(self.expand(F.silu(state)))))
        return F.silu(state + residual)

class _ECRWavePyramid(nn.Module):

    def __init__(self, input_channels: int, dropout: float) -> None:
        super().__init__()
        channels = (128, 192, 256, 320)
        strides = (1, 2, 2, 2)
        dilations = (1, 1, 2, 4)
        stages = []
        current = input_channels
        for output, stride, dilation in zip(channels, strides, dilations):
            stages.append(_ECRPyramidStage(current, output, stride, dilation, dropout))
            current = output
        self.stages = nn.ModuleList(stages)
        decoder_width = 128
        self.lateral = nn.ModuleList((nn.Conv1d(channel, decoder_width, 1) for channel in channels))
        self.smooth = nn.ModuleList((_WaveGrammarBlock(decoder_width, 1, dropout) for _ in channels))
        self.output_norm = nn.LayerNorm(decoder_width)
        self.emission = nn.Conv1d(decoder_width, 6, 1)
        with torch.no_grad():
            self.emission.weight[3:].zero_()
            self.emission.bias[:3].fill_(-2.0)
            self.emission.bias[3:].zero_()

    def forward(self, role_state: torch.Tensor) -> torch.Tensor:
        features = []
        for stage in self.stages:
            role_state = stage(role_state)
            features.append(role_state)
        fused = self.smooth[-1](self.lateral[-1](features[-1]))
        for index in range(len(features) - 2, -1, -1):
            fused = F.interpolate(fused, size=features[index].shape[-1], mode='linear', align_corners=False)
            fused = self.smooth[index](fused + self.lateral[index](features[index]))
        fused = self.output_norm(fused.transpose(1, 2)).transpose(1, 2)
        return self.emission(F.silu(fused))

class _RoleFormationProjection(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv1d(input_channels, output_channels, 1, bias=False)
        self.norm = nn.GroupNorm(1, output_channels)
        self.mix = nn.Conv1d(output_channels, output_channels, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = self.norm(self.project(state))
        return self.mix(F.silu(state))

class _FormationTransitionBlock(nn.Module):

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.norm = nn.GroupNorm(1, width)
        self.depthwise = nn.Conv1d(width, width, 7, padding=3 * dilation, dilation=dilation, groups=width, bias=False)
        self.expand = nn.Conv1d(width, 4 * width, 1)
        self.project = nn.Conv1d(4 * width, width, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.full((1, width, 1), 0.1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(self.norm(state))
        update = self.project(self.dropout(F.gelu(self.expand(update))))
        return state + self.layer_scale.to(dtype=state.dtype) * update

class _DenseFormationDecoder(nn.Module):

    def __init__(self, input_channels: int, dropout: float) -> None:
        super().__init__()
        width = 256
        self.stem = nn.Conv1d(input_channels, width, 1, bias=False)
        self.stem_norm = nn.GroupNorm(1, width)
        self.transitions = nn.ModuleList((_FormationTransitionBlock(width, dilation, dropout) for dilation in (1, 1, 2, 4, 8, 16, 4, 1)))
        self.output_norm = nn.GroupNorm(1, width)
        self.emission = nn.Conv1d(width, 6, 1)
        with torch.no_grad():
            self.emission.weight[3:].zero_()
            self.emission.bias[:3].fill_(-2.0)
            self.emission.bias[3:].zero_()

    def forward(self, role_state: torch.Tensor) -> torch.Tensor:
        state = F.gelu(self.stem_norm(self.stem(role_state)))
        for transition in self.transitions:
            state = transition(state)
        return self.emission(F.gelu(self.output_norm(state)))

class RoleBoundEventReadout(nn.Module):

    def __init__(self, population: int, time_scale: int=1, state_refinement: bool=False, spatial_refinement: bool=False, recruitment_refinement: bool=False, p_his_completion: bool=False, biatrial_refinement: bool=False, subgrid_interpolation: bool=False, subgrid_waves: tuple[str, ...] | list[str] | None=None, wave_support_ms: tuple[float, float, float] | list[float] | None=None, upsample_factor: int=1, cycle_grammar_refinement: bool=False, rhythm_conditioning: bool=False, lead_visibility_refinement: bool=False, dense_formation_refinement: bool=False, readout_width: int | None=None, dropout: float=0.0):
        super().__init__()
        if upsample_factor < 1:
            raise ValueError('upsample_factor must be positive')
        self.population = population
        self.upsample_factor = int(upsample_factor)
        self.state_refinement = state_refinement
        self.cycle_grammar_refinement = cycle_grammar_refinement
        self.rhythm_conditioning = rhythm_conditioning
        self.lead_visibility_refinement = lead_visibility_refinement
        self.dense_formation_refinement = dense_formation_refinement
        self.spatial_refinement = spatial_refinement
        self.recruitment_refinement = recruitment_refinement
        self.p_his_completion = p_his_completion
        self.biatrial_refinement = biatrial_refinement
        selected_subgrid_waves = tuple((str(wave).upper() for wave in subgrid_waves)) if subgrid_waves is not None else ('P', 'QRS', 'T') if subgrid_interpolation else ()
        unknown_subgrid_waves = set(selected_subgrid_waves).difference({'P', 'QRS', 'T'})
        if unknown_subgrid_waves:
            raise ValueError(f'Unknown subgrid waves: {sorted(unknown_subgrid_waves)}')
        self.subgrid_waves = frozenset(selected_subgrid_waves)
        self.subgrid_interpolation = bool(self.subgrid_waves)
        if (spatial_refinement or recruitment_refinement) and (not state_refinement):
            raise ValueError('spatial/recruitment refinement requires state_refinement')
        if self.subgrid_interpolation and (not state_refinement):
            raise ValueError('subgrid interpolation requires state_refinement')
        if cycle_grammar_refinement and (not state_refinement):
            raise ValueError('cycle grammar refinement requires state_refinement')
        if rhythm_conditioning and (not cycle_grammar_refinement):
            raise ValueError('rhythm conditioning requires cycle grammar refinement')
        if lead_visibility_refinement and (not spatial_refinement):
            raise ValueError('lead visibility refinement requires spatial refinement')
        if dense_formation_refinement and (not cycle_grammar_refinement):
            raise ValueError('dense formation requires cycle grammar refinement')
        if dense_formation_refinement and rhythm_conditioning:
            raise ValueError('dense formation already supplies cycle context')
        if wave_support_ms is None:
            p_kernel = 15 * time_scale + (time_scale - 1)
            qrs_kernel = 8 * time_scale + 1
            t_kernel = 31 * time_scale + (time_scale - 1)
        else:
            if len(wave_support_ms) != 3 or any((value <= 0 for value in wave_support_ms)):
                raise ValueError('wave_support_ms must contain three positive durations')
            latent_rate = 50.0 * time_scale
            kernels = [max(1, 2 * int(round((float(duration) * latent_rate / 1000.0 - 1.0) / 2.0)) + 1) for duration in wave_support_ms]
            p_kernel, qrs_kernel, t_kernel = kernels
        if state_refinement:
            p_spatial = 16 if spatial_refinement else 0
            qrs_spatial = 32 if spatial_refinement else 0
            t_spatial = 32 if spatial_refinement else 0
            p_visible = 16 if lead_visibility_refinement else 0
            qrs_visible = 32 if lead_visibility_refinement else 0
            t_visible = 32 if lead_visibility_refinement else 0
            p_recruitment = population if recruitment_refinement else 0
            p_channels = (5 if biatrial_refinement else 4) * population + p_spatial + p_visible + p_recruitment
            qrs_channels = 12 * population + qrs_spatial + qrs_visible
            t_channels = 6 * population + t_spatial + t_visible
            if cycle_grammar_refinement:
                width = int(readout_width or population)
                if width < 1:
                    raise ValueError('readout_width must be positive')
                self.readout_width = width
                if dense_formation_refinement:
                    self.role_widths = (96, 160, 128)
                    self.p = _RoleFormationProjection(p_channels, self.role_widths[0])
                    self.qrs = _RoleFormationProjection(qrs_channels, self.role_widths[1])
                    self.t = _RoleFormationProjection(t_channels, self.role_widths[2])
                else:
                    self.role_widths = (width, width, width)
                    self.p = _WaveStateEncoder(p_channels, width, p_kernel)
                    self.qrs = _WaveStateEncoder(qrs_channels, width, qrs_kernel)
                    self.t = _WaveStateEncoder(t_channels, width, t_kernel)
                if rhythm_conditioning:
                    self.cycle_clock = CardiacCycleStateMixer(4, width=64, cycle_slots=4, dilations=(1, 4, 16, 64), heads=4, dropout=dropout)
                    self.role_clock_gate = nn.Conv1d(64, 3 * width, 1)
                    nn.init.zeros_(self.role_clock_gate.weight)
                    nn.init.zeros_(self.role_clock_gate.bias)
                else:
                    self.cycle_clock = None
                    self.role_clock_gate = None
                self.cycle_pyramid = _DenseFormationDecoder(sum(self.role_widths), dropout) if dense_formation_refinement else _ECRWavePyramid(3 * width, dropout)
            else:
                self.readout_width = population
                self.p = PhysiologicalEvidenceFilter(p_channels, population, p_kernel, output_channels=2 if 'P' in self.subgrid_waves else 1)
                self.qrs = PhysiologicalEvidenceFilter(qrs_channels, population, qrs_kernel, output_channels=2 if 'QRS' in self.subgrid_waves else 1)
                self.t = PhysiologicalEvidenceFilter(t_channels, population, t_kernel, output_channels=2 if 'T' in self.subgrid_waves else 1)
            if self.subgrid_interpolation and (not cycle_grammar_refinement):
                with torch.no_grad():
                    for head in (self.p, self.qrs, self.t):
                        if head[-1].out_channels == 2:
                            head[-1].weight[1:].zero_()
                            head[-1].bias[1:].zero_()
        else:
            self.p = nn.Conv1d(population, 1, p_kernel, padding=p_kernel // 2)
            self.qrs = nn.Conv1d(2 * population, 1, qrs_kernel, padding=qrs_kernel // 2)
            self.t = nn.Conv1d(2 * population, 1, t_kernel, padding=t_kernel // 2)
        if cycle_grammar_refinement and self.subgrid_interpolation:
            self.polyphase_lift = nn.ConvTranspose1d(6, 3, kernel_size=self.upsample_factor, stride=self.upsample_factor, groups=3)
            nn.init.zeros_(self.polyphase_lift.weight)
            nn.init.zeros_(self.polyphase_lift.bias)
        else:
            self.polyphase_lift = None

    @staticmethod
    def _hermite_interpolate(evidence_and_residual: torch.Tensor, output_length: int) -> torch.Tensor:
        evidence = evidence_and_residual[:, :1]
        if output_length == evidence.shape[-1]:
            return evidence
        residual = evidence_and_residual[:, 1:2]
        output_dtype = evidence.dtype
        evidence = evidence.float()
        residual = residual.float()
        previous = torch.cat((evidence[..., :1], evidence[..., :-1]), dim=-1)
        following = torch.cat((evidence[..., 1:], evidence[..., -1:]), dim=-1)
        tangent = 0.5 * (following - previous) + 0.25 * torch.tanh(residual)
        latent_steps = evidence.shape[-1]
        position = ((torch.arange(output_length, device=evidence.device, dtype=torch.float32) + 0.5) * (float(latent_steps) / float(output_length)) - 0.5).clamp(0.0, float(latent_steps - 1))
        left = position.floor().long()
        right = (left + 1).clamp_max(latent_steps - 1)
        fraction = (position - left.to(position.dtype)).view(1, 1, -1)
        y0 = evidence.index_select(-1, left)
        y1 = evidence.index_select(-1, right)
        m0 = tangent.index_select(-1, left)
        m1 = tangent.index_select(-1, right)
        fraction2 = fraction.square()
        fraction3 = fraction2 * fraction
        h00 = 2.0 * fraction3 - 3.0 * fraction2 + 1.0
        h10 = fraction3 - 2.0 * fraction2 + fraction
        h01 = -2.0 * fraction3 + 3.0 * fraction2
        h11 = fraction3 - fraction2
        return (h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1).to(dtype=output_dtype)

    def forward(self, graph: torch.Tensor, output_length: int, membrane_phase: torch.Tensor | None=None, adaptation: torch.Tensor | None=None, av_block: torch.Tensor | None=None, bundle_block: torch.Tensor | None=None, myocardial_block: torch.Tensor | None=None, restitution: torch.Tensor | None=None, multipoles: torch.Tensor | None=None, observed_multipoles: torch.Tensor | None=None, recruitment: torch.Tensor | None=None, left_atrial: torch.Tensor | None=None, phase_anchors: torch.Tensor | None=None, observation_support: torch.Tensor | None=None) -> torch.Tensor:
        h = self.population
        output_length *= self.upsample_factor
        if self.state_refinement:
            required = (membrane_phase, adaptation, av_block, bundle_block, myocardial_block, restitution)
            if any((value is None for value in required)):
                raise ValueError('State-refined role readout requires all physiological states')
            p_completion = graph[:, h:2 * h] if self.p_his_completion else av_block
            p_state = torch.cat((graph[:, :h], membrane_phase[:, :h], adaptation[:, :h], p_completion), dim=1)
            if self.biatrial_refinement:
                if left_atrial is None:
                    raise ValueError('Biatrial P-wave readout requires left-atrial state')
                p_state = torch.cat((p_state, left_atrial), dim=1)
            qrs_state = torch.cat((graph[:, 2 * h:4 * h], graph[:, 4 * h:6 * h], membrane_phase[:, h:], adaptation[:, h:], bundle_block, myocardial_block), dim=1)
            t_state = torch.cat((graph[:, 6 * h:8 * h], graph[:, 8 * h:], restitution), dim=1)
            if self.spatial_refinement:
                if multipoles is None:
                    raise ValueError('Spatially refined role readout requires multipole states')
                p_state = torch.cat((p_state, multipoles[:, :16]), dim=1)
                qrs_state = torch.cat((qrs_state, multipoles[:, 16:48]), dim=1)
                t_state = torch.cat((t_state, multipoles[:, 48:80]), dim=1)
            if self.lead_visibility_refinement:
                if observed_multipoles is None:
                    raise ValueError('Lead-visible readout requires observed multipole states')
                p_state = torch.cat((p_state, observed_multipoles[:, :16]), dim=1)
                qrs_state = torch.cat((qrs_state, observed_multipoles[:, 16:48]), dim=1)
                t_state = torch.cat((t_state, observed_multipoles[:, 48:80]), dim=1)
            if self.recruitment_refinement:
                if recruitment is None:
                    raise ValueError('Recruitment-refined role readout requires IAF population recruitment')
                p_state = torch.cat((p_state, recruitment[:, :h]), dim=1)
        else:
            p_state = graph[:, :h]
            qrs_state = graph[:, 4 * h:6 * h]
            t_state = graph[:, 8 * h:10 * h]
        wave_evidence = (self.p(p_state), self.qrs(qrs_state), self.t(t_state))
        if self.rhythm_conditioning:
            if phase_anchors is None or observation_support is None:
                raise ValueError('Rhythm-conditioned readout requires phase anchors and support')
            assert self.cycle_clock is not None
            assert self.role_clock_gate is not None
            clock = self.cycle_clock(phase_anchors, observation_support).trajectory
            gates = self.role_clock_gate(clock).chunk(3, dim=1)
            wave_evidence = tuple((evidence * (1.0 + 0.5 * torch.tanh(gate)) for evidence, gate in zip(wave_evidence, gates)))
        if self.cycle_grammar_refinement:
            emission = self.cycle_pyramid(torch.cat(wave_evidence, dim=1))
            evidence, slope = emission.split(3, dim=1)
            wave_evidence = tuple((torch.cat((evidence[:, index:index + 1], slope[:, index:index + 1]), dim=1) for index in range(3)))
        if self.subgrid_interpolation:
            wave_names = ('P', 'QRS', 'T')
            logits = torch.cat(tuple((self._hermite_interpolate(evidence, output_length) if name in self.subgrid_waves else F.interpolate(evidence[:, :1], size=output_length, mode='linear', align_corners=False) for name, evidence in zip(wave_names, wave_evidence))), dim=1)
            if self.polyphase_lift is not None:
                role_value_slope = torch.cat(wave_evidence, dim=1)
                phase_correction = 0.5 * torch.tanh(self.polyphase_lift(role_value_slope))
                if phase_correction.shape[-1] != output_length:
                    phase_correction = F.interpolate(phase_correction, size=output_length, mode='linear', align_corners=False)
                logits = logits + phase_correction
            return logits
        logits = torch.cat(tuple((evidence[:, :1] for evidence in wave_evidence)), dim=1)
        return F.interpolate(logits, size=output_length, mode='linear', align_corners=False)

class ConductionSpike(nn.Module):

    def __init__(self, num_classes: int, task: str='classification', population_width: int=64, dropout: float=0.15, max_delay_steps: int=12, surrogate_slope: float=8.0, iaf_mode: str='straight_through', lead_field_rank: int=64, evidence_kernels: tuple[int, ...] | list[int]=(15, 31, 31), evidence_dilations: tuple[int, ...] | list[int]=(1, 2, 8), input_stride: int | None=None, physiology_time_scale: int | None=None, mask_unobserved: bool=False, electrotonic_response: bool=False, charge_conserving_response: bool=False, isoelectric_centering: bool=False, gap_junction_prepotential: bool=False, cable_coupled_iaf: bool=False, charge_residual_transport: bool=False, iaf_threshold_spread: float=0.0, state_refinement: bool=False, spatial_refinement: bool=False, recruitment_refinement: bool=False, p_his_completion: bool=False, etiology_roles: tuple[str, ...] | list[str] | None=None, source_state_refinement: bool=False, phase_geometry_refinement: bool=False, exclusive_segmentation: bool=False, limb_inverse_mode: str='pseudoinverse', observed_av_delay: bool=False, biatrial_conduction: bool=False, biatrial_av_convergence: bool=False, dendritic_compartments: int=1, cardiac_phase_folding: bool=False, polarity_invariant_drive: bool=False, anatomical_source_currents: bool=False, conserved_source_routing: bool=False, conserved_source_context: bool=False, iaf_phase_tiling: bool=False, iaf_encoder_group_size: int=1, stationary_wavelet_drive: bool=False, refractory_state_readout: bool=False, renewal_phase_response: bool=False, population_renewal_clock: bool=False, conduction_velocity_restitution: bool=False, velocity_max_extra_delay: int=3, fitzhugh_nagumo_tissue: bool=False, transmembrane_current_state: bool=False, conjugate_membrane_current_state: bool=False, atrial_voltage_current_state: bool=False, fractional_transmembrane_current_state: bool=False, kirchhoff_transmembrane_current_state: bool=False, phase_response_state_readout: bool=False, exchangeable_population_readout: bool=False, population_moment_extrema: bool=False, multipole_observability_readout: bool=False, phase_density_coordinates: bool=False, tempotron_readout: bool=False, tempotron_compartments: int=4, tempotron_multi_spike: bool=False, tempotron_biphasic_psp: bool=False, tempotron_response_modes: bool=False, subgrid_interpolation: bool=False, subgrid_waves: tuple[str, ...] | list[str] | None=None, wave_support_ms: tuple[float, float, float] | list[float] | None=None, segmentation_upsample_factor: int=1, cycle_grammar_refinement: bool=False, rhythm_conditioning: bool=False, lead_visibility_refinement: bool=False, dense_formation_refinement: bool=False, segmentation_readout_width: int | None=None, soft_etiology_routing: bool=False, etiology_prior_strength: float=2.0, bounded_group_recruitment: bool=False, adaptive_threshold: bool=True, disable_excitation_dynamics: bool=False, use_conduction_delays: bool=True, use_refractory_gating: bool=True, use_ventricular_dynamics: bool=True, shared_response_weights: bool=False, free_spatial_weights: bool=False, **_: object) -> None:
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.population_width = population_width
        if iaf_encoder_group_size < 1 or population_width % iaf_encoder_group_size:
            raise ValueError('IAF encoder group size must be a positive divisor of population width')
        if iaf_encoder_group_size > 1 and (not iaf_phase_tiling):
            raise ValueError('Grouped time encoding requires tiled IAF phases')
        if iaf_encoder_group_size > 1 and (anatomical_source_currents or conserved_source_routing or cable_coupled_iaf):
            raise ValueError('Grouped time encoding is isolated from other source/IAF ablations')
        self.iaf_encoder_group_size = iaf_encoder_group_size
        self.iaf_source_sites = population_width // iaf_encoder_group_size
        time_scale = physiology_time_scale or (2 if task == 'segmentation' else 1)
        drive_stride = input_stride or 2 // time_scale
        if drive_stride < 1:
            raise ValueError('input_stride must be positive')
        self.time_scale = time_scale
        self.mask_unobserved = mask_unobserved
        self.disable_excitation_dynamics = disable_excitation_dynamics
        self.use_electrotonic_response = electrotonic_response
        self.use_charge_conserving_response = charge_conserving_response
        if electrotonic_response and charge_conserving_response:
            raise ValueError('Choose one tissue event-to-membrane response')
        self.use_gap_junction_prepotential = gap_junction_prepotential
        self.use_cable_coupled_iaf = cable_coupled_iaf
        self.use_charge_residual_transport = charge_residual_transport
        if sum((gap_junction_prepotential, cable_coupled_iaf, charge_residual_transport)) > 1:
            raise ValueError('Choose one IAF information-refinement mechanism per ablation')
        self.source_state_refinement = source_state_refinement
        self.phase_geometry_refinement = phase_geometry_refinement
        self.exclusive_segmentation = exclusive_segmentation
        self.biatrial_conduction = biatrial_conduction
        self.refractory_state_readout = refractory_state_readout
        self.renewal_phase_response = renewal_phase_response
        self.phase_response_state_readout = task == 'classification' and phase_response_state_readout
        self.exchangeable_population_readout = task == 'classification' and exchangeable_population_readout
        self.population_moment_extrema = population_moment_extrema
        self.conjugate_membrane_current_state = conjugate_membrane_current_state
        self.atrial_voltage_current_state = atrial_voltage_current_state
        if sum((transmembrane_current_state, conjugate_membrane_current_state, atrial_voltage_current_state, fractional_transmembrane_current_state, kirchhoff_transmembrane_current_state)) > 1:
            raise ValueError('Choose one membrane/current tissue-state representation')
        if population_moment_extrema and (not exchangeable_population_readout):
            raise ValueError('Population extrema require the exchangeable population readout')
        self.multipole_observability_readout = task == 'classification' and multipole_observability_readout
        self.phase_density_coordinates = task == 'classification' and phase_density_coordinates
        if phase_response_state_readout and (not renewal_phase_response):
            raise ValueError('Phase-response state readout requires renewal phase dynamics')
        if population_renewal_clock and (not renewal_phase_response):
            raise ValueError('Population renewal clock requires renewal dynamics')
        self.tempotron_readout = task == 'classification' and tempotron_readout
        if (tempotron_multi_spike or tempotron_biphasic_psp or tempotron_response_modes) and (not self.tempotron_readout):
            raise ValueError('Tempotron refinements require the Tempotron readout')
        if stationary_wavelet_drive:
            if polarity_invariant_drive:
                raise ValueError('Wavelet and polarity drive are separate ablations')
            self.drive_encoder = StationaryWaveletCardiacDrive(population_width, dropout, stride=drive_stride, isoelectric_centering=isoelectric_centering, limb_inverse_mode=limb_inverse_mode)
        else:
            self.drive_encoder = MultiScaleCardiacDrive(population_width, dropout, stride=drive_stride, isoelectric_centering=isoelectric_centering, limb_inverse_mode=limb_inverse_mode, polarity_invariant_drive=polarity_invariant_drive)
        source_kernel = 5
        if task == 'segmentation' and time_scale != 2:
            target_kernel = 5.0 * time_scale / 2.0
            source_kernel = 2 * int(round((target_kernel - 1.0) / 2.0)) + 1
        if anatomical_source_currents and conserved_source_routing:
            raise ValueError('Anatomical source scales and conserved routing are separate ablations')
        self.conserved_source_routing = conserved_source_routing
        if conserved_source_context and (not conserved_source_routing):
            raise ValueError('Contextual source localization requires conserved source routing')
        if conserved_source_routing:
            self.source_drive = ConservedAnatomicalSourceRouter(population_width, contextual=conserved_source_context)
        elif anatomical_source_currents:
            self.source_drive = AnatomicalSourceCurrent(population_width, time_scale=time_scale)
        else:
            self.source_drive = nn.Sequential(nn.Conv1d(population_width, population_width, source_kernel, padding=source_kernel // 2, groups=population_width, bias=False), nn.SiLU(), nn.Conv1d(population_width, 3 * self.iaf_source_sites, 1))
            initial_rate = torch.tensor([0.05, 0.01, 0.005]).repeat_interleave(self.iaf_source_sites)
            nn.init.normal_(self.source_drive[-1].weight, mean=0.0, std=0.01)
            with torch.no_grad():
                self.source_drive[-1].bias.copy_(torch.logit(initial_rate))
        self.recruitment_calibrator = BoundedGroupRecruitmentCalibrator(population_width) if bounded_group_recruitment else None
        self.recovery_control = PostIAFRecoveryControl(population_width)
        iaf_args = {'sources': 3 * population_width, 'slope': surrogate_slope, 'mode': iaf_mode, 'integration_step': 1.0 / time_scale, 'threshold_spread': iaf_threshold_spread, 'phase_tiling': iaf_phase_tiling, 'phase_tile_size': iaf_encoder_group_size if iaf_encoder_group_size > 1 else None, 'adaptive_threshold': adaptive_threshold}
        self.iaf = CableCoupledIAF(population_width=population_width, **iaf_args) if cable_coupled_iaf else AdaptiveIAF(population_width=population_width, **iaf_args)
        self.prepotential_logit = nn.Parameter(torch.full((3 * population_width,), -2.2)) if gap_junction_prepotential else None
        self.residual_transport_logit = nn.Parameter(torch.full((3,), -1.1)) if charge_residual_transport else None
        self.graph = RefractoryConductionGraph(max_delay=max_delay_steps, population_size=population_width, time_scale=time_scale, observed_av_delay=observed_av_delay, biatrial_conduction=biatrial_conduction, biatrial_av_convergence=biatrial_av_convergence, renewal_phase_response=renewal_phase_response, population_renewal_clock=population_renewal_clock, conduction_velocity_restitution=conduction_velocity_restitution, velocity_max_extra_delay=velocity_max_extra_delay, fitzhugh_nagumo_tissue=fitzhugh_nagumo_tissue, use_conduction_delays=use_conduction_delays, use_refractory_gating=use_refractory_gating, use_ventricular_dynamics=use_ventricular_dynamics)
        self.electrotonic = ElectrotonicTissueResponse(population_width, time_scale=time_scale) if electrotonic_response else None
        self.charge_conserving_tissue = ChargeConservingTissueResponse(population_width, time_scale=time_scale) if charge_conserving_response else None
        self.current_source = TransmembraneCurrentSource(population_width, fractional=fractional_transmembrane_current_state, time_scale=time_scale, kirchhoff=kirchhoff_transmembrane_current_state) if transmembrane_current_state or conjugate_membrane_current_state or atrial_voltage_current_state or fractional_transmembrane_current_state or kirchhoff_transmembrane_current_state else None
        self.multipole = FreeSpatialMultipoleProjection(population_width) if free_spatial_weights else AnatomicalMultipoleProjection(population_width)
        self.decoder = MultipoleLeadField(response_length=31 * time_scale + (time_scale - 1), shared_response_weights=shared_response_weights)
        self.etiology_bound = task == 'classification' and etiology_roles is not None
        self.soft_etiology_routing = self.etiology_bound and soft_etiology_routing
        self.cardiac_phase_folding = task == 'classification' and cardiac_phase_folding
        if self.cardiac_phase_folding and self.etiology_bound:
            raise ValueError('Cardiac phase folding and etiology routing are separate ablations')
        if self.tempotron_readout and (self.cardiac_phase_folding or self.etiology_bound):
            raise ValueError('Tempotron, phase folding, and etiology routing are separate readouts')
        if self.exchangeable_population_readout and (self.etiology_bound or self.cardiac_phase_folding or self.tempotron_readout or self.phase_response_state_readout or biatrial_conduction or refractory_state_readout or source_state_refinement):
            raise ValueError('Exchangeable population readout is an isolated readout ablation')
        if self.multipole_observability_readout and (self.exchangeable_population_readout or self.etiology_bound or self.cardiac_phase_folding or self.tempotron_readout or self.phase_response_state_readout or biatrial_conduction or refractory_state_readout or source_state_refinement):
            raise ValueError('Multipole observability readout is an isolated readout ablation')
        if self.phase_density_coordinates and (self.exchangeable_population_readout or self.multipole_observability_readout or self.etiology_bound or self.cardiac_phase_folding or self.tempotron_readout or self.phase_response_state_readout):
            raise ValueError('Phase-density coordinates are an isolated state representation')
        phase_state_channels = 10 if phase_geometry_refinement else 3
        classifier_state_channels = (21 + phase_state_channels + (6 if source_state_refinement else 0) + (1 if biatrial_conduction else 0) + (5 if refractory_state_readout else 0)) * population_width + 80
        if self.conjugate_membrane_current_state:
            classifier_state_channels += 10 * population_width
        if self.phase_response_state_readout:
            classifier_state_channels += 8 * population_width + 1
        if conduction_velocity_restitution:
            classifier_state_channels += 5
        if self.phase_density_coordinates:
            classifier_state_channels += 3 * 4 * 2
        if self.exchangeable_population_readout:
            classifier_state_channels = 170 if self.population_moment_extrema else 128
        if self.multipole_observability_readout:
            classifier_state_channels = 216
        if task != 'classification':
            self.classifier = None
        elif self.etiology_bound:
            if len(etiology_roles) != num_classes:
                raise ValueError('etiology_roles must contain one role per class')
            role_channels = {'rhythm': (4 + (1 if biatrial_conduction else 0) + (1 if refractory_state_readout else 0) + (2 if self.phase_response_state_readout else 0)) * population_width + (1 if self.phase_response_state_readout else 0), 'conduction': (15 + (5 if refractory_state_readout else 0)) * population_width + 40, 'ectopic': (8 + (2 if refractory_state_readout else 0) + (4 if self.phase_response_state_readout else 0)) * population_width + 16 + (1 if self.phase_response_state_readout else 0), 'structural': 7 * population_width + 48, 'recovery': (12 + (2 if refractory_state_readout else 0) + (4 if self.phase_response_state_readout else 0)) * population_width + 48 + (1 if self.phase_response_state_readout else 0), 'global': classifier_state_channels}
            if self.soft_etiology_routing:
                self.classifier = SoftEtiologyStateReadout(role_channels, etiology_roles, kernel_sizes=tuple(evidence_kernels), dilations=tuple(evidence_dilations), prior_strength=etiology_prior_strength)
            else:
                self.classifier = EtiologyBoundStateReadout(role_channels, etiology_roles, kernel_sizes=tuple(evidence_kernels), dilations=tuple(evidence_dilations))
        elif self.tempotron_readout:
            self.classifier = MultiCompartmentTempotronReadout(classifier_state_channels, num_classes, compartments=tempotron_compartments, multi_spike=tempotron_multi_spike, biphasic_psp=tempotron_biphasic_psp, response_mode_pooling=tempotron_response_modes)
        elif self.cardiac_phase_folding:
            self.classifier = CardiacPhaseFoldedReadout(classifier_state_channels, num_classes, kernel_sizes=tuple(evidence_kernels), dilations=tuple(evidence_dilations))
        else:
            self.classifier = StateEvidenceReadout(classifier_state_channels, num_classes, kernel_sizes=tuple(evidence_kernels), dilations=tuple(evidence_dilations), dendritic_compartments=dendritic_compartments)
        self.segmenter = RoleBoundEventReadout(population_width, time_scale=time_scale, state_refinement=state_refinement, spatial_refinement=spatial_refinement, recruitment_refinement=recruitment_refinement, p_his_completion=p_his_completion, biatrial_refinement=biatrial_conduction, subgrid_interpolation=subgrid_interpolation, subgrid_waves=subgrid_waves, wave_support_ms=wave_support_ms, upsample_factor=segmentation_upsample_factor, cycle_grammar_refinement=cycle_grammar_refinement, rhythm_conditioning=rhythm_conditioning, lead_visibility_refinement=lead_visibility_refinement, dense_formation_refinement=dense_formation_refinement, readout_width=segmentation_readout_width, dropout=dropout) if task == 'segmentation' else None

    def set_training_progress(self, progress: float) -> None:
        self.iaf.set_training_progress(progress)

    def _observation_support(self, x: torch.Tensor, output_length: int, valid_time_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_time_mask is None or not self.mask_unobserved:
            return x.new_ones((x.shape[0], 1, output_length))
        if valid_time_mask.ndim == 2:
            valid_time_mask = valid_time_mask.unsqueeze(1)
        if valid_time_mask.ndim != 3 or valid_time_mask.shape[1] != 1:
            raise ValueError('valid_time_mask must have shape [batch, 1, time]')
        return F.interpolate(valid_time_mask.to(device=x.device, dtype=x.dtype), size=output_length, mode='nearest').clamp(0.0, 1.0).detach()

    def _population_moments(self, state: torch.Tensor, groups: int) -> torch.Tensor:
        batch, channels, steps = state.shape
        expected = groups * self.population_width
        if channels != expected:
            raise ValueError(f'Expected {expected} channels for {groups} populations, got {channels}')
        cells = state.view(batch, groups, self.population_width, steps)
        mean = cells.mean(dim=2)
        variance = (cells.square().mean(dim=2) - mean.square()).clamp_min(0.0)
        moments = [mean, (variance + 1e-06).sqrt()]
        if self.population_moment_extrema:
            moments.extend((cells.amax(dim=2), cells.amin(dim=2)))
        return torch.cat(moments, dim=1)

    def forward(self, x: torch.Tensor, valid_time_mask: torch.Tensor | None=None, observed_lead_mask: torch.Tensor | None=None) -> dict[str, torch.Tensor]:
        if observed_lead_mask is not None:
            if observed_lead_mask.ndim == 2:
                observed_lead_mask = observed_lead_mask.unsqueeze(-1)
            if observed_lead_mask.ndim != 3 or observed_lead_mask.shape[1] != x.shape[1]:
                raise ValueError('observed_lead_mask must have shape [batch, leads, 1]')
            x = x * observed_lead_mask.to(device=x.device, dtype=x.dtype)
        encoded = self.drive_encoder(x, observed_lead_mask=observed_lead_mask)
        observation_support = self._observation_support(x, encoded.shape[-1], valid_time_mask)
        source_projection = self.source_drive(encoded)
        site_drive = (source_projection if self.conserved_source_routing else torch.sigmoid(source_projection)) * observation_support
        if self.iaf_encoder_group_size > 1:
            drive = site_drive.view(x.shape[0], 3, self.iaf_source_sites, site_drive.shape[-1]).repeat_interleave(self.iaf_encoder_group_size, dim=2).reshape(x.shape[0], 3 * self.population_width, site_drive.shape[-1])
        else:
            drive = site_drive
        if self.recruitment_calibrator is None:
            recruitment_gain = drive.new_ones(3)
        else:
            drive = self.recruitment_calibrator(drive)
            recruitment_gain = self.recruitment_calibrator.bounded_gain().to(device=drive.device, dtype=drive.dtype)
        calibrated_source_drive = drive
        if self.disable_excitation_dynamics:
            spikes = drive.clamp(0.0, 1.0)
            iaf_state = {'probability': spikes, 'drive': drive, 'hard_spike': spikes,
                         'voltage': drive.new_zeros(drive.shape), 'phase': drive.new_zeros(drive.shape),
                         'charge_phase': drive.new_zeros(drive.shape),
                         'initial_phase': drive.new_zeros((1, drive.shape[1], 1)),
                         'adaptation': drive.new_zeros(drive.shape)}
        else:
            spikes, iaf_state = self.iaf(drive)
        encoder_spikes = spikes
        if self.iaf_encoder_group_size > 1:
            batch, _, steps = spikes.shape
            group = self.iaf_encoder_group_size
            hard_encoder = iaf_state['hard_spike'].view(batch, 3, self.iaf_source_sites, group, steps)
            probability_encoder = iaf_state['probability'].view(batch, 3, self.iaf_source_sites, group, steps)
            hard_site = (hard_encoder.sum(dim=3) > 0).to(spikes.dtype)
            probability_site = 1.0 - (1.0 - probability_encoder).prod(dim=3)
            synchronized_site = hard_site + probability_site - probability_site.detach()
            spikes = synchronized_site.repeat_interleave(group, dim=2).reshape(batch, 3 * self.population_width, steps)
            iaf_state['hard_spike'] = hard_site.repeat_interleave(group, dim=2).reshape(batch, 3 * self.population_width, steps)
            iaf_state['probability'] = probability_site.repeat_interleave(group, dim=2).reshape(batch, 3 * self.population_width, steps)
        charge_residual = None
        residual_gain = None
        if self.prepotential_logit is not None:
            gain = torch.sigmoid(self.prepotential_logit).view(1, -1, 1)
            prepotential = iaf_state['probability'] * (1.0 - iaf_state['hard_spike'])
            exact_event = iaf_state['hard_spike'] + (spikes - spikes.detach())
            transmitted_sources = (exact_event + gain * prepotential).clamp(0.0, 1.0)
        elif self.residual_transport_logit is not None:
            initial_phase = iaf_state['initial_phase'].expand(iaf_state['phase'].shape[0], -1, -1)
            previous_phase = torch.cat((initial_phase, iaf_state['phase'][..., :-1]), dim=-1)
            charge_residual = iaf_state['phase'] - previous_phase
            residual_gain = torch.sigmoid(self.residual_transport_logit).repeat_interleave(self.population_width).view(1, -1, 1)
            transmitted_sources = (spikes + residual_gain * charge_residual).clamp(0.0, 1.0)
        else:
            transmitted_sources = spikes
        conduction = self.graph(transmitted_sources, self.recovery_control(transmitted_sources, iaf_state['phase'], iaf_state['adaptation']), source_probability=iaf_state['probability'])
        event_graph = conduction.graph
        if self.electrotonic is not None:
            physiological_graph = self.electrotonic(event_graph)
        elif self.charge_conserving_tissue is not None:
            physiological_graph = self.charge_conserving_tissue(event_graph)
        else:
            physiological_graph = event_graph
        electrical_source_graph = self.current_source(physiological_graph) if self.current_source is not None else physiological_graph
        if self.conjugate_membrane_current_state:
            serial_tissue_state = torch.cat((physiological_graph, electrical_source_graph), dim=1)
        elif self.atrial_voltage_current_state:
            serial_tissue_state = torch.cat((physiological_graph[:, :self.population_width], electrical_source_graph[:, self.population_width:]), dim=1)
        else:
            serial_tissue_state = electrical_source_graph
        multipole_state = self.multipole(electrical_source_graph)
        dipole_state = multipole_state.view(x.shape[0], 10, 8, multipole_state.shape[-1])[:, :, :3].reshape(x.shape[0], 30, multipole_state.shape[-1])
        volume_response_state = self.decoder.filter_response(multipole_state)
        reconstruction = self.decoder.project_leads(volume_response_state, x.shape[-1])
        observed_multipole_state = self.decoder.observed_components(volume_response_state, observed_lead_mask)
        source_phase = iaf_state['charge_phase'] if self.iaf_encoder_group_size > 1 else iaf_state['phase']
        membrane_phase = 2.0 * source_phase - 1.0
        phase_angle = 2.0 * torch.pi * source_phase
        phase_sine = torch.sin(phase_angle)
        phase_cosine = torch.cos(phase_angle)
        h = self.population_width
        atrial_angle = phase_angle[:, :h]
        relative_angles = torch.cat((phase_angle[:, h:2 * h] - atrial_angle, phase_angle[:, 2 * h:] - atrial_angle), dim=1)
        relative_phase = torch.cat((torch.sin(relative_angles), torch.cos(relative_angles)), dim=1)
        phase_geometry = torch.cat((phase_sine, phase_cosine, relative_phase), dim=1)
        adaptation = iaf_state['adaptation'] / (1.0 + iaf_state['adaptation'])
        branches = physiological_graph[:, 2 * h:4 * h]
        myocardium = physiological_graph[:, 4 * h:6 * h]
        action_potential = physiological_graph[:, 6 * h:8 * h]
        recovery = physiological_graph[:, 8 * h:10 * h]
        av_block = conduction.av_incoming * (1.0 - conduction.av_gate)
        bundle_block = conduction.bundle_incoming * (1.0 - conduction.bundle_gate)
        myocardial_block = conduction.myocardial_incoming * (1.0 - conduction.myocardial_gate)
        laterality = myocardium[:, :h] - myocardium[:, h:]
        restitution = myocardium * recovery
        cardiac_parts = [phase_geometry if self.phase_geometry_refinement else membrane_phase, adaptation, serial_tissue_state, av_block, bundle_block, myocardial_block, laterality, restitution]
        intrinsic_rate_state = (40.0 * self.time_scale / conduction.record_intrinsic_period.clamp_min(1.0)).expand(-1, 1, physiological_graph.shape[-1])
        if self.phase_response_state_readout:
            cardiac_parts.extend((conduction.atrial_prematurity, conduction.atrial_cycle_phase, conduction.myocardial_prematurity, conduction.myocardial_cycle_phase, conduction.ventricular_prematurity, intrinsic_rate_state))
        if self.biatrial_conduction:
            cardiac_parts.append(conduction.left_atrial)
        if self.refractory_state_readout:
            cardiac_parts.extend((conduction.av_availability, conduction.bundle_availability, conduction.myocardial_availability))
        if self.source_state_refinement:
            cardiac_parts.extend((iaf_state['probability'], transmitted_sources))
        if self.graph.conduction_velocity_restitution:
            cardiac_parts.extend((conduction.av_velocity_restitution, conduction.bundle_velocity_restitution, conduction.myocardial_velocity_restitution))
        cardiac_parts.append(multipole_state)
        cell_resolved_cardiac_state = torch.cat(cardiac_parts, dim=1)
        if self.exchangeable_population_readout:
            circular_phase_state = torch.cat((phase_sine.view(x.shape[0], 3, h, phase_sine.shape[-1]).mean(dim=2), phase_cosine.view(x.shape[0], 3, h, phase_cosine.shape[-1]).mean(dim=2)), dim=1)
            cardiac_state = torch.cat((circular_phase_state, self._population_moments(adaptation, 3), self._population_moments(physiological_graph, 10), self._population_moments(av_block, 1), self._population_moments(bundle_block, 2), self._population_moments(myocardial_block, 2), self._population_moments(laterality, 1), self._population_moments(restitution, 2), multipole_state), dim=1)
        elif self.multipole_observability_readout:
            source_stages = (0, 1, 4)
            cardiac_state = torch.cat((self.multipole.project(phase_sine, source_stages), self.multipole.project(phase_cosine, source_stages), self.multipole.project(adaptation, source_stages), multipole_state, self.multipole.project(av_block, (1,)), self.multipole.project(bundle_block, (2, 3)), self.multipole.project(myocardial_block, (4, 5)), self.multipole.project(laterality, (4,)), self.multipole.project(restitution, (8, 9))), dim=1)
        else:
            cardiac_state = cell_resolved_cardiac_state
        if self.phase_density_coordinates:
            population_angles = phase_angle.view(x.shape[0], 3, h, phase_angle.shape[-1])
            phase_density = torch.cat(tuple((coordinate for harmonic in range(1, 5) for coordinate in (torch.cos(harmonic * population_angles).mean(dim=2), torch.sin(harmonic * population_angles).mean(dim=2)))), dim=1)
            cardiac_state = torch.cat((cardiac_state, phase_density), dim=1)
        rhythm_parts = [membrane_phase[:, :h], adaptation[:, :h], physiological_graph[:, :h], av_block]
        if self.biatrial_conduction:
            rhythm_parts.append(conduction.left_atrial)
        if self.refractory_state_readout:
            rhythm_parts.append(conduction.av_availability)
        if self.phase_response_state_readout:
            rhythm_parts.extend((conduction.atrial_prematurity, conduction.atrial_cycle_phase, intrinsic_rate_state))
        conduction_parts = [membrane_phase[:, h:], adaptation[:, h:], physiological_graph[:, h:6 * h], av_block, bundle_block, myocardial_block, laterality, multipole_state[:, 8:6 * 8]]
        ectopic_parts = [membrane_phase[:, 2 * h:], adaptation[:, 2 * h:], transmitted_sources[:, 2 * h:], myocardium, myocardial_block, laterality, multipole_state[:, 4 * 8:6 * 8]]
        recovery_parts = [membrane_phase[:, 2 * h:], adaptation[:, 2 * h:], physiological_graph[:, 4 * h:], myocardial_block, restitution, multipole_state[:, 4 * 8:]]
        if self.refractory_state_readout:
            conduction_parts.extend((conduction.av_availability, conduction.bundle_availability, conduction.myocardial_availability))
            ectopic_parts.append(conduction.myocardial_availability)
            recovery_parts.append(conduction.myocardial_availability)
        if self.phase_response_state_readout:
            ectopic_parts.extend((conduction.ventricular_prematurity, conduction.myocardial_cycle_phase, intrinsic_rate_state))
            recovery_parts.extend((conduction.myocardial_prematurity, conduction.myocardial_cycle_phase, intrinsic_rate_state))
        role_states = {'rhythm': torch.cat(rhythm_parts, dim=1), 'conduction': torch.cat(conduction_parts, dim=1), 'ectopic': torch.cat(ectopic_parts, dim=1), 'structural': torch.cat((physiological_graph[:, 2 * h:8 * h], laterality, multipole_state[:, 2 * 8:8 * 8]), dim=1), 'recovery': torch.cat(recovery_parts, dim=1), 'global': cardiac_state}
        cardiac_phase_anchors = torch.stack((physiological_graph[:, :h].mean(dim=1), physiological_graph[:, h:2 * h].mean(dim=1), myocardium.mean(dim=1), recovery.mean(dim=1)), dim=1)
        if self.segmenter is None:
            segment_logits = cardiac_state.new_zeros((x.shape[0], self.num_classes, x.shape[-1]))
        else:
            segment_logits = self.segmenter(electrical_source_graph, x.shape[-1], membrane_phase=membrane_phase, adaptation=adaptation, av_block=av_block, bundle_block=bundle_block, myocardial_block=myocardial_block, restitution=restitution, multipoles=multipole_state, observed_multipoles=observed_multipole_state, recruitment=iaf_state['probability'], left_atrial=conduction.left_atrial, phase_anchors=cardiac_phase_anchors, observation_support=observation_support)
        if self.classifier is None:
            global_logits = cardiac_state.new_zeros((x.shape[0], self.num_classes))
            evidence_trajectory = cardiac_state.new_zeros((x.shape[0], self.num_classes, cardiac_state.shape[-1]))
        elif self.etiology_bound:
            global_logits, evidence_trajectory = self.classifier(role_states, observation_support)
        elif self.cardiac_phase_folding:
            global_logits, evidence_trajectory = self.classifier(cardiac_state, cardiac_phase_anchors, observation_support)
        else:
            global_logits, evidence_trajectory = self.classifier(cardiac_state, observation_support)
        output = {'logits': segment_logits if self.task == 'segmentation' else global_logits, 'global_logits': global_logits, 'segment_logits': segment_logits, 'evidence_trajectory': evidence_trajectory, 'cardiac_state': cardiac_state, 'cell_resolved_cardiac_state': cell_resolved_cardiac_state, 'reconstruction': reconstruction, 'observed_multipole_state': observed_multipole_state, 'source_spikes': spikes, 'encoder_source_spikes': encoder_spikes, 'source_hard_spikes': iaf_state['hard_spike'], 'source_probability': iaf_state['probability'], 'source_drive': iaf_state['drive'], 'calibrated_source_drive': calibrated_source_drive, 'recruitment_gain': recruitment_gain, 'transmitted_sources': transmitted_sources, 'observation_support': observation_support, 'source_voltage': iaf_state['voltage'], 'phase_geometry': phase_geometry, 'relative_phase': relative_phase, 'cardiac_phase_anchors': cardiac_phase_anchors, 'graph_state': physiological_graph, 'electrical_source_state': electrical_source_graph, 'serial_tissue_state': serial_tissue_state, 'event_graph_state': event_graph, 'multipole_state': multipole_state, 'dipole_state': dipole_state, 'action_potential_state': action_potential, 'recovery_state': recovery, 'av_gate': conduction.av_gate, 'bundle_gate': conduction.bundle_gate, 'myocardial_gate': conduction.myocardial_gate, 'expected_delays': conduction.expected_delays, 'record_av_delay': conduction.record_av_delay, 'right_atrial_state': conduction.right_atrial, 'left_atrial_state': conduction.left_atrial, 'av_availability': conduction.av_availability, 'bundle_availability': conduction.bundle_availability, 'myocardial_availability': conduction.myocardial_availability, 'atrial_prematurity': conduction.atrial_prematurity, 'atrial_cycle_phase': conduction.atrial_cycle_phase, 'myocardial_prematurity': conduction.myocardial_prematurity, 'myocardial_cycle_phase': conduction.myocardial_cycle_phase, 'ventricular_prematurity': conduction.ventricular_prematurity, 'record_intrinsic_period': conduction.record_intrinsic_period, 'av_velocity_restitution': conduction.av_velocity_restitution, 'bundle_velocity_restitution': conduction.bundle_velocity_restitution, 'myocardial_velocity_restitution': conduction.myocardial_velocity_restitution, 'intrinsic_rate_state': intrinsic_rate_state, 'state_logits': global_logits}
        if 'gap_current' in iaf_state:
            output['source_gap_current'] = iaf_state['gap_current']
            output['source_uncoupled_phase'] = iaf_state['uncoupled_phase']
            output['source_uncoupled_drive'] = iaf_state['uncoupled_drive']
            output['gap_conductance'] = iaf_state['gap_conductance']
        if charge_residual is not None:
            output['source_charge_residual'] = charge_residual
            output['residual_transport_gain'] = residual_gain
        if self.task == 'segmentation' and self.exclusive_segmentation:
            background = torch.zeros_like(segment_logits[:, :1])
            output['probabilities'] = torch.softmax(torch.cat((background, segment_logits), dim=1), dim=1)[:, 1:]
        return output
