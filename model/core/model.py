from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .cycle_state import CardiacCycleStateMixer, PathologyEvidenceQueryHead
from .cardiac_core import ConductionSpike
ARCHITECTURE_VERSION = 'strict-conduction-spike-v8-heterogeneous-population'

class StrictConductionSpikeV8(ConductionSpike):
    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, num_classes: int, task: str, sample_rate: int, population_width: int=64, dropout: float=0.15, dendritic_compartments: int=1, refractory_state_readout: bool=False, adaptive_threshold: bool=True, disable_excitation_dynamics: bool=False, use_conduction_delays: bool=True, use_refractory_gating: bool=True, use_ventricular_dynamics: bool=True, shared_response_weights: bool=False, free_spatial_weights: bool=False) -> None:
        if task not in {'classification', 'segmentation'}:
            raise ValueError(f'Unsupported task: {task}')
        if sample_rate != 100:
            raise ValueError('StrictConductionSpikeV8 is locked to a 100-Hz input')
        if dendritic_compartments < 1:
            raise ValueError('dendritic_compartments must be positive')
        _legacy_readout_options = (dendritic_compartments, refractory_state_readout)
        input_stride = 1
        super().__init__(num_classes=num_classes, task=task, population_width=population_width, dropout=dropout, max_delay_steps=24, surrogate_slope=6.0 if task == 'segmentation' else 8.0, iaf_mode='population' if task == 'segmentation' else 'straight_through', iaf_threshold_spread=0.8 if task == 'segmentation' else 0.0, evidence_kernels=(15, 31, 31), evidence_dilations=(1, 2, 8), input_stride=input_stride, physiology_time_scale=2, mask_unobserved=True, isoelectric_centering=False, limb_inverse_mode='pseudoinverse', state_refinement=task == 'segmentation', spatial_refinement=task == 'segmentation', p_his_completion=task == 'segmentation', subgrid_interpolation=task == 'segmentation', subgrid_waves=('P', 'QRS', 'T') if task == 'segmentation' else None, wave_support_ms=(120.0, 100.0, 240.0), segmentation_upsample_factor=5 if task == 'segmentation' else 1, cycle_grammar_refinement=task == 'segmentation', rhythm_conditioning=False, lead_visibility_refinement=False, dense_formation_refinement=False, segmentation_readout_width=128, dendritic_compartments=1, refractory_state_readout=False, bounded_group_recruitment=True, polarity_invariant_drive=False, stationary_wavelet_drive=False, gap_junction_prepotential=False, charge_residual_transport=False, iaf_phase_tiling=False, electrotonic_response=False, transmembrane_current_state=False, adaptive_threshold=adaptive_threshold, disable_excitation_dynamics=disable_excitation_dynamics, use_conduction_delays=use_conduction_delays, use_refractory_gating=use_refractory_gating, use_ventricular_dynamics=use_ventricular_dynamics, shared_response_weights=shared_response_weights, free_spatial_weights=free_spatial_weights)
        self.classifier = None
        self.cycle_state_mixer: CardiacCycleStateMixer | None
        self.task_head: nn.Module | None
        if task == 'classification':
            self.segmenter = None
            cardiac_channels = 24 * population_width + 80
            self.cycle_state_mixer = CardiacCycleStateMixer(cardiac_channels, width=192, cycle_slots=8, dropout=dropout)
            self.task_head = PathologyEvidenceQueryHead(self.cycle_state_mixer.width, num_classes, queries_per_class=4)
        else:
            self.cycle_state_mixer = None
            self.task_head = None
        self.register_buffer('wave_salience_log_prior', torch.tensor((2.0 / 3.0, 6.0 / 5.0, 3.0 / 2.0)).log() if task == 'segmentation' else None, persistent=False)

    def forward(self, x: torch.Tensor, valid_time_mask: torch.Tensor | None=None, observed_lead_mask: torch.Tensor | None=None, *, return_states: bool=False) -> dict[str, torch.Tensor]:
        masked_x = x
        time_mask = x.new_ones((x.shape[0], 1, x.shape[-1])) if valid_time_mask is None else (valid_time_mask.unsqueeze(1) if valid_time_mask.ndim == 2 else valid_time_mask).to(device=x.device, dtype=x.dtype)
        if time_mask.shape != (x.shape[0], 1, x.shape[-1]):
            raise ValueError('valid_time_mask must have shape [batch, 1, time]')
        if observed_lead_mask is not None:
            lead_mask = observed_lead_mask.unsqueeze(-1) if observed_lead_mask.ndim == 2 else observed_lead_mask
            masked_x = masked_x * lead_mask.to(device=x.device, dtype=x.dtype)
        valid = time_mask >= 0.5
        lengths = valid.sum(dim=-1)
        if torch.any(lengths < 1):
            raise ValueError('Every ECG must contain at least one valid time sample')
        positions = torch.arange(x.shape[-1], device=x.device).view(1, 1, -1)
        if not torch.equal(valid, positions < lengths.unsqueeze(-1)):
            raise ValueError('valid_time_mask must describe one right-padded prefix')
        ordered = masked_x.masked_fill(~valid, float('inf')).sort(dim=-1).values
        lower_index = ((lengths - 1) // 2).expand(-1, x.shape[1]).unsqueeze(-1)
        upper_index = (lengths // 2).expand(-1, x.shape[1]).unsqueeze(-1)
        lower = ordered.gather(-1, lower_index).squeeze(-1)
        upper = ordered.gather(-1, upper_index).squeeze(-1)
        isoelectric = (0.5 * (lower + upper)).unsqueeze(-1).detach()
        masked_x = (masked_x - isoelectric) * time_mask
        reconstruction_target = masked_x.detach()
        canvas_length = x.shape[-1]
        model_length = int(lengths.max().item())
        masked_x = masked_x[..., :model_length]
        model_time_mask = time_mask[..., :model_length]
        output = super().forward(masked_x, valid_time_mask=model_time_mask, observed_lead_mask=observed_lead_mask)
        if self.task == 'classification':
            assert self.cycle_state_mixer is not None
            assert self.task_head is not None
            cycle_state = self.cycle_state_mixer(output['cardiac_state'], output['observation_support'])
            pathology_evidence = self.task_head(cycle_state.trajectory, output['observation_support'])
            logits = pathology_evidence.logits
            output['global_logits'] = logits
            output['pathology_attention'] = pathology_evidence.attention
            output['pathology_attentive_mean'] = pathology_evidence.attentive_mean
            output['pathology_attentive_std'] = pathology_evidence.attentive_std
            output['cycle_state_trajectory'] = cycle_state.trajectory
            output['cycle_memory'] = cycle_state.cycle_memory
            output['cycle_record_state'] = cycle_state.record_state
            output['cycle_projection_energy'] = cycle_state.projection_energy
        else:
            logits = output['segment_logits']
            if not self.training:
                assert self.wave_salience_log_prior is not None
                logits = logits + self.wave_salience_log_prior.to(dtype=logits.dtype).view(1, -1, 1)
        output['logits'] = logits
        output['state_logits'] = output['global_logits']
        for name in ('logits', 'reconstruction'):
            value = output[name]
            if value.ndim == 3 and value.shape[-1] < canvas_length:
                output[name] = F.pad(value, (0, canvas_length - value.shape[-1]))
        output['reconstruction_target'] = reconstruction_target
        if return_states:
            return output
        return {name: output[name] for name in ('logits', 'reconstruction', 'reconstruction_target')}
