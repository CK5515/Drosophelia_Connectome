"""Spectral mixture-of-experts: one damped-wave expert per graph-frequency band, gated by input.

Each layer transforms the node signal into the graph-Fourier domain, evolves each band with its own
expert's damped wave dynamics, mixes the bands with an input-dependent gate, returns to node space,
and applies a pointwise nonlinearity. Every weight is shared across neurons: the model can only tell
neurons apart through the eigenbasis and through where the stimulus enters.
"""
import contextlib
import math
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flybrain.models.wave import wave_solve, zoh_propagator

STIM_SCALE = 200.0     # Hz; the stimulus sampler's maximum rate


def inverse_softplus(y: float) -> float:
    return float(math.log(math.expm1(y)))


@dataclass(frozen=True)
class MoEConfig:
    n_layers: int = 2
    channels: int = 8
    n_experts: int = 4
    gate_hidden: int = 32
    n_stim: int = 5
    nonlinear: bool = True
    input_dependent_gate: bool = True
    readout_softplus: bool = True
    gamma_init: float = 0.3
    alpha_init: float = 1.0
    init_noise: float = 0.01


def moe_full() -> MoEConfig:
    return MoEConfig()


def moe_linear() -> MoEConfig:
    """Control for the non-collapse criterion: affine in the stimulus, not purely linear.

    It keeps its biases (an intercept) on purpose. It is meant to be the strongest model that is still
    linear in the stimulus; an intercept lets it fit the mean offset, which makes the control stronger
    and the criterion more conservative. Dropping the biases would weaken it and flatter the full model.
    """
    return replace(MoEConfig(), nonlinear=False, input_dependent_gate=False, readout_softplus=False)


def moe_single() -> MoEConfig:
    return replace(MoEConfig(), n_experts=1, input_dependent_gate=False)


class SpectralMoE(nn.Module):
    def __init__(self, eigenvalues, eigenvectors, bands, input_idx, group_of_input,
                 cfg: MoEConfig = MoEConfig()):
        super().__init__()
        self.cfg = cfg
        u = torch.as_tensor(eigenvectors)
        if not u.is_complex():
            u = u.to(torch.complex64)
        mu = torch.as_tensor(eigenvalues).to(u.real.dtype)
        if cfg.n_experts == 1:
            bands = [np.arange(mu.numel())]
        if len(bands) != cfg.n_experts:
            raise ValueError(f"{len(bands)} bands for {cfg.n_experts} experts")
        band_of_mode = torch.zeros(mu.numel(), dtype=torch.long)
        for k, idx in enumerate(bands):
            band_of_mode[torch.as_tensor(np.asarray(idx), dtype=torch.long)] = k

        # Structural data, not learned state: non-persistent so that load_state_dict never overwrites
        # a model's own basis or input wiring (e.g. a relabelled copy) with another model's.
        self.register_buffer("U", u.contiguous(), persistent=False)
        self.register_buffer("Uh", u.conj().T.contiguous(), persistent=False)
        self.register_buffer("mu", mu, persistent=False)
        self.register_buffer("band_of_mode", band_of_mode, persistent=False)
        self.register_buffer("input_idx", torch.as_tensor(np.asarray(input_idx), dtype=torch.long),
                             persistent=False)
        self.register_buffer("group_of_input",
                             torch.as_tensor(np.asarray(group_of_input), dtype=torch.long),
                             persistent=False)
        # Test-time band switch (see knockout); not learned state, so it stays out of checkpoints.
        self.register_buffer("band_scale", torch.ones(cfg.n_experts), persistent=False)

        c, k = cfg.channels, cfg.n_experts
        g0, a0 = inverse_softplus(cfg.gamma_init), inverse_softplus(cfg.alpha_init)
        noise = cfg.init_noise
        self.gamma_tilde = nn.ParameterList(
            [nn.Parameter(torch.full((k, c), g0) + noise * abs(g0) * torch.randn(k, c))
             for _ in range(cfg.n_layers)])
        self.alpha_tilde = nn.ParameterList(
            [nn.Parameter(torch.full((k, c), a0) + noise * abs(a0) * torch.randn(k, c))
             for _ in range(cfg.n_layers)])
        self.mix_real = nn.ParameterList([nn.Parameter(torch.randn(c, c) / math.sqrt(c)) for _ in range(cfg.n_layers)])
        self.mix_imag = nn.ParameterList([nn.Parameter(torch.randn(c, c) / math.sqrt(c)) for _ in range(cfg.n_layers)])
        self.mix_bias = nn.ParameterList([nn.Parameter(torch.zeros(c)) for _ in range(cfg.n_layers)])
        if cfg.input_dependent_gate:
            self.gate = nn.ModuleList([nn.Sequential(nn.Linear(k + cfg.n_stim, cfg.gate_hidden),
                                                     nn.GELU(),
                                                     nn.Linear(cfg.gate_hidden, k))
                                       for _ in range(cfg.n_layers)])
        else:
            self.gate_logits = nn.ParameterList([nn.Parameter(torch.zeros(k)) for _ in range(cfg.n_layers)])
        self.lift = nn.Parameter(torch.ones(c) / math.sqrt(c))
        self.readout = nn.Parameter(torch.randn(c) / math.sqrt(c))
        self.readout_bias = nn.Parameter(torch.zeros(1))

    def _apply(self, fn, *args, **kwargs):
        # Module.to(float64) casts complex buffers to real and silently drops the imaginary part.
        # Hold the basis out of the cast and restore it as the complex dtype matching mu.
        u, uh = self._buffers.pop("U"), self._buffers.pop("Uh")
        try:
            super()._apply(fn, *args, **kwargs)
        finally:
            ref = self._buffers["mu"]
            cdtype = torch.complex128 if ref.dtype == torch.float64 else torch.complex64
            self._buffers["U"] = u.to(device=ref.device, dtype=cdtype)
            self._buffers["Uh"] = uh.to(device=ref.device, dtype=cdtype)
        return self

    # --- transforms -------------------------------------------------------
    def _to_spectral(self, h: torch.Tensor) -> torch.Tensor:
        b, n, c, t = h.shape
        x = h.permute(1, 0, 2, 3).reshape(n, b * c * t).to(self.Uh.dtype)
        return (self.Uh @ x).reshape(-1, b, c, t).permute(1, 0, 2, 3)

    def _to_node(self, x_hat: torch.Tensor) -> torch.Tensor:
        b, m, c, t = x_hat.shape
        x = x_hat.permute(1, 0, 2, 3).reshape(m, b * c * t)
        return (self.U @ x).reshape(-1, b, c, t).permute(1, 0, 2, 3)

    # --- pieces -----------------------------------------------------------
    def _node_drive(self, stim: torch.Tensor, n_bins: int, stim_bins: int) -> torch.Tensor:
        rates = stim[:, self.group_of_input] / STIM_SCALE                       # (B, n_in)
        on = (torch.arange(n_bins, device=stim.device) < stim_bins).to(stim.dtype)
        drive_in = rates[:, :, None] * on                                       # (B, n_in, T)
        out = torch.zeros(stim.shape[0], self.U.shape[0], self.cfg.channels, n_bins,
                          dtype=stim.dtype, device=stim.device)
        out[:, self.input_idx] = drive_in[:, :, None, :] * self.lift.view(1, 1, -1, 1)
        return out

    def _band_energies(self, h_hat: torch.Tensor) -> torch.Tensor:
        power = h_hat.abs().pow(2).sum(dim=(2, 3))                              # (B, M)
        per_band = torch.zeros(h_hat.shape[0], self.cfg.n_experts,
                               dtype=power.dtype, device=power.device)
        per_band.index_add_(1, self.band_of_mode, power)
        return torch.log(per_band + 1e-8)

    def _gate_weights(self, h_hat: torch.Tensor, stim: torch.Tensor, layer: int) -> torch.Tensor:
        if not self.cfg.input_dependent_gate:
            weights = F.softmax(self.gate_logits[layer], dim=0).expand(stim.shape[0], -1)
        else:
            features = torch.cat([self._band_energies(h_hat), stim / STIM_SCALE], dim=-1)
            weights = F.softmax(self.gate[layer](features), dim=-1)
        return weights * self.band_scale.to(weights.dtype)

    @contextlib.contextmanager
    def knockout(self, expert: int):
        """Switch one expert off at test time without renormalising the others."""
        saved = self.band_scale.clone()
        self.band_scale[expert] = 0.0
        try:
            yield
        finally:
            self.band_scale.copy_(saved)

    def forward(self, stim: torch.Tensor, n_bins: int, stim_bins: int, return_aux: bool = False):
        h_node = self._node_drive(stim, n_bins, stim_bins)
        h_hat = self._to_spectral(h_node)
        gates = []
        for layer in range(self.cfg.n_layers):
            g = self._gate_weights(h_hat, stim, layer)
            gates.append(g)
            gamma = F.softplus(self.gamma_tilde[layer])[self.band_of_mode]       # (M, C)
            alpha = F.softplus(self.alpha_tilde[layer])[self.band_of_mode]
            x_hat = wave_solve(h_hat, zoh_propagator(self.mu, gamma, alpha))
            x_hat = x_hat * g[:, self.band_of_mode].unsqueeze(-1).unsqueeze(-1).to(x_hat.dtype)
            y = self._to_node(x_hat)
            z = (torch.einsum("bnct,dc->bndt", y.real, self.mix_real[layer])
                 + torch.einsum("bnct,dc->bndt", y.imag, self.mix_imag[layer])
                 + self.mix_bias[layer].view(1, 1, -1, 1))
            h_node = z + h_node
            if self.cfg.nonlinear:
                h_node = F.gelu(h_node)
            if layer < self.cfg.n_layers - 1:
                h_hat = self._to_spectral(h_node)
        out = torch.einsum("bnct,c->bnt", h_node, self.readout) + self.readout_bias
        if self.cfg.readout_softplus:
            out = F.softplus(out)
        return (out, {"gates": gates}) if return_aux else out
