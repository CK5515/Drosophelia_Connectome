"""Batched GPU port of Shiu et al.'s whole-brain LIF model (same equations, constants and step order)."""
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch


@dataclass(frozen=True)
class LIFParams:
    dt_ms: float = 0.1
    v0_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_th_mv: float = -45.0
    tau_mbr_ms: float = 20.0
    tau_syn_ms: float = 5.0
    t_refrac_ms: float = 2.2
    t_delay_ms: float = 1.8
    w_syn_mv: float = 0.275
    f_poisson: float = 250.0

    @property
    def refrac_steps(self) -> int:
        return int(round(self.t_refrac_ms / self.dt_ms))

    @property
    def delay_steps(self) -> int:
        return int(round(self.t_delay_ms / self.dt_ms))


def propagator(params: LIFParams) -> np.ndarray:
    """Exact update of [v - v0, g] over one dt for dv/dt = (v0 - v + g)/t_mbr, dg/dt = -g/tau."""
    A = np.array([[-1.0 / params.tau_mbr_ms, 1.0 / params.tau_mbr_ms],
                  [0.0, -1.0 / params.tau_syn_ms]])
    return scipy.linalg.expm(A * params.dt_ms)


class LIFNetwork:
    def __init__(self, W: sp.spmatrix, params: LIFParams = LIFParams(), device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        W = sp.csr_matrix(W)
        W.sort_indices()
        self.n = W.shape[0]
        self.params = params
        self.device = torch.device(device)
        self.dtype = dtype
        self.indptr = torch.as_tensor(W.indptr.astype(np.int64), device=self.device)
        self.indices = torch.as_tensor(W.indices.astype(np.int64), device=self.device)
        self.weights = torch.as_tensor(W.data.astype(np.float64) * params.w_syn_mv, dtype=dtype, device=self.device)
        self.P = propagator(params)

    def _deliver(self, g: torch.Tensor, not_ref: torch.Tensor, batch_idx: torch.Tensor,
                 pre_idx: torch.Tensor) -> None:
        if pre_idx.numel() == 0:
            return
        starts = self.indptr[pre_idx]
        n_out = self.indptr[pre_idx + 1] - starts
        total = int(n_out.sum())
        if total == 0:
            return
        owner = torch.repeat_interleave(torch.arange(len(pre_idx), device=self.device), n_out, output_size=total)
        first = torch.cumsum(n_out, 0) - n_out
        offsets = torch.arange(total, device=self.device) - first[owner] + starts[owner]
        flat = batch_idx[owner] * self.n + self.indices[offsets]
        # Brian2 discards synaptic input that arrives while the target is refractory
        g.view(-1).index_add_(0, flat, self.weights[offsets] * not_ref.view(-1)[flat].to(self.dtype))

    @torch.no_grad()
    def run(self, input_idx, n_steps: int, record_idx, bin_steps: int, *, rates_hz=None, stim_steps=None,
            input_events=None, seed: int = 0) -> torch.Tensor:
        if (rates_hz is None) == (input_events is None):
            raise ValueError("pass exactly one of rates_hz or input_events")
        p, dev = self.params, self.device
        input_idx = torch.as_tensor(np.asarray(input_idx), dtype=torch.long, device=dev)
        record_idx = torch.as_tensor(np.asarray(record_idx), dtype=torch.long, device=dev)
        if rates_hz is not None:
            rates_hz = torch.as_tensor(rates_hz, dtype=self.dtype, device=dev)
            batch = rates_hz.shape[0]
            p_event = rates_hz * (p.dt_ms * 1e-3)
            stimulated = rates_hz > 0
            stim_steps = n_steps if stim_steps is None else stim_steps
        else:
            input_events = torch.as_tensor(np.asarray(input_events), dtype=torch.bool, device=dev)
            batch = input_events.shape[1]
            stimulated = torch.ones((batch, len(input_idx)), dtype=torch.bool, device=dev)

        n, n_bins = self.n, n_steps // bin_steps
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        p00, p01, p11 = float(self.P[0, 0]), float(self.P[0, 1]), float(self.P[1, 1])
        kick = p.w_syn_mv * p.f_poisson

        v = torch.full((batch, n), p.v0_mv, dtype=self.dtype, device=dev)
        g = torch.zeros_like(v)
        last_spike = torch.full((batch, n), -(10 ** 6), dtype=torch.int32, device=dev)
        refrac = torch.full((batch, n), p.refrac_steps, dtype=torch.int32, device=dev)
        refrac[:, input_idx] = (~stimulated).to(torch.int32) * p.refrac_steps
        ring = [None] * p.delay_steps
        counts = torch.zeros((batch, n_bins, len(record_idx)), dtype=torch.int16, device=dev)

        for s in range(n_steps):
            # 1. state update (frozen while refractory)
            not_ref = (s - last_spike) >= refrac
            u = v - p.v0_mv
            v = torch.where(not_ref, p00 * u + p01 * g + p.v0_mv, v)
            g = torch.where(not_ref, p11 * g, g)
            # 2. threshold
            spikes = (v > p.v_th_mv) & not_ref
            last_spike.masked_fill_(spikes, s)
            # 3. synapses: delayed recurrent delivery, then external kicks
            slot = s % p.delay_steps
            arriving = ring[slot]
            ring[slot] = spikes.nonzero(as_tuple=True)
            if arriving is not None:
                self._deliver(g, not_ref, *arriving)
            if input_events is not None:
                events = input_events[s]
            elif s < stim_steps:
                events = torch.rand(p_event.shape, generator=gen, device=dev, dtype=self.dtype) < p_event
            else:
                events = None
            if events is not None:
                v[:, input_idx] += events.to(self.dtype) * kick
            # 4. reset
            v.masked_fill_(spikes, p.v_reset_mv)
            g.masked_fill_(spikes, 0.0)
            b = s // bin_steps
            if b < n_bins:
                counts[:, b] += spikes[:, record_idx].to(torch.int16)
        return counts
