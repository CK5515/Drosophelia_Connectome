"""Baselines: the floor, brute force, and a standard spatio-temporal graph network."""
import math
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

STIM_SCALE = 200.0


class MeanPredictor:
    """Per-neuron, per-bin mean of the training conditions. The floor every model must clear."""

    def __init__(self, mean_trace: np.ndarray):
        self.mean_trace = np.asarray(mean_trace, dtype=np.float32)

    @classmethod
    def fit(cls, rates: np.ndarray) -> "MeanPredictor":
        return cls(np.asarray(rates, dtype=np.float32).mean(axis=0))

    def predict(self, n: int) -> np.ndarray:
        return np.broadcast_to(self.mean_trace, (n, *self.mean_trace.shape)).copy()


class MLPBaseline(nn.Module):
    """Stimulus -> every scored neuron x bin. Cannot generalise to held-out neurons by construction."""

    def __init__(self, n_stim: int, n_out: int, n_bins: int, hidden: int = 512):
        super().__init__()
        self.n_out, self.n_bins = n_out, n_bins
        self.net = nn.Sequential(nn.Linear(n_stim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_out * n_bins))

    def forward(self, stim: torch.Tensor) -> torch.Tensor:
        flat = self.net(stim / STIM_SCALE)
        return F.softplus(flat.view(stim.shape[0], self.n_out, self.n_bins))


class ChebGRUBaseline(nn.Module):
    """GRU over bins whose matrix multiplications are order-K Chebyshev filters of L_sym - I."""

    def __init__(self, laplacian, input_idx, group_of_input, hidden: int = 16, order: int = 3):
        super().__init__()
        lap = sp.csr_matrix(laplacian) - sp.eye(sp.csr_matrix(laplacian).shape[0])
        coo = lap.tocoo()
        idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        val = torch.tensor(coo.data, dtype=torch.float32)
        # Graph structure is not learnable: keep it out of the state dict (as the MoE does).
        # Built once, CSR, as a non-persistent buffer so it follows .to(device) without entering the
        # checkpoint. One CPU forward at N=5000, ~1M nnz, hidden 16, 50 bins, batch 4 took:
        # 44.2 s rebuilding COO per call, 20.9 s with cached COO, 1.23 s with cached CSR.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*CSR.*beta.*")  # only torch's beta notice
            lap_sparse = torch.sparse_coo_tensor(idx, val, tuple(lap.shape)).coalesce().to_sparse_csr()
        self.register_buffer("lap_sparse", lap_sparse, persistent=False)
        self.n_nodes = lap.shape[0]
        self.hidden, self.order = hidden, order
        self.register_buffer("input_idx", torch.as_tensor(np.asarray(input_idx), dtype=torch.long),
                             persistent=False)
        self.register_buffer("group_of_input", torch.as_tensor(np.asarray(group_of_input), dtype=torch.long),
                             persistent=False)
        in_dim = 1 + hidden
        def block():
            return nn.Parameter(torch.randn(order, in_dim, hidden) / math.sqrt(in_dim * order))
        self.w_z, self.w_r, self.w_h = block(), block(), block()
        self.b_z = nn.Parameter(torch.zeros(hidden))
        self.b_r = nn.Parameter(torch.zeros(hidden))
        self.b_h = nn.Parameter(torch.zeros(hidden))
        self.readout = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.readout_bias = nn.Parameter(torch.zeros(1))

    def _spmm(self, y: torch.Tensor) -> torch.Tensor:
        b, n, d = y.shape
        flat = y.permute(1, 0, 2).reshape(n, b * d)
        return torch.sparse.mm(self.lap_sparse, flat).reshape(n, b, d).permute(1, 0, 2)

    def _cheb(self, y: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        t0 = y
        t1 = self._spmm(y)
        out = t0 @ weights[0] + t1 @ weights[1]
        prev, cur = t0, t1
        for k in range(2, self.order):
            nxt = 2.0 * self._spmm(cur) - prev
            out = out + nxt @ weights[k]
            prev, cur = cur, nxt
        return out + bias

    def forward(self, stim: torch.Tensor, n_bins: int, stim_bins: int) -> torch.Tensor:
        b = stim.shape[0]
        rates = stim[:, self.group_of_input] / STIM_SCALE
        h = torch.zeros(b, self.n_nodes, self.hidden, dtype=stim.dtype, device=stim.device)
        outputs = []
        for t in range(n_bins):
            x = torch.zeros(b, self.n_nodes, 1, dtype=stim.dtype, device=stim.device)
            if t < stim_bins:
                x[:, self.input_idx, 0] = rates
            inp = torch.cat([x, h], dim=-1)
            z = torch.sigmoid(self._cheb(inp, self.w_z, self.b_z))
            r = torch.sigmoid(self._cheb(inp, self.w_r, self.b_r))
            cand = torch.tanh(self._cheb(torch.cat([x, r * h], dim=-1), self.w_h, self.b_h))
            h = (1.0 - z) * h + z * cand
            outputs.append(F.softplus(h @ self.readout + self.readout_bias))
        return torch.stack(outputs, dim=-1)
