"""Exact per-bin solution of the damped graph wave equation, one oscillator per (mode, channel).

Mode i with eigenvalue mu_i obeys  x'' + 2*gamma*x' + alpha*mu_i*x = f(t).  The drive is held
constant across a bin (zero-order hold), so one matrix exponential of the augmented system gives an
exact update covering the underdamped, critically damped and overdamped regimes without branching.
"""
import numpy as np
import torch


def zoh_propagator(mu: torch.Tensor, gamma: torch.Tensor, alpha: torch.Tensor,
                   dt: float = 1.0) -> torch.Tensor:
    if gamma.shape != alpha.shape:
        raise ValueError(f"gamma {tuple(gamma.shape)} and alpha {tuple(alpha.shape)} must match")
    stiffness = alpha * mu.reshape(-1, *([1] * (gamma.dim() - 1)))      # (M, C)
    zero = torch.zeros_like(stiffness)
    one = torch.ones_like(stiffness)
    rows = [
        torch.stack([zero, one, zero], dim=-1),
        torch.stack([-stiffness, -2.0 * gamma, one], dim=-1),
        torch.stack([zero, zero, zero], dim=-1),
    ]
    a = torch.stack(rows, dim=-2) * dt                                  # (M, C, 3, 3)
    return torch.linalg.matrix_exp(a)[..., :2, :]                       # (M, C, 2, 3)


def wave_solve(drive: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
    p = prop.to(drive.dtype) if drive.is_complex() else prop
    p00, p01, q0 = p[..., 0, 0], p[..., 0, 1], p[..., 0, 2]
    p10, p11, q1 = p[..., 1, 0], p[..., 1, 1], p[..., 1, 2]
    x = torch.zeros(drive.shape[:-1], dtype=drive.dtype, device=drive.device)
    v = torch.zeros_like(x)
    out = []
    for t in range(drive.shape[-1]):
        f = drive[..., t]
        x, v = p00 * x + p01 * v + q0 * f, p10 * x + p11 * v + q1 * f
        out.append(x)
    return torch.stack(out, dim=-1)


def impulse_response(mu: float, gamma: float, alpha: float, n_steps: int,
                     dt: float = 1.0) -> np.ndarray:
    prop = zoh_propagator(torch.tensor([float(mu)], dtype=torch.float64),
                          torch.tensor([[float(gamma)]], dtype=torch.float64),
                          torch.tensor([[float(alpha)]], dtype=torch.float64), dt=dt)
    drive = torch.zeros(1, 1, 1, n_steps, dtype=torch.float64)
    drive[..., 0] = 1.0
    return wave_solve(drive, prop)[0, 0, 0].numpy()
