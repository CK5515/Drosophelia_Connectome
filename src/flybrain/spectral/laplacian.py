"""Signed normalized Laplacians of the subcircuit (degrees from |A_s| so the spectrum stays in [0, 2])."""
import numpy as np
import scipy.sparse as sp


def _normalized_adjacency(W: np.ndarray) -> np.ndarray:
    A_s = 0.5 * (W + W.T)
    degree = np.abs(A_s).sum(axis=1)
    if np.any(degree == 0):
        raise ValueError(f"zero-degree nodes after symmetrization: {np.flatnonzero(degree == 0)[:10].tolist()}")
    d = 1.0 / np.sqrt(degree)
    return d[:, None] * A_s * d[None, :]


def signed_symmetric_laplacian(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    return np.eye(W.shape[0]) - _normalized_adjacency(W)


def signed_magnetic_laplacian(W: np.ndarray, q: float = 0.25) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    B = (W != 0).astype(np.float64)
    theta = 2.0 * np.pi * q * (B - B.T)
    return np.eye(W.shape[0]) - _normalized_adjacency(W) * np.exp(1j * theta)


def opposite_sign_reciprocal_pairs(W: sp.spmatrix) -> int:
    W = sp.csr_matrix(W)
    product = W.multiply(W.T)            # (i,j) entry is W_ij * W_ji
    return int((product.data < 0).sum() // 2)
