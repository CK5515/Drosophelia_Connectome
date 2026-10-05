"""Full dense eigendecomposition of a subcircuit Laplacian, with sanity checks and caching."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from flybrain.spectral.laplacian import signed_magnetic_laplacian, signed_symmetric_laplacian


@dataclass(frozen=True)
class SpectralBasis:
    kind: str                 # "symmetric" | "magnetic"
    q: float                  # 0.0 for symmetric
    eigenvalues: np.ndarray   # (N,) float64, ascending
    eigenvectors: np.ndarray  # (N, N) float32 or complex64; column j is mode j
    nodes: np.ndarray | None = None  # (N,) full-connectome indices of the subcircuit ordering, if known


def compute_basis(W: sp.spmatrix, kind: str, q: float = 0.25, device: str = "cuda",
                  nodes: np.ndarray | None = None) -> SpectralBasis:
    dense = sp.csr_matrix(W).toarray().astype(np.float64)
    if kind == "symmetric":
        L, q, store = signed_symmetric_laplacian(dense), 0.0, np.float32
    elif kind == "magnetic":
        L, store = signed_magnetic_laplacian(dense, q), np.complex64
    else:
        raise ValueError(f"unknown basis kind: {kind}")
    evals, evecs = torch.linalg.eigh(torch.as_tensor(L, device=device))
    return SpectralBasis(kind=kind, q=float(q), eigenvalues=evals.cpu().numpy().astype(np.float64),
                         eigenvectors=evecs.cpu().numpy().astype(store),
                         nodes=None if nodes is None else np.asarray(nodes))


def basis_checks(basis: SpectralBasis, device: str = "cuda", tol_eig: float = 1e-6,
                 tol_orth: float = 1e-3) -> dict:
    U = torch.as_tensor(basis.eigenvectors, device=device)
    gram = U.conj().T @ U
    err = float((gram - torch.eye(U.shape[0], dtype=U.dtype, device=device)).abs().max())
    lo, hi = float(basis.eigenvalues.min()), float(basis.eigenvalues.max())
    return {"eig_min": lo, "eig_max": hi, "eig_in_range": bool(lo >= -tol_eig and hi <= 2 + tol_eig),
            "orthonormality_max_err": err, "orthonormal": bool(err < tol_orth)}


def save_basis(basis: SpectralBasis, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {} if basis.nodes is None else {"nodes": basis.nodes}
    np.savez(path, kind=np.array(basis.kind), q=np.array(basis.q), eigenvalues=basis.eigenvalues,
             eigenvectors=basis.eigenvectors, **extra)


def load_basis(path: Path) -> SpectralBasis:
    z = np.load(path)
    return SpectralBasis(kind=str(z["kind"]), q=float(z["q"]), eigenvalues=z["eigenvalues"],
                         eigenvectors=z["eigenvectors"], nodes=z["nodes"] if "nodes" in z.files else None)
