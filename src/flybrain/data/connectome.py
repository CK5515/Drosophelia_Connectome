"""The whole-brain signed, directed connectome as a sparse matrix."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flybrain.paths import PROCESSED


@dataclass(frozen=True)
class Connectome:
    W: sp.csr_matrix          # (N, N) float32; W[pre, post] = predicted sign x synapse count
    flywire_ids: np.ndarray   # (N,) int64; row/column index -> FlyWire ID

    @property
    def n(self) -> int:
        return self.W.shape[0]

    def index_of(self, ids) -> np.ndarray:
        lookup = {int(f): i for i, f in enumerate(self.flywire_ids)}
        missing = [int(f) for f in ids if int(f) not in lookup]
        if missing:
            raise KeyError(f"FlyWire IDs not in connectome: {missing[:5]}")
        return np.array([lookup[int(f)] for f in ids], dtype=np.int64)


def build_connectome(connectivity_path: Path, completeness_path: Path) -> Connectome:
    ids = pd.read_csv(completeness_path, index_col=0).index.to_numpy(dtype=np.int64)
    df = pd.read_parquet(connectivity_path,
                         columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    n = len(ids)
    W = sp.csr_matrix(
        (df["Excitatory x Connectivity"].to_numpy(np.float32),
         (df["Presynaptic_Index"].to_numpy(), df["Postsynaptic_Index"].to_numpy())),
        shape=(n, n),
    )
    W.sort_indices()
    return Connectome(W=W, flywire_ids=ids)


def save_connectome(conn: Connectome, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=conn.W.data, indices=conn.W.indices, indptr=conn.W.indptr,
             shape=np.array(conn.W.shape), flywire_ids=conn.flywire_ids)


def load_connectome(path: Path) -> Connectome:
    z = np.load(path)
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return Connectome(W=W, flywire_ids=z["flywire_ids"])


def load_or_build_connectome(connectivity_path: Path, completeness_path: Path,
                             cache_path: Path = PROCESSED / "connectome_v630.npz") -> Connectome:
    if cache_path.exists():
        return load_connectome(cache_path)
    conn = build_connectome(connectivity_path, completeness_path)
    save_connectome(conn, cache_path)
    return conn
