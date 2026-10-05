"""Feeding subcircuit: neurons on short paths from gustatory inputs to MN9 (graph-defined only)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


class SubcircuitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Subcircuit:
    nodes: np.ndarray                    # (n,) sorted full-connectome indices
    W: sp.csr_matrix                     # (n, n) float32 signed, W[pre, post]
    hops: int                            # k used for the d_in/d_out cut
    n_candidates: int                    # |S_k| before truncation
    d_in: np.ndarray                     # (n,) hops from any input neuron
    d_out: np.ndarray                    # (n,) hops to either MN9
    input_local: dict[str, np.ndarray]   # group -> local indices
    mn9_local: np.ndarray                # (2,) local indices


def hop_distances(adj: sp.spmatrix, sources: np.ndarray, max_hops: int) -> np.ndarray:
    n = adj.shape[0]
    reach = (sp.csr_matrix(adj) != 0).astype(np.int32).T.tocsr()  # reach[post, pre]
    dist = np.full(n, max_hops + 1, dtype=np.int32)
    dist[sources] = 0
    frontier = np.zeros(n, dtype=np.int32)
    frontier[sources] = 1
    for k in range(1, max_hops + 1):
        new = (reach @ frontier > 0) & (dist > max_hops)
        if not new.any():
            break
        dist[new] = k
        frontier = new.astype(np.int32)
    return dist


def has_directed_path(adj: sp.spmatrix, sources: np.ndarray, target: int) -> bool:
    n = adj.shape[0]
    return bool(hop_distances(adj, sources, max_hops=n)[target] <= n)


def extract_subcircuit(W: sp.spmatrix, input_groups: dict[str, np.ndarray], outputs: np.ndarray,
                       min_size: int = 2000, max_size: int = 5000, max_hops: int = 6) -> Subcircuit:
    W = sp.csr_matrix(W, dtype=np.float32)
    all_inputs = np.concatenate(list(input_groups.values()))
    protected = np.union1d(all_inputs, outputs)
    d_in = hop_distances(W, all_inputs, max_hops)
    d_out = hop_distances(W.T.tocsr(), outputs, max_hops)

    for k in range(1, max_hops + 1):
        mask = (d_in <= k) & (d_out <= k)
        mask[protected] = True
        if mask.sum() >= min_size:
            break
    else:
        raise SubcircuitError(f"no k <= {max_hops} reaches {min_size} neurons")

    cand = np.flatnonzero(mask)
    n_candidates = len(cand)
    if n_candidates > max_size:
        absW = abs(W)
        strength = np.asarray(absW.sum(axis=1)).ravel() + np.asarray(absW.sum(axis=0)).ravel()
        not_protected = ~np.isin(cand, protected)
        path_len = (d_in + d_out)[cand]
        order = np.lexsort((-strength[cand], path_len, not_protected))  # last key is primary
        cand = np.sort(cand[order[:max_size]])

    local_W = W[cand][:, cand].tocsr()
    _, labels = connected_components(local_W, directed=True, connection="weak")
    keep = labels == np.argmax(np.bincount(labels))
    if not keep[np.isin(cand, protected)].all():
        raise SubcircuitError("largest weakly connected component dropped an input or MN9 neuron")
    nodes = cand[keep]
    local_W = local_W[keep][:, keep].tocsr()

    return Subcircuit(
        nodes=nodes,
        W=local_W,
        hops=k,
        n_candidates=n_candidates,
        d_in=d_in[nodes],
        d_out=d_out[nodes],
        input_local={g: np.searchsorted(nodes, idx).astype(np.int64) for g, idx in input_groups.items()},
        mn9_local=np.searchsorted(nodes, outputs).astype(np.int64),
    )


def save_subcircuit(sub: Subcircuit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(nodes=sub.nodes, W_data=sub.W.data, W_indices=sub.W.indices, W_indptr=sub.W.indptr,
                  W_shape=np.array(sub.W.shape), hops=np.array(sub.hops), n_candidates=np.array(sub.n_candidates),
                  d_in=sub.d_in, d_out=sub.d_out, mn9_local=sub.mn9_local,
                  group_names=np.array(list(sub.input_local)))
    for name, idx in sub.input_local.items():
        arrays[f"input_{name}"] = idx
    np.savez(path, **arrays)


def load_subcircuit(path: Path) -> Subcircuit:
    z = np.load(path)
    return Subcircuit(
        nodes=z["nodes"],
        W=sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"])),
        hops=int(z["hops"]),
        n_candidates=int(z["n_candidates"]),
        d_in=z["d_in"],
        d_out=z["d_out"],
        input_local={str(g): z[f"input_{g}"] for g in z["group_names"]},
        mn9_local=z["mn9_local"],
    )
