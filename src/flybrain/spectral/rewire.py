"""Degree- and sign-preserving directed rewiring: the control that asks whether this connectome matters."""
import numpy as np
import scipy.sparse as sp


def rewire_preserving_degree_and_sign(W: sp.spmatrix, seed: int,
                                      swaps_per_edge: float = 10.0) -> sp.csr_matrix:
    w = sp.coo_matrix(W)
    pre, post, data = w.row.copy(), w.col.copy(), w.data.copy()
    rng = np.random.default_rng(seed)
    existing = {(int(a), int(b)) for a, b in zip(pre, post)}
    for sign in (1.0, -1.0):
        group = np.flatnonzero(np.sign(data) == sign)
        if len(group) < 2:
            continue
        for _ in range(int(swaps_per_edge * len(group))):
            i, j = rng.choice(group, size=2, replace=False)
            a, b, c, d = int(pre[i]), int(post[i]), int(pre[j]), int(post[j])
            if a == d or c == b:
                continue
            if (a, d) in existing or (c, b) in existing:
                continue
            existing.discard((a, b))
            existing.discard((c, d))
            existing.add((a, d))
            existing.add((c, b))
            post[i], post[j] = d, b
    n_dup = len(pre) - len(set(zip(pre.tolist(), post.tolist())))
    if n_dup:
        raise RuntimeError(f"rewiring produced {n_dup} duplicate edges; the swap bookkeeping is broken")
    return sp.csr_matrix((data, (pre, post)), shape=w.shape)


def rewiring_report(original: sp.spmatrix, rewired: sp.spmatrix) -> dict:
    a, b = sp.csr_matrix(original), sp.csr_matrix(rewired)
    def degrees(m):
        binary = m.copy()
        binary.data = np.ones_like(binary.data)
        return (np.asarray(binary.sum(axis=1)).ravel(), np.asarray(binary.sum(axis=0)).ravel())
    out_a, in_a = degrees(a)
    out_b, in_b = degrees(b)
    pairs_a = {(int(i), int(j)) for i, j in zip(*a.nonzero())}
    pairs_b = {(int(i), int(j)) for i, j in zip(*b.nonzero())}
    coo = sp.coo_matrix(b)
    return {
        "out_degree_preserved": bool(np.array_equal(out_a, out_b)),
        "in_degree_preserved": bool(np.array_equal(in_a, in_b)),
        "sign_counts_preserved": bool((a.data > 0).sum() == (b.data > 0).sum()
                                      and (a.data < 0).sum() == (b.data < 0).sum()),
        "weight_multiset_preserved": bool(np.array_equal(np.sort(a.data), np.sort(b.data))),
        "frac_edges_displaced": float(1.0 - len(pairs_a & pairs_b) / max(len(pairs_a), 1)),
        "n_self_loops": int((coo.row == coo.col).sum()),
        # CSR conversion already sums duplicates, so literal duplicate entries cannot be seen here.
        # The key now detects edge loss through coalescing: any duplicate would shrink nnz.
        "n_duplicate_pairs": int(a.nnz - b.nnz),
    }
