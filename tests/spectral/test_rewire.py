import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.spectral.rewire import rewire_preserving_degree_and_sign, rewiring_report


def signed_graph(n=60, density=0.08, seed=0):
    rng = np.random.default_rng(seed)
    dense = (rng.random((n, n)) < density) * rng.choice([-1.0, 1.0], size=(n, n), p=[0.4, 0.6]) \
        * rng.integers(1, 30, size=(n, n))
    np.fill_diagonal(dense, 0.0)
    return sp.csr_matrix(dense)


def test_rewiring_preserves_degrees_signs_and_weights():
    w = signed_graph()
    w_before = w.copy()
    r = rewire_preserving_degree_and_sign(w, seed=0)
    rep = rewiring_report(w, r)
    assert r.shape == w.shape                                  # node indices never relabelled
    assert (w != w_before).nnz == 0 and w.nnz == w_before.nnz  # input not mutated
    assert r.nnz == w.nnz                                      # no edge lost to coalesced duplicates
    assert rep["in_degree_preserved"] and rep["out_degree_preserved"]
    assert rep["sign_counts_preserved"] and rep["weight_multiset_preserved"]
    assert rep["n_self_loops"] == 0 and rep["n_duplicate_pairs"] == 0


def test_rewiring_actually_moves_most_edges():
    # Sparse like the real subcircuit (2% density): on a dense 60-node toy graph the chance-overlap
    # floor of a degree-preserving shuffle is ~10%, so >0.9 is unreachable there regardless of effort.
    w = signed_graph(n=200, density=0.03)
    rep = rewiring_report(w, rewire_preserving_degree_and_sign(w, seed=1))
    assert rep["frac_edges_displaced"] > 0.9


def test_rewiring_is_seeded():
    w = signed_graph()
    a = rewire_preserving_degree_and_sign(w, seed=2)
    b = rewire_preserving_degree_and_sign(w, seed=2)
    c = rewire_preserving_degree_and_sign(w, seed=3)
    assert (a != b).nnz == 0
    assert (a != c).nnz > 0


def test_excitatory_and_inhibitory_edges_never_mix():
    w = signed_graph()
    r = rewire_preserving_degree_and_sign(w, seed=4)
    for name, mask in (("excitatory", lambda m: m > 0), ("inhibitory", lambda m: m < 0)):
        for axis in (0, 1):
            before = np.asarray(mask(w).sum(axis=axis)).ravel()
            after = np.asarray(mask(r).sum(axis=axis)).ravel()
            np.testing.assert_array_equal(before, after, err_msg=f"{name} axis {axis}")


def test_empty_and_single_edge_graphs_are_handled():
    empty = sp.csr_matrix((5, 5))
    assert rewire_preserving_degree_and_sign(empty, seed=0).nnz == 0
    one = sp.csr_matrix(([2.0], ([0], [1])), shape=(5, 5))
    out = rewire_preserving_degree_and_sign(one, seed=0)
    assert out.nnz == 1 and out[0, 1] == 2.0                  # nothing to swap with
