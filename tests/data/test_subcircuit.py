import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.data.subcircuit import (SubcircuitError, extract_subcircuit, has_directed_path,
                                      hop_distances, load_subcircuit, save_subcircuit)

EDGES = [(0, 2, 1), (2, 9, 1), (1, 3, 5), (3, 4, 1), (4, 9, 1), (0, 5, 1), (6, 9, 1), (7, 8, 1)]


def graph(edges):
    pre, post, w = zip(*edges)
    return sp.csr_matrix((np.array(w, np.float32), (pre, post)), shape=(10, 10))


GROUPS = {"a": np.array([0]), "b": np.array([1])}
OUT = np.array([9])


def test_hop_distances_forward_and_backward():
    W = graph(EDGES)
    d_in = hop_distances(W, np.array([0, 1]), max_hops=6)
    assert d_in.tolist() == [0, 0, 1, 1, 2, 1, 7, 7, 7, 2]
    d_out = hop_distances(W.T.tocsr(), OUT, max_hops=6)
    assert d_out.tolist() == [2, 3, 1, 2, 1, 7, 1, 7, 7, 0]


def test_has_directed_path():
    W = graph(EDGES)
    assert has_directed_path(W, np.array([1]), 9)
    assert not has_directed_path(W, np.array([9]), 0)


def test_extract_picks_smallest_k_reaching_min_size():
    sub = extract_subcircuit(graph(EDGES), GROUPS, OUT, min_size=6, max_size=100, max_hops=6)
    assert sub.hops == 2
    assert sub.nodes.tolist() == [0, 1, 2, 3, 4, 9]
    assert sub.n_candidates == 6
    assert sub.input_local["a"].tolist() == [0]
    assert sub.input_local["b"].tolist() == [1]
    assert sub.mn9_local.tolist() == [5]
    np.testing.assert_array_equal(sub.W.toarray(), graph(EDGES).toarray()[np.ix_(sub.nodes, sub.nodes)])
    assert sub.d_in.tolist() == [0, 0, 1, 1, 2, 2]


def test_truncation_keeps_protected_then_short_paths_then_strength():
    # extra edge 2->3 keeps node 3 weakly connected after node 4 is dropped
    W = graph(EDGES + [(2, 3, 1)])
    sub = extract_subcircuit(W, GROUPS, OUT, min_size=6, max_size=5, max_hops=6)
    # non-protected: 2 (d_in+d_out=2), 3 (sum 3, strength 7), 4 (sum 3, strength 2) -> keep 2 and 3
    assert sub.nodes.tolist() == [0, 1, 2, 3, 9]
    assert sub.n_candidates == 6


def test_largest_component_dropping_a_protected_node_raises():
    # without 2->3, truncation keeps 3 and drops 4, which cuts {1,3} off from {0,2,9}
    with pytest.raises(SubcircuitError, match="dropped"):
        extract_subcircuit(graph(EDGES), GROUPS, OUT, min_size=6, max_size=5, max_hops=6)


def test_save_load_roundtrip(tmp_path):
    sub = extract_subcircuit(graph(EDGES), GROUPS, OUT, min_size=6, max_size=100, max_hops=6)
    save_subcircuit(sub, tmp_path / "sub.npz")
    got = load_subcircuit(tmp_path / "sub.npz")
    assert got.nodes.tolist() == sub.nodes.tolist()
    assert list(got.input_local) == ["a", "b"]
    assert got.hops == 2 and got.n_candidates == 6
    np.testing.assert_array_equal(got.W.toarray(), sub.W.toarray())
    np.testing.assert_array_equal(got.d_out, sub.d_out)
