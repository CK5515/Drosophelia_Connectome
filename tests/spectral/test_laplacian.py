import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.spectral.laplacian import (opposite_sign_reciprocal_pairs, signed_magnetic_laplacian,
                                         signed_symmetric_laplacian)


def random_signed_digraph(n=30, seed=0):
    rng = np.random.default_rng(seed)
    W = (rng.random((n, n)) < 0.2) * rng.integers(1, 20, size=(n, n)) * rng.choice([-1, 1], size=(n, n))
    np.fill_diagonal(W, 0)
    for i in range(n):                      # ring guarantees no zero-degree node
        W[i, (i + 1) % n] = abs(W[i, (i + 1) % n]) + 1
    return W.astype(np.float64)


@pytest.mark.parametrize("builder", [signed_symmetric_laplacian, signed_magnetic_laplacian])
def test_hermitian_with_spectrum_in_0_2(builder):
    L = builder(random_signed_digraph())
    np.testing.assert_allclose(L, L.conj().T, atol=1e-12)
    eig = np.linalg.eigvalsh(L)
    assert eig.min() >= -1e-6 and eig.max() <= 2 + 1e-6


def test_magnetic_with_q_zero_equals_symmetric():
    W = random_signed_digraph()
    np.testing.assert_allclose(signed_magnetic_laplacian(W, q=0.0), signed_symmetric_laplacian(W), atol=1e-12)


def test_transposing_graph_conjugates_magnetic_laplacian():
    W = random_signed_digraph(seed=1)
    np.testing.assert_allclose(signed_magnetic_laplacian(W.T), signed_magnetic_laplacian(W).conj(), atol=1e-12)


def test_magnetic_phase_encodes_direction():
    W = np.array([[0.0, 1.0], [0.0, 0.0]])  # single edge 0 -> 1
    H = signed_magnetic_laplacian(W, q=0.25)
    # A_s = 0.5, degree 0.5 each, normalized off-diagonal = 1, phase 2*pi*0.25*(1-0) = pi/2
    np.testing.assert_allclose(H[0, 1], -np.exp(1j * np.pi / 2), atol=1e-12)
    np.testing.assert_allclose(H[1, 0], -np.exp(-1j * np.pi / 2), atol=1e-12)


def test_zero_degree_raises():
    W = np.zeros((3, 3))
    W[0, 1] = 1.0
    with pytest.raises(ValueError, match="zero-degree"):
        signed_symmetric_laplacian(W)


def test_opposite_sign_cancellation_is_counted_and_can_zero_a_degree():
    W = np.zeros((3, 3))
    W[0, 1], W[1, 0] = 2.0, -2.0   # cancels in A_s
    W[1, 2], W[2, 1] = 1.0, 1.0
    assert opposite_sign_reciprocal_pairs(sp.csr_matrix(W)) == 1
    with pytest.raises(ValueError, match="zero-degree"):
        signed_symmetric_laplacian(W)   # node 0 only has the cancelled pair
