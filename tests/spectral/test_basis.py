import numpy as np
import pytest
import scipy.sparse as sp

from flybrain.spectral.basis import basis_checks, compute_basis, load_basis, save_basis
from flybrain.spectral.laplacian import signed_magnetic_laplacian


def small_graph(n=40, seed=0):
    rng = np.random.default_rng(seed)
    W = (rng.random((n, n)) < 0.2) * rng.integers(1, 20, size=(n, n)) * rng.choice([-1, 1], size=(n, n))
    np.fill_diagonal(W, 0)
    for i in range(n):
        W[i, (i + 1) % n] = abs(W[i, (i + 1) % n]) + 1
    return sp.csr_matrix(W.astype(np.float32))


@pytest.mark.parametrize("kind,dtype", [("symmetric", np.float32), ("magnetic", np.complex64)])
def test_compute_basis_diagonalizes_laplacian(kind, dtype):
    W = small_graph()
    b = compute_basis(W, kind, device="cpu")
    assert b.eigenvectors.dtype == dtype
    assert np.all(np.diff(b.eigenvalues) >= 0)
    checks = basis_checks(b, device="cpu")
    assert checks["eig_in_range"] and checks["orthonormal"]
    if kind == "magnetic":
        H = signed_magnetic_laplacian(W.toarray())
        U = b.eigenvectors.astype(np.complex128)
        np.testing.assert_allclose(U @ np.diag(b.eigenvalues) @ U.conj().T, H, atol=1e-4)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        compute_basis(small_graph(), "laplacian-of-vibes", device="cpu")


def test_save_load_roundtrip(tmp_path):
    nodes = np.arange(40)[::-1] * 3
    b = compute_basis(small_graph(), "magnetic", device="cpu", nodes=nodes)
    save_basis(b, tmp_path / "m.npz")
    got = load_basis(tmp_path / "m.npz")
    np.testing.assert_array_equal(got.nodes, nodes)
    assert got.kind == "magnetic" and got.q == 0.25
    np.testing.assert_array_equal(got.eigenvectors, b.eigenvectors)
    np.testing.assert_array_equal(got.eigenvalues, b.eigenvalues)


def test_nodes_default_none_and_old_files_load(tmp_path):
    b = compute_basis(small_graph(), "symmetric", device="cpu")
    assert b.nodes is None
    save_basis(b, tmp_path / "s.npz")
    assert load_basis(tmp_path / "s.npz").nodes is None
