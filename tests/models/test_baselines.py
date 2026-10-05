import numpy as np
import pytest
import scipy.sparse as sp
import torch

from flybrain.models.baselines import ChebGRUBaseline, MeanPredictor, MLPBaseline


def test_mean_predictor_returns_the_training_mean_trace():
    rates = np.stack([np.full((4, 3), 1.0), np.full((4, 3), 3.0)]).astype(np.float32)  # (2,4,3)
    mp = MeanPredictor.fit(rates)
    out = mp.predict(5)
    assert out.shape == (5, 4, 3)
    assert np.allclose(out, 2.0)


def test_mlp_predicts_only_the_requested_neurons():
    model = MLPBaseline(n_stim=5, n_out=7, n_bins=50)
    out = model(torch.rand(3, 5))
    assert out.shape == (3, 7, 50)
    assert torch.all(out >= 0)                      # softplus readout


def test_mlp_parameter_count_scales_with_outputs_not_all_neurons():
    small = sum(p.numel() for p in MLPBaseline(5, 10, 50).parameters())
    big = sum(p.numel() for p in MLPBaseline(5, 20, 50).parameters())
    assert big > small


def chain_laplacian(n=10):
    rows = [(i, (i + 1) % n, 1.0) for i in range(n)]
    pre, post, w = zip(*rows)
    a = sp.csr_matrix((w, (pre, post)), shape=(n, n))
    a = ((a + a.T) * 0.5).tocsr()
    deg = np.asarray(abs(a).sum(axis=1)).ravel()
    d = sp.diags(1.0 / np.sqrt(deg))
    return (sp.eye(n) - d @ a @ d).tocsr()


def test_chebgru_shape_and_nonnegative():
    model = ChebGRUBaseline(chain_laplacian(), input_idx=np.array([0, 1]),
                            group_of_input=np.array([0, 2]))
    out = model(torch.rand(2, 5) * 200, n_bins=6, stim_bins=3)
    assert out.shape == (2, 10, 6)
    assert torch.all(out >= 0) and torch.isfinite(out).all()


def test_chebgru_is_node_agnostic_under_relabelling():
    """Same structural guarantee the MoE has: no per-neuron weights."""
    torch.manual_seed(0)
    n = 10
    lap = chain_laplacian(n)
    perm = np.random.default_rng(0).permutation(n)
    a = ChebGRUBaseline(lap, np.array([0, 1]), np.array([0, 2]))
    b = ChebGRUBaseline(lap[perm][:, perm], perm.argsort()[np.array([0, 1])], np.array([0, 2]))
    graph_keys = {"lap_idx", "lap_val", "lap_sparse", "input_idx", "group_of_input"}
    assert not graph_keys & set(a.state_dict()), "graph must stay out of the state dict"
    b.load_state_dict(a.state_dict())
    stim = torch.rand(2, 5) * 150
    out_a = a(stim, 5, 3).detach().numpy()
    out_b = b(stim, 5, 3).detach().numpy()
    # lap[perm][:, perm] makes new index i the original neuron perm[i], so out_a[:, perm] == out_b.
    np.testing.assert_allclose(out_a[:, perm], out_b, rtol=1e-5, atol=1e-6)
    # Negative control: the other plausible pairing (right only for an involution) must fail,
    # so this test can actually detect a wrong mapping. Guard against a vacuous pass.
    assert not np.array_equal(perm.argsort(), perm)
    assert not np.allclose(out_a[:, perm.argsort()], out_b, rtol=1e-5, atol=1e-6)


def test_chebgru_gradients_reach_every_parameter():
    model = ChebGRUBaseline(chain_laplacian(), np.array([0]), np.array([0]))
    model(torch.rand(1, 5) * 200, 4, 2).sum().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chebgru_cached_laplacian_moves_to_gpu():
    model = ChebGRUBaseline(chain_laplacian(), np.array([0, 1]), np.array([0, 2])).to("cuda")
    assert model.lap_sparse.is_cuda and model.lap_sparse.layout == torch.sparse_csr
    out = model(torch.rand(2, 5, device="cuda") * 200, 4, 2)
    assert out.is_cuda and torch.isfinite(out).all()


def test_chebgru_double_precision_converts_cached_laplacian():
    model = ChebGRUBaseline(chain_laplacian(), np.array([0, 1]), np.array([0, 2])).double()
    assert model.lap_sparse.dtype == torch.float64
    out = model(torch.rand(2, 5, dtype=torch.float64) * 200, 4, 2)
    assert out.dtype == torch.float64 and torch.isfinite(out).all()
