import numpy as np
import pytest
import torch

from flybrain.models.moe import SpectralMoE, moe_full, moe_linear, moe_single
from flybrain.spectral.bands import equal_count_bands

N, M, K = 12, 12, 4


def tiny_basis(seed=0, dtype=torch.complex128):
    """A random unitary basis with eigenvalues in [0, 2] - stands in for a real Laplacian basis."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(N, M, generator=g, dtype=torch.float64) + 1j * torch.randn(N, M, generator=g, dtype=torch.float64)
    u, _ = torch.linalg.qr(a)
    mu = torch.linspace(0.3, 1.7, M, dtype=torch.float64)
    return mu, u.to(dtype), equal_count_bands(M, K)


def build(cfg=None, seed=0, n_in=3, dtype=torch.complex128):
    mu, u, bands = tiny_basis(seed, dtype)
    model = SpectralMoE(mu, u, bands, input_idx=np.arange(n_in),
                        group_of_input=np.array([0, 1, 4][:n_in]), cfg=cfg or moe_full())
    return model.to(torch.float64)


def test_forward_shape_and_finiteness():
    model = build()
    out = model(torch.rand(2, 5, dtype=torch.float64) * 200, n_bins=7, stim_bins=4)
    assert out.shape == (2, N, 7)
    assert torch.isfinite(out).all()


def test_all_zero_stimulus_stays_finite():
    # review focus 2: ~7 of 3000 real conditions drive nothing; log-energy must not become -inf
    model = build()
    out = model(torch.zeros(1, 5, dtype=torch.float64), n_bins=6, stim_bins=3, return_aux=True)
    rates, aux = out
    assert torch.isfinite(rates).all()
    for g in aux["gates"]:
        assert torch.isfinite(g).all()
        assert torch.allclose(g.sum(-1), torch.ones(1, dtype=torch.float64))


def test_relabelling_neurons_permutes_the_output():
    """No per-node parameters: the model sees neurons only through the basis rows."""
    torch.manual_seed(0)
    mu, u, bands = tiny_basis()
    perm = torch.randperm(N)
    assert not torch.equal(perm, torch.arange(N))
    assert not torch.equal(perm[perm[:3]], torch.arange(3))     # not an involution on the inputs
    stim = torch.rand(2, 5, dtype=torch.float64) * 150
    a = SpectralMoE(mu, u, bands, np.arange(3), np.array([0, 1, 4]), moe_full()).to(torch.float64)
    inv = torch.argsort(perm)     # b's row i is a's node perm[i], so a's node j sits at row inv[j]
    b = SpectralMoE(mu, u[perm], bands, inv.numpy()[np.arange(3)], np.array([0, 1, 4]),
                    moe_full()).to(torch.float64)
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(a(stim, 6, 3)[:, perm, :], b(stim, 6, 3))


def test_linear_config_is_affine_in_the_stimulus():
    # The control is deliberately affine, not purely linear: it keeps its biases (an intercept), making
    # it the strongest model that is still linear in the stimulus and the non-collapse criterion more
    # conservative. So randomise the biases and test affinity: g(s) = f(s) - f(0) must be linear.
    model = build(moe_linear())
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in list(model.mix_bias) + [model.readout_bias]:
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float64))
    zero = torch.zeros(1, 5, dtype=torch.float64)
    s1 = torch.rand(1, 5, dtype=torch.float64) * 200
    s2 = torch.rand(1, 5, dtype=torch.float64) * 200
    f0 = model(zero, 6, 3)
    assert f0.abs().max() > 1e-3            # the biases are genuinely in play
    def g(s):
        return model(s, 6, 3) - f0
    torch.testing.assert_close(g(2.0 * s1 - 3.0 * s2), 2.0 * g(s1) - 3.0 * g(s2), rtol=1e-9, atol=1e-9)


def test_full_config_is_not_linear():
    model = build(moe_full())
    s1 = torch.rand(1, 5, dtype=torch.float64) * 200
    s2 = torch.rand(1, 5, dtype=torch.float64) * 200
    lhs = model(s1 + s2, 6, 3)
    rhs = model(s1, 6, 3) + model(s2, 6, 3)
    assert not torch.allclose(lhs, rhs, rtol=1e-3, atol=1e-3)


def test_gate_responds_to_the_stimulus():
    model = build(moe_full())
    _, aux_a = model(torch.tensor([[200.0, 0, 0, 0, 0]], dtype=torch.float64), 6, 3, return_aux=True)
    _, aux_b = model(torch.tensor([[0, 0, 200.0, 0, 0]], dtype=torch.float64), 6, 3, return_aux=True)
    assert not torch.allclose(aux_a["gates"][0], aux_b["gates"][0], atol=1e-6)


def test_single_expert_config_has_one_band_worth_of_parameters():
    model = build(moe_single())
    assert model.gamma_tilde[0].shape[0] == 1
    assert int(model.band_of_mode.max()) == 0


def test_experts_start_identical_up_to_init_noise():
    model = build(moe_full())
    for layer in range(model.cfg.n_layers):
        for p in (model.gamma_tilde[layer], model.alpha_tilde[layer]):
            spread = p.detach().std(dim=0).mean() / p.detach().abs().mean()
            assert spread < 0.05          # 1% noise, so well under 5%


def test_parameter_count_does_not_depend_on_neuron_count():
    small = build()
    counts = sum(p.numel() for p in small.parameters())
    mu, u, bands = tiny_basis()
    big_u = torch.zeros(N * 3, M, dtype=torch.complex128)
    big_u[:N] = u
    big = SpectralMoE(mu, big_u, bands, np.arange(3), np.array([0, 1, 4]), moe_full())
    assert sum(p.numel() for p in big.parameters()) == counts


def test_gradients_reach_every_parameter():
    model = build()
    out = model(torch.rand(2, 5, dtype=torch.float64) * 200, 6, 3)
    out.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert missing == []


def _small(dtype=torch.complex64):
    mu, u, bands = tiny_basis(dtype=dtype)
    return SpectralMoE(mu.to(u.real.dtype), u, bands, np.arange(3), np.array([0, 1, 4]), moe_full())


@pytest.mark.parametrize("convert", [lambda m: m.to(torch.float32), lambda m: m.float()])
def test_float32_conversion_keeps_complex64_basis(convert):
    model = convert(_small())
    for buf in (model.U, model.Uh):
        assert buf.is_complex() and buf.dtype == torch.complex64
    out = model(torch.rand(2, 5) * 200, 6, 3)
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_double_round_trip_keeps_imaginary_part():
    model = _small().double()
    assert model.U.dtype == torch.complex128 and model.Uh.dtype == torch.complex128
    assert model.U.imag.abs().max() > 0 and model.Uh.imag.abs().max() > 0


def test_state_dict_holds_only_learnable_parameters_and_round_trips():
    model = _small()
    keys = set(model.state_dict().keys())
    assert keys == {n for n, _ in model.named_parameters()}
    for name in ("U", "Uh", "mu", "band_of_mode", "input_idx", "group_of_input"):
        assert name not in keys
    fresh = _small()
    for p in fresh.parameters():                 # make sure loading really overwrites
        torch.nn.init.normal_(p)
    result = fresh.load_state_dict(model.state_dict())
    assert not result.missing_keys and not result.unexpected_keys
    stim = torch.rand(2, 5) * 200
    torch.testing.assert_close(fresh(stim, 6, 3), model(stim, 6, 3))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_transfer_keeps_complex_basis_on_device():
    model = _small().to("cuda")
    for buf in (model.U, model.Uh):
        assert buf.is_complex() and buf.dtype == torch.complex64 and buf.device.type == "cuda"
    out = model(torch.rand(2, 5, device="cuda") * 200, 6, 3)
    assert out.device.type == "cuda" and torch.isfinite(out).all()


def test_band_scale_switches_an_expert_off():
    model = build(moe_full())
    stim = torch.rand(2, 5, dtype=torch.float64) * 200
    base = model(stim, 6, 3)
    with model.knockout(0):
        knocked = model(stim, 6, 3)
    assert not torch.allclose(base, knocked, atol=1e-8)
    after = model(stim, 6, 3)
    torch.testing.assert_close(base, after)        # the context manager restores the buffer
    with pytest.raises(ValueError):                # ... and restores it when the body raises, too
        with model.knockout(0):
            raise ValueError("boom")
    torch.testing.assert_close(base, model(stim, 6, 3))
