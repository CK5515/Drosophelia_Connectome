import numpy as np
import pytest
import torch

from flybrain.analysis.kernels import expert_kernels, specialisation
from flybrain.models.moe import SpectralMoE, inverse_softplus, moe_full
from flybrain.spectral.bands import equal_count_bands


def model_with(gammas):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(8, 8, generator=g) + 1j * torch.randn(8, 8, generator=g)
    u, _ = torch.linalg.qr(a)
    m = SpectralMoE(torch.linspace(0.3, 1.7, 8), u.to(torch.complex64), equal_count_bands(8, 4),
                    np.arange(2), np.array([0, 1]), moe_full())
    with torch.no_grad():
        for k, val in enumerate(gammas):
            m.gamma_tilde[0][k] = inverse_softplus(val)
    return m


def test_kernels_report_timescales_and_regimes():
    out = expert_kernels(model_with([0.1, 0.2, 0.4, 0.8]), n_steps=40, bin_ms=20.0)
    layer0 = out["layers"][0]["experts"]
    assert len(layer0) == 4
    assert layer0[0]["timescale_ms"] > layer0[3]["timescale_ms"]       # smaller gamma = slower
    assert all(e["regime"] in ("underdamped", "critical", "overdamped") for e in layer0)
    assert len(layer0[0]["impulse"]) == 40


def test_specialisation_detects_a_two_fold_timescale_spread():
    spread = specialisation(expert_kernels(model_with([0.1, 0.2, 0.4, 0.8])))
    assert spread["specialised"] and spread["max_ratio"] >= 2.0
    flat = specialisation(expert_kernels(model_with([0.30, 0.31, 0.29, 0.30])))
    assert not flat["specialised"] and flat["max_ratio"] < 2.0


def test_regime_omega_and_band_median_match_hand_computation():
    eigs = np.linspace(0.3, 1.7, 8)                        # the basis built in model_with; 2 modes per band
    m = model_with([0.3, 0.3, 0.3, 0.3])
    with torch.no_grad():
        m.gamma_tilde[0][0] = inverse_softplus(0.1)        # expert 0: alpha*mu >> gamma^2
        m.alpha_tilde[0][0] = inverse_softplus(1.0)
        m.gamma_tilde[0][3] = inverse_softplus(3.0)        # expert 3: alpha*mu << gamma^2
        m.alpha_tilde[0][3] = inverse_softplus(0.1)
    experts = expert_kernels(m)["layers"][0]["experts"]
    for k in (0, 3):
        assert experts[k]["band_median_mu"] == pytest.approx(float(np.median(eigs[2 * k:2 * k + 2])), rel=1e-5)
    mu0, mu3 = experts[0]["band_median_mu"], experts[3]["band_median_mu"]
    assert experts[0]["regime"] == "underdamped"
    assert experts[0]["omega_per_bin"] == pytest.approx(np.sqrt(1.0 * mu0 - 0.1 ** 2), rel=1e-4)
    assert experts[3]["regime"] == "overdamped"
    assert experts[3]["omega_per_bin"] == 0.0
    assert 0.1 * mu3 < 3.0 ** 2                            # the overdamped case is genuinely so
