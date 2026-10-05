import numpy as np
import pytest
import torch
from dataclasses import replace

from flybrain.analysis.gates import categorise, gate_summary
from flybrain.models.moe import SpectralMoE, moe_full
from flybrain.spectral.bands import equal_count_bands

GROUPS = ["sugar_R", "sugar_L", "bitter", "water", "ir94e"]


def test_categorise_labels_dominant_mixed_and_silent():
    stim = np.array([[200.0, 0, 0, 0, 0],       # sugar_R dominant
                     [0, 0, 0, 0, 0],           # nothing
                     [100.0, 95.0, 0, 0, 0]])   # mixed: within 1.2x
    assert categorise(stim, GROUPS).tolist() == ["sugar_R", "none", "mixed"]


def test_gate_summary_reports_weights_per_category_and_entropy():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(8, 8, generator=g) + 1j * torch.randn(8, 8, generator=g)
    u, _ = torch.linalg.qr(a)
    model = SpectralMoE(torch.linspace(0.3, 1.7, 8), u.to(torch.complex64), equal_count_bands(8, 4),
                        np.arange(2), np.array([0, 2]), moe_full())
    stim = np.array([[200.0, 0, 0, 0, 0], [0, 0, 180.0, 0, 0], [90.0, 0, 85.0, 0, 0]], dtype=np.float32)
    out = gate_summary(model, stim, n_bins=6, stim_bins=3, group_names=GROUPS, device="cpu")
    assert set(out["per_category"]).issubset(set(GROUPS) | {"mixed", "none"})
    for weights in out["per_category"].values():
        assert len(weights[0]) == 4
        assert abs(sum(weights[0]) - 1.0) < 1e-4
    assert len(out["entropy_per_layer"]) == model.cfg.n_layers
    assert all(0.0 <= e <= np.log(4) + 1e-6 for e in out["entropy_per_layer"])


def _basis_model(cfg):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(8, 8, generator=g) + 1j * torch.randn(8, 8, generator=g)
    u, _ = torch.linalg.qr(a)
    return SpectralMoE(torch.linspace(0.3, 1.7, 8), u.to(torch.complex64), equal_count_bands(8, 4),
                       np.arange(2), np.array([0, 2]), cfg)


def _entropy(p):
    return float(-(p * np.log(p)).sum(axis=-1).mean())


def test_uniform_gate_has_per_sample_entropy_of_exactly_ln4():
    model = _basis_model(replace(moe_full(), input_dependent_gate=False))
    stim = np.array([[200.0, 0, 0, 0, 0], [0, 0, 180.0, 0, 0]], dtype=np.float32)
    out = gate_summary(model, stim, n_bins=6, stim_bins=3, group_names=GROUPS, device="cpu")
    np.testing.assert_allclose(out["entropy_per_layer"], np.log(4), atol=1e-6)


def test_entropy_is_the_mean_of_per_sample_entropies_not_entropy_of_the_mean_gate():
    torch.manual_seed(2)
    model = _basis_model(moe_full())
    with torch.no_grad():
        for head in model.gate:
            head[-1].weight.mul_(10.0)          # make the two stimuli route differently
    stim = np.array([[200.0, 0, 0, 0, 0], [0, 0, 180.0, 0, 0]], dtype=np.float32)
    out = gate_summary(model, stim, n_bins=6, stim_bins=3, group_names=GROUPS, device="cpu")
    with torch.no_grad():
        _, aux = model(torch.tensor(stim), 6, 3, return_aux=True)
    for layer, g in enumerate(aux["gates"]):
        g = g.numpy().astype(np.float64)
        per_sample = _entropy(g)
        of_mean = float(-(g.mean(axis=0) * np.log(g.mean(axis=0))).sum())
        assert out["entropy_per_layer"][layer] == pytest.approx(per_sample, abs=1e-5)
        assert not np.allclose(g[0], g[1], atol=1e-2)                 # the two stimuli really route differently
        assert out["entropy_per_layer"][layer] < of_mean - 0.03       # averaging first would give of_mean
