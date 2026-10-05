import warnings
import numpy as np
import pytest

from flybrain.train.metrics import normalised_r2, pooled_r2, r2_per_neuron, summarise, trace_r2


def test_perfect_and_mean_predictions_bracket_r2():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(20, 5, 3)).astype(np.float32)
    assert np.allclose(r2_per_neuron(target, target), 1.0)
    mean_pred = np.broadcast_to(target.reshape(-1, 3).mean(axis=0), target.shape)
    assert np.allclose(r2_per_neuron(np.ascontiguousarray(mean_pred), target), 0.0, atol=1e-6)


def test_zero_variance_neuron_is_nan_not_inf():
    target = np.zeros((10, 4, 2), dtype=np.float32)
    target[..., 0] = np.arange(40).reshape(10, 4)
    r2 = r2_per_neuron(np.zeros_like(target), target)
    assert np.isnan(r2[1])
    assert np.isfinite(r2[0])


def test_pooled_r2_uses_only_masked_neurons():
    target = np.zeros((8, 3, 2), dtype=np.float32)
    target[..., 0] = np.arange(24).reshape(8, 3)
    pred = target.copy()
    pred[..., 1] = 99.0                      # garbage on the unmasked neuron
    assert pooled_r2(pred, target, np.array([True, False])) == pytest.approx(1.0)


def test_trace_r2_scores_one_neuron():
    target = np.zeros((6, 4, 3), dtype=np.float32)
    target[..., 2] = np.arange(24).reshape(6, 4)
    assert trace_r2(target, target, 2) == pytest.approx(1.0)


def test_normalised_r2_divides_by_ceiling_and_skips_unreliable():
    r2 = np.array([0.4, 0.2, np.nan, 0.9])
    ceiling = np.array([0.8, 0.05, 0.9, np.nan])     # only index 0 qualifies
    mean, used = normalised_r2(r2, ceiling, min_ceiling=0.1)
    assert used == 1 and mean == pytest.approx(0.5)


def test_summarise_reports_every_required_block():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(12, 5, 4)).astype(np.float32)
    pred = target + rng.normal(scale=0.1, size=target.shape).astype(np.float32)
    masks = {"observed_active": np.array([True, True, False, False]),
             "heldout_active": np.array([False, False, True, True])}
    out = summarise(pred, target, masks, ceiling=np.full(4, 0.9), mn9_idx=[0, 1])
    expected_keys = {
        "r2_observed_active_mean", "r2_observed_active_median", "pooled_r2_observed_active",
        "normalised_r2_observed_active", "n_observed_active", "n_observed_active_ceiling_usable",
        "r2_heldout_active_mean", "r2_heldout_active_median", "pooled_r2_heldout_active",
        "normalised_r2_heldout_active", "n_heldout_active", "n_heldout_active_ceiling_usable",
        "mn9_trace_r2"
    }
    assert set(out) == expected_keys
    assert out["n_observed_active"] == 2
    assert len(out["mn9_trace_r2"]) == 2
    assert out["r2_observed_active_mean"] > 0.9


def test_summarise_all_silent_mask_raises_no_warning():
    """All-silent mask: every selected neuron has zero-variance target."""
    rng = np.random.default_rng(2)
    target = np.zeros((8, 5, 3), dtype=np.float32)
    target[..., 0] = np.arange(40).reshape(8, 5)  # only neuron 0 has variance
    pred = target.copy()
    masks = {"silent": np.array([False, True, True])}  # select only silent neurons 1, 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = summarise(pred, target, masks, ceiling=np.full(3, 0.9), mn9_idx=[])
    assert np.isnan(out["r2_silent_mean"])
    assert np.isnan(out["r2_silent_median"])
    assert np.isnan(out["pooled_r2_silent"])
    assert out["n_silent"] == 2


def test_summarise_empty_mask_raises_no_warning():
    """Empty mask: no neurons selected."""
    rng = np.random.default_rng(3)
    target = rng.normal(size=(8, 5, 3)).astype(np.float32)
    pred = target.copy()
    masks = {"none": np.array([False, False, False])}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = summarise(pred, target, masks, ceiling=np.full(3, 0.9), mn9_idx=[])
    assert np.isnan(out["r2_none_mean"])
    assert np.isnan(out["r2_none_median"])
    assert np.isnan(out["pooled_r2_none"])
    assert out["n_none"] == 0


def test_pooled_r2_zero_variance_is_nan():
    """pooled_r2 on zero-variance data returns NaN, not division error."""
    target = np.zeros((8, 5, 2), dtype=np.float32)
    pred = np.zeros_like(target)
    mask = np.array([True, True])
    result = pooled_r2(pred, target, mask)
    assert np.isnan(result)
