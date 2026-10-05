import json

import numpy as np
import pytest

from flybrain.train.dataset import (active_mask, condition_splits, ever_active_mask, load_bundle,
                                    neuron_splits, noise_ceiling, trial_mean_rates)


def make_bundle_dir(tmp_path, counts, conditions, bin_ms=20.0):
    d = tmp_path / "ds"
    d.mkdir()
    np.save(d / "counts.npy", counts.astype(np.uint8))
    np.save(d / "conditions.npy", conditions.astype(np.float32))
    (d / "meta.json").write_text(json.dumps({
        "n_conditions": counts.shape[0], "n_trials": counts.shape[1], "n_bins": counts.shape[2],
        "bin_ms": bin_ms, "record_idx": list(range(counts.shape[3])), "mn9_local": [0, 1],
        "group_names": ["a", "b", "c", "d", "e"],
    }))
    return d


def test_load_bundle_exposes_shapes_and_bin_width(tmp_path):
    counts = np.zeros((6, 4, 10, 7), dtype=np.uint8)
    b = load_bundle(make_bundle_dir(tmp_path, counts, np.zeros((6, 5))))
    assert (b.n_conditions, b.n_trials, b.n_bins, b.n_neurons) == (6, 4, 10, 7)
    assert b.bin_s == pytest.approx(0.02)


def test_condition_splits_are_disjoint_sized_and_seeded():
    tr, va, te = condition_splits(3000, seed=0)
    assert len(tr) == 2100 and len(va) == 450 and len(te) == 450
    assert len(set(tr) | set(va) | set(te)) == 3000
    assert np.array_equal(tr, condition_splits(3000, seed=0)[0])
    assert not np.array_equal(tr, condition_splits(3000, seed=1)[0])
    for s in (tr, va, te):
        assert np.all(np.diff(s) > 0)


def test_neuron_splits_stratify_by_hop_and_protect():
    d_in = np.array([0] * 88 + [1] * 600 + [2] * 4312)
    protected = np.arange(90)
    observed, heldout = neuron_splits(d_in, protected, seed=0)
    assert len(set(observed) & set(heldout)) == 0
    assert len(observed) + len(heldout) == len(d_in)
    assert set(protected).issubset(set(observed))
    for level in (1, 2):
        members = np.flatnonzero(d_in == level)
        frac = len(set(members) & set(heldout)) / len(members)
        assert 0.15 < frac < 0.25            # ~20% per stratum
    # review focus 5: a held-out neuron may also be ever-active; the split does not care
    assert len(heldout) > 0


def test_trial_mean_rates_converts_counts_to_hz(tmp_path):
    counts = np.zeros((3, 4, 5, 2), dtype=np.uint8)
    counts[1, :, 0, 0] = 2          # 2 spikes in a 20 ms bin = 100 Hz, in every trial
    counts[1, 0, 1, 1] = 4          # 4 spikes in one of four trials = 50 Hz mean
    b = load_bundle(make_bundle_dir(tmp_path, counts, np.zeros((3, 5))))
    rates = trial_mean_rates(b, np.array([1]))
    assert rates.shape == (1, 5, 2) and rates.dtype == np.float32
    assert rates[0, 0, 0] == pytest.approx(100.0)
    assert rates[0, 1, 1] == pytest.approx(50.0)
    assert rates[0, 2, 0] == 0.0


def test_masks_identify_signal_carrying_neurons(tmp_path):
    counts = np.zeros((4, 4, 5, 3), dtype=np.uint8)
    counts[:, :, 0, 0] = 1                     # neuron 0: fires, but identically everywhere
    counts[2, :, 3, 1] = 3                     # neuron 1: fires only in condition 2
    b = load_bundle(make_bundle_dir(tmp_path, counts, np.zeros((4, 5))))
    idx = np.arange(4)
    ever = ever_active_mask(b, idx)
    assert ever.tolist() == [True, True, False]
    rates = trial_mean_rates(b, idx)
    act = active_mask(rates, threshold_hz=0.5)
    assert act[1]                              # varies across conditions
    assert not act[2]                          # never fires


def test_noise_ceiling_is_high_for_reliable_and_nan_for_silent(tmp_path):
    rng = np.random.default_rng(0)
    n_cond, n_trials, n_bins = 40, 4, 6
    counts = np.zeros((n_cond, n_trials, n_bins, 3), dtype=np.uint8)
    signal = rng.integers(0, 6, size=(n_cond, n_bins))
    for r in range(n_trials):
        counts[:, r, :, 0] = signal                                   # perfectly reliable
        counts[:, r, :, 1] = rng.integers(0, 6, size=(n_cond, n_bins))  # pure noise
    b = load_bundle(make_bundle_dir(tmp_path, counts, np.zeros((n_cond, 5))))
    ceil = noise_ceiling(b, np.arange(n_cond))
    assert ceil[0] > 0.95
    assert ceil[1] < 0.5
    assert np.isnan(ceil[2])                                          # silent -> undefined


def test_noise_ceiling_chunking_with_near_constant_neuron(tmp_path):
    """Exercise chunking with non-divisor chunk size and near-constant high-rate neuron."""
    n_cond, n_trials, n_bins = 40, 4, 6
    counts = np.zeros((n_cond, n_trials, n_bins, 4), dtype=np.uint8)

    # Neuron 0: reliable signal
    signal = np.random.default_rng(0).integers(0, 6, size=(n_cond, n_bins))
    for r in range(n_trials):
        counts[:, r, :, 0] = signal

    # Neuron 1: pure noise (each trial independent)
    for r in range(n_trials):
        counts[:, r, :, 1] = np.random.default_rng(r + 1).integers(0, 6, size=(n_cond, n_bins))

    # Neuron 2: silent (no spikes)
    pass

    # Neuron 3: near-constant high-rate neuron
    # 40 spikes per bin in almost every trial, with tiny variation (+1 in one bin of one trial)
    counts[:, :, :, 3] = 40
    counts[5, 2, 3, 3] = 41  # Single +1 to create tiny variance

    b = load_bundle(make_bundle_dir(tmp_path, counts, np.zeros((n_cond, 5))))
    cond_idx = np.arange(n_cond)

    # Compute with different chunk sizes; they should agree
    ceil_small_chunk = noise_ceiling(b, cond_idx, chunk=7)     # non-divisor
    ceil_large_chunk = noise_ceiling(b, cond_idx, chunk=10**6)  # single chunk

    # trial_mean_rates and ever_active_mask should also match
    rates_small = trial_mean_rates(b, cond_idx, chunk=7)
    rates_large = trial_mean_rates(b, cond_idx, chunk=10**6)
    np.testing.assert_allclose(rates_small, rates_large)

    ever_small = ever_active_mask(b, cond_idx, chunk=7)
    ever_large = ever_active_mask(b, cond_idx, chunk=10**6)
    assert np.array_equal(ever_small, ever_large)

    # Noise ceiling should match with equal_nan=True (handles NaN for silent neurons)
    np.testing.assert_allclose(ceil_small_chunk, ceil_large_chunk, equal_nan=True)

    # Assert the near-constant neuron's ceiling is valid (NaN or in [0, 1], never negative)
    nc_ceil = ceil_small_chunk[3]
    assert np.isnan(nc_ceil) or (0.0 <= nc_ceil <= 1.0), \
        f"Near-constant neuron ceiling {nc_ceil} should be NaN or in [0, 1]"
