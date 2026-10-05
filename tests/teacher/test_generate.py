import json
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from flybrain.teacher.generate import generate_dataset
from flybrain.teacher.lif import LIFNetwork, LIFParams
from flybrain.teacher.stimuli import StimulusConfig, sample_conditions

CFG = StimulusConfig(t_on_ms=10.0, t_total_ms=20.0, bin_ms=2.0)  # 200 steps, stim 100, 10 bins


def tiny_setup():
    rng = np.random.default_rng(0)
    dense = (rng.random((12, 12)) < 0.3) * rng.choice([-30.0, 80.0], size=(12, 12))
    dense[:, :3] = 0.0            # nothing drives the input neurons except the stimulus
    np.fill_diagonal(dense, 0.0)
    net = LIFNetwork(sp.csr_matrix(dense), LIFParams(), device="cpu")
    return net, np.array([0, 1, 2]), np.array([0, 1, 4]), np.arange(12)


def run(out_dir, conditions):
    net, inp, groups, rec = tiny_setup()
    return generate_dataset(net, inp, groups, rec, conditions, n_trials=2, cfg=CFG, batch_size=4, seed=7,
                            out_dir=out_dir, extra_meta={"note": "unit"})


def test_writes_expected_files_and_shapes(tmp_path):
    conditions = sample_conditions(6, StimulusConfig(), seed=1)
    out = run(tmp_path / "ds", conditions)
    counts = np.load(out / "counts.npy")
    assert counts.shape == (6, 2, 10, 12) and counts.dtype == np.uint8
    np.testing.assert_array_equal(np.load(out / "conditions.npy"), conditions)
    assert json.loads((out / "progress.json").read_text())["conditions_done"] == 6
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_trials"] == 2 and meta["bin_ms"] == 2.0 and meta["note"] == "unit"


def test_inputs_fall_silent_after_stimulus_off(tmp_path):
    conditions = np.full((4, 5), 200.0, dtype=np.float32)
    counts = np.load(run(tmp_path / "ds", conditions) / "counts.npy")
    assert counts[:, :, :5, :3].sum() > 0
    # a kick on the last stimulus step (99) fires at step 100 = bin 5, so silence is guaranteed from bin 6
    assert counts[:, :, 6:, :3].sum() == 0


def test_deterministic_and_resumable(tmp_path):
    conditions = sample_conditions(6, StimulusConfig(), seed=2)
    full = np.load(run(tmp_path / "a", conditions) / "counts.npy")
    np.testing.assert_array_equal(full, np.load(run(tmp_path / "b", conditions) / "counts.npy"))

    partial_dir = run(tmp_path / "c", conditions)
    counts = np.load(partial_dir / "counts.npy", mmap_mode="r+")
    counts[2:] = 0
    counts.flush()
    del counts
    (partial_dir / "progress.json").write_text(json.dumps({"conditions_done": 2}))
    resumed = np.load(run(tmp_path / "c", conditions) / "counts.npy")
    np.testing.assert_array_equal(full, resumed)


def test_condition_to_trial_mapping(tmp_path):
    """Verify condition/trial mapping through repeat/reshape is not scrambled.
    
    The reshape line converts (B, n_bins, n_rec) from net.run() into
    (n_cond, n_trials, n_bins, n_rec). If reshape axes are swapped
    (e.g. to (n_trials, n_cond, ...)), the per-condition spike patterns
    would be scrambled.
    
    This test uses asymmetric n_cond and n_trials (3 vs 2) so that
    axis swaps change memory layout. Each condition stimulates exactly
    one group at high rate (200 Hz), and that group's input neurons
    must show activity while others remain silent (zero-rate -> zero spikes).
    Assertions check each (condition, trial) cell individually.
    """
    # Create 3 conditions, each stimulating only one group at high rate
    conditions = np.zeros((3, 5), dtype=np.float32)
    conditions[0, 0] = 200.0  # condition 0: group 0 only
    conditions[1, 1] = 200.0  # condition 1: group 1 only
    conditions[2, 2] = 200.0  # condition 2: group 2 only
    
    net, inp, groups, rec = tiny_setup()
    out_dir = tmp_path / "trial_map"
    generate_dataset(net, inp, groups, rec, conditions, n_trials=2, cfg=CFG,
                     batch_size=4, seed=77, out_dir=out_dir)
    
    counts = np.load(out_dir / "counts.npy")
    assert counts.shape == (3, 2, 10, 12), f"Expected (3, 2, 10, 12), got {counts.shape}"
    
    # For each condition, check that ONLY the stimulated group's input neurons fire
    # Condition 0 maps to group 0 -> input neuron 0
    for trial in range(2):
        assert counts[0, trial, :, 0].sum() > 0, f"cond 0 trial {trial}: neuron 0 (group 0) should fire"
        assert counts[0, trial, :, 1].sum() == 0, f"cond 0 trial {trial}: neuron 1 (group 1) should be silent"
        assert counts[0, trial, :, 2].sum() == 0, f"cond 0 trial {trial}: neuron 2 (group 4) should be silent"
    
    # Condition 1 maps to group 1 -> input neuron 1
    for trial in range(2):
        assert counts[1, trial, :, 0].sum() == 0, f"cond 1 trial {trial}: neuron 0 (group 0) should be silent"
        assert counts[1, trial, :, 1].sum() > 0, f"cond 1 trial {trial}: neuron 1 (group 1) should fire"
        assert counts[1, trial, :, 2].sum() == 0, f"cond 1 trial {trial}: neuron 2 (group 4) should be silent"
    
    # Condition 2 maps to group 2 (no input neuron) -> all input neurons silent
    for trial in range(2):
        assert counts[2, trial, :, :3].sum() == 0, f"cond 2 trial {trial}: all input neurons should be silent"


def test_resume_rejects_batch_size_change(tmp_path):
    """Resuming with a different batch_size must raise ValueError."""
    conditions = sample_conditions(3, StimulusConfig(), seed=3)
    out_dir = tmp_path / "batch_mismatch"

    # Generate with batch_size=4
    net, inp, groups, rec = tiny_setup()
    generate_dataset(net, inp, groups, rec, conditions, n_trials=2, cfg=CFG,
                     batch_size=4, seed=7, out_dir=out_dir)

    # Try to resume with batch_size=2
    with pytest.raises(ValueError, match="batch_size"):
        net, inp, groups, rec = tiny_setup()
        generate_dataset(net, inp, groups, rec, conditions, n_trials=2, cfg=CFG,
                         batch_size=2, seed=7, out_dir=out_dir)


def test_overflow_raises_error(tmp_path):
    """OverflowError must be raised when spike counts exceed 255."""
    # Create a stub network that returns values > 255
    @dataclass
    class StubParams:
        dt_ms: float = 0.1

    class StubLIFNetwork:
        params = StubParams()

        def run(self, input_idx, n_steps, record_idx, bin_steps, **kwargs):
            # Return a tensor of shape (B, n_bins, n_record) with max value 300
            batch_size = kwargs['rates_hz'].shape[0]
            n_bins = n_steps // bin_steps
            n_record = len(record_idx)
            result = np.full((batch_size, n_bins, n_record), 300, dtype=np.int16)
            return torch.from_numpy(result)

    conditions = sample_conditions(2, StimulusConfig(), seed=4)
    out_dir = tmp_path / "overflow"
    net = StubLIFNetwork()
    inp = np.array([0, 1, 2])
    groups = np.array([0, 1, 4])
    rec = np.arange(12)

    with pytest.raises(OverflowError, match="exceeds uint8"):
        generate_dataset(net, inp, groups, rec, conditions, n_trials=1, cfg=CFG,
                         batch_size=4, seed=42, out_dir=out_dir)
