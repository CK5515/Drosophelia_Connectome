import numpy as np

from flybrain.teacher.stimuli import StimulusConfig, expand_to_inputs, sample_conditions, schedule_steps

CFG = StimulusConfig()


def test_sampled_rates_follow_spec_distribution():
    c = sample_conditions(20000, CFG, seed=0)
    assert c.shape == (20000, 5) and c.dtype == np.float32
    zero_frac = (c == 0).mean()
    assert abs(zero_frac - 0.3) < 0.01
    nonzero = c[c > 0]
    assert nonzero.max() <= 200.0
    assert abs(nonzero.mean() - 100.0) < 2.0


def test_sampling_is_seeded():
    np.testing.assert_array_equal(sample_conditions(10, CFG, 3), sample_conditions(10, CFG, 3))
    assert not np.array_equal(sample_conditions(10, CFG, 3), sample_conditions(10, CFG, 4))


def test_schedule_steps_for_default_protocol():
    assert schedule_steps(CFG, dt_ms=0.1) == (10000, 5000, 200)


def test_expand_to_inputs_copies_group_rate_to_each_member():
    conditions = np.array([[10, 20, 30, 40, 50], [1, 2, 3, 4, 5]], dtype=np.float32)
    got = expand_to_inputs(conditions, np.array([0, 0, 2, 4]))
    np.testing.assert_array_equal(got, [[10, 10, 30, 50], [1, 1, 3, 5]])
