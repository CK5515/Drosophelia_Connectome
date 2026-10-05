import pytest

from flybrain.analysis.criteria import beats


def test_clear_consistent_win_passes():
    out = beats([0.50, 0.52, 0.51], [0.30, 0.31, 0.29])
    assert out["beats"] and out["all_same_sign"]
    assert out["mean_diff"] == pytest.approx(0.21, abs=0.01)


def test_noisy_difference_fails():
    assert not beats([0.5, 0.2, 0.8], [0.4, 0.6, 0.3])["beats"]


def test_one_seed_of_opposite_sign_fails_even_when_the_mean_is_large():
    a = [1.0] * 9 + [0.0]
    b = [0.0] * 9 + [0.001]
    out = beats(a, b)
    assert out["mean_diff"] > 2 * out["std_diff"]      # the magnitude rule alone would pass
    assert not out["all_same_sign"] and not out["beats"]


def test_mismatched_or_too_short_input_raises():
    with pytest.raises(ValueError):
        beats([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        beats([0.1], [0.2])
