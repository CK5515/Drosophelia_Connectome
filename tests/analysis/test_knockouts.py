import numpy as np

from flybrain.analysis.knockouts import group_delta_by_hop


def test_delta_r2_is_grouped_by_hop_distance():
    delta = np.array([0.5, 0.1, 0.2, np.nan])
    d_in = np.array([0, 1, 1, 2])
    mask = np.array([True, True, True, True])
    out = group_delta_by_hop(delta, d_in, mask)
    assert out["0"]["mean"] == 0.5 and out["0"]["n"] == 1
    assert np.isclose(out["1"]["mean"], 0.15) and out["1"]["n"] == 2
    assert out["2"]["n"] == 0                       # only a NaN there
    assert np.isnan(out["2"]["mean"]) and np.isnan(out["2"]["median"])
