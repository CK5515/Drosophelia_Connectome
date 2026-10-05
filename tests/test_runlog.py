import json

import numpy as np

from flybrain.runlog import make_run_dir, write_metrics


def test_make_run_dir_writes_config_with_git_commit(tmp_path):
    run_dir = make_run_dir("unit_test", {"alpha": 1}, runs_root=tmp_path)
    meta = json.loads((run_dir / "config.json").read_text())
    assert meta["experiment"] == "unit_test"
    assert meta["config"] == {"alpha": 1}
    assert "git_commit" in meta
    assert run_dir.parent == tmp_path / "unit_test"


def test_two_runs_in_same_second_get_distinct_dirs(tmp_path):
    a = make_run_dir("unit_test", {}, runs_root=tmp_path)
    b = make_run_dir("unit_test", {}, runs_root=tmp_path)
    assert a != b


def test_write_metrics_handles_numpy_types(tmp_path):
    write_metrics(tmp_path, {"r2": np.float32(0.5), "n": np.int64(3), "ok": np.bool_(True)})
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics == {"r2": 0.5, "n": 3, "ok": True}
