import pytest

from flybrain.paths import ROOT, ensure_free_space


def test_root_is_repo_root():
    assert (ROOT / "pyproject.toml").exists()


def test_free_space_guard_raises_when_threshold_unreachable():
    with pytest.raises(RuntimeError, match="GB free"):
        ensure_free_space(min_gb=1e9)


def test_free_space_guard_returns_free_gb():
    assert ensure_free_space(min_gb=0.0) > 0.0
