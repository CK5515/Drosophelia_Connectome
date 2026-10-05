"""Run folders: every script records its resolved config, git commit and metrics."""
import json
import subprocess
import time
from pathlib import Path

from flybrain.paths import ROOT, RUNS


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _jsonable(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def make_run_dir(experiment: str, config: dict, runs_root: Path = RUNS) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / experiment / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = runs_root / experiment / f"{stamp}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    meta = {"experiment": experiment, "timestamp": stamp, "git_commit": git_commit(), "config": config}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, default=_jsonable))
    return run_dir


def write_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_jsonable))
