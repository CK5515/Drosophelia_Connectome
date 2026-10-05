"""Repository paths and the free-disk guard used by every data-writing script."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"


def ensure_free_space(path: Path = ROOT, min_gb: float = 10.0) -> float:
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free at {path}; need >= {min_gb} GB")
    return free_gb
