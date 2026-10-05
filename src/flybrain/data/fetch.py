"""Download FlyWire v630 connectivity files shipped with Shiu et al.'s model."""
import urllib.request
from pathlib import Path

from flybrain.paths import RAW, ensure_free_space

BASE_URL = "https://github.com/philshiu/Drosophila_brain_model/raw/main/"
FILES = {
    "connectivity": ("2023_03_23_connectivity_630_final.parquet", 86630944),
    "completeness": ("2023_03_23_completeness_630_final.csv", 3057611),
}


def fetch_raw(raw_dir: Path = RAW) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, expected_bytes) in FILES.items():
        path = raw_dir / name
        if not path.exists() or path.stat().st_size != expected_bytes:
            ensure_free_space()
            urllib.request.urlretrieve(BASE_URL + name, path)
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"{name}: expected {expected_bytes} bytes, got {path.stat().st_size}")
        paths[key] = path
    return paths
