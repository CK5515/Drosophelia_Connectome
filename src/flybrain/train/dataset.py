"""Load the Plan 1 dataset and derive the splits, masks and noise ceiling every run shares."""
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flybrain.paths import PROCESSED


@dataclass(frozen=True)
class Bundle:
    counts: np.ndarray        # (C, R, T, N) uint8 memmap
    conditions: np.ndarray    # (C, 5) float32
    meta: dict
    bin_s: float

    @property
    def n_conditions(self) -> int:
        return self.counts.shape[0]

    @property
    def n_trials(self) -> int:
        return self.counts.shape[1]

    @property
    def n_bins(self) -> int:
        return self.counts.shape[2]

    @property
    def n_neurons(self) -> int:
        return self.counts.shape[3]


def load_bundle(out_dir: Path = PROCESSED / "dataset_v1") -> Bundle:
    meta = json.loads((Path(out_dir) / "meta.json").read_text())
    return Bundle(
        counts=np.load(Path(out_dir) / "counts.npy", mmap_mode="r"),
        conditions=np.load(Path(out_dir) / "conditions.npy"),
        meta=meta,
        bin_s=float(meta["bin_ms"]) / 1000.0,
    )


def condition_splits(n_conditions: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n_conditions)
    n_train = int(round(0.70 * n_conditions))
    n_val = int(round(0.15 * n_conditions))
    parts = (order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:])
    return tuple(np.sort(p) for p in parts)


def neuron_splits(d_in: np.ndarray, protected: np.ndarray, seed: int = 0,
                  frac_heldout: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    protected = np.asarray(protected)
    heldout = []
    for level in np.unique(d_in):
        members = np.setdiff1d(np.flatnonzero(d_in == level), protected)
        k = int(round(frac_heldout * len(members)))
        if k:
            heldout.extend(rng.choice(members, size=k, replace=False).tolist())
    heldout = np.sort(np.array(heldout, dtype=np.int64))
    observed = np.setdiff1d(np.arange(len(d_in)), heldout)
    return observed, heldout


def trial_mean_rates(bundle: Bundle, cond_idx: np.ndarray, chunk: int = 256) -> np.ndarray:
    cond_idx = np.asarray(cond_idx)
    out = np.empty((len(cond_idx), bundle.n_bins, bundle.n_neurons), dtype=np.float32)
    for start in range(0, len(cond_idx), chunk):
        block = np.asarray(bundle.counts[cond_idx[start:start + chunk]], dtype=np.float32)
        out[start:start + len(block)] = block.mean(axis=1) / bundle.bin_s
    return out


def ever_active_mask(bundle: Bundle, cond_idx: np.ndarray, chunk: int = 256) -> np.ndarray:
    cond_idx = np.asarray(cond_idx)
    out = np.zeros(bundle.n_neurons, dtype=bool)
    for start in range(0, len(cond_idx), chunk):
        block = np.asarray(bundle.counts[cond_idx[start:start + chunk]])
        out |= block.any(axis=(0, 1, 2))
    return out


def active_mask(rates: np.ndarray, threshold_hz: float = 0.5) -> np.ndarray:
    flat = rates.reshape(-1, rates.shape[-1])
    return flat.std(axis=0) > threshold_hz


def _half_means(chunk: np.ndarray, a: tuple[int, int], b: tuple[int, int], bin_s: float):
    first = chunk[:, list(a)].mean(axis=1) / bin_s
    second = chunk[:, list(b)].mean(axis=1) / bin_s
    return first.reshape(-1, chunk.shape[-1]), second.reshape(-1, chunk.shape[-1])


def noise_ceiling(bundle: Bundle, cond_idx: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Split-half reliability over the three 2-vs-2 trial pairings, Spearman-Brown corrected."""
    cond_idx = np.asarray(cond_idx)
    pairings = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))

    per_split = []
    for a, b in pairings:
        # Pass 1: Compute means in float64
        sum_x = np.zeros(bundle.n_neurons, dtype=np.float64)
        sum_y = np.zeros(bundle.n_neurons, dtype=np.float64)
        n_samples = 0

        for start in range(0, len(cond_idx), chunk):
            chunk_data = np.asarray(bundle.counts[cond_idx[start:start + chunk]], dtype=np.float32)
            x, y = _half_means(chunk_data, a, b, bundle.bin_s)
            sum_x += np.asarray(x, dtype=np.float64).sum(axis=0)
            sum_y += np.asarray(y, dtype=np.float64).sum(axis=0)
            n_samples += len(x)

        mean_x = sum_x / n_samples
        mean_y = sum_y / n_samples

        # Pass 2: Accumulate centred quantities in float64
        sum_xx = np.zeros(bundle.n_neurons, dtype=np.float64)
        sum_yy = np.zeros(bundle.n_neurons, dtype=np.float64)
        sum_xy = np.zeros(bundle.n_neurons, dtype=np.float64)

        for start in range(0, len(cond_idx), chunk):
            chunk_data = np.asarray(bundle.counts[cond_idx[start:start + chunk]], dtype=np.float32)
            x, y = _half_means(chunk_data, a, b, bundle.bin_s)
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            xc = x - mean_x
            yc = y - mean_y
            sum_xx += (xc ** 2).sum(axis=0)
            sum_yy += (yc ** 2).sum(axis=0)
            sum_xy += (xc * yc).sum(axis=0)

        denom = np.sqrt(sum_xx * sum_yy)
        with np.errstate(invalid="ignore", divide="ignore"):
            r_split = np.where(denom > 0, sum_xy / denom, np.nan)
        per_split.append(r_split)

    with warnings.catch_warnings():
        # all-NaN columns are the intended silent-neuron case, not an error
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.nanmean(np.stack(per_split), axis=0)
    with np.errstate(invalid="ignore"):
        corrected = 2.0 * r / (1.0 + r)
    return np.clip(corrected, 0.0, 1.0) * np.where(np.isnan(r), np.nan, 1.0)
