"""Scoring: per-neuron R2 over held-out stimuli, pooled, MN9 traces, noise-ceiling normalisation."""
import warnings
import numpy as np


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])


def r2_per_neuron(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p, t = _flat(pred), _flat(target)
    sse = ((t - p) ** 2).sum(axis=0)
    sst = ((t - t.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sst > 0, 1.0 - sse / sst, np.nan)


def pooled_r2(pred: np.ndarray, target: np.ndarray, neuron_mask: np.ndarray) -> float:
    p, t = _flat(pred)[:, neuron_mask], _flat(target)[:, neuron_mask]
    sse = ((t - p) ** 2).sum()
    sst = ((t - t.mean()) ** 2).sum()
    return float(1.0 - sse / sst) if sst > 0 else float("nan")


def trace_r2(pred: np.ndarray, target: np.ndarray, neuron_idx: int) -> float:
    return float(r2_per_neuron(pred[..., [neuron_idx]], target[..., [neuron_idx]])[0])


def normalised_r2(r2: np.ndarray, ceiling: np.ndarray, min_ceiling: float = 0.1) -> tuple[float, int]:
    usable = np.isfinite(r2) & np.isfinite(ceiling) & (ceiling > min_ceiling)
    if not usable.any():
        return float("nan"), 0
    return float(np.mean(r2[usable] / ceiling[usable])), int(usable.sum())


def summarise(pred: np.ndarray, target: np.ndarray, masks: dict, ceiling: np.ndarray,
              mn9_idx: list) -> dict:
    r2 = r2_per_neuron(pred, target)
    out = {}
    for name, mask in masks.items():
        vals = r2[mask]
        if mask.any():
            with warnings.catch_warnings():
                # an all-NaN slice means every selected neuron is silent or the model diverged -> NaN by design
                warnings.simplefilter("ignore", RuntimeWarning)
                out[f"r2_{name}_mean"] = float(np.nanmean(vals))
                out[f"r2_{name}_median"] = float(np.nanmedian(vals))
        else:
            out[f"r2_{name}_mean"] = float("nan")
            out[f"r2_{name}_median"] = float("nan")
        out[f"pooled_r2_{name}"] = pooled_r2(pred, target, mask) if mask.any() else float("nan")
        norm, used = normalised_r2(np.where(mask, r2, np.nan), ceiling)
        out[f"normalised_r2_{name}"] = norm
        out[f"n_{name}"] = int(mask.sum())
        out[f"n_{name}_ceiling_usable"] = used
    out["mn9_trace_r2"] = [trace_r2(pred, target, int(i)) for i in mn9_idx]
    return out
