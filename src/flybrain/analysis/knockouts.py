"""Switch one frequency band off and see which neurons notice."""
import numpy as np


def group_delta_by_hop(delta: np.ndarray, d_in: np.ndarray, mask: np.ndarray) -> dict:
    """Average the R2 lost per neuron within each hop distance from the input. Positive means the band helped."""
    out = {}
    for level in np.unique(d_in):
        sel = mask & (d_in == level) & np.isfinite(delta)
        kept = delta[sel]
        out[str(int(level))] = {"mean": float(kept.mean()) if kept.size else float("nan"),
                                "median": float(np.median(kept)) if kept.size else float("nan"),
                                "n": int(kept.size)}
    return out


def band_knockout(model, tensors, cfg, predict_fn, r2_fn, d_in: np.ndarray) -> dict:
    """Per band: how much per-neuron R2 the model loses when that band's expert is switched off at test time."""
    mask = tensors.metric_masks["observed_active"]
    base_r2 = r2_fn(predict_fn(model, tensors.stim_test, tensors, cfg), tensors.target_test)
    results = {}
    for k in range(model.cfg.n_experts):
        with model.knockout(k):
            r2 = r2_fn(predict_fn(model, tensors.stim_test, tensors, cfg), tensors.target_test)
        delta = base_r2 - r2
        # observed_active is never empty on real data, so nanmean cannot hit an all-NaN slice here
        results[str(k)] = {"by_hop": group_delta_by_hop(delta, d_in, mask),
                           "mean_delta_observed_active": float(np.nanmean(delta[mask]))}
    return {"baseline_mean_r2": float(np.nanmean(base_r2[mask])), "per_band": results}
