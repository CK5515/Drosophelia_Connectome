"""The pre-registered claim criterion from spec section 8.4. Nothing else decides a comparison."""
import numpy as np


def beats(a, b) -> dict:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired inputs must match: {x.shape} vs {y.shape}")
    if x.size < 2:
        raise ValueError("need at least two seeds to estimate the spread")
    diff = x - y
    mean, sd = float(diff.mean()), float(diff.std(ddof=1))
    same_sign = bool(np.all(diff > 0) or np.all(diff < 0))
    return {"mean_diff": mean, "std_diff": sd, "all_same_sign": same_sign,
            "beats": bool(mean > 2.0 * sd and np.all(diff > 0)), "per_seed": diff.tolist()}
