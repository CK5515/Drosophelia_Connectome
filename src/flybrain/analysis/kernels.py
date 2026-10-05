"""What each expert learned: decay timescale, oscillation frequency, damping regime, impulse response."""
import numpy as np
import torch
import torch.nn.functional as F

from flybrain.models.wave import impulse_response


def expert_kernels(model, n_steps: int = 50, bin_ms: float = 20.0) -> dict:
    mu = model.mu.detach().cpu().numpy()
    band_of_mode = model.band_of_mode.detach().cpu().numpy()
    layers = []
    for layer in range(model.cfg.n_layers):
        gamma = F.softplus(model.gamma_tilde[layer].detach()).cpu().numpy()      # (K, C)
        alpha = F.softplus(model.alpha_tilde[layer].detach()).cpu().numpy()
        experts = []
        for k in range(gamma.shape[0]):
            g, a = float(np.median(gamma[k])), float(np.median(alpha[k]))
            band_mu = float(np.median(mu[band_of_mode == k])) if (band_of_mode == k).any() else 0.0
            disc = a * band_mu - g ** 2
            regime = "underdamped" if disc > 1e-9 else ("critical" if abs(disc) <= 1e-9 else "overdamped")
            experts.append({"expert": k, "gamma": g, "alpha": a, "band_median_mu": band_mu,
                            "timescale_ms": float(bin_ms / g) if g > 0 else float("inf"),
                            "omega_per_bin": float(np.sqrt(disc)) if disc > 0 else 0.0,
                            "regime": regime,
                            "gamma_per_channel": gamma[k].tolist(),
                            "alpha_per_channel": alpha[k].tolist(),
                            "impulse": impulse_response(band_mu, g, a, n_steps).tolist()})
        layers.append({"layer": layer, "experts": experts})
    return {"layers": layers, "bin_ms": bin_ms}


def specialisation(kernels: dict, factor: float = 2.0) -> dict:
    best = {"specialised": False, "max_ratio": 0.0, "layer": -1}
    for layer in kernels["layers"]:
        scales = [e["timescale_ms"] for e in layer["experts"] if np.isfinite(e["timescale_ms"])]
        if len(scales) < 2:
            continue
        ratio = max(scales) / min(scales)
        if ratio > best["max_ratio"]:
            best = {"specialised": bool(ratio >= factor), "max_ratio": float(ratio),
                    "layer": layer["layer"]}
    return best
