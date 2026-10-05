"""Where the router sends each kind of taste, and how sharp its choices are."""
import numpy as np
import torch


def categorise(stim: np.ndarray, group_names: list, dominance: float = 1.2) -> np.ndarray:
    labels = []
    for row in np.asarray(stim, dtype=float):
        if not np.any(row > 0):
            labels.append("none")
            continue
        order = np.argsort(row)[::-1]
        top, second = row[order[0]], row[order[1]]
        labels.append(group_names[order[0]] if second == 0 or top >= dominance * second else "mixed")
    return np.array(labels)


def gate_summary(model, stim: np.ndarray, n_bins: int, stim_bins: int, group_names: list,
                 device: str = "cuda", batch_size: int = 16) -> dict:
    model.to(device).eval()
    per_layer = [[] for _ in range(model.cfg.n_layers)]
    with torch.no_grad():
        for start in range(0, len(stim), batch_size):
            chunk = torch.tensor(np.asarray(stim[start:start + batch_size], dtype=np.float32), device=device)
            _, aux = model(chunk, n_bins, stim_bins, return_aux=True)
            for layer, g in enumerate(aux["gates"]):
                per_layer[layer].append(g.float().cpu().numpy())
    gates = [np.concatenate(chunks, axis=0) for chunks in per_layer]
    labels = categorise(stim, group_names)
    per_category = {}
    for label in sorted(set(labels)):
        sel = labels == label
        per_category[label] = [g[sel].mean(axis=0).tolist() for g in gates]
    entropy = []
    for g in gates:
        p = np.clip(g, 1e-12, 1.0)
        entropy.append(float((-(p * np.log(p)).sum(axis=1)).mean()))
    return {"per_category": per_category, "entropy_per_layer": entropy,
            "counts": {label: int((labels == label).sum()) for label in sorted(set(labels))}}
