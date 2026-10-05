"""Split the sorted spectrum into K contiguous bands with equal mode counts."""
import numpy as np


def equal_count_bands(n_modes: int, k: int) -> list[np.ndarray]:
    if n_modes < k:
        raise ValueError(f"need at least {k} modes, got {n_modes}")
    sizes = [n_modes // k] * k
    sizes[-1] += n_modes - sum(sizes)
    edges = np.cumsum([0] + sizes)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]
