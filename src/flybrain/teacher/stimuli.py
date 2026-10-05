"""Stimulus conditions: one Poisson rate per gustatory group, on for 500 ms then off."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StimulusConfig:
    p_zero: float = 0.3
    max_rate_hz: float = 200.0
    t_on_ms: float = 500.0
    t_total_ms: float = 1000.0
    bin_ms: float = 20.0


def sample_conditions(n: int, cfg: StimulusConfig, seed: int, n_groups: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rates = cfg.max_rate_hz - rng.uniform(0.0, cfg.max_rate_hz, size=(n, n_groups))  # (0, max]
    rates[rng.random((n, n_groups)) < cfg.p_zero] = 0.0
    return rates.astype(np.float32)


def schedule_steps(cfg: StimulusConfig, dt_ms: float) -> tuple[int, int, int]:
    return (int(round(cfg.t_total_ms / dt_ms)), int(round(cfg.t_on_ms / dt_ms)), int(round(cfg.bin_ms / dt_ms)))


def expand_to_inputs(conditions: np.ndarray, group_of_input: np.ndarray) -> np.ndarray:
    return conditions[:, group_of_input]
