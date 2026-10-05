"""Write the supervised dataset: per-trial binned spike counts for every subcircuit neuron."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from flybrain.teacher.lif import LIFNetwork
from flybrain.teacher.stimuli import StimulusConfig, expand_to_inputs, schedule_steps


def generate_dataset(net: LIFNetwork, input_idx: np.ndarray, group_of_input: np.ndarray, record_idx: np.ndarray,
                     conditions: np.ndarray, n_trials: int, cfg: StimulusConfig, batch_size: int, seed: int,
                     out_dir: Path, extra_meta: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_steps, stim_steps, bin_steps = schedule_steps(cfg, net.params.dt_ms)
    n_bins, n_cond, n_rec = n_steps // bin_steps, len(conditions), len(record_idx)
    counts_path, progress_path = out_dir / "counts.npy", out_dir / "progress.json"

    if counts_path.exists() and progress_path.exists():
        counts = np.load(counts_path, mmap_mode="r+")
        done = json.loads(progress_path.read_text())["conditions_done"]
        meta = json.loads((out_dir / "meta.json").read_text())
        if meta["n_conditions"] != n_cond:
            raise ValueError(f"cannot resume: n_conditions was {meta['n_conditions']}, now {n_cond}")
        if meta["n_trials"] != n_trials:
            raise ValueError(f"cannot resume: n_trials was {meta['n_trials']}, now {n_trials}")
        if meta["n_bins"] != n_bins:
            raise ValueError(f"cannot resume: n_bins was {meta['n_bins']}, now {n_bins}")
        if meta["batch_size"] != batch_size:
            raise ValueError(f"cannot resume: batch_size was {meta['batch_size']}, now {batch_size}")
        if meta["seed"] != seed:
            raise ValueError(f"cannot resume: seed was {meta['seed']}, now {seed}")
        on_disk_conditions = np.load(out_dir / "conditions.npy")
        if not np.array_equal(on_disk_conditions, np.asarray(conditions, dtype=np.float32)):
            raise ValueError("cannot resume: conditions array differs from on-disk conditions.npy")
    else:
        counts = np.lib.format.open_memmap(counts_path, mode="w+", dtype=np.uint8,
                                           shape=(n_cond, n_trials, n_bins, n_rec))
        np.save(out_dir / "conditions.npy", conditions.astype(np.float32))
        meta = {"n_conditions": n_cond, "n_trials": n_trials, "n_bins": n_bins, "bin_ms": cfg.bin_ms,
                "stimulus": asdict(cfg), "lif": asdict(net.params), "seed": seed, "batch_size": batch_size,
                "input_idx": np.asarray(input_idx).tolist(), "group_of_input": np.asarray(group_of_input).tolist(),
                "record_idx": np.asarray(record_idx).tolist(), **(extra_meta or {})}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        done = 0
        progress_path.write_text(json.dumps({"conditions_done": 0}))

    per_chunk = max(1, batch_size // n_trials)
    for start in range(done, n_cond, per_chunk):
        stop = min(start + per_chunk, n_cond)
        rates = expand_to_inputs(np.repeat(conditions[start:stop], n_trials, axis=0), group_of_input)
        out = net.run(input_idx, n_steps, record_idx, bin_steps, rates_hz=torch.as_tensor(rates),
                      stim_steps=stim_steps, seed=seed * 1_000_003 + start).cpu().numpy()
        if out.max() > 255:
            raise OverflowError(f"bin count {out.max()} exceeds uint8 in conditions {start}:{stop}")
        counts[start:stop] = out.reshape(stop - start, n_trials, n_bins, n_rec).astype(np.uint8)
        counts.flush()
        progress_path.write_text(json.dumps({"conditions_done": stop}))
    return out_dir
