"""Part 3 - generate the dataset. 3000 taste cocktails x 4 trials x 50 bins of 20 ms x 5000 neurons.

About 3 GB, mostly zeros - 73.4% of the circuit never fires at all. Refuses to run unless Part 2 passed.
"""
import argparse
import json
import sys
import time

import numpy as np

from flybrain.data.connectome import load_or_build_connectome
from flybrain.data.fetch import fetch_raw
from flybrain.data.neurons import INPUT_GROUPS
from flybrain.data.subcircuit import load_subcircuit
from flybrain.paths import PROCESSED, ensure_free_space
from flybrain.runlog import git_commit, make_run_dir, write_metrics
from flybrain.teacher.generate import generate_dataset
from flybrain.teacher.lif import LIFNetwork, LIFParams
from flybrain.teacher.stimuli import StimulusConfig, sample_conditions


def main() -> int:
    gate = json.loads((PROCESSED / "r2_gate.json").read_text())
    if not gate["pass"]:
        print("R2 gate has not passed; refusing to generate the dataset.")
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conditions", type=int, default=3000)
    ap.add_argument("--n-trials", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=gate["batch_size"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROCESSED / "dataset_v1"))
    args = ap.parse_args()
    if args.n_conditions < 1000:
        print("spec floor is 1000 conditions")
        return 1

    ensure_free_space()
    paths = fetch_raw()
    conn = load_or_build_connectome(paths["connectivity"], paths["completeness"])
    sub = load_subcircuit(PROCESSED / "subcircuit.npz")
    group_names = list(INPUT_GROUPS)
    input_idx = np.concatenate([conn.index_of(INPUT_GROUPS[g]) for g in group_names])
    group_of_input = np.concatenate([np.full(len(INPUT_GROUPS[g]), i) for i, g in enumerate(group_names)])
    cfg = StimulusConfig()
    conditions = sample_conditions(args.n_conditions, cfg, seed=args.seed, n_groups=len(group_names))
    run_dir = make_run_dir("r2_dataset", vars(args))

    t0 = time.time()
    out = generate_dataset(
        LIFNetwork(conn.W, LIFParams(), device="cuda"), input_idx, group_of_input, sub.nodes, conditions,
        n_trials=args.n_trials, cfg=cfg, batch_size=args.batch_size, seed=args.seed, out_dir=args.out,
        extra_meta={"group_names": group_names, "subcircuit_flywire_ids": conn.flywire_ids[sub.nodes].tolist(),
                    "mn9_local": sub.mn9_local.tolist(),
                    "input_local": {g: v.tolist() for g, v in sub.input_local.items()},
                    "git_commit": git_commit()})
    wall = time.time() - t0

    counts = np.load(out / "counts.npy", mmap_mode="r")
    rate_hz = counts.mean(axis=(0, 1, 2), dtype=np.float64) / (cfg.bin_ms / 1000)
    # NOTE: MN9 stats use only the first of the two MN9 neurons (sub.mn9_local[0], the one Shiu et al. report, FlyWire 720575940660219265).
    mn9_on = counts[:, :, :int(cfg.t_on_ms / cfg.bin_ms), sub.mn9_local[0]].mean(axis=(1, 2), dtype=np.float64) / (cfg.bin_ms / 1000)
    metrics = {"wall_seconds": wall, "n_conditions": args.n_conditions, "n_trials": args.n_trials,
               "shape": list(counts.shape), "bytes": int(counts.nbytes),
               "frac_neurons_silent": float((rate_hz == 0).mean()),
               "frac_neurons_below_1hz": float((rate_hz < 1).mean()),
               "median_neuron_rate_hz": float(np.median(rate_hz)),
               "mn9_on_rate_hz_percentiles": {p: float(np.percentile(mn9_on, p)) for p in (10, 50, 90)},
               "max_bin_count": int(counts.max())}
    write_metrics(run_dir, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"run: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
