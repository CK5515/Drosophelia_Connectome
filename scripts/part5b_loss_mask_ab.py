"""Part 5 1/2 - I was wrong about silent neurons.

Two-arm A/B on moe_full: train on the 1,062 ever-active neurons, or on all 4,018 observed ones including
the permanently silent. I assumed the silent ones would flatten the signal. They do the opposite.
Writes the winning policy to data/processed/p2_loss_mask.json, which every later script reads.
"""
import json
import sys

import numpy as np

from flybrain.paths import PROCESSED, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.train.loop import TrainConfig, train_one

SEEDS = (0, 1, 2)
KEY = "normalised_r2_observed_active"


def main() -> int:
    ensure_free_space()
    lr = json.loads((PROCESSED / "p2_lr.json").read_text())["moe"]
    run_dir = make_run_dir("p2_loss_mask_ab", {"seeds": list(SEEDS), "lr": lr})
    arms = {}
    for policy in ("ever_active", "all_observed"):
        rows = []
        for seed in SEEDS:
            m = train_one(TrainConfig(model="moe_full", loss_mask=policy, lr=lr, seed=seed))
            rows.append({"seed": seed, KEY: m[KEY], "r2_observed_active_mean": m["r2_observed_active_mean"],
                         "r2_heldout_active_mean": m["r2_heldout_active_mean"],
                         "n_loss_mask": m["n_loss_mask"], "run_dir": m["run_dir"]})
            print(f"  {policy} seed {seed}: {KEY}={rows[-1][KEY]}")
        arms[policy] = rows

    a = np.array([r[KEY] for r in arms["ever_active"]], dtype=float)
    b = np.array([r[KEY] for r in arms["all_observed"]], dtype=float)
    spread = float(np.sqrt(a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2))
    diff = float(a.mean() - b.mean())
    policy = "ever_active" if diff > spread else "all_observed"
    reason = (f"ever_active minus all_observed = {diff:.4f} on {KEY}; pooled seed spread = {spread:.4f}; "
              f"adopted {policy} by the rule fixed before the runs")
    (PROCESSED / "p2_loss_mask.json").write_text(json.dumps({"policy": policy, "reason": reason,
                                                             "diff": diff, "spread": spread}, indent=2))
    write_metrics(run_dir, {"arms": arms, "diff": diff, "pooled_spread": spread, "policy": policy,
                            "reason": reason})
    print(f"run: {run_dir}\n{reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
