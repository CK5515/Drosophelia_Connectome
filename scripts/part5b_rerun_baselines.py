"""Part 5 1/2 - re-run every baseline under the mask the A/B picked, so the comparison is apples to apples.

Not a gated step, just a correction of my own mistake. Ordered so that stopping it early still leaves
something useful, and the summary is rewritten after every run with complete:false until the last one lands.
"""
import json
import sys

from flybrain.paths import PROCESSED, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.train.loop import TrainConfig, train_one

LOSS_MASK = "all_observed"
SEEDS = (0, 1, 2)
ORDER = (("moe_linear", SEEDS), ("moe_single", SEEDS), ("mlp", SEEDS), ("mean", (0,)), ("chebgru", SEEDS))
KEY = "normalised_r2_observed_active"
FIELDS = (KEY, "r2_observed_active_mean", "r2_observed_active_median", "pooled_r2_observed_active",
          "r2_heldout_active_mean", "n_parameters", "n_scored_without_prediction", "mn9_trace_r2")


def lr_for(table: dict, model: str) -> float:
    key = "moe" if model.startswith("moe") else model
    return float(table[key] if model != "mean" else table["moe"])


def main() -> int:
    ensure_free_space()
    lr_file = PROCESSED / "p2_lr.json"
    lr_table = json.loads(lr_file.read_text())
    lr_source = "p2_lr.json (extended pilot with controller override: moe=0.01, chebgru=0.01)"
    run_dir = make_run_dir("p2_baselines_all_observed",
                           {"loss_mask": LOSS_MASK, "order": [[m, list(s)] for m, s in ORDER],
                            "learning_rates": lr_table, "lr_source": lr_source})
    print("run:", run_dir, "| loss_mask", LOSS_MASK, "| learning rates", lr_table, flush=True)
    results: dict = {}
    for model, seeds in ORDER:
        lr = lr_for(lr_table, model)
        per_seed = []
        results[model] = per_seed
        for seed in seeds:
            print(f"start {model} seed {seed} lr {lr}", flush=True)
            m = train_one(TrainConfig(model=model, lr=lr, seed=seed, loss_mask=LOSS_MASK))
            row = {k: m.get(k) for k in FIELDS} | {"best_epoch": m["history"]["best_epoch"],
                                                   "run_dir": m["run_dir"], "seed": seed, "lr": lr}
            per_seed.append(row)
            print(f"done {model} seed {seed}: {KEY}={row[KEY]} median={row['r2_observed_active_median']} "
                  f"best_epoch={row['best_epoch']} dir={row['run_dir']}", flush=True)
            # rewrite the summary after every run so a partial sweep is still readable
            write_metrics(run_dir, {"loss_mask": LOSS_MASK, "learning_rates": lr_table, "lr_source": lr_source,
                                    "complete": False, "results": results})
    write_metrics(run_dir, {"loss_mask": LOSS_MASK, "learning_rates": lr_table, "lr_source": lr_source,
                            "complete": True, "results": results})
    print(f"sweep complete: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
