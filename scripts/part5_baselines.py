"""Part 5 - the baselines, and the check that the task is worth running at all.

Picks a learning rate, trains mean / mlp / chebgru / moe_single / moe_linear on three seeds each, and
verifies the per-neuron-mean floor scores below 0.8 - otherwise a trivial predictor already sits at the
ceiling and nothing downstream means anything. Pass --replot to redraw the figure from saved metrics.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flybrain import plotting
from flybrain.paths import PROCESSED, RUNS, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.train.loop import TrainConfig, train_one

PILOT_LRS = (3e-4, 1e-3, 3e-3)
PILOT_CONDITIONS = 300
PILOT_EPOCHS = 6
SEEDS = (0, 1, 2)
BASELINES = ("mean", "mlp", "chebgru", "moe_linear", "moe_single")
KEY = "normalised_r2_observed_active"


def pilot(model: str) -> tuple[float, list]:
    rows = []
    for lr in PILOT_LRS:
        m = train_one(TrainConfig(model=model, lr=lr, seed=0, n_conditions=PILOT_CONDITIONS,
                                  max_epochs=PILOT_EPOCHS, patience=PILOT_EPOCHS, tag="_pilot"))
        val = m["history"]["val_loss"][-1] if m["history"]["val_loss"] else float("nan")
        rows.append({"lr": lr, "final_val_loss": val, "run_dir": m["run_dir"]})
        print(f"  pilot {model} lr={lr}: val {val:.5f}")
    best = min(rows, key=lambda r: r["final_val_loss"])
    return float(best["lr"]), rows


PANELS = [("normalised_r2_observed_active", "noise-normalised R² (headline)"),
          ("r2_observed_active_mean", "per-neuron R², mean"),
          ("r2_observed_active_median", "per-neuron R², median"),
          ("pooled_r2_observed_active", "pooled R² (all neurons at once)")]


def make_figure(results: dict) -> None:
    """Four views of the same runs: the mean is dominated by a tail of near-silent neurons, so show median and pooled too."""
    per_run = {name: [json.loads((Path(r["run_dir"]) / "metrics.json").read_text()) for r in rs]
               for name, rs in results.items()}
    plotting.apply_style()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(13, 3.8))
    names = list(BASELINES)
    for ax, (key, title) in zip(axes, PANELS):
        for i, name in enumerate(names):
            vals = [m[key] for m in per_run[name] if m.get(key) is not None and np.isfinite(m[key])]
            if not vals:
                continue
            ax.bar(i, np.mean(vals), color=plotting.SERIES[0], width=0.6, edgecolor=plotting.SURFACE)
            ax.scatter([i] * len(vals), vals, color=plotting.TEXT_SECONDARY, zorder=3, s=16)
        floor = per_run["mean"][0][key]
        ax.axhline(floor, color=plotting.SERIES[1], lw=1.2, ls="--", zorder=2)
        ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("held-out stimuli, observed-active neurons")
    fig.suptitle("R4 baselines: bars are seed means, dots are seeds, dashed line is the per-neuron-mean floor", y=1.03)
    fig.tight_layout()
    plotting.save_figure(fig, "r4_baselines.png")


def main() -> int:
    if "--replot" in sys.argv:  # redraw from an existing run, never train
        # a bare --replot means "the newest run"; passing a run dir picks that one instead. Falling through
        # to training here once cost 17 hours of GPU and a pile of junk run folders, so it fails loudly now.
        rest = [a for a in sys.argv[1:] if a != "--replot"]
        if rest:
            run = Path(rest[0])
        else:
            runs = sorted(RUNS.glob("p2_r4_baselines/*/metrics.json"))
            if not runs:
                raise FileNotFoundError("nothing to replot: no p2_r4_baselines run has a metrics.json yet")
            run = runs[-1].parent
        make_figure(json.loads((run / "metrics.json").read_text())["results"])
        print(f"redrew the figure from {run}")
        return 0
    ensure_free_space()
    run_dir = make_run_dir("p2_r4_baselines", {"pilot_lrs": list(PILOT_LRS), "seeds": list(SEEDS),
                                              "baselines": list(BASELINES)})
    lr_file = PROCESSED / "p2_lr.json"
    pilots = {}
    if lr_file.exists():
        lr_table = json.loads(lr_file.read_text())
        lr_source = "p2_lr.json (extended pilot with controller override: moe=0.01, chebgru=0.01)"
        print("reusing learning rates from", lr_file, lr_table)
    else:
        lr_table = {}
        lr_source = "internal 3-point pilot"
        for model in ("moe_linear", "chebgru", "mlp"):
            lr, rows = pilot(model)
            pilots[model] = rows
            lr_table["moe" if model == "moe_linear" else model] = lr
        lr_file.write_text(json.dumps(lr_table, indent=2))
        print("chosen learning rates:", lr_table)

    results = {}
    for model in BASELINES:
        lr = lr_table.get(model, lr_table["moe"]) if model != "mean" else lr_table["moe"]
        per_seed = []
        for seed in SEEDS if model != "mean" else (0,):
            m = train_one(TrainConfig(model=model, lr=lr, seed=seed))
            per_seed.append({k: m[k] for k in (KEY, "r2_observed_active_mean", "r2_heldout_active_mean",
                                               "pooled_r2_observed_active", "n_parameters",
                                               "mn9_trace_r2")} | {"run_dir": m["run_dir"], "seed": seed})
            print(f"  {model} seed {seed}: {KEY}={per_seed[-1][KEY]}")
        results[model] = per_seed

    floor = results["mean"][0][KEY]
    checks = {"mean_predictor_below_0.8": bool(np.isfinite(floor) and floor < 0.8)}
    metrics = {"checks": checks, "pass": all(checks.values()), "learning_rates": lr_table, "lr_source": lr_source,
               "pilots": pilots, "results": results, "floor_normalised_r2": floor}
    write_metrics(run_dir, metrics)

    make_figure(results)

    print(f"run: {run_dir}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name} (floor = {floor})")
    print("R4 GATE:", "PASS" if metrics["pass"] else "FAIL")
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
