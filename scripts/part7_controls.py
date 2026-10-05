"""Part 7 - does the connectome matter, does direction matter, and which bands carry the signal?

Retrains the full model on the rewired graphs and on the direction-blind symmetric basis, then switches each
frequency band off at test time on the trained seed-0 model. The knockouts run first: they need nothing from
the retraining, and finding a broken checkpoint 14 GPU-hours in would be a bad evening.
Pass --replot to redraw both figures from saved metrics.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flybrain import plotting
from flybrain.analysis.criteria import beats
from flybrain.analysis.knockouts import band_knockout
from flybrain.data.subcircuit import load_subcircuit
from flybrain.paths import PROCESSED, RUNS, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.train.loop import TrainConfig, load_run_model, predict, train_one
from flybrain.train.metrics import r2_per_neuron

SEEDS = (0, 1, 2)
KEY = "normalised_r2_observed_active"
REWIRING_KEYS = ("frac_edges_displaced", "in_degree_preserved", "out_degree_preserved",
                 "sign_counts_preserved", "weight_multiset_preserved")

DISCLOSURES = [
    "reciprocity is not preserved: 52.9% of reciprocal pairs in the real subcircuit have opposite signs, and the "
    "swap procedure does not protect that structure, so a real-beats-rewired gap means 'something beyond degree "
    "and sign', not specifically reciprocity",
    "unweighted in-degree and out-degree are preserved exactly, and weighted out-strength is preserved because "
    "weights travel with their source edge, but weighted in-strength is not",
    "swaps_per_edge=10.0 counts attempted swaps, not accepted ones; swaps that would create a self-loop or a "
    "duplicate edge are skipped, so the realised displacement is the number that matters and is reported per seed",
    "one rewired graph is drawn per seed, so the three rewired runs differ in both initialisation and graph; this "
    "is deliberate - a single fixed rewiring would conflate 'this graph is easy' with 'rewired graphs are easy'",
]


def latest_metrics(pattern: str, with_dir: bool = False):
    paths = sorted(glob.glob(str(RUNS / pattern / "*" / "metrics.json")))
    if not paths:
        raise FileNotFoundError(f"no run matching {pattern}")
    metrics = json.loads(Path(paths[-1]).read_text())
    return (metrics, str(Path(paths[-1]).parent)) if with_dir else metrics


def floor_normalised_r2() -> float | None:
    """The mean predictor's score, for a reference line: all three arms are far below it and should look it."""
    try:
        sweep = latest_metrics("p2_baselines_all_observed")
        value = float(sweep["results"]["mean"][0][KEY])
        return value if np.isfinite(value) else None
    except (FileNotFoundError, KeyError, IndexError, TypeError, ValueError):
        return None


def check_controls_ready() -> dict:
    """Refuse before any training if a control graph is missing or the rewiring run failed; return its provenance."""
    rewired = PROCESSED / "rewired"
    missing = [str(rewired / f"{stem}_seed{s}.npz") for s in SEEDS for stem in ("W", "basis_magnetic")
               if not (rewired / f"{stem}_seed{s}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"rewired control inputs missing, run scripts/part7_rewire_graph.py first: {missing}")
    summary, folder = latest_metrics("p2_rewired", with_dir=True)
    if summary.get("pass") is not True:
        raise RuntimeError(f"the rewiring run {folder} did not pass its checks (pass={summary.get('pass')!r}); "
                           "refusing to train on its graphs")
    # JSON turns the integer seed keys of the rewiring reports into strings
    absent = [s for s in SEEDS if str(s) not in summary["reports"]]
    if absent:
        raise RuntimeError(f"the rewiring run {folder} has no report for seeds {absent}")
    return {"run_dir": folder,
            "per_seed": {str(s): {k: summary["reports"][str(s)][k] for k in REWIRING_KEYS} for s in SEEDS}}


def make_figures(metrics: dict, floor: float | None) -> None:
    """Draw both R6 figures from a metrics dict, so they can be redrawn without retraining anything."""
    real, arms, knock = metrics["real_normalised_r2"], metrics["arms"], metrics["knockouts"]
    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = ["real graph\nmagnetic", "rewired\nmagnetic", "real graph\nsymmetric"]
    series = [real, [r[KEY] for r in arms["rewired"]], [r[KEY] for r in arms["symmetric"]]]
    for i, vals in enumerate(series):
        ax.bar(i, np.mean(vals), width=0.6, color=plotting.SERIES[i % len(plotting.SERIES)],
               edgecolor=plotting.SURFACE)
        ax.scatter([i] * len(vals), vals, color=plotting.TEXT_SECONDARY, zorder=3, s=18)
    if floor is not None:
        # without the floor on the plot, "real beats rewired" reads as a win rather than as less bad
        ax.axhline(floor, color=plotting.TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(2.45, floor, f"mean-predictor floor {floor:.3f}", fontsize=7, va="bottom", ha="right",
                color=plotting.TEXT_SECONDARY)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("noise-normalised R²")
    ax.set_title("R6 controls: bars are seed means, dots are seeds")
    plotting.save_figure(fig, "r6_controls.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    # the by_hop keys are strings, so sort them as integers or hop 10 would land before hop 2
    hops = sorted(knock["per_band"]["0"]["by_hop"], key=int)
    styles = ["-", "--", "-.", ":"]
    for k in sorted(knock["per_band"], key=int):
        means = [knock["per_band"][k]["by_hop"][h]["mean"] for h in hops]
        # bands that the gate never uses sit exactly on zero and would hide under one another
        ax.plot([int(h) for h in hops], means, marker="o", linestyle=styles[int(k) % len(styles)],
                color=plotting.SERIES[int(k) % len(plotting.SERIES)], label=f"band {int(k) + 1}")
    ax.axhline(0.0, color=plotting.TEXT_SECONDARY, linewidth=1)
    ax.set_xticks([int(h) for h in hops])
    ax.set_xlabel("hops from the taste neurons")
    ax.set_ylabel("R² lost when the band is switched off\n(negative: the model does better without it)")
    ax.set_title("Switching each band off, seed 0")
    ax.legend(fontsize=8)
    plotting.save_figure(fig, "r6_knockouts.png")


def main() -> int:
    ensure_free_space()
    rewiring = check_controls_ready()
    lr = json.loads((PROCESSED / "p2_lr.json").read_text())["moe"]
    policy = json.loads((PROCESSED / "p2_loss_mask.json").read_text())["policy"]
    r5, r5_dir = latest_metrics("p2_r5_moe", with_dir=True)
    real = r5["headline_normalised_r2"]
    # beats() pairs by position, so the real scores must be in seed order
    if [r["seed"] for r in r5["headline_rows"]] != list(SEEDS):
        raise RuntimeError(f"{r5_dir}: headline rows are not seeds {list(SEEDS)} in order")
    run_dir = make_run_dir("p2_r6_controls", {"seeds": list(SEEDS), "lr": lr, "loss_mask": policy,
                                              "r5_source": r5_dir, "rewiring_source": rewiring["run_dir"]})

    # The knockouts need only R5's already-trained model, so they run BEFORE the control training. Left at the
    # end, a stale run folder or a fingerprint mismatch would surface 14 GPU-hours in and throw away the arms.
    sub = load_subcircuit(PROCESSED / "subcircuit.npz")
    knockout_dir = r5["headline_rows"][0]["run_dir"]
    model, tensors, cfg = load_run_model(knockout_dir)
    knock = band_knockout(model, tensors, cfg, predict, r2_per_neuron, sub.d_in)
    del model, tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()          # hand the headline model's memory back before the control runs
    print(f"  knockouts done on {knockout_dir}: baseline mean R2 {knock['baseline_mean_r2']:.4f}")

    fixed = {"real_normalised_r2": real, "knockouts": knock, "knockout_run_dir": knockout_dir,
             "rewiring": rewiring, "disclosures": DISCLOSURES}
    arms: dict = {}
    write_metrics(run_dir, fixed | {"complete": False, "arms": arms, "verdicts": {}})

    for label, kwargs in (("rewired", {"graph": "rewired"}), ("symmetric", {"basis": "symmetric"})):
        rows = []
        arms[label] = rows
        for seed in SEEDS:
            m = train_one(TrainConfig(model="moe_full", loss_mask=policy, lr=lr, seed=seed, **kwargs))
            rows.append({"seed": seed, KEY: m[KEY], "run_dir": m["run_dir"]})
            print(f"  {label} seed {seed}: {KEY}={rows[-1][KEY]}", flush=True)
            # rewrite after every run so an interrupted R6 is still readable
            write_metrics(run_dir, fixed | {"complete": False, "arms": arms, "verdicts": {}})

    verdicts = {"real_beats_rewired": beats(real, [r[KEY] for r in arms["rewired"]]),
                "magnetic_beats_symmetric": beats(real, [r[KEY] for r in arms["symmetric"]])}
    write_metrics(run_dir, fixed | {"complete": True, "arms": arms, "verdicts": verdicts})

    make_figures(fixed | {"arms": arms}, floor_normalised_r2())

    print(f"run: {run_dir}")
    for name, v in verdicts.items():
        print(f"  {name}: {v['beats']} (mean diff {v['mean_diff']:.4f}, 2sd {2 * v['std_diff']:.4f})")
    return 0


if __name__ == "__main__":
    if "--replot" in sys.argv:
        plotted, where = latest_metrics("p2_r6_controls", with_dir=True)
        make_figures(plotted, floor_normalised_r2())
        print(f"redrew both figures from {where}")
        sys.exit(0)
    sys.exit(main())
