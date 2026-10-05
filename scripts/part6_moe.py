"""Part 6 - the thing the whole project was built for.

Scores the full spectral MoE against its own ablations under the two rules I fixed before looking at any of
it, then reads the trained weights: what timescale each expert learned, and where the router sends its
weight. No training happens here. Pass --replot to redraw both figures from saved metrics.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flybrain import plotting
from flybrain.analysis.criteria import beats
from flybrain.analysis.gates import gate_summary
from flybrain.analysis.kernels import expert_kernels, specialisation
from flybrain.paths import PROCESSED, RUNS, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.train.loop import load_run_model

KEY = "normalised_r2_observed_active"


def latest_metrics(pattern: str, with_dir: bool = False):
    paths = sorted(glob.glob(str(RUNS / pattern / "*" / "metrics.json")))
    if not paths:
        raise FileNotFoundError(f"no run matching {pattern}")
    metrics = json.loads(Path(paths[-1]).read_text())
    return (metrics, str(Path(paths[-1]).parent)) if with_dir else metrics


def make_figures(metrics: dict) -> None:
    """Draw both R5 figures from a metrics dict, so they can be redrawn without re-running the analysis."""
    kernels, gates = metrics["kernels"], metrics["gates"]
    seeds = [r["seed"] for r in metrics["headline_rows"]]
    plotting.apply_style()

    # one row per layer, one column per seed: the layer-2 kernels carry criterion 2's qualifying ratios,
    # and a legend per panel because each seed learned its own timescales
    n_layers = len(kernels[0]["layers"])
    fig, axes = plt.subplots(n_layers, len(kernels), figsize=(4.2 * len(kernels), 3.0 * n_layers),
                             sharex=True, squeeze=False)
    for col, (k, seed) in enumerate(zip(kernels, seeds)):
        for row, layer in enumerate(k["layers"]):
            ax = axes[row][col]
            for e in layer["experts"]:
                ax.plot(np.arange(len(e["impulse"])) * k["bin_ms"], e["impulse"],
                        color=plotting.SERIES[e["expert"] % len(plotting.SERIES)],
                        label=f"band {e['expert'] + 1}: {e['timescale_ms']:.0f} ms, {e['regime'][:5]}")
            ax.set_title(f"seed {seed}, layer {row + 1}")
            ax.legend(fontsize=6)
            if col == 0:
                ax.set_ylabel("impulse response")
            if row == n_layers - 1:
                ax.set_xlabel("time (ms)")
    plotting.save_figure(fig, "r5_expert_kernels.png")

    # One panel per seed per layer. Averaging gate weights ACROSS seeds would be a lie: each seed collapses
    # onto its own band, and the mean of three different collapses looks like soft routing that never happened.
    cats = sorted(gates[0]["per_category"])
    n_bands = len(gates[0]["per_category"][cats[0]][0])
    width = 0.8 / n_bands
    fig, axes = plt.subplots(n_layers, len(gates), figsize=(4.6 * len(gates), 3.0 * n_layers),
                             sharey=True, squeeze=False)
    for col, (g, seed) in enumerate(zip(gates, seeds)):
        for row in range(n_layers):
            ax = axes[row][col]
            weights = np.array([g["per_category"][c][row] for c in cats])
            for band in range(n_bands):
                ax.bar(np.arange(len(cats)) + band * width, weights[:, band], width=width,
                       color=plotting.SERIES[band % len(plotting.SERIES)], label=f"band {band + 1}",
                       edgecolor=plotting.SURFACE)
            ticks = np.arange(len(cats)) + (n_bands - 1) * width / 2
            # only the bottom row carries category labels, or they collide with the next row's titles
            ax.set_xticks(ticks, cats if row == n_layers - 1 else [""] * len(cats), rotation=30, ha="right",
                          fontsize=7)
            ax.set_title(f"seed {seed}, layer {row + 1}: entropy {g['entropy_per_layer'][row]:.3f} "
                         f"of max {np.log(n_bands):.3f}", fontsize=9)
            if col == 0:
                ax.set_ylabel("mean gate weight")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    plotting.save_figure(fig, "r5_gates.png")


def main() -> int:
    ensure_free_space()
    policy = json.loads((PROCESSED / "p2_loss_mask.json").read_text())["policy"]
    ab = latest_metrics("p2_loss_mask_ab")
    # criterion 1 is a paired comparison, so the arms must share one loss-mask policy: R4's arms were trained
    # under ever_active, which the A/B showed to be the losing policy, so using them would measure the mask,
    # not the architecture. Use the all_observed sweep instead.
    comparison_source = "p2_baselines_all_observed"
    sweep, comparison_dir = latest_metrics(comparison_source, with_dir=True)
    if sweep["complete"] is not True:
        raise RuntimeError(f"{comparison_dir} is not complete; wait for the sweep to finish before running R5")
    # a paired comparison across two different loss masks measures the mask, not the architecture
    if sweep["loss_mask"] != policy:
        raise RuntimeError(f"comparison sweep {comparison_dir} was trained with loss mask {sweep['loss_mask']!r} "
                           f"but the headline runs use {policy!r}")
    full_rows = ab["arms"][policy]
    run_dir = make_run_dir("p2_r5_moe", {"loss_mask": policy,
                                         "headline_runs": [r["run_dir"] for r in full_rows]})

    full = [r[KEY] for r in full_rows]
    linear = [r[KEY] for r in sweep["results"]["moe_linear"]]
    single = [r[KEY] for r in sweep["results"]["moe_single"]]
    verdicts = {"criterion1_full_beats_linear": beats(full, linear),
                "full_beats_single": beats(full, single),
                "full_beats_chebgru": beats(full, [r[KEY] for r in sweep["results"]["chebgru"]]),
                "full_beats_mlp": beats(full, [r[KEY] for r in sweep["results"]["mlp"]])}

    meta = json.loads((PROCESSED / "dataset_v1" / "meta.json").read_text())
    group_names = list(meta["group_names"])
    kernels_per_seed, spec_per_seed, gates_per_seed = [], [], []
    for row in full_rows:
        model, tensors, cfg = load_run_model(row["run_dir"])
        # bin_ms comes from the dataset, not expert_kernels' default: the timescales are only in
        # milliseconds because the teacher was binned at this width
        k = expert_kernels(model, n_steps=tensors.n_bins, bin_ms=meta["bin_ms"])
        kernels_per_seed.append(k)
        spec_per_seed.append(specialisation(k))
        gates_per_seed.append(gate_summary(model, tensors.stim_test, tensors.n_bins, tensors.stim_bins,
                                           group_names, device=cfg.device))
    n_specialised = sum(1 for s in spec_per_seed if s["specialised"])
    criterion2 = {"n_seeds_specialised": n_specialised, "threshold": 2,
                  "passes": bool(n_specialised >= 2),
                  "max_ratio_per_seed": [s["max_ratio"] for s in spec_per_seed]}

    metrics = {"loss_mask_policy": policy, "comparison_source": comparison_dir,
               "timescale_caveat": ("timescale_ms = bin_ms / gamma is the exponential decay time only in the "
                                    "underdamped and critically damped regimes; for an overdamped expert the "
                                    "slow root decays more slowly than this, so read the regime field alongside it"),
               "headline_normalised_r2": full,
               "criterion1": verdicts["criterion1_full_beats_linear"], "criterion2": criterion2,
               "other_comparisons": {k: v for k, v in verdicts.items()
                                    if k != "criterion1_full_beats_linear"},
               "kernels": kernels_per_seed, "specialisation": spec_per_seed, "gates": gates_per_seed,
               "headline_rows": full_rows}
    write_metrics(run_dir, metrics)

    make_figures(metrics)

    print(f"run: {run_dir}")
    print("  criterion 1 (full beats linear):", verdicts["criterion1_full_beats_linear"]["beats"])
    print("  criterion 2 (specialisation):", criterion2["passes"], criterion2["max_ratio_per_seed"])
    return 0


if __name__ == "__main__":
    if "--replot" in sys.argv:
        plotted, where = latest_metrics("p2_r5_moe", with_dir=True)
        make_figures(plotted)
        print(f"redrew both figures from {where}")
        sys.exit(0)
    sys.exit(main())
