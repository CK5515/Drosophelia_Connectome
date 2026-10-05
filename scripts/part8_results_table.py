"""Part 8 - collect every run into one table, one figure and docs/plan2-results.md.

A reporting tool, so it still works mid-project: it records which parts it could not find and says so
loudly rather than quietly printing a short table that looks complete.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flybrain import plotting
from flybrain.paths import ROOT, RUNS
from flybrain.runlog import make_run_dir, write_metrics

KEY = "normalised_r2_observed_active"
ORDER = ["mean", "mlp", "chebgru", "moe_linear", "moe_single", "moe_full", "moe_full_rewired",
         "moe_full_symmetric"]
RUNGS = ("baselines", "loss_mask_ab", "r5", "r6", "epoch_cap_probe")
# the baseline arms come from the all_observed sweep, not R4: R4 trained under ever_active, the losing mask,
# and a table mixing the two would compare masks instead of architectures
BASELINE_SOURCE = "p2_baselines_all_observed"
EXPECTED_SEEDS = {name: 3 for name in ORDER} | {"mean": 1}
LABELS = {"mean": "mean predictor (floor)"}    # the sweep's key stays "mean"; "mean" is also a column
POPULATION = ("a per-neuron R2 averaged over the observed_active mask, each neuron's R2 divided by that "
              "neuron's split-half noise ceiling")
PROBE_PATTERN = "p2_moe_full_magnetic_real_*_seed0_probe*"


def latest(pattern: str, with_dir: bool = False):
    """Newest metrics.json under runs/<pattern>/*/; {} (and no folder) when there is none."""
    paths = sorted(glob.glob(str(RUNS / pattern / "*" / "metrics.json")))
    if not paths:
        return ({}, None) if with_dir else {}
    metrics = json.loads(Path(paths[-1]).read_text())
    return (metrics, str(Path(paths[-1]).parent)) if with_dir else metrics


def finite(values) -> list:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def scored(rows) -> list:
    """(seed, value) pairs for the rows whose score is finite, so a dropped seed stays traceable."""
    return [(r["seed"], float(r[KEY])) for r in rows if r.get(KEY) is not None and np.isfinite(r[KEY])]


def run_mask(run_dir: str) -> str:
    return json.loads((Path(run_dir) / "config.json").read_text())["config"]["loss_mask"]


def resolve_policy(r5: dict, r5_dir, sweep: dict, sweep_dir, ab: dict, ab_dir) -> str | None:
    """The loss mask the headline numbers were trained under; never defaulted."""
    if not r5:
        return None
    if "loss_mask_policy" not in r5:
        raise RuntimeError(f"{r5_dir} has no 'loss_mask_policy'; refusing to guess which mask its numbers use")
    policy = r5["loss_mask_policy"]
    if sweep and sweep["loss_mask"] != policy:
        raise RuntimeError(f"R5 {r5_dir} used loss mask {policy!r} but the baseline sweep {sweep_dir} used "
                           f"{sweep['loss_mask']!r}; the table would compare masks, not architectures")
    if ab and ab["policy"] != policy:
        raise RuntimeError(f"R5 {r5_dir} used loss mask {policy!r} but the loss-mask A/B {ab_dir} adopted "
                           f"{ab['policy']!r}")
    return policy


def check_r6_masks(r6: dict, r6_dir, policy) -> None:
    """R6 writes no top-level loss_mask, so read each control run's own config.json."""
    if policy is None:
        return
    for label, rows in r6.get("arms", {}).items():
        for r in rows:
            got = run_mask(r["run_dir"])
            if got != policy:
                raise RuntimeError(f"R6 {r6_dir} arm {label!r} run {r['run_dir']} used loss mask {got!r}, "
                                   f"but the headline uses {policy!r}")


def epoch_cap_probe(policy, headline: list, r5_rows: list):
    """The single-seed high-epoch-cap run under the headline mask, or None. Shows whether the 40-epoch cap
    was load-bearing."""
    folders = sorted({str(Path(p).parents[1]) for p in glob.glob(str(RUNS / PROBE_PATTERN / "*" / "metrics.json"))})
    for folder in reversed(folders):
        metrics, run_dir = latest(Path(folder).name, with_dir=True)
        cfg = metrics["config"]
        if policy is not None and cfg["loss_mask"] != policy:
            continue
        seed0 = finite([r[KEY] for r in r5_rows if r.get("seed") == 0])
        return {"run_dir": run_dir, KEY: metrics[KEY], "best_epoch": metrics["history"]["best_epoch"],
                "max_epochs": cfg["max_epochs"], "seed": cfg["seed"], "loss_mask": cfg["loss_mask"],
                "headline_mean_40_epoch": float(np.mean(headline)) if headline else None,
                "headline_seed0_40_epoch": seed0[0] if seed0 else None}
    if folders:
        print(f"  note: probe folders exist but none used loss mask {policy!r}: {folders}")
    return None


def rel(path) -> str:
    """Repo-relative, so the published page does not leak an absolute home directory."""
    if not path:
        return "unknown"
    text = str(path)
    root = str(ROOT)
    return text[len(root) + 1:] if text.startswith(root + "/") else text


def verdict_row(name: str, v: dict) -> str:
    per_seed = " / ".join(f"{d:+.2f}" for d in v["per_seed"])
    return (f"| {name} | {per_seed} | {v['mean_diff']:.3f} | {2 * v['std_diff']:.3f} | "
            f"{'yes' if v['all_same_sign'] else 'no'} | **{'PASS' if v['beats'] else 'no claim'}** |")


def main() -> int:
    ab, ab_dir = latest("p2_loss_mask_ab", with_dir=True)
    sweep, sweep_dir = latest(BASELINE_SOURCE, with_dir=True)
    r5, r5_dir = latest("p2_r5_moe", with_dir=True)
    r6, r6_dir = latest("p2_r6_controls", with_dir=True)
    unusable = {}
    if sweep and sweep["complete"] is not True:
        unusable["baselines"] = f"{sweep_dir} is not complete (sweep still running or interrupted)"
        sweep, sweep_dir = {}, None
    if r6 and r6.get("complete") is not True:
        unusable["r6"] = f"{r6_dir} is not complete"
        r6, r6_dir = {}, None
    policy = resolve_policy(r5, r5_dir, sweep, sweep_dir, ab, ab_dir)
    check_r6_masks(r6, r6_dir, policy)
    head_rows = r5.get("headline_rows", [])
    headline = [v for _, v in scored(head_rows)]
    probe = epoch_cap_probe(policy, headline, head_rows)
    probe_dir = probe["run_dir"] if probe else None
    sources = dict(zip(RUNGS, (sweep_dir, ab_dir, r5_dir, r6_dir, probe_dir)))
    missing = [name for name, src in sources.items() if src is None]

    table = {name: scored(rows) for name, rows in sweep.get("results", {}).items()}
    table["moe_full"] = scored(head_rows)
    table["moe_full_rewired"] = scored(r6.get("arms", {}).get("rewired", []))
    table["moe_full_symmetric"] = scored(r6.get("arms", {}).get("symmetric", []))
    rows, shortfalls = [], []
    for name in ORDER:
        pairs = table.get(name, [])
        vals = [v for _, v in pairs]
        if len(vals) != EXPECTED_SEEDS[name]:
            shortfalls.append({"model": name, "found": len(vals), "expected": EXPECTED_SEEDS[name]})
        if vals:
            rows.append({"model": name, "mean": float(np.mean(vals)), "n_seeds": len(vals),
                         "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                         "per_seed": vals, "seeds_kept": [s for s, _ in pairs]})
    if missing:
        print(f"MISSING RUNGS (table is partial): {', '.join(missing)}")
        for name, why in unusable.items():
            print(f"  {name}: {why}")
    if not rows:
        print("nothing to tabulate: no rung has usable results yet; wrote nothing")
        return 1
    if shortfalls:
        print(f"SHORT ROWS: {shortfalls}")

    # floor: the sweep's own mean-predictor arm, so it shares the mask and seed pipeline of the other arms
    # (the original R4 metrics carry a top-level floor_normalised_r2, but those runs used ever_active)
    floor_vals = finite([r[KEY] for r in sweep.get("results", {}).get("mean", []) if r.get(KEY) is not None])
    floor = float(np.mean(floor_vals)) if floor_vals else None
    ceiling_stats, ceiling_dir = latest("p2_ceiling_stats", with_dir=True)
    # these are raw (not normalised) split-half ceilings; in normalised units the ceiling is 1 by construction
    ceiling = ({"source": ceiling_dir,
                "observed_active_median_raw_ceiling": ceiling_stats["observed_active"]["median_ceiling"],
                "scored_union_median_raw_ceiling": ceiling_stats["scored_union"]["median_ceiling"]}
               if ceiling_stats else None)
    complete = not missing and not unusable and not shortfalls
    label = policy if policy is not None else "unknown (R5 missing)"
    partial = bool(missing or unusable or shortfalls)

    # figure first: a plotting failure must not leave a summary page that cites a figure never written
    plotting.apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, r in enumerate(rows):
        ax.bar(i, r["mean"], width=0.6, color=plotting.SERIES[0], edgecolor=plotting.SURFACE)
        ax.scatter([i] * len(r["per_seed"]), r["per_seed"], color=plotting.TEXT_SECONDARY, s=18, zorder=3)
    if floor is not None:
        ax.axhline(floor, color=plotting.SERIES[1], linestyle="--", linewidth=1.2, label="mean-predictor floor")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(rows)), [LABELS.get(r["model"], r["model"]) for r in rows], rotation=20, ha="right")
    ax.set_ylabel("noise-normalised R²")
    ax.set_title(f"Every model, every seed (loss mask: {label})" + (" - partial" if partial else ""))
    plotting.save_figure(fig, "summary_table.png")

    summary = {"loss_mask_policy": policy, "metric": KEY, "metric_population": POPULATION, "complete": complete,
               "sources": sources, "missing_rungs": missing, "unusable_rungs": unusable,
               "shortfalls": shortfalls, "rows": rows,
               "floor_normalised_r2": floor, "floor_n_seeds": len(floor_vals), "noise_ceiling": ceiling,
               "epoch_cap_probe": probe,
               "loss_mask_ab": ({"policy": ab["policy"], "diff": ab["diff"], "pooled_spread": ab["pooled_spread"]}
                                if ab else None),
               "criterion1": r5.get("criterion1"), "criterion2": r5.get("criterion2"),
               "controls": r6.get("verdicts"), "rewiring": r6.get("rewiring"),
               "disclosures": r6.get("disclosures"), "knockouts": r6.get("knockouts")}
    run_dir = make_run_dir("p2_summary", {"metric": KEY, "sources": sources})
    write_metrics(run_dir, summary)

    lines = ["# Plan 2 results", ""]
    if partial:
        what = sorted(set(missing) | set(unusable) | {f"{s['model']} short" for s in shortfalls})
        lines += [f"**PARTIAL: {', '.join(what)}.**", ""]
    lines += [f"Metric: {KEY} (loss mask: {label}): {POPULATION}.", "",
              "| model | mean | std | seeds |", "|---|---|---|---|"]
    lines += [f"| {LABELS.get(r['model'], r['model'])} | {r['mean']:.4f} | "
              f"{'n/a' if r['std'] is None else format(r['std'], '.4f')} | {r['n_seeds']} |" for r in rows]
    lines += ["", f"Floor (mean predictor, {len(floor_vals)} seed(s), observed_active mask, loss mask "
              f"{label}): {floor:.4f}" if floor is not None else "Floor: not available",
              f"Noise ceiling over the scored population, RAW R2 and NOT normalised: "
              f"{ceiling.get('observed_active_median_raw_ceiling', float('nan')):.4f} median over "
              f"observed_active, {ceiling.get('scored_union_median_raw_ceiling', float('nan')):.4f} over the "
              f"scored union ({rel(ceiling.get('source'))})." if ceiling else "Noise ceiling: not available",
              ""]
    lines += ["## Epoch-cap probe", ""]
    if probe:
        lines += [f"| quantity | value |", "|---|---|",
                  f"| probe, seed {probe['seed']} at {probe['max_epochs']} epochs | "
                  f"{probe['normalised_r2_observed_active']:.4f} |",
                  f"| same seed at 40 epochs | {probe['headline_seed0_40_epoch']:.4f} |",
                  f"| 3-seed headline mean at 40 epochs | {probe['headline_mean_40_epoch']:.4f} |",
                  f"| best epoch reached | {probe['best_epoch']} of {probe['max_epochs']} |", "",
                  f"Run: `{rel(probe['run_dir'])}`.", ""]
    else:
        lines += ["not run", ""]
    lines += ["## Pre-registered verdicts", "",
              "A claim counts only if the mean paired difference across seeds exceeds twice their standard "
              "deviation AND every seed agrees on the sign.", "",
              "| claim | per-seed difference | mean | 2 x std | same sign | verdict |", "|---|---|---|---|---|---|"]
    lines += [verdict_row(name, v) for name, v in
              [("criterion 1: full beats linear", summary["criterion1"])] +
              sorted((summary["controls"] or {}).items())]
    c2 = summary["criterion2"]
    lines += ["", f"Criterion 2 (expert specialisation): **{'PASS' if c2['passes'] else 'FAIL'}** - "
              f"{c2['n_seeds_specialised']} of 3 seeds reach the {c2['threshold']}x timescale ratio "
              f"(per-seed maxima {', '.join(format(r, '.2f') for r in c2['max_ratio_per_seed'])}).", "",
              "## Sources", ""]
    lines += [f"- {name}: `{rel(src)}`" for name, src in sources.items()]
    (ROOT / "docs" / "plan2-results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["rows"], indent=2))
    print("run:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
