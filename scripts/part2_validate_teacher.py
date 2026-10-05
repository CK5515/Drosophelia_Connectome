"""Part 2 - does my GPU fly agree with Shiu's fly?

Runs the original Brian2 model and my PyTorch port side by side and writes the verdict to
data/processed/r2_gate.json. Part 3 refuses to generate data unless this passed.
"""
import argparse
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from flybrain import plotting
from flybrain.data.connectome import load_or_build_connectome
from flybrain.data.fetch import fetch_raw
from flybrain.data.neurons import INPUT_GROUPS, MN9_IDS
from flybrain.data.subcircuit import load_subcircuit
from flybrain.paths import DATA, PROCESSED, ROOT, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.teacher.lif import LIFNetwork, LIFParams

DOSE_HZ = [50, 100, 150, 200]
BITTER_HZ = [0, 50, 100, 150, 200]
SUGAR_FOR_BITTER_HZ = 100
N_TRIALS = 30
T_RUN_S = 1.0
N_STEPS = 10000
BRIAN_DIR = DATA / "brian2_ref"


def experiments():
    exps = [dict(name=f"sugarR_{f}Hz", sugar=f, bitter=0) for f in DOSE_HZ]
    exps += [dict(name=f"sugarR_{SUGAR_FOR_BITTER_HZ}Hz_bitter_{b}Hz", sugar=SUGAR_FOR_BITTER_HZ, bitter=b)
             for b in BITTER_HZ]
    return exps


def run_brian2(exps, paths, n_proc):
    sys.path.insert(0, str(ROOT / "third_party"))
    from brian2 import Hz
    from shiu_model import default_params, run_exp
    BRIAN_DIR.mkdir(parents=True, exist_ok=True)
    for e in exps:
        params = dict(default_params)
        params.update(r_poi=e["sugar"] * Hz, r_poi2=e["bitter"] * Hz, n_run=N_TRIALS)
        run_exp(exp_name=e["name"], neu_exc=list(INPUT_GROUPS["sugar_R"]),
                neu_exc2=list(INPUT_GROUPS["bitter"]) if e["bitter"] > 0 else [],
                path_res=BRIAN_DIR, path_comp=paths["completeness"], path_con=paths["connectivity"],
                params=params, n_proc=n_proc)


def brian2_rates(name, conn):
    df = pd.read_parquet(BRIAN_DIR / f"{name}.parquet")
    if int(df["trial"].max()) >= N_TRIALS:
        raise ValueError(f"stale Brian2 parquet for experiment {name}: max trial {int(df['trial'].max())} "
                         f">= N_TRIALS={N_TRIALS}; delete it and rerun")
    rates = np.zeros((N_TRIALS, conn.n))
    np.add.at(rates, (df["trial"].to_numpy(), conn.index_of(df["flywire_id"].to_numpy())), 1.0)
    return rates / T_RUN_S


def port_rates(net, conn, e, seed):
    groups = [("sugar_R", e["sugar"])] + ([("bitter", e["bitter"])] if e["bitter"] > 0 else [])
    input_idx = np.concatenate([conn.index_of(INPUT_GROUPS[g]) for g, _ in groups])
    rates = np.concatenate([np.full(len(INPUT_GROUPS[g]), hz, dtype=np.float32) for g, hz in groups])
    out = net.run(input_idx, N_STEPS, np.arange(conn.n), N_STEPS,
                  rates_hz=torch.as_tensor(np.tile(rates, (N_TRIALS, 1))), seed=seed)
    return out[:, 0, :].double().cpu().numpy() / T_RUN_S


def mean_se(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def measure_throughput(net, conn, record_idx):
    all_inputs = np.concatenate([conn.index_of(ids) for ids in INPUT_GROUPS.values()])
    best = None
    for batch in (64, 128, 256):
        try:
            rates = torch.full((batch, len(all_inputs)), 100.0)
            torch.cuda.synchronize()
            t0 = time.time()
            net.run(all_inputs, N_STEPS, record_idx, 200, rates_hz=rates, stim_steps=N_STEPS // 2, seed=0)
            torch.cuda.synchronize()
            tph = batch / (time.time() - t0) * 3600
            print(f"  batch {batch}: {tph:,.0f} trials/hour")
            if best is None or tph > best[1]:
                best = (batch, tph)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  batch {batch}: out of memory")
            break
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-brian2", action="store_true", help="reuse existing data/brian2_ref parquet files")
    ap.add_argument("--n-proc", type=int, default=30)
    args = ap.parse_args()

    ensure_free_space()
    paths = fetch_raw()
    conn = load_or_build_connectome(paths["connectivity"], paths["completeness"])
    sub = load_subcircuit(PROCESSED / "subcircuit.npz")
    exps = experiments()
    run_dir = make_run_dir("r2_teacher_validation", {"dose_hz": DOSE_HZ, "bitter_hz": BITTER_HZ,
                                                     "n_trials": N_TRIALS, "n_proc": args.n_proc})
    brian2_seconds = "skipped (reused existing parquet files)"
    if not args.skip_brian2:
        t0 = time.time()
        run_brian2(exps, paths, args.n_proc)
        brian2_seconds = time.time() - t0
        print(f"Brian2 reference done in {brian2_seconds:.0f} s")

    net = LIFNetwork(conn.W, LIFParams(), device="cuda")
    brian, port = {}, {}
    for i, e in enumerate(exps):
        brian[e["name"]] = brian2_rates(e["name"], conn)
        port[e["name"]] = port_rates(net, conn, e, seed=1000 + i)
        print(f"  simulated {e['name']}")

    mn9 = int(conn.index_of([MN9_IDS[0]])[0])

    dose = []
    for f in DOSE_HZ:
        name = f"sugarR_{f}Hz"
        pm, ps = mean_se(port[name][:, mn9])
        bm, bs = mean_se(brian[name][:, mn9])
        dose.append(dict(hz=f, port_mean=pm, port_se=ps, brian_mean=bm, brian_se=bs,
                         within_2se=bool(abs(pm - bm) <= 2 * np.hypot(ps, bs))))
    port_dose = np.array([d["port_mean"] for d in dose])
    crit_dose = (all(d["within_2se"] for d in dose) and bool(np.all(np.diff(port_dose) >= 0))
                 and bool(port_dose[-1] > port_dose[0]))

    bitter = []
    for b in BITTER_HZ:
        name = f"sugarR_{SUGAR_FOR_BITTER_HZ}Hz_bitter_{b}Hz"
        pm, ps = mean_se(port[name][:, mn9])
        bm, bs = mean_se(brian[name][:, mn9])
        bitter.append(dict(hz=b, port_mean=pm, port_se=ps, brian_mean=bm, brian_se=bs))
    rho_port = float(spearmanr(BITTER_HZ, [d["port_mean"] for d in bitter]).statistic)
    rho_brian = float(spearmanr(BITTER_HZ, [d["brian_mean"] for d in bitter]).statistic)
    crit_bitter = bool(rho_port < -0.8 and rho_brian < -0.8)

    rp, rb = port["sugarR_100Hz"].mean(0), brian["sugarR_100Hz"].mean(0)
    active = (rp > 1.0) | (rb > 1.0)
    r_net = float(pearsonr(rp[active], rb[active]).statistic)
    crit_net = bool(r_net > 0.9)
    # reporting-only statistics (not part of any criterion)
    rho_net = float(spearmanr(rp[active], rb[active]).statistic)
    absdiff = np.abs(rp - rb)[active]
    stim = np.zeros(conn.n, dtype=bool)
    stim[conn.index_of(INPUT_GROUPS["sugar_R"])] = True
    act_x = active & ~stim
    extra = {"network_spearman_rho": rho_net,
             "network_median_abs_diff_hz": float(np.median(absdiff)),
             "network_p90_abs_diff_hz": float(np.percentile(absdiff, 90)),
             "excl_inputs_pearson_r": float(pearsonr(rp[act_x], rb[act_x]).statistic),
             "excl_inputs_spearman_rho": float(spearmanr(rp[act_x], rb[act_x]).statistic),
             "excl_inputs_median_abs_diff_hz": float(np.median(np.abs(rp - rb)[act_x])),
             "excl_inputs_n_active": int(act_x.sum())}

    batch, tph = measure_throughput(net, conn, sub.nodes)
    checks = {"mn9_dose_response": crit_dose, "bitter_suppression": crit_bitter, "network_rates_r_gt_0.9": crit_net}
    metrics = {"checks": checks, "pass": all(checks.values()), "mn9_flywire_id": MN9_IDS[0], "dose": dose,
               "bitter": bitter, "spearman_bitter_port": rho_port, "spearman_bitter_brian": rho_brian,
               "network_pearson_r": r_net, "network_n_active": int(active.sum()),
               "throughput_batch_size": batch, "trials_per_hour": tph, "conditions_per_hour_R4": tph / 4,
               "brian2_seconds": brian2_seconds, **extra}
    write_metrics(run_dir, metrics)
    (PROCESSED / "r2_gate.json").write_text(json.dumps({
        "pass": metrics["pass"], "batch_size": batch, "trials_per_hour": tph,
        "conditions_per_hour": tph / 4, "run_dir": str(run_dir.relative_to(ROOT))}, indent=2))

    plotting.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, rows, key, xlabel, title in [
        (axes[0], dose, "hz", "sugar GRN rate (Hz)", "MN9 dose response"),
        (axes[1], bitter, "hz", f"bitter GRN rate (Hz), sugar at {SUGAR_FOR_BITTER_HZ} Hz", "Bitter suppression"),
    ]:
        x = [r[key] for r in rows]
        for label, color, m, s in [("Brian2 (Shiu)", plotting.SERIES[1], "brian_mean", "brian_se"),
                                   ("GPU port", plotting.SERIES[0], "port_mean", "port_se")]:
            y = [r[m] for r in rows]
            ax.errorbar(x, y, yerr=[2 * r[s] for r in rows], color=color, marker="o", capsize=3, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MN9 rate (Hz), ±2 SE")
        ax.set_title(title)
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right")
    lim = max(rp[active].max(), rb[active].max()) * 1.05
    axes[2].plot([0, lim], [0, lim], color=plotting.TEXT_SECONDARY, linewidth=1, linestyle="--")
    axes[2].scatter(rb[active], rp[active], s=12, color=plotting.SERIES[0], edgecolors=plotting.SURFACE,
                    linewidths=0.5)
    axes[2].set_xlabel("Brian2 rate (Hz)")
    axes[2].set_ylabel("GPU port rate (Hz)")
    axes[2].set_title(f"All active neurons, sugar 100 Hz (r = {r_net:.4f})")
    plotting.save_figure(fig, "r2_teacher_validation.png")

    print(f"run: {run_dir}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  throughput: batch {batch}, {tph:,.0f} trials/hour = {tph / 4:,.0f} conditions/hour at R=4")
    print("R2 GATE:", "PASS" if metrics["pass"] else "FAIL")
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
