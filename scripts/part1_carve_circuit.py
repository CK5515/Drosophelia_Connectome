"""Part 1 - carve the feeding circuit out of the connectome.

Keeps the 88 taste neurons and both MN9 no matter what, then neurons within k hops of both. k=2 gave 5,516
candidates, so the 5,000 cap decides the rest: shorter taste-to-MN9 paths first, ties broken by synaptic
strength. Wiring only - the teacher never touches this.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from flybrain import plotting
from flybrain.data.connectome import load_or_build_connectome
from flybrain.data.fetch import fetch_raw
from flybrain.data.neurons import INPUT_GROUPS, MN9_IDS
from flybrain.data.subcircuit import extract_subcircuit, has_directed_path, save_subcircuit
from flybrain.paths import PROCESSED, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics

CONFIG = {"flywire_version": 630, "min_size": 2000, "max_size": 5000, "max_hops": 6}


def main() -> int:
    ensure_free_space()
    paths = fetch_raw()
    conn = load_or_build_connectome(paths["connectivity"], paths["completeness"])
    groups = {g: conn.index_of(ids) for g, ids in INPUT_GROUPS.items()}
    outputs = conn.index_of(MN9_IDS)
    run_dir = make_run_dir("r1_subcircuit", CONFIG)

    sub = extract_subcircuit(conn.W, groups, outputs, min_size=CONFIG["min_size"],
                             max_size=CONFIG["max_size"], max_hops=CONFIG["max_hops"])
    save_subcircuit(sub, PROCESSED / "subcircuit.npz")

    W = sub.W
    n = W.shape[0]
    all_inputs = np.concatenate(list(sub.input_local.values()))
    out_deg = np.diff(W.indptr)
    in_deg = np.bincount(W.indices, minlength=n)
    # R3's Laplacians divide by sum_j |(W + W^T)/2|_ij; catch cancelling-only neurons here, before the dataset exists
    sym_degree = np.asarray(abs((W + W.T) * 0.5).sum(axis=1)).ravel()
    checks = {
        "size_in_range": bool(CONFIG["min_size"] <= n <= CONFIG["max_size"]),
        "all_88_inputs_present": bool(len(np.unique(all_inputs)) == 88),
        "both_mn9_present": bool(len(np.unique(sub.mn9_local)) == 2),
        "path_to_each_mn9": all(has_directed_path(W, all_inputs, int(m)) for m in sub.mn9_local),
        "no_zero_symmetrized_degree": bool(np.all(sym_degree > 0)),
    }
    metrics = {
        "checks": checks,
        "pass": all(checks.values()),
        "n_neurons": n,
        "hops_k": sub.hops,
        "n_candidates_before_truncation": sub.n_candidates,
        "n_edges": int(W.nnz),
        "frac_inhibitory_edges": float((W.data < 0).mean()),
        "excitatory_synapses": float(W.data[W.data > 0].sum()),
        "inhibitory_synapses": float(-W.data[W.data < 0].sum()),
        "out_degree_median": float(np.median(out_deg)), "out_degree_max": int(out_deg.max()),
        "in_degree_median": float(np.median(in_deg)), "in_degree_max": int(in_deg.max()),
        "d_in_counts": {int(k): int(v) for k, v in zip(*np.unique(sub.d_in, return_counts=True))},
        "d_out_counts": {int(k): int(v) for k, v in zip(*np.unique(sub.d_out, return_counts=True))},
    }
    write_metrics(run_dir, metrics)

    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, deg, label in zip(axes, [out_deg, in_deg], ["out-degree", "in-degree"]):
        bins = np.logspace(0, np.log10(deg.max() + 1), 30)
        ax.hist(deg[deg > 0], bins=bins, color=plotting.SERIES[0], edgecolor=plotting.SURFACE, linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(f"{label} within subcircuit")
    axes[0].set_ylabel("neurons")
    fig.suptitle(f"Feeding subcircuit: {n} neurons, {W.nnz} edges", color=plotting.TEXT_PRIMARY)
    plotting.save_figure(fig, "r1_degree_distribution.png")

    print(f"run: {run_dir}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("R1 GATE:", "PASS" if metrics["pass"] else "FAIL")
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
