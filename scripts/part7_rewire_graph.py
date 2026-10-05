"""Part 7 - build the rewired control graphs, one per seed.

Shuffles edges while preserving every neuron's in-degree, out-degree, sign counts and weight multiset
exactly, then rebuilds the eigenbasis from scratch. A fresh rewiring per seed on purpose: one fixed
rewiring would confuse "this graph is easy" with "rewired graphs are easy".
"""
import sys

import scipy.sparse as sp

from flybrain.data.subcircuit import load_subcircuit
from flybrain.paths import PROCESSED, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.spectral.basis import basis_checks, compute_basis, save_basis
from flybrain.spectral.rewire import rewire_preserving_degree_and_sign, rewiring_report

SEEDS = (0, 1, 2)


def main() -> int:
    ensure_free_space()
    sub = load_subcircuit(PROCESSED / "subcircuit.npz")
    out_dir = PROCESSED / "rewired"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir("p2_rewired", {"seeds": list(SEEDS), "swaps_per_edge": 10.0})
    reports = {}
    for seed in SEEDS:
        w = rewire_preserving_degree_and_sign(sub.W, seed=seed)
        rep = rewiring_report(sub.W, w)
        sp.save_npz(out_dir / f"W_seed{seed}.npz", w)
        basis = compute_basis(w, "magnetic", q=0.25, device="cuda", nodes=sub.nodes)
        checks = basis_checks(basis, device="cuda")
        save_basis(basis, out_dir / f"basis_magnetic_seed{seed}.npz")
        reports[seed] = {**rep, **checks,
                         "eigenvalue_range": [float(basis.eigenvalues.min()), float(basis.eigenvalues.max())]}
        print(f"  seed {seed}: displaced {rep['frac_edges_displaced']:.3f}, "
              f"eig [{basis.eigenvalues.min():.4f}, {basis.eigenvalues.max():.4f}], "
              f"in-degrees preserved={rep['in_degree_preserved']}")
    ok = all(r["out_degree_preserved"] and r["in_degree_preserved"] and r["sign_counts_preserved"]
             and r["weight_multiset_preserved"] and r["n_self_loops"] == 0
             and r["n_duplicate_pairs"] == 0 and r["frac_edges_displaced"] > 0.9
             and r["eig_in_range"] and r["orthonormal"] for r in reports.values())
    write_metrics(run_dir, {"reports": reports, "pass": ok})
    print(f"run: {run_dir}\nREWIRING CHECKS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
