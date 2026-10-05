"""Part 4 - the spectral half of the spectral MoE.

Both Laplacians of the circuit: signed symmetric (throws direction away) and signed magnetic (keeps it as a
complex phase, q=0.25). Full eigendecomposition of each, then four equal-count bands, one per future expert.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from flybrain import plotting
from flybrain.data.subcircuit import load_subcircuit
from flybrain.paths import PROCESSED, ensure_free_space
from flybrain.runlog import make_run_dir, write_metrics
from flybrain.spectral.bands import equal_count_bands
from flybrain.spectral.basis import basis_checks, compute_basis, save_basis
from flybrain.spectral.laplacian import opposite_sign_reciprocal_pairs

K = 4
Q = 0.25
KINDS = ("symmetric", "magnetic")


def band_enrichment(basis, bands, hop):
    """log2 of mean mode energy per hop-distance group, relative to a uniformly spread mode."""
    energy = np.abs(basis.eigenvectors) ** 2                    # (N, N): energy[i, j] of mode j on neuron i
    n = energy.shape[0]
    groups = np.unique(hop)
    out = np.zeros((len(bands), len(groups)))
    for bi, band in enumerate(bands):
        e = energy[:, band]                                       # (N, modes in band)
        for gi, gval in enumerate(groups):
            members = hop == gval
            out[bi, gi] = e[members].sum(axis=0).mean() / (members.sum() / n)
    sizes = [int((hop == g).sum()) for g in groups]
    return np.log2(out), groups, sizes


def n_reciprocal_pairs(W) -> int:
    """Count unordered pairs {i, j} with both i->j and j->i present (diagonal ignored)."""
    A = (W != 0).astype(np.int8).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return int(A.multiply(A.T).nnz // 2)


def main() -> int:
    ensure_free_space()
    sub = load_subcircuit(PROCESSED / "subcircuit.npz")
    n = sub.W.shape[0]
    run_dir = make_run_dir("r3_spectrum", {"K": K, "q": Q, "n_neurons": n})
    bands = equal_count_bands(n, K)

    metrics = {"n_neurons": n, "opposite_sign_reciprocal_pairs": opposite_sign_reciprocal_pairs(sub.W),
               "n_reciprocal_pairs": n_reciprocal_pairs(sub.W), "bases": {}, "enrichment": {}}
    bases = {}
    for kind in KINDS:
        b = compute_basis(sub.W, kind, q=Q, device="cuda", nodes=sub.nodes)
        checks = basis_checks(b, device="cuda")
        ok = checks["eig_in_range"] and checks["orthonormal"]
        save_basis(b, PROCESSED / "bases" / (f"{kind}.npz" if ok else f"{kind}.failed.npz"))
        metrics["bases"][kind] = {**checks, "band_ranges": [[float(b.eigenvalues[bd[0]]), float(b.eigenvalues[bd[-1]])]
                                                             for bd in bands]}
        bases[kind] = b
        metrics["enrichment"][kind] = {}
        for hop_name, hop in (("d_in", sub.d_in), ("d_out", sub.d_out)):
            enrich, groups, sizes = band_enrichment(b, bands, hop)
            metrics["enrichment"][kind][hop_name] = {
                "groups": [int(g) for g in groups], "group_sizes": sizes,
                "log2_enrichment": enrich.tolist(), "ratio_vs_uniform": (2.0 ** enrich).tolist()}
    metrics["pass"] = all(metrics["bases"][k]["eig_in_range"] and metrics["bases"][k]["orthonormal"] for k in KINDS)
    write_metrics(run_dir, metrics)

    plotting.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, kind in zip(axes, KINDS):
        ev = bases[kind].eigenvalues
        ax.hist(ev, bins=80, color=plotting.SERIES[0], edgecolor=plotting.SURFACE, linewidth=0.4)
        for bi, bd in enumerate(bands):
            if bi > 0:
                ax.axvline(0.5 * (ev[bd[0] - 1] + ev[bd[0]]), ymax=0.68, color=plotting.TEXT_SECONDARY,
                           linestyle="--", linewidth=1)
            ax.annotate(f"band {bi + 1}", (float(np.median(ev[bd])), 0.95 - 0.07 * bi),
                        xycoords=("data", "axes fraction"), ha="center", va="center",
                        color=plotting.TEXT_SECONDARY, fontsize=9)
        ax.set_xlabel("graph frequency μ (Laplacian eigenvalue)")
        ax.set_title(f"signed {kind} Laplacian", pad=10)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.6)
    axes[0].set_ylabel("modes")
    plotting.save_figure(fig, "r3_spectra.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    cmap = plotting.diverging_cmap()
    for row, kind in enumerate(KINDS):
        for col, (hop, label) in enumerate([(sub.d_in, "hops from taste neurons"), (sub.d_out, "hops to MN9")]):
            enrich = np.array(metrics["enrichment"][kind]["d_in" if col == 0 else "d_out"]["log2_enrichment"])
            groups = metrics["enrichment"][kind]["d_in" if col == 0 else "d_out"]["groups"]
            lim = max(np.abs(enrich).max(), 1e-3)
            ax = axes[row, col]
            im = ax.imshow(enrich, cmap=cmap, norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto")
            ax.grid(False)
            ax.set_xticks(range(len(groups)), [str(g) for g in groups])
            ax.set_yticks(range(K), [f"band {i + 1}" for i in range(K)])
            ax.set_xlabel(label)
            ax.set_title(f"{kind}", fontsize=10)
            for (i, j), val in np.ndenumerate(enrich):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                        color=plotting.SURFACE if abs(val) > 0.8 * lim else plotting.TEXT_PRIMARY)
            fig.colorbar(im, ax=ax, label="log2 energy vs uniform", shrink=0.85)
    fig.suptitle("Where does each frequency band live in the circuit?", color=plotting.TEXT_PRIMARY)
    fig.tight_layout()
    plotting.save_figure(fig, "r3_band_localization.png")

    print(f"run: {run_dir}")
    for kind in KINDS:
        m = metrics["bases"][kind]
        print(f"  {kind}: eig [{m['eig_min']:.2e}, {m['eig_max']:.4f}] in range={m['eig_in_range']}, "
              f"orth err={m['orthonormality_max_err']:.1e}")
    print(f"  opposite-sign reciprocal pairs: {metrics['opposite_sign_reciprocal_pairs']} "
          f"of {metrics['n_reciprocal_pairs']} reciprocal pairs")
    print("R3 GATE:", "PASS" if metrics["pass"] else "FAIL")
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
