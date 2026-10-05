"""Part 5 - the noise ceiling, so the R2 numbers have something to be read against.

Split-half over the 4 trials with a Spearman-Brown correction, on the test split. CPU only, no training.
"""
import numpy as np

from flybrain.runlog import make_run_dir, write_metrics
from flybrain.train.loop import TrainConfig, prepare


def main() -> int:
    t, _ = prepare(TrainConfig(model="mean"))
    c = np.asarray(t.ceiling, dtype=float)
    obs, held = t.metric_masks["observed_active"], t.metric_masks["heldout_active"]
    scored = obs | held
    use = c >= 0.1
    stats = lambda m: {"n": int(m.sum()), "median_ceiling": float(np.nanmedian(c[m])),
                       "n_ceiling_usable": int((m & use).sum())}
    metrics = {"n_neurons": int(c.size), "median_ceiling_all_neurons": float(np.nanmedian(c)),
               "n_ceiling_ge_0.1_all_neurons": int(use.sum()),
               "observed_active": stats(obs), "heldout_active": stats(held), "scored_union": stats(scored),
               "step0_cost": {
                   "source": "run folder timestamps of runs/p2_moe_linear_magnetic_real_ever_active_seed0_step0/20260930-164702 "
                             "(config.json 16:47:02 to metrics.json 16:50:50), not a recorded metric; upper bound, includes evaluation",
                   "epoch_seconds_upper_bound": 228, "steps_per_epoch": 132, "seconds_per_step_upper_bound": 228 / 132,
                   "gpu_peak_torch_allocated_gb": 4.57,
                   "gpu_peak_nvidia_smi_mib_incl_cuda_context": 7723,
                   "gpu_peak_source": "untracked runs/step0_gpumem.log (nvidia-smi samples; max 7723 MiB including CUDA context); "
                                      "the 4.57 GB torch-allocated figure came from the step-zero agent's report"}}
    d = make_run_dir("p2_ceiling_stats", {"split": "test", "script": "scripts/part5_noise_ceiling.py"})
    write_metrics(d, metrics)
    print(d); print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
