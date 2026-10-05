# Plan 2 results - cleaned and prepared by ai

Metric: normalised_r2_observed_active (loss mask: all_observed): a per-neuron R2 averaged over the observed_active mask, each neuron's R2 divided by that neuron's split-half noise ceiling.

| model | mean | std | seeds |
|---|---|---|---|
| mean predictor (floor) | 0.1449 | n/a | 1 |
| mlp | 0.8451 | 0.0145 | 3 |
| chebgru | -1.3998 | 0.1985 | 3 |
| moe_linear | -3.6379 | 0.0798 | 3 |
| moe_single | -2.3221 | 0.1199 | 3 |
| moe_full | -1.5916 | 1.2512 | 3 |
| moe_full_rewired | -7.0502 | 2.2673 | 3 |
| moe_full_symmetric | -1.3674 | 0.2925 | 3 |

Floor (mean predictor, 1 seed(s), observed_active mask, loss mask all_observed): 0.1449
Noise ceiling over the scored population, RAW R2 and NOT normalised: 0.4396 median over observed_active, 0.4183 over the scored union (runs/p2_ceiling_stats/20261001-165138).

## Epoch-cap probe

| quantity | value |
|---|---|
| probe, seed 0 at 120 epochs | -2.5169 |
| same seed at 40 epochs | -2.9468 |
| 3-seed headline mean at 40 epochs | -1.5916 |
| best epoch reached | 114 of 120 |

Run: `runs/p2_moe_full_magnetic_real_all_observed_seed0_probe120/20261004-032529`.

## Pre-registered verdicts

A claim counts only if the mean paired difference across seeds exceeds twice their standard deviation AND every seed agrees on the sign.

| claim | per-seed difference | mean | 2 x std | same sign | verdict |
|---|---|---|---|---|---|
| criterion 1: full beats linear | +0.61 / +3.17 / +2.37 | 2.046 | 2.617 | yes | **no claim** |
| magnetic_beats_symmetric | -1.31 / +0.94 / -0.29 | -0.224 | 2.254 | no | **no claim** |
| real_beats_rewired | +4.73 / +4.06 / +7.59 | 5.459 | 3.757 | yes | **PASS** |

Criterion 2 (expert specialisation): **PASS** - 2 of 3 seeds reach the 2x timescale ratio (per-seed maxima 1.67, 2.11, 9.48).

## Sources

- baselines: `runs/p2_baselines_all_observed/20261002-143604`
- loss_mask_ab: `runs/p2_loss_mask_ab/20261001-165320`
- r5: `runs/p2_r5_moe/20261003-124145`
- r6: `runs/p2_r6_controls/20261003-130247`
- epoch_cap_probe: `runs/p2_moe_full_magnetic_real_all_observed_seed0_probe120/20261004-032529`
