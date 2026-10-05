# Where the code for each part of the write-up lives

Every part of the [blog post](../README.md) has a script here. Run them in order and you get the whole thing
from scratch. Each one writes a timestamped folder under `runs/` with the config it actually ran and the
metrics it produced, so every number in the write-up can be chased back to the run that made it.

Nothing here takes arguments you have to guess at. The scripts that draw figures take `--replot`, which
redraws from the saved metrics without retraining anything.

| part of the post | script | writes to `runs/` | figures |
|---|---|---|---|
| 1. A circuit, not the whole brain | `part1_carve_circuit.py` | `r1_subcircuit/` | `r1_degree_distribution.png` |
| 2. A GPU fly that agrees with the real fly | `part2_validate_teacher.py` | `r2_teacher_validation/` | `r2_teacher_validation.png` |
| 3. The dataset | `part3_make_dataset.py` | `r2_dataset/` | — |
| 4. The MORGAN part | `part4_spectrum.py` | `r3_spectrum/` | `r3_spectra.png`, `r3_band_localization.png` |
| 5. Baselines | `part5_baselines.py` | `p2_r4_baselines/` | `r4_baselines.png` |
| 5. …and what to read them against | `part5_noise_ceiling.py` | `p2_ceiling_stats/` | — |
| 5½. I was wrong about silent neurons | `part5b_loss_mask_ab.py` | `p2_loss_mask_ab/` | — |
| 5½. …so run it all back | `part5b_rerun_baselines.py` | `p2_baselines_all_observed/` | — |
| 6. The MoE level | `part6_moe.py` | `p2_r5_moe/` | `r5_expert_kernels.png`, `r5_gates.png` |
| 7. Build the rewired controls | `part7_rewire_graph.py` | `p2_rewired/` | — |
| 7. The connectome matters | `part7_controls.py` | `p2_r6_controls/` | `r6_controls.png`, `r6_knockouts.png` |
| 8. Price the epoch cap | `part8_epoch_probe.py` | `p2_moe_full_…_probe120/` | — |
| 8. A conclusion | `part8_results_table.py` | `p2_summary/` | `summary_table.png` |

## What depends on what

Parts 1 → 2 → 3 are a chain and the later ones refuse to run if an earlier gate failed — part 3 will not
generate a dataset unless part 2's validation passed, which is the one place I wanted a hard stop rather than
a warning I'd skim past.

After that, part 4 needs the circuit from part 1, and everything from part 5 onward needs the dataset from
part 3. Part 5½ writes the loss-mask decision to `data/processed/p2_loss_mask.json` and **every later script
reads it**, so the parts after it are all scored under the same policy. Part 7's controls need part 6's run
folder and part 7's own rewired graphs, and it checks for both before spending a single GPU-hour.

## The library underneath

The scripts are thin. The actual work is in `src/flybrain/`:

| module | what it does | parts |
|---|---|---|
| `data/` | fetch FlyWire, build the connectome, carve the subcircuit | 1 |
| `teacher/` | the LIF port, the stimuli, dataset generation | 2, 3 |
| `spectral/` | the two Laplacians, eigenbases, frequency bands, graph rewiring | 4, 7 |
| `models/` | the damped-wave solver, the spectral MoE, the baselines | 5, 6, 7 |
| `train/` | splits, masks, the noise ceiling, metrics, the training loop | all of them |
| `analysis/` | the claim rule, expert kernels, gate summaries, band knockouts | 6, 7, 8 |

Two bits worth knowing if you read the source:

`analysis/criteria.py` is the whole pre-registered rule, in about ten lines. A difference only counts if the
mean paired difference across seeds beats twice their standard deviation *and* every seed agrees on the sign.
It is deliberately the only thing that decides a comparison, so there is one place to check whether I moved
the goalposts.

`models/wave.py` solves the damped wave equation exactly per time bin with a 3×3 matrix exponential, rather
than stepping it. That one choice is why there is no separate code path for underdamped, critically damped and
overdamped experts — the same expression covers all three, including the μ = 0 case that would otherwise
divide by zero.

## Running it

Needs a CUDA GPU for parts 2 through 8 (mine was a 3060) and about 60 GB free for the dataset and the
eigenbases. Parts 5 through 8 are roughly 60 GPU-hours end to end, most of it the two baseline sweeps and the
R6 controls. `pip install -e .` then run the parts in order.
