# Fractional Motion Repair

This directory contains the reproducibility code and aggregate evidence for
`Endpoint-Constrained Multi-Band Fractional Memory for Repairing Long-Duration
Limb Occlusions in Dance Motion Capture`.

The method repairs long, localized gaps in 3D dance skeletons. It applies three
truncated Grunwald--Letnikov fractional-integral kernels to visible motion
increments, mixes bidirectional context, and projects the residual correction
to zero at both observed endpoints of each missing interval.

## Evidence boundary

- The frozen confirmatory claim concerns 72-frame one-limb gaps at 60 Hz.
- The complete fractional system improved MPJPE over a matched finite bridge by
  0.279% (95% CI 0.167% to 0.387%).
- A post-confirmatory loss-matched control retained a 0.293% gain (95% CI
  0.184% to 0.400%) without the fractional trajectory loss.
- The duration sweep, loss-matched control, and order ablation are explicitly
  post-confirmatory.
- Uniform full-body gaps favor finite memory. The method is not presented as a
  universal replacement for finite temporal models.

## Layout

```text
src/        Training, testing, bootstrap, ablation, latency, and figure code
protocols/  Frozen and post-confirmatory protocol records
results/    Aggregate values reported in the manuscript, including the
            post-confirmatory duration metric tables
```

Raw AIST++ files, sequence-level test errors, checkpoints, and generated caches
are not redistributed. AIST++ remains subject to its providers' terms.

## Environment

The reported experiments used Python 3.10, PyTorch, CUDA, and one NVIDIA RTX
3090 with 24 GB of VRAM. Install the repository-level requirements and run all
commands from this directory so that `src/` imports resolve consistently.

## Data preparation

Download the official AIST++ 3D keypoint and multi-view 2D keypoint releases.
Place them at:

```text
data/aistpp/annotations/keypoints3d/*.pkl
data/keypoints2d.zip
```

Build and verify the native-rate cache:

```bash
python src/build_raw_cache.py
python src/verify_raw_cache.py
```

The cache builder refuses to overwrite existing outputs unless `--force` is
provided.

## Grouped folds for the Mathematics revision

Generate the five sequence manifests with the frozen split seed from the
[revision protocol](protocols/mathematics_revision_protocol.md).
The published [split manifests](results/revision/splits) list the sequences
in the primary and sensitivity folds.

```bash
python src/revision_splits.py --raw-cache-dir data/raw60 \
  --ignore-list data/downloads/ignore_list.txt --out-dir runs/revision/splits
```

Pass `--fold 0` through `--fold 4` and `--ignore-list` to `raw_suite.py` and
`fractional_screen.py` to use a fold for training and inner validation. Each
run writes `revision_split_manifest.json`. `fractional_screen.py` checks the base
experts against that manifest and the requested training seed. Use a separate
output directory with `--include-ignored` for the all-sequence sensitivity
split. Legacy evaluation entry points reject fold runs.

`fractional_screen.py --variants K0,K1,K2,K3,K4,K5,K6,K7` selects the eight
kernel variants in the revision protocol. The comparison uses the base loss.
Use `--variants continued,K5,B3_K5` to train the B0, B2, and B3 comparisons
from the same base checkpoint. The `B3_` prefix enables projection of the TCN
residual at the observed endpoints. A bounded gap subtracts the linear trend
between endpoint residuals. A one-sided gap subtracts its observed endpoint
residual. A gap with no observed endpoint uses the TCN residual directly. The
saved variant name identifies the projection when loading a checkpoint.

### Milestone B bridge ablation

Train `matched_tcn,gap_router` with `raw_suite.py` for each of the five folds
and seeds 20261201–20261203. Train `continued,K5,B3_K5` with
`fractional_screen.py` using the corresponding base run. Store each pair
under `runs/revision/foldF/seedS/{base,bridges}`. For example, evaluate the
first pair and combine the complete set of 15 evaluations with these commands.

```bash
python src/bridge_ablation.py evaluate --fold 0 --seed 20261201 \
  --raw-cache-dir data/raw60 --ignore-list data/downloads/ignore_list.txt \
  --base-run runs/revision/fold0/seed20261201/base \
  --bridge-run runs/revision/fold0/seed20261201/bridges \
  --out-dir runs/revision/fold0/seed20261201/validation \
  --aist-motion-zip data/downloads/motions.zip
python src/bridge_ablation.py summarize --runs-dir runs/revision \
  --out-dir results/revision
```

The evaluator uses the inner validation sequences, 128-frame windows at stride
128, and the ten conditions and mask seeds in the frozen protocol. It records
normalized MPJPE averaged first within each sequence and then across sequences,
available millimeter coverage, and the total deployed parameters. The summary
requires all 450 fold × seed × bridge × condition cells. Its printed bridge means
give each fold, training seed, and condition equal weight.

### Numerical stability and training audit

The TCN uses automatic mixed precision. Visibility-weighted convolutions and
their normalization run with autocast disabled in at least float32, retaining
float64 inputs. This prevents overflow in the backward pass when visible kernel
support is small. Kernel initialization, parameterization, and learning-rate
multiplier follow the revision protocol.

Each extension training step checks the loss, unscaled gradients, gradient norm,
and updated parameters for finite values. Validation scores are also checked.
A non-finite value stops training and records the epoch, batch, stage, affected
tensors, and AMP scale in `VARIANT_stability.json`. Completed epochs are written
to `VARIANT_history.json`, including per-band beta and tempering decay.

Audit the original 45 training runs from their saved histories and selected
checkpoints before retraining.

```bash
python src/bridge_stability.py --original-runs-dir runs/revision \
  --out-dir results/revision
```

Keep the corrected runs in a separate directory. After all 45 corrected runs
finish, add their paired records to the audit with this command.

```bash
python src/bridge_stability.py --original-runs-dir runs/revision \
  --retrained-runs-dir runs/revision_stable_20261117 --out-dir results/revision
```

`bridge_training_stability.csv` contains one row for each fold, seed, and bridge.
It records original and corrected checkpoint epochs, validation scores, finite
epoch counts, checkpoint hashes, and per-band beta and lambda. The original
first logged non-finite epoch refers to the saved epoch summaries. Original
batch gradients were unrecorded. An original last-finite lambda is available
when that epoch supplied the selected checkpoint. Other missing historical
values are blank and their source is `not_recorded`.

### Selected kernel diagnostics

`kernel_diagnostics.py` exports each selected model's normalized base-kernel taps
and per-band mean absolute lag, zero-lag mass, beta and decay where defined.
The lag statistic is `sum(k * w[k])` in frames, before visibility normalization
and endpoint projection.

After the five-fold, three-seed K6 training finishes, generate the exploratory
K5/K6 shapes and K6 fits from the validation-selected checkpoints.

```bash
python src/kernel_shapes.py --runs-dir runs/kernel_ablation_20261118 \
  --stable-runs-dir runs/revision_stable_20261117 --out-dir results/revision
```

The command checks the training manifests, selected epochs and checkpoint
hashes. It writes `kernel_shape_fit.csv`, `kernel_shapes.png`,
`kernel_shapes.pdf` and `kernel_shape_manifest.json`. The figure shows checkpoint
means and the 10th–90th percentile range across fold–seed pairs. `--preview`
uses the currently completed K6 runs and marks the filenames and manifest.

Each K6 band is fit on log tap values with a free intercept using power-law,
exponential and tempered-power-law profiles, with nonnegative decay slopes.
AIC uses the Gaussian log-residual working model and includes the variance
parameter, giving parameter counts of 3, 3 and 4. Fit statuses preserve
unavailable fits for nonpositive tap values. The fits describe kernel shape
under the exploratory T4.1 analysis.

The completed report uses all 15 selected checkpoints for each of K5 and K6:
[shape fits](results/revision/kernel_shape_fit.csv),
[checkpoint and analysis manifest](results/revision/kernel_shape_manifest.json),
[PNG figure](results/revision/kernel_shapes.png) and
[PDF figure](results/revision/kernel_shapes.pdf).
The three mean lags, calculated per checkpoint before averaging the 15 runs,
are 0.040/0.201/0.517 frames for K5 and 9.590/20.910/32.090 frames for K6.
The power-law working profile has the lowest AIC for all 45 K6 band cases;
the tempered fits select zero exponential decay and have AIC higher by 2.
The actual K6 profiles are nonmonotone. This relative fit comparison describes
the finite 97-tap kernels; it does not establish a power-law mechanism or
predictive superiority. Predictive comparisons and G1 decisions remain pending
the frozen milestone C tests.
Twelve of the 15 K6 checkpoints selected epoch 8, the preregistered limit.

K6 is initialized from the K3 Grünwald–Letnikov weights, so its lowest-AIC
power-law profile largely reflects that initialization after an eight-epoch
budget. The K5 and K6 mean lags differ by two orders of magnitude, so the
learned memory length depends on the parameterization. Both moved toward
shorter memory than their initial kernels: the fitted K6 exponents steepened
from 0.775/0.5/0.225 to 1.24/0.86/0.53. Learned shapes are therefore not
evidence about the intrinsic memory of the motion data; whether memory shape
affects accuracy is decided by the frozen milestone C tests.

Run the diagnostic regression checks on the CPU with
`python src/smoke_fractional.py --cpu-only`.

### Frozen milestone C test evaluation

Claude approved B3 in PR #15. Reuse that review's 15 B0 and 15 B3_K5
checkpoints; train B3_K0–K4, B3_K6 and B3_K7 for all five folds and
three seeds. The commands below require the complete source directory and
the recorded training evidence. They preserve the eight-epoch protocol.

After all 105 new trainings finish, freeze the full 135-long-checkpoint grid
and 15 short checkpoints before constructing any test dataset:

```bash
python src/kernel_test.py freeze --raw-cache-dir data/raw60 \
  --ignore-list data/downloads/ignore_list.txt --base-runs-dir runs/revision \
  --stable-runs-dir runs/revision_stable_20261117 \
  --runs-dir runs/kernel_ablation_20261118 --out-dir runs/kernel_test \
  --protocol protocols/mathematics_revision_protocol.md \
  --decisions ../docs/DECISIONS.md --aist-motion-zip data/downloads/motions.zip
python src/kernel_test.py evaluate --freeze runs/kernel_test/test_freeze.json \
  --fold 0 --seed 20261201 --device cuda
```

Run `evaluate` once for each of the 15 fold/seed pairs. The evaluator checks
the complete freeze, all checkpoint hashes and evidence, sequence lists,
approved B3 decision, source hashes and fixed masks before test access.
Completed manifests record the device, PyTorch/CUDA/cuDNN versions and
precision/determinism settings; summarization requires the same runtime
configuration across all 15 runs.
It uses 128-frame fixed windows, stride 128, batch size 32 and zero workers.
Each method/condition has a saved transaction. An interrupted transaction
or failed run stays locked for review; completed runs are verified and skipped.

```bash
python src/kernel_test.py summarize --freeze runs/kernel_test/test_freeze.json \
  --out-dir runs/kernel_test/summary
python src/kernel_inference.py --freeze runs/kernel_test/test_freeze.json \
  --results-dir runs/kernel_test/summary --out-dir results/revision
python src/smoke_kernel_inference.py --work-dir tmp/kernel-inference-smoke
```

`kernel_ablation.csv` has 1,350 fold/seed/method/condition rows, including total
router parameters and each selected kernel band's mean lag, zero-lag mass,
beta and decay where applicable. These statistics describe the normalized
base kernel before visibility normalization, mixing and endpoint projection.
`kernel_sequence_results.csv` and the full freeze remain on the experiment
server. Do not commit sequence-level results, data or checkpoints.

The primary score is `mpjpe_window_mean`: average windows within sequences,
then average three training seeds, then average endpoint conditions.
Inference samples whole genre/choreography groups uniformly with replacement,
retaining all member sequences. All comparisons share 10,000 draws with seed
20261114. This retains equal sequence weighting within each resample.
`kernel_hypotheses.csv` reports raw 95% and 90% percentile intervals, bootstrap
tail evidence with a +1 Monte Carlo correction, and Holm decisions: two H1
comparisons and four H2 comparisons. Difference and intersection-union
equivalence evidence receive separate four-comparison H2 corrections.
The rank-specific critical intervals accompany those decisions; they are not
simultaneous confidence regions. H3 uses the raw 95% lower limit and −0.5%
margin. K0 equivalence is explicitly labelled exploratory G1 evidence.
`kernel_ablation_summary.csv` and `kernel_inference_manifest.json` record
aggregation and decision evidence. The author makes the G1 decision.

### Additional evaluation metrics

`evaluate_detailed(..., extended_metrics=True)` computes the following fields
for the ordered windows supplied to the evaluator.

| Field | Definition |
|---|---|
| `mpjpe_window_mean` | Mean missing-joint position error per window, averaged across windows within each sequence. The aggregate gives each sequence equal weight. |
| `mpjpe_raw_units` | The same score multiplied by the sequence's torso scale in `metadata.npz`, yielding AIST++ reconstruction units. |
| `mpjpe_mm` | The reconstruction-unit score multiplied by `1000 / smpl_scaling`, yielding millimeters at the SMPL reference scale. The aggregate averages the sequences with a conversion and reports their count in `mpjpe_mm_sequences`. |
| `endpoint_velocity_jump_error` | Mean norm of the predicted minus true velocity jump at observed boundary frames. Each eligible joint has a missing neighbor on one side and an observed neighbor on the other. The unit is torso lengths per frame. |
| `acceleration_error` | Mean norm of the error in the centered second difference over joint triplets containing a missing frame. The unit is torso lengths per frame squared. |
| `mpjpe_joint_00`–`mpjpe_joint_16` | Normalized MPJPE at missing frames for each joint in the cached COCO order. |

The last three metrics pool eligible observations and leave unsupported values
blank in CSV output. The `mpjpe` field pools missing joint-frames.

For millimeters, pass the per-sequence scales returned by
`load_aist_smpl_scales` in `revision_metrics.py` as `smpl_scales`.
The archived evaluator `fractional_test.py` exposes the same diagnostics with
`--extended-metrics`. To load windows in the evaluator process, pass
`--num-workers 0`. The `--aist-motion-zip data/downloads/motions.zip` option enables
the diagnostics and reads scales from a trusted AIST++ motion archive.
The current release covers 411 of the 1,408 cached sequences. Sequences
without a conversion have a blank millimeter value.
[AIST++ motion release](https://github.com/google/aistplusplus_dataset/releases/tag/v1.0)

The conversion uses the dataset's documented SMPL fit and represents errors
at the reference body's scale. [AIST++ scale explanation](https://github.com/google/aistplusplus_api/issues/9#issuecomment-782814093)

## Smoke test

```bash
python src/smoke_fractional.py
```

The smoke test checks the fractional definition, endpoint projection,
observation preservation, hidden-target invariance, finite gradients, and
matched parameter count. It also checks the K0–K7 kernel family, including
mean-lag matching and the uniform-window limit of the tempered kernel.

## Training outline

Train the matched short and long baseline checkpoints for each seed:

```bash
python src/raw_suite.py \
  --raw-cache-dir data/raw60 \
  --out-dir runs/base_seed20260829 \
  --seed 20260829 --split-seed 20260829 --mask-seed 20270829 \
  --variants matched_tcn,gap_router
```

Train the finite and fractional long-gap experts:

```bash
python src/fractional_screen.py \
  --raw-cache-dir data/raw60 \
  --base-run runs/base_seed20260829 \
  --out-dir runs/fractional_seed20260829 \
  --seed 20260829 --split-seed 20260829 \
  --variants integer_memory,fractional_memory,fractional_memory_loss
```

Repeat with seeds 20260830 and 20260831. The frozen test requires aligned
`--base-runs` and `--fractional-runs`; see `python src/fractional_test.py -h`.
Use `fractional_memory_loss` for the frozen complete-system test and
`fractional_memory` with `--fractional-checkpoint fractional_memory_best.pt`
for the post-confirmatory loss-matched control.

## Aggregate analysis

```bash
python src/aggregate_fractional_test.py --run-dir RUN_DIR --out-dir REPORT_DIR
python src/fractional_gap_sweep.py -h
python src/aggregate_fractional_gap_sweep.py -h
python src/fractional_order_ablation.py -h
python src/benchmark_fractional_latency.py -h
```

Protocol files distinguish development, frozen confirmation, and exploratory
analysis. Do not reinterpret post-confirmatory results as independent tests.

## License and data

Code is released under the repository MIT license. AIST++ data are available
from https://google.github.io/aistplusplus_dataset/ and are not covered by this
repository license.
