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
