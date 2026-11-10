# Fractional DanceRouter development protocol

Outcome note: this file records the development-time plan. The later order
ablation did not support a joint-adaptive interpretation; the manuscript treats
multi-band fractional memory, rather than learned anatomical order, as the
supported contribution.

## Scope

This extension targets partial-body occlusion repair rather than future-motion
prediction. It preserves observed coordinates exactly and retains the original
duration-aware routing rule. The original DanceRouter artifacts remain frozen.

## Motivation and novelty boundary

Recent work has already used Grunwald-Letnikov fractional differences as a
training loss for human-motion prediction, and fractional total variation has
been used for 3D pose estimation. Therefore, the extension will not claim the
first use of fractional calculus in human-motion analysis.

The testable contribution is narrower: a learnable, occlusion-aware multi-order
fractional increment-memory bank inside the long-gap expert, combined with
joint-wise missing-run duration routing for partial-body motion completion.
Only increments whose two endpoint frames are visible enter the memory
operator, so hidden ground-truth coordinates cannot enter a completion.
Within every bounded missing run, projected increment residuals are centered
and integrated so that the resulting position correction is exactly zero at
both observed endpoints.
The three memory bands are mixed per coordinate channel. Fractional orders are
learned per joint and shared across that joint's XYZ coordinates, allowing the
limbs and torso to adopt different power-law decay while preserving coordinate
symmetry. The finite-window control retains the same parameter count.

## Development-only screen

- Split seed: 20260829.
- Training seed: 20260829.
- Development validation only; test windows are not evaluated during selection.
- Initial state: the previously trained long-gap TCN checkpoint.
- Fine-tuning: at most eight epochs, identical masks, batches, learning rate,
  and validation seeds across variants.
- The fractional and finite-memory adapters share a 10x adapter learning-rate
  multiplier; their inherited TCN backbones use the continued-training rate.
- Primary validation conditions: limb48, limb72, and mixed.
- All variants use the same long-gap training curriculum: temporal32,
  temporal48, limb48, limb72, and mixed masks sampled uniformly per window.

## Variants

1. Continued TCN: controls for additional optimization steps.
2. Fractional-loss TCN: alpha 0.5, truncation window 5, weight 0.1, following
   the closest human-motion prediction precedent.
3. Finite-memory TCN: a parameter-matched bank of ordinary symmetric temporal
   averages over 9, 33, and 97 frames.
4. Fractional-memory TCN: three learnable order bands constrained to
   [0.10, 0.35], [0.35, 0.65], and [0.65, 0.90]. Each band applies a
   bidirectional 97-frame GL fractional integral over visible frame increments
   only. The projected residual increments are integrated as an
   endpoint-constrained bridge for missing joints.
5. Fractional-memory plus fractional-loss TCN.
6. Fixed-order fractional-memory TCN: the three order-band midpoints are held
   at 0.225, 0.500, and 0.775 while channel mixing, gates, and projection remain
   learnable. This tests whether unconstrained order drift shortens memory too
   aggressively.

## Advancement gate

Advance a frozen fractional design to final testing only if it:

- improves mean normalized primary validation MPJPE by at least 0.5% over the
  original DanceRouter;
- beats the strongest continued-training, fractional-loss, and parameter-matched
  finite-memory control by at least 0.2%; and
- has no screened condition worse than the original by more than 1.5%.

If the gate fails, the test set remains untouched while the architecture is
revised or the Fractal and Fractional venue route is abandoned.
