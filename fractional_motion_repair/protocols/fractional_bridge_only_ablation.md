# Post-confirmatory bridge-only ablation

## Purpose

The frozen confirmatory fractional system used both the multi-band fractional
memory bridge and a short-window fractional trajectory loss. The finite-window
bridge used the base completion loss. This post-confirmatory ablation removes
the fractional trajectory loss from the fractional bridge to determine whether
the long-limb result depends materially on the additional loss term.

## Status and interpretation

- This analysis was specified after the focused test had been opened.
- It is a mechanism-attribution ablation, not a second confirmatory test.
- It cannot replace or redefine the frozen primary analysis.
- Checkpoints are selected on the same development validation split and are
  evaluated without additional test-based tuning.

## Fixed comparison

- Dataset split: choreography-grouped AIST++ split seed 20260829.
- Training seeds: 20260829, 20260830, and 20260831.
- Test mask seed: 20270829.
- Finite model: three ordinary windows of 9, 33, and 97 frames, base loss.
- Fractional model: three 97-frame GL integral bands, base loss only.
- Shared components: frozen short expert, endpoint bridge, duration router,
  training masks, optimizer, epoch budget, and nominal parameter count.
- Primary descriptive condition: one-limb 72-frame outage.
- Statistics: paired sequence-level 10,000-resample bootstrap, reported as a
  post-confirmatory estimate.
