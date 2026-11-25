# F1 brief

## Scope

Method overview of the endpoint-constrained kernel bridge with learnable tempered GL kernels (ECB-TGL) for long-gap skeletal motion imputation. This stage pipeline and model anatomy follow the project's CVPR/ECCV Figure 1 style direction, with the physical size set by the Mathematics template.

## Components and materials

| Component | Role | Drawn material |
|---|---|---|
| Masked poses | Input | Original schematic poses with an absent arm in the middle frame |
| Linear interpolation | Existing | Two anchors and their connecting segment |
| Tempered kernel bands | New | Three schematic tap profiles in one hue family |
| Input endpoint bridge | New | Schematic correction that meets its anchors |
| Long-gap TCN | Existing | Compact temporal convolution glyph |
| Output endpoint projection | New | Raw residual and constrained residual meeting both anchors |
| Short-gap TCN | Existing | Parallel temporal convolution glyph |
| Duration routing | Existing | Two paths joining into one |
| Completed poses | Output | Matched schematic poses with the middle arm restored |

## Reading order

1. Read from the masked sequence into the linear interpolation.
2. Follow the long-gap path through the kernel features, the input bridge, the TCN and the output projection.
3. The short-gap TCN shares the linear starting point. Duration routing combines the experts and the completed sequence retains observed joints.

## Main point

Learnable temporal features and explicit endpoint constraints modify the long-gap expert of a duration-routed TCN.

## Visual direction

A left-to-right main path with a restrained lower branch for the short-gap expert. Use muted teal for the learned kernel bands and the output endpoint projection. Keep pre-existing operations and pose drawings neutral. Leave architecture internals and implementation details to the caption and methods text.

## Working assumptions

- Show a two-sided single-limb gap as the representative schematic. The test protocol also includes whole-body gaps.
- The geometric pose and tap glyphs are synthetic and carry no measured values. Their role is to identify the inputs, outputs and operations.
- The figures use the full span of the archived MDPI class with the adjustwidth environment. Confirm the live Mathematics class if the manuscript changes its geometry.
- Both endpoint bridges operate on the long-gap expert. The output projection constrains the TCN residual, and the interpolant is added back before duration routing.

Sources: FIGURE_BRIEF.md; WRITING_BRIEF.md §3 and §8; MATHEMATICS_REVISION_PLAN.md §5.2; fractional_models.py; pilot.py.

## Caption draft

ECB-TGL completes masked 128-frame AIST++ skeletal windows by applying learnable tempered Grünwald–Letnikov (GL) kernel features and input and output endpoint constraints to the long-gap temporal convolutional network (TCN), then routing the two experts by gap duration while preserving observed joints.
