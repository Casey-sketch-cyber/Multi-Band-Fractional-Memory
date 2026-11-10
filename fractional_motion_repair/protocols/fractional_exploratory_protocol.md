# Post-Confirmatory Fractional Memory Analyses

Date frozen: 2026-08-16, after the focused limb-72 confirmatory test.

## Status

These analyses are explicitly exploratory. They do not alter the frozen models,
the confirmatory limb-72 endpoint, or the decision rule recorded in
`fractional_confirmatory_protocol.md`.

## A. Limb-Gap Duration Response

- Dataset split: choreography-grouped AIST++ test split, split seed 20260829.
- Models: the already frozen parameter-matched finite-window bridge and learned
  fractional bridge from training seeds 20260829, 20260830, and 20260831.
- Windowing: native 60 Hz, 128-frame windows, stride 128, including the final
  valid window of every sequence.
- Limb-gap lengths: 24, 36, 48, 60, 72, 84, and 96 frames.
- Mask repeats: three fixed repeats beginning at seeds 20270829, 20270830, and
  20270831. A model pair always receives exactly the same mask realization.
- Primary descriptive quantity: sequence-averaged relative MPJPE gain of the
  fractional bridge over the finite-window bridge.
- Uncertainty: 10,000 paired sequence bootstrap resamples after averaging the
  three training seeds and three mask repeats within each held-out sequence.
- Interpretation: the curve is evidence about the duration response of the
  frozen operator, not a second confirmatory hypothesis test.

## B. Learned Fractional Memory

For each frozen fractional checkpoint, report the three learned GL integral
orders per joint, the softmax mixture assigned to the three order bands, the
residual gate, and the normalized kernel's expected lag. Summaries are averaged
over xyz coordinates and then across the three training seeds. No test targets
are used for this analysis.

## Guardrails

1. No model or hyperparameter is changed after viewing confirmatory test data.
2. No gap length is selected as a new primary endpoint.
3. Negative or non-monotone duration-sweep results remain in the report.
4. Confirmatory and exploratory findings are labeled separately in the paper.
