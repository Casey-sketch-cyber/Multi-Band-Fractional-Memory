# Reported aggregate results

These files contain rounded aggregate values used in the manuscript. They do
not contain AIST++ raw data, per-frame coordinates, checkpoints, or private
participant information.

- `main_results.csv`: point-weighted held-out errors across three seeds.
- `paired_comparisons.csv`: sequence-level paired bootstrap comparisons.
- `scope_results.csv`: frozen secondary conditions for the complete system.
- `duration_sweep.csv`: post-confirmatory localized-limb duration response.
- `order_ablation.csv`: post-confirmatory validation ablation.
- `efficiency.csv`: strict RTX 3090 latency and memory measurements.
- `split_summary.json`: data, split, mask, and test-window counts.

Positive relative gain means that the second method has lower MPJPE. The
loss-matched control and duration sweep were designed after the focused test and
must remain labeled post-confirmatory.
