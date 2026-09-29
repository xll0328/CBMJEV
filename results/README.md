# Released development evidence

These are selected aggregates and diagnostics, not a complete archive of
training runs. All reported empirical comparisons are development-validation
evidence; no result here should be promoted to a locked-test finding.

| Question | Artifact |
| --- | --- |
| Four-seed comparison with the exact-K16 BRiG-style reference | [Per-seed diagnostic](diagnostics/cub_brig_rle_k16_gate_60_63_v1.json) |
| Comparison at equal mean declared query cost | [Full static-mixture grid](main/cub_value_full_grid_equal_mean_cost_static_60_63_v1.json) |
| Canonical fitted-static versus dynamic replay | [Four-seed aggregate directory](main/cub_canonical_fixed_replay_60_63/) |
| Early seed-60 budget sweep | [Single-seed diagnostics](main/cub_seed60_budget_grid/) |

The paper's tables include both negative results and bounded pilot diagnostics.
Read each artifact's split, mode, seed set, comparator, and aggregation before
combining numbers. Retrospective static mixtures are analysis references, not
deployable policies selected before evaluation. Replay timing is not deployed
latency. Synthetic smoke outputs are not included as paper evidence.

Raw data, checkpoints, response caches, and all per-example trajectories are
not distributed. You can inspect these summaries without data; reproducing
the training/evaluation requires the data and configurations described in the
[quick start](../docs/QUICKSTART.md). The full manuscript is not distributed
in this code release.
