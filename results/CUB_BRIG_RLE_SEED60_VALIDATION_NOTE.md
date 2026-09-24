# CUB BRiG RLE seed-60 K16 validation note

Date: 2026-09-24. Status: exploratory development-validation result only.

## Protocol and provenance

The completed run is the run-length-state-encoding implementation of the same
budget-specific BRiG Q objective (`runs/cub_brig_rle_seed60_fold{0,1,2}_K16_v1`).
All three fold receipts are `COMPLETE`; the training reports use seed 60,
hidden width 128, 8 epochs per budget, batch size 64, Adam LR 0.001, and
budgets 1–16. Policy fitting uses only the nested training folds. The replay
(`runs/cub_brig_rle_seed60_eval_K16_v1`) is explicitly `split=validation`,
`mode=offline_replay`, `paper_evidence=false`, with 594 samples. No test split
was loaded for this evaluation.

Before comparing results, the BRiG, canonical-fixed, and value-grid metric
files were checked to share identical hashes for the merged receipt, responder
receipt, validation-cache manifest, task-head component, controller component,
responder checkpoint, and plan binding. Thus these are matched seed-60
development-validation comparisons, not independent replications. The BRiG
metrics SHA-256 is
`afe81df84881922c551099215a794552e0f006c5d478916448d4b82c8fef1b2f`.

## Results

| Policy | Groups | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Canonical schema-order fixed, K=16 | 16 | 0.26094 | 0.22719 |
| Fitted static order, K=16 | 16 | 0.29461 | 0.26735 |
| Value-singleton, nominal K=16 | 11.78 | 0.26599 | 0.23486 |
| BRiG RLE, K=16 | 16 | 0.28788 | 0.25784 |
| All-concept endpoint, K=28 | 28 | 0.32660 | 0.29683 |

BRiG RLE is +2.69 accuracy points and +3.07 macro-F1 points over canonical
fixed, and +2.19 accuracy points over value-singleton in this seed. It remains
0.67 accuracy points and 0.95 macro-F1 points below fitted static; the
all-concept endpoint is higher still. These are descriptive deltas on one
repeatedly inspected validation set, with no uncertainty estimate. They do not
show a JEV-specific gain (there is no matched non-JEV same-capacity sequential
ablation), general adaptive superiority, test performance, or latency gain.

## Interpretation and next gate

This is a useful signal that BRiG's budget-specific state/action controller
may recover some performance over schema order at K=16. The stronger fitted
static baseline prevents claiming that the adaptive controller is best. The
follow-up trusted-cache fold-0 training matches the checked data, Q1,
rollout-state and risk-target semantics, but its full GPU checkpoint diverges
from RLE from budget 2 onward. A mixed replay using trusted-cache only for fold
0 and RLE for folds 1/2 scores 27.9461% accuracy / 24.9583% macro-F1, versus
28.7879% / 25.7840% for all-RLE (−0.842 / −0.826 pp). This complete 594-row
validation replay is explicitly `paper_evidence=false`; it is not a clean
method comparison or a statistical claim. The drift can affect policy
behavior, so trusted-cache is not established as behavior-preserving. Diagnose
and control optimizer-level numerical divergence before training the remaining
folds with it; then require a clean all-fold comparison, multi-seed validation,
and a matched non-JEV state/action MLP. Only after selection and protocol are
frozen should the locked test be opened once.

Authoritative server artifacts:

- Fold receipts/checkpoints/reports: `runs/cub_brig_rle_seed60_fold{0,1,2}_K16_v1/`.
- Replay receipt/metrics/traces: `runs/cub_brig_rle_seed60_eval_K16_v1/`.
- Same-system canonical fixed: `runs/cub_eval_canonical_fixed_seed60_v1/`.
- Same-system fitted-static/value controls: `runs/cub_eval_value_budget_grid_seed60_v1/`.
