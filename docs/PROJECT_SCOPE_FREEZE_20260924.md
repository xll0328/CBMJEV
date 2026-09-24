# CBMJev scope freeze and decision gate — 2026-09-24

## Decision

Freeze broad architecture search and stop adding datasets, modules, or loosely
motivated hyperparameter runs. Current evidence does **not** support the initial
broad claim that a JEV-style dynamic concept policy improves over a strong
learned-static concept selector. Keep the project active, but treat the current
method as an unproven candidate and narrow the next phase to one falsifiable
question: *does conditioning acquisition on the observed concept history add
value beyond a matched-cost learned-static policy when concepts come from the
same imperfect responder?*

This is a scope decision, not a claim that sequential acquisition is generally
ineffective and not yet a final paper-positioning decision.

## Evidence motivating the freeze

- On the four-seed CUB development-validation grid, neither dynamic value
  variant has a positive mean accuracy delta over the expected-cost-matched
  fitted-static frontier at any nontrivial inspected cap K=2, 4, 8, 16, or 28.
  No cap has positive deltas in all four seeds. This is not an equivalence test
  and does not establish general inferiority.
- In the single-seed CUB BRiG/RLE K16 replay, RLE reaches 28.79% accuracy and
  25.78% macro-F1: above canonical schema order (26.09% / 22.72%), but below
  fitted static (29.46% / 26.74%). It is not multi-seed evidence or a JEV
  ablation.
- The old trusted-cache implementation was not checkpoint-identical to RLE;
  a one-fold mixed validation replay was lower by 0.842 accuracy points. A
  source-bound audit traced an immediate budget-2 discrepancy to batch-order
  floating-point variation in cached risks. The updated terminal-state cache
  subsequently passed receipt integrity but matched RLE checkpoints only at
  budgets 1--7, not 8--16; it is excluded from the current claim gate. See
  `results/exploratory/CUB_BRIG_TRUSTED_CACHE_PARITY_20260924.md`.

Sources: `results/main/CUB_VALUE_FULL_GRID_EQUAL_MEAN_COST_20260924.md`,
`findings.md` (CUB BRiG RLE section), and
`docs/BRIG_TRUSTED_CACHE_ORDER_AUDIT_20260924.md`.

## Bounded next actions

1. **Completed:** seed-60/fold-0 terminal-cache run and all-budget checkpoint
   comparison. Receipt integrity passes; exact parity fails from budget 8.
   Retire this variant from the claim gate and preserve the diagnosis. Do not
   launch rescue runs or use its metrics as method evidence.
2. **In progress:** train the nine remaining reference RLE folds for seeds
   61--63 on authorized physical GPUs 0/1.
3. Complete one finite strong-adaptive-baseline gate at **exact K=16** on CUB
   validation: preserve the completed BRiG/RLE seed 60, train seeds 61--63
   with the same frozen hidden width 128, eight epochs per budget, Adam
   learning rate 0.001, and unchanged responder/head/fold sources, then compare
   with each seed's existing fitted-static K16 result. Report accuracy first,
   macro-F1 second, every seed delta, and paired image/group uncertainty.
   Exact K removes stopping as an explanation. This is a development decision
   gate, not a final test result or a JEV-specific ablation.
4. Treat the positive-superiority method line as **no-go** unless BRiG has at
   least a +1.0 percentage-point four-seed mean accuracy delta over fitted
   static, positive deltas in at least three of four seeds, and a nonnegative
   mean macro-F1 delta. This operational threshold was fixed after the
   exploratory seed-60 result but before seeds 61--63 completed; it is a
   development resource decision, not a statistical significance test.
   If it passes, separately test a capacity/objective-matched non-JEV
   sequential controller and observed-history ablation before attributing
   anything to JEV. If it fails, end model training for this formulation and
   complete the bounded negative/measurement study. Hold the official test
   set until the final question and protocol are frozen.

## Go/no-go for expanding the research scope

The completed four-seed value-policy results already reject the broad current
positive claim at the development gate. The exact-K BRiG comparison above is
the last bounded check for a stronger adaptive controller, not an invitation
to tune against the same validation set. If it fails, preserve the
negative/boundary findings and revise the paper position. CVPR 2027 remains a
target, not a promised outcome; current evidence is not submission-ready.

## Resource closure

On 2026-09-24, project-owned legacy BRiG workers with no checkpoint or
COMPLETE receipt after roughly 20--23 hours were terminated by SIGTERM:
nine original crossfit folds, three fast folds, two accelerated folds, and the
fast-run automatic handoff watcher. Their logs remain on the server; no result
files were deleted. The completed source-bound seed-60 RLE fold/evaluation
artifacts remain intact. The terminal-cache parity run completed on physical
GPU 1 but failed all-budget checkpoint identity. Nine new RLE folds were
launched on physical GPUs 0/1; the launcher and receipts are tracked by
`scripts/run_cub_brig_rle_k16_gate.sh`.
