# Four-seed canonical fixed replay — CUB development validation

Date: 2026-09-24. This is a same-system offline validation replay correcting
the fixed/static alias described in
`CUB_BUDGET_GRID_FIXED_STATIC_ALIAS_CORRECTION_20260924.md`. It is not a test
result, significance test, paper-evidence artifact, or speed measurement.

## Protocol and provenance

For seeds 60–63, the canonical schema-order `fixed` policy was evaluated at
K={0,1,2,4,8,16,28} on the same frozen merged controller, responder, and
validation response cache used by the historical four-seed budget grids. Each
run was bound to its corresponding historical grid by source hashes and
verified that (a) the old fitted static order was noncanonical and (b) the
historical `fixed` and `static` rows were exact aliases. The new replay did not
load test data or change any fitted component. It writes
`paper_evidence=false` and
`OFFLINE_VALIDATION_DEVELOPMENT_NOT_PAPER_OR_LATENCY_EVIDENCE` in its metrics
and receipt.

The output directories are `runs/cub_eval_canonical_fixed_seed{60,61,62,63}_v1/`.
Each receipt is `COMPLETE` for 594 validation rows × 7 budgets. The per-seed
`metrics.json` SHA-256 values are:

| Seed | SHA-256 |
|---:|---|
| 60 | `0768c859e1a8c9fbdd86506e161760be86d289e2dccbfe488d0747883465f5b7` |
| 61 | `bb23ad53dc36d9822d11b5e51296f9891c7a1a82cda8a47bb489085bc11eb436` |
| 62 | `b784d7e43b6c00730c8f0364586f184a7d2dba7ce74933e491e3d5827f036174` |
| 63 | `d39795220cb64da101e367ce824c6bda39a11b6ba002d913f6b3100ce385b53d` |

Evaluator script SHA-256: `4d8ae39cce382e8b1496003d0414205c6239099f55d766cc203bc9235f37f437`.
The final plan-file hash check is compatible with the older deployed
`cbmjev.crossfit_evaluation` helper API; a unit regression test covers both
hash preservation and drift rejection.

## Results

Numbers are mean ± sample standard deviation across the four seeds. `value`,
`value_singleton`, `static`, and `static_value` are the historical matched
validation policies; the old `fixed` row is intentionally omitted because it
is an alias of `static`, not canonical schema order. Accuracy and macro-F1 are
percent. “Groups” is mean number queried under that policy (fixed-budget
policies query exactly K; stop policies may stop early).

| Policy | K | Accuracy (%) | Macro-F1 (%) | Groups |
|---|---:|---:|---:|---:|
| Canonical fixed | 4 | 15.74 ± 1.98 | 12.08 ± 1.56 | 4.00 |
| Fitted static | 4 | 18.18 ± 0.36 | 14.32 ± 0.58 | 4.00 |
| Value | 4 | 15.11 ± 1.89 | 11.49 ± 1.65 | 4.00 |
| Value-singleton | 4 | 15.61 ± 2.76 | 12.00 ± 2.53 | 4.00 |
| Canonical fixed | 8 | 18.69 ± 2.49 | 15.52 ± 2.48 | 8.00 |
| Fitted static | 8 | 25.00 ± 1.79 | 20.78 ± 1.63 | 8.00 |
| Value | 8 | 21.55 ± 2.14 | 17.65 ± 2.00 | 7.26 |
| Value-singleton | 8 | 22.60 ± 2.99 | 18.67 ± 2.74 | 7.46 |
| Static-value (learned order + stopping) | 8 | 23.91 ± 2.10 | 19.81 ± 2.53 | 6.82 |
| Canonical fixed | 16 | 25.67 ± 1.76 | 21.72 ± 2.02 | 16.00 |
| Fitted static | 16 | 29.67 ± 1.65 | 25.67 ± 1.77 | 16.00 |
| Value | 16 | 24.41 ± 2.30 | 20.74 ± 1.94 | 9.10 |
| Value-singleton | 16 | 24.83 ± 3.51 | 20.99 ± 3.43 | 9.58 |
| Static-value (learned order + stopping) | 16 | 25.72 ± 2.86 | 21.78 ± 3.34 | 7.74 |
| Canonical fixed | 28 | 31.14 ± 1.95 | 27.43 ± 2.19 | 28.00 |
| Fitted static | 28 | 31.14 ± 1.95 | 27.43 ± 2.19 | 28.00 |
| Value | 28 | 27.61 ± 2.66 | 23.91 ± 2.32 | 18.70 |
| Value-singleton | 28 | 25.04 ± 3.52 | 21.13 ± 3.46 | 9.65 |
| Static-value (learned order + stopping) | 28 | 25.76 ± 2.91 | 21.81 ± 3.37 | 7.79 |

### Paired seed deltas versus canonical fixed

Positive values favor the named policy. The table gives accuracy percentage
point deltas for each seed; it is descriptive and does not test statistical
significance.

| Policy | K=4 | K=8 | K=16 | K=28 |
|---|---:|---:|---:|---:|
| Fitted static | +2.44 ± 2.10 (3/4 wins) | +6.31 ± 1.92 (4/4) | +4.00 ± 0.84 (4/4) | 0.00 (same full prefix) |
| Value | −0.63 ± 1.03 (1/4) | +2.86 ± 1.88 (4/4) | −1.26 ± 0.92 (0/4; one tie) | −3.54 ± 4.27 (0/4; two ties) |
| Value-singleton | −0.13 ± 2.51 (2/4) | +3.91 ± 0.91 (4/4) | −0.84 ± 1.82 (2/4) | −6.10 ± 1.70 (0/4) |
| Static-value | +2.44 ± 2.10 (3/4) | +5.22 ± 0.46 (4/4) | +0.04 ± 1.42 (2/4) | −5.39 ± 2.25 (0/4) |

## Interpretation

1. Correcting the alias changes the story in a useful but limited way: the
   value-driven dynamic policies beat the canonical schema-order baseline at
   K=8 in all four seeds (+2.86 pp for `value`, +3.91 pp for
   `value_singleton`). This is evidence that they do more than reproduce
   schema order at that budget.
2. The stronger matched fitted-static ordering remains better at K=8: 25.00%
   accuracy versus 21.55% (`value`) and 22.60% (`value_singleton`). It wins
   against canonical fixed in all four seeds and against each dynamic method
   in all four seeds at this K. Thus current results do **not** show dynamic
   selection beating the strongest learned static acquisition baseline.
3. At K=16, both value policies are slightly below canonical fixed on mean
   accuracy; at K=28, `value` is 3.54 pp below canonical fixed and
   `value_singleton` is 6.10 pp below. The broad claim that adaptive acquisition
   improves the accuracy-budget frontier is unsupported by these results.
4. `static_value` suggests a possible stopping/efficiency tradeoff, not a
   dynamic-selection win: at nominal K=16 it averages 25.72% accuracy with
   7.74 groups, close to canonical fixed's 25.67% at 16 groups, but below the
   fitted-static fixed-budget result of 29.67%. Its utility and cost claims
   need paired uncertainty and full compute-cost accounting.
5. These are repeatedly inspected validation results on one dataset, from one
   family of fitted systems. They are not a locked-test evaluation, a JEV
   versus non-JEV ablation, a significance/equivalence finding, or sufficient
   evidence for paper conclusions.

## Next research action

Do not scale the present dynamic-value configuration across more seeds yet.
First inspect why the learned static prefix is so strong and why policy
quality degrades at larger nominal budgets; verify which state/action/value
signals drive the K=8 gain, and run a prespecified mechanism diagnostic against
the same canonical and fitted-static controls. Any revisions remain
training/validation-only. Add a second dataset and a non-JEV sequential control
before making a JEV-specific claim; keep final test data locked until the
protocol and model selection are frozen.
