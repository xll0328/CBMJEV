# CUB value-policy full-grid equal-mean-cost diagnostic (2026-09-24)

The four-seed, 594-validation-image-per-seed CUB grid was reanalyzed at each
policy's **observed mean declared concept cost**, not just its nominal cap K.
For each dynamic point, a per-case independent randomized mixture of the two
adjacent fitted-static budgets has the same *expected* declared cost. Expected
accuracy is a linear interpolation of the two static accuracies. This is a
retrospective development-validation diagnostic, not a deployed policy or a
test-set result.

The source-hashed, machine-readable report is
[`cub_value_full_grid_equal_mean_cost_static_60_63_v1.json`](cub_value_full_grid_equal_mean_cost_static_60_63_v1.json).
It analyzes both dynamic value variants at every predeclared cap over seeds
60--63. Positive entries mean dynamic accuracy exceeded the cost-matched
fitted-static mixture; `+ seeds` counts strictly positive seed deltas.

| Cap K | Value delta (pp) | Value + seeds | Singleton delta (pp) | Singleton + seeds |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 0/4 | 0.00 | 0/4 |
| 1 | 0.00 | 0/4 | 0.00 | 0/4 |
| 2 | -0.63 | 1/4 | -0.29 | 2/4 |
| 4 | -3.07 | 0/4 | -2.56 | 1/4 |
| 8 | -2.15 | 1/4 | -1.57 | 0/4 |
| 16 | -0.94 | 1/4 | -1.02 | 2/4 |
| 28 | -0.05 | 1/4 | -0.86 | 2/4 |

The K0/K1 ties are trivial, not dynamic-policy evidence. At K8, the unrestricted
value policy costs 6.28--8.00 groups across seeds and loses 2.15 pp on average
to the equal-mean-cost comparator. At K16, its cost is 7.09--10.95 groups and
its mean deficit is 0.94 pp. At K28, seeds 61 and 62 query all 28 groups and
tie the all-concepts static endpoint by construction; that near-zero aggregate
does not indicate successful adaptive acquisition. The other K28 seed deltas
are -1.29 and +1.10 pp. Thus no nontrivial budget has a consistent positive
four-seed point estimate for either variant. This does **not** prove
inferiority, equivalence, or absence of benefit for a redesigned policy.

All four source metrics have distinct SHA-256 bindings in the JSON. In each
source, the historical `fixed` and `static` per-group metrics are identical
for all seven shared budgets, and the static order is noncanonical. They are
one fitted-static comparator, not two independent controls. A corrected
canonical fixed-order evaluation is still pending. Macro-F1 is not interpolated
because it is nonlinear in the confusion matrix.

Selected unrestricted-value K8, K16, and K28 image-group paired-bootstrap
files are [`K8`](cub_value_k8_equal_mean_cost_static_bootstrap_60_63_v1.json),
[`K16`](cub_value_k16_equal_mean_cost_static_bootstrap_60_63_v1.json), and
[`K28`](cub_value_k28_equal_mean_cost_static_bootstrap_60_63_v1.json).
At K8, the four conditional 95% percentile intervals in seed order are
[-5.50, 0.73], [-8.39, -1.67], [-4.47, 1.13], and [-2.40, 3.02] pp.
The seed-61 interval excludes zero, but it is **not a confirmatory discovery**
after examining this grid; this does not overturn the multi-seed pattern.
At K28, the seed-61/62 intervals collapse to [0, 0] because both policies
select all concepts, not because uncertainty about dynamic value is zero.
These intervals re-match cost inside each resample,
but do not account for training-seed variability, randomized mixture draws,
many inspected operating points, or the use of validation costs to determine
mixture weights. Equal **mean declared cost** does not imply equal realized
cost on every image or equal wall-clock latency.

## Reproduce and next decision

Run `python3 -m scripts.analyze_cub_cost_matched_frontier` with the four
`runs/cub_eval_value_budget_grid_seed{60,61,62,63}_v1/metrics.json` inputs,
`--adaptive-policies value_K0 value_singleton_K0 value_K1
value_singleton_K1 value_K2 value_singleton_K2 value_K4 value_singleton_K4
value_K8 value_singleton_K8 value_K16 value_singleton_K16 value_K28
value_singleton_K28`, and `--baseline-method static`. The CLI requires a new
output path; do not overwrite the source-hashed artifact above. Selected
bootstrap reports use `--adaptive-policy value_K8` (and K16/K28) with the
same sources and comparator.

This diagnostic argues against promoting the current dynamic value controller
as a positive CUB result. The next server experiments should first restore a
genuinely distinct canonical fixed-order control, then separate stopping from
query choice with the existing forced-budget controller, and finally evaluate
a redesigned policy against fitted static and strong adaptive acquisition
baselines at predeclared equal-cost operating points. Freeze model selection
before any locked test evaluation. The server was unreachable by SSH during
this analysis, so no new GPU job was launched; only physical GPU 0/1 are
authorized for subsequent work.
