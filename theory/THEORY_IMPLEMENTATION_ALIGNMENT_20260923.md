# Theory--Implementation Alignment Audit

Date: 2026-09-23.  Scope: current CBMJev implementation, completed four-seed
CUB value frontier, CEBaB/ISIC development evidence, and the CVPR draft.

This audit keeps theorem language from drifting beyond what the code and
experiments currently support.  It is not a new theorem and not a claim that the
project is submission-ready.

## Summary

| Theory block | Mathematical status | Current implementation/evidence status | Allowed manuscript wording now |
|---|---|---|---|
| T1 information boundary | Proven under stated interface assumptions | Method and runtime are designed around replay/live environments; no full release audit yet | "The framework enforces a typed evidence interface and records checks for no-bypass-style failures." |
| T2 decision sensitivity | Conditional deterministic bound | Four-seed CUB value policy is negative/static-equivalent; no uniform risk/value estimation bound | "The present one-step value policy does not establish a benefit over matched static selection; the theorem gives no empirical regret guarantee." |
| T3 shared-computation cost identity | Proven for specified cost models | Offline CUB frontier measures query budgets; seed61/62 shared-GPU live profiles show little all-versus-batch difference | "Query count and measured latency are separate; an isolated speedup result remains pending." |
| T4 noisy complementarity | Proven for toy distributions | No accepted real-data complementarity evidence yet; CEBaB/ISIC are boundary evidence, not positive mechanism confirmation | "The toy result motivates diagnostics for error dependence and complementary concepts." |
| T5 finite frozen-family risk | Standard finite-family bound | No frozen final candidate family or independent certification run yet | "A certification protocol is specified; no final risk certificate has been issued." |
| T6 selection/calibration caution | Proven as examples/conditions | No accepted selected/stop calibration study yet | "Post-selection reliability must be evaluated separately." |

## Implementation-facing side conditions

### T1: information boundary

Implementation hooks currently aligned:

- `ReplayEnvironment` and `LiveEnvironment` expose acquired typed observations,
  not raw input, to policy rollouts.
- Candidate generation is based on the observed state and schema.
- Offline replay attaches labels, sample IDs, and groups after inference traces
  are produced.
- Live profiling computes input digests after measured episodes and labels those
  fields as external instrumentation.

Still pending before a release claim:

- A final no-bypass audit over the exact frozen code snapshot.
- A check that every paper-claimed policy uses the same allowed interface.
- Static review that no new helper introduced a raw-input, sample-ID, timing, or
  hidden-cache channel.

Current paper status: T1 supports the protocol definition, not a theorem that
concepts are semantically faithful or causal.

### T2: value/risk controller sensitivity

Implementation hooks currently aligned:

- Risk and value objectives are separate configurations/checkpoints.
- `value`, `value_singleton`, and `static_value` are rejected unless the merged
  controller objective is actually `value`.
- The CUB value branch uses separate clean value-objective checkpoints for
  seeds 60/61/62/63 rather than reusing a risk-objective controller.
- The completed four-seed validation aggregate is
  `negative_or_static_equivalent` for both accuracy and macro-F1. Forced K16
  diagnostics and trajectory analysis show mixed seed outcomes and no stable
  conditional-ordering benefit.

Still pending:

- Margins/error diagnostics for chosen vs. runner-up actions.
- A matched long-horizon adaptive baseline; the fold-safe BRiG adaptation is
  training on GPU0 but has no validation result yet.
- Any empirical support for uniform estimation error.  The theory file does not
  provide this and the paper must not imply it.

Current paper status: value controllers can be evaluated as empirical policies,
not as Bellman-optimal or regret-certified policies.

### T3: actual cost vs. query budget

Implementation hooks currently aligned:

- Offline grid evaluators report declared query units and are explicitly marked
  as not latency evidence.
- The profile entrypoint measures live validation episodes with all-at-once and
  split-batch configurations.
- A seed60 historical profile attempt was rejected by the responder identity
  guard; this is correct and is not used as evidence.
- Seed61/62 live validation profiles are complete and included as development
  cost-accounting evidence. They were measured on shared GPUs.

Still pending:

- Isolated final hardware protocol for any speedup claim.
- A cost model fit or explicit decision to avoid fitting one because residuals
  are too large.

Current paper status: cost-accounting protocol exists; speedup claim is pending.

### T4: complementarity and error dependence

Implementation hooks currently aligned:

- The analytical noisy parity figure is labeled as a toy illustration.
- Boundary evidence table prevents overclaiming when value policies save queries
  but lose macro-F1 or collapse to majority-class behavior.

Still pending:

- A real-data error-dependence diagnostic connecting responder errors to
  task-loss improvements.
- A controlled synthetic/mechanism experiment generated from the frozen script
  rather than hand-written claims.

Current paper status: T4 is a mechanism lens and a falsification target, not
evidence that CUB/CEBaB contain exploitable complementarity.

### T5: finite frozen-family certification

Implementation hooks currently aligned:

- The paper evidence state remains `DRAFT_NOT_SUBMISSION_READY`.
- Release-gate language distinguishes validation development evidence from final
  paper claims.

Still pending:

- Frozen candidate family, including exact policies, thresholds, responders,
  task heads, and failure handling.
- Independent certification split with iid/group assumptions documented.
- Machine-readable candidate count `M`, risk target `alpha`, `delta`, and
  complete trace files.

Current paper status: include only as a proposed guardrail/protocol unless a
real certification run exists.

### T6: post-selection calibration

Implementation hooks currently aligned:

- Current evaluation summaries retain stop coverage/query counts, which are
  prerequisites for selected-slice diagnostics.

Still pending:

- Calibration metrics for pooled, selected, and stop-time subsets.
- Minimum slice sample counts and a decision on whether the calibration story is
  central or just a limitation.

Current paper status: use as a caution, not as a demonstrated empirical finding.

## Current evidence classification

| Evidence | Supports | Does not support |
|---|---|---|
| `cub_seed60_budget_grid_nostatic_v1` | CUB automatic-concept query-budget frontier for fixed/random/all references | Dynamic value/JEV gain, speedup, multi-seed stability, test result |
| `cub_value_policy_60_61_62_63_v1` | Four-seed CUB validation negative/static-equivalent result for current dynamic value policy | Broad dynamic superiority, JEV necessity, locked test result |
| `boundary_evidence_cebab_isic_v1` | CEBaB query-saving/macro-F1 boundary; ISIC majority-class failure and balanced-head low-specificity all-concept case | Positive algorithmic claim, medical reliability, JEV necessity |
| `cub_live_profile_61_62_v1` | Shared-GPU development cost-accounting and batch consistency | Isolated deployment speedup |

## Required edits if future results are weak

- The current CUB value policies do not beat matched static/fixed frontiers;
  the paper must retain this negative result and cannot use them as its main
  positive method evidence. A revised method needs new matched experiments.
- If value saves query groups but latency is flat or worse, the paper should use
  T3 as a negative-cost explanation, not hide behind query budgets.
- If ISIC remains majority-class behavior, all medical-language claims must stay
  in limitations/boundary evidence.
- If CEBaB macro-F1 loss persists, report it as a budget-quality trade-off, not
  "same performance with fewer concepts."
- If certification is empty, report `no certificate`; do not weaken the risk
  target after seeing the result.

## Paper insertion guidance

Safe current wording:

> Our analysis gives conditions under which adaptive concept measurement can or
> cannot improve a task--cost trade-off.  Four-seed CUB validation replay finds
> the current one-step value policy negative/static-equivalent under matched
> concept budgets. CEBaB and ISIC remain boundary cases, and shared-GPU live
> profiles support cost accounting rather than a speedup claim.

Unsafe current wording:

> CBMJev is certified, faster, and more accurate than static masks on visual
> benchmarks.

The latter still requires a revised method with matched positive results,
strong adaptive baselines, isolated cost measurement, and locked evaluation.
