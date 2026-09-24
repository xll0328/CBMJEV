# CBMJev: core-method decision, not a result claim

Date: 2026-09-23. Current server authorization: physical GPU 0 only.
This memo refines `METHOD_RETHINK_20260923.md`. It does not change the running
BRiG adapter or declare a paper method frozen.

Follow-up development evidence: `results/main/CUB_CONDITIONAL_SECOND_QUERY_20260923.md`
tests a fixed-first, conditional-second query on four CUB seeds. The
`research/AWESOME_JEV_METHOD_INSIGHTS_20260923.md` scan separates a Choice
candidate ranker from an absolute STOP gate and highlights batching costs.
The stronger training-OOF-fit/validation-evaluate follow-up is recorded in
`results/main/CUB_OOF_CONDITIONAL_SECOND_QUERY_20260923.md`: a simple
conditional lookup lowers K2 CE in four seeds, while accuracy is mixed and
no JEV/K16 effect has yet been demonstrated. This raises the baseline bar
without changing the evidence gates below.
An OOF-selected additive conditional-risk model then lowers CE versus the
fixed second query in all four seeds but does not reliably beat that lookup:
three of four CE comparisons favor additive, yet the descriptive CE mean is
only −0.0145 and accuracy mean is −0.25 pp against lookup. See
`results/main/CUB_OOF_K2_ADDITIVE_RISK_20260923.md` and the independent
result-to-claim verdict in `findings.md`. This argues against escalating
model complexity on K2 evidence alone.
The public CUB gold attributes also cannot supply an all-sample, complete
28-group oracle: only 118/594 validation cases have all groups annotated,
and those cases cover 92/200 classes with class-distribution TV 0.539 versus
the full validation split. Any gold comparison must be paired on an identical
case set and labeled as complete-case sensitivity or hybrid intervention,
not an all-population oracle ceiling. See
`results/main/CUB_GOLD_COVERAGE_ORACLE_DECISION_20260923.md`.

## Verdict first

**Reject the current headline** "JEV makes a dynamic CBM that is more accurate
and faster than a learned mask." The four-seed CUB validation result does not
support accuracy; a shared encoder/all-head responder does not establish saved
compute; and sequential feature acquisition is established prior work.
**Continue the project with a narrower, falsifiable hypothesis:** a typed,
semantic, noisy-measurement controller may exploit *outcome-dependent* value of
information, especially when candidate menus change, while staying within an
auditable concept-only interface. If that conditional signal is absent, a
well-learned static mask is the correct answer for this regime.

This is an `Accept with Revisions` for the research program and a `Reject and
Pivot` for the current paper headline. The public NanoJev implementation
supports complete candidate-set Choice plus Boolean/Score questions, but it
does not prove that a large language decision model is needed for fixed CUB
groups. See [NanoJev](https://github.com/TianyuCodings/NanoJev) and the
[BRiG-AFA preprint](https://arxiv.org/abs/2608.02305) for the relevant
comparison; the latter already learns nonmyopic, budget-specific Q-functions.

## The hidden mismatch in our motivating analogy

In the doctor story, the first visual impression is an observation. Our strict
controller starts with an empty semantic history. Therefore a deterministic
first query cannot depend on the patient/image. Any patient-specific first
query requires (i) a charged, explicitly acquired semantic triage observation,
or (ii) raw-image access to the controller, which is a *relaxed* concept
bottleneck and requires matched raw-image baselines. We should keep (i) as
the main setting and (ii) only as a separately labeled sensitivity analysis.

There are two distinct resources:

1. **Semantic disclosure/annotation budget:** how many concepts the task head
   may see or how many external human/measurement responses are requested.
2. **Deployment compute and latency:** encoder, concept heads, controller,
   repeated responder calls, and batching.

When one shared encoder and every concept head are evaluated on the first
call, sequential masking can save the first resource but not automatically the
second. If a public dataset's gold concepts simulate external measurements,
call the experiment *simulated oracle acquisition*, not clinical deployment.
Automatic-responder acquisition is instead a constrained semantic interface:
its value is interpretability/early sufficient evidence, not the creation of
new raw input information. Only a genuinely lazy or heterogeneous responder
can ground a compute-saving claim.

## An exact test for whether adaptivity has room to help

Fix an initial, paid-for semantic observation `U` and a frozen task head. For
each legal continuation action `a`, define

`q_a(u) = E[terminal loss + future acquisition cost | U=u, take a]`.

For one remaining choice, the static and adaptive optima are

`R_static = min_a E_U[q_a(U)]`,
`R_dynamic = E_U[min_a q_a(U)]`.

Hence the *conditional-ranking gap* is

`G = R_static - R_dynamic >= 0`.

For two actions, writing `d(U)=q_1(U)-q_2(U)`, the exact identity is

`G = (E|d(U)| - |E d(U)|)/2`.

This is zero if one action is optimal for almost every observed outcome
(allowing ties). Merely varying action order or obtaining many distinct masks
does not imply `G>0`; the *ranking of expected downstream risks* must cross.
For a learned `q_hat` with uniform error at most `epsilon`, the plug-in action
has conditional excess risk at most `2 epsilon`. Thus a tiny `G` cannot survive
even modest value-estimation error. These are elementary decision identities,
not claims of a new theorem or of a measured positive gap.

The test must be honest: fit conditional risks on cross-fit controller data,
choose the initial observation and action family without inspecting the
evaluation fold, and estimate actual policy loss on disjoint held-out cases.
Report the *realized paired policy gap*, uncertainty by original image/group,
and calibration/ranking error. Per-example hindsight winners use hidden labels
and unseen responses and are only an unattainable oracle ceiling; they are not
the deployable conditional gap.

## Why the present controller can fail, and what to test

| Mechanism | Current evidence | Targeted test |
|---|---|---|
| STOP/value scale | Four-seed CUB dynamic policy loses to fixed K16; seed60 only improves near full forced budget | Compare fixed-budget trajectories before any learned STOP; calibrate STOP separately at the same operating objective |
| Weak conditional signal | Seed61 forced K16 uses the same group set for 588/594 cases; seed62 varies more but loses | Held-out conditional-ranking gap above, not trajectory entropy alone |
| Noisy privileged teacher | One-step realized weighted CE differences and Choice softmax over realized errors use unseen outcomes/labels in training targets | Regress *expected* conditional risk, compare hard/soft teachers, measure ranking calibration on unseen cases |
| Short horizon | One-step gain misses concepts whose value appears after later queries | Matched BRiG long-horizon Q and short rollout; do not call generic Bellman learning a JEV contribution |
| Head-mask shift | Head trained with random masks, deployment selects correlated masks | Report selected-mask calibration/error; optionally train a policy-mixture head, then retrain *all* compared policies against the same head protocol |
| Weak/noisy responder | ISIC balanced seed70 automatic concepts are weak and all-concept specificity is poor | Automatic-versus-oracle concepts under distribution-matched head fitting; if ceiling is low, improve responder or drop dataset as main evidence |
| Fake speedup | Shared encoding and batched all-head prediction dominate first-call cost | Separate semantic budget, incremental measurement cost, and measured latency; no speedup claim from replay query count |

## Minimal coherent method, conditional on headroom

**Typed Belief-and-Choice CBM (working label, not a final method name).**

1. A responder maps `(x, requested concept group)` to a typed answer. Its
   training uses only public concept supervision. Its cost contract says
   whether queries are cached, lazy, human-simulated, or batched.
2. The controller sees only acquired typed answers, their identities/statuses,
   the remaining budget, and public candidate descriptions. A *belief model*
   estimates distributions over unqueried **automatic responder outputs**.
   Beliefs are policy-side predictions, never silently passed to the final
   task head as acquired evidence.
3. A budget-conditioned Q/risk model estimates expected continuation risk for
   each legal candidate. A complete-set Choice head may compare the *entire
   current candidate menu*. A separate STOP decision compares current risk
   with cost-adjusted continuation risk. Match its decision loss to the
   paper's metric/operating point; class-weighted CE is not automatically
   optimized for top-1 accuracy or macro-F1.
4. The final predictor consumes only actually revealed typed answers plus the
   mask; it is trained and audited on masks relevant to the evaluation policy.
   Report a mask-only control and a shuffled-response control to identify
   action-path information and reliance on actual concept values.

The belief model is one route, **not an obligatory module**. If BRiG already
captures the attainable gain, use it as the policy baseline and investigate
only the semantic candidate-set representation. If a simple MLP with the same
state, teacher, and budget performs equally, remove JEV necessity from the
claim rather than layering more capacity onto the system.

## Where a JEV-style architecture can be nontrivial

Its potential contribution is *typed, set-valued question evaluation*, not
"a transformer chooses the next concept." A fair ladder is:

1. Fixed-order/static mask and simple conditional MLP.
2. Candidate-independent shared scorer with concept descriptions.
3. Complete-set Choice scorer with candidate interactions.
4. Same Choice architecture with pretrained versus random/frozen semantic
   representations, matched parameter/access/supervision budgets.

Test changed menus (subsets, additions, unavailable groups, costs) separately
from genuinely unseen concepts. Subset/reordering robustness is **not**
unseen-concept transfer. A real unseen-ontology claim additionally needs a
responder and task head that can interpret new concept descriptions; a
fixed-ID CUB head cannot establish it. If those components cannot be built
without extra labels, do not promise cross-ontology generalization.

## Next experimental gates

1. Finish the already-running GPU0 BRiG folds and compare at exact K16 with
   the same final validation cache/head, including paired per-case outcomes.
2. Compute held-out conditional-ranking gap after a fixed paid-for triage
   query, on both automatic and existing gold concept responses, keeping the
   two regimes explicitly separate. Bootstrap at original-image/group level.
3. Measure automatic-versus-oracle full-concept ceilings and selected-mask
   error. If the automatic gap is huge, work on the responder/head before
   building a larger policy.
4. If `G` is credibly positive and BRiG gains, compare matched expected-risk,
   nonmyopic Q, and Choice representations. If `G` is near zero, stop
   policy-complexity escalation and frame the boundary result honestly.
5. Only after a real Choice increment exists, add menu-shift controls and a
   second credible visual benchmark. Freeze selection before touching test.

Theoretical contribution to pursue is not `G>=0` itself (standard), but a
specific result tied to noisy semantic answers, conditional ranking margins,
measurement cost, and the error of a deployable typed candidate scorer. It
belongs in the paper only if its assumptions and predicted regime are checked.

## Idea-evaluator snapshot

Fatal flaws of the *current headline*: `CRITICAL` for claiming compute savings
from a shared all-head responder; `MAJOR` for novelty collision with active
feature acquisition/BRiG; `MAJOR` for no measured adaptive headroom. The
revised, evidence-gated program is a `Yellow` fit: available GPU resources and
public concept labels suffice for strong diagnostics, but data/regime and
novelty risks remain. Provisional 1--10 scores for the *revised hypothesis*,
not measured outcomes: Higher 4, Faster 2, Stronger 6, Cheaper 5, Broader 7.
The plausible paper dimensions are reliability under noisy concepts and
generalization across candidate menus, not a promised speedup.

The potential paradigm change is modest unless the work proves that typed
semantic decision models transfer across concept menus *and* that this yields
real task-cost gains against strong AFA controls. Sequence alone and JEV
branding do not meet that bar.
