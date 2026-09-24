# What candidate-set interaction can and cannot establish

Date: 2026-09-23. Status: deterministic representation result and research-design
constraint, **not** a novelty theorem or an empirical benefit claim.

## Exact question

Our structured Choice head receives a complete feasible menu. Each row already
contains the same observed concept state, remaining budget, cost weight, and one
action with its declared incremental cost. The `none` arm scores each row with
a shared scalar head; the `attention` arm additionally mixes candidate rows.
Both arms use the same visible information and the same action labels. Does
menu interaction create a decision that an unrestricted independent scorer
could not express?

## Representation-level result

Let `h` be a visible history, `A(h)` a nonempty finite feasible action menu
determined entirely by `h` and frozen public protocol, and `g(h,A,a)` any
deterministic finite real score for `a in A`. Define

`f(h,a) := g(h,A(h),a)` for every feasible `(h,a)`.

Then for every `h`, the complete-menu argmax of `g`, with the same deterministic
tie rule, equals the argmax of the candidate-independent `f(h,a)`. This is
substitution of the deterministic menu into `g`, so it is an exact pointwise
claim, with no probability, optimization, approximation, or asymptotic limit.
It also covers STOP when STOP is an ordinary member of the menu. A sufficiently
expressive scorer with access to an injective representation of `(h,a)` can
therefore reproduce any deterministic set-attention **choice policy** under
these assumptions.

This does **not** say our finite LayerNorm-plus-linear scalar arm can fit that
`f`, or that SGD finds it. The actual `none` arm is deliberately weak; the
attention arm adds parameters and nonlinear candidate mixing. It also does
not equate their calibrated choice probabilities, runtime, or sample
complexity. A softmax denominator can vary with the menu even when scores are
candidate-independent.

If the menu contains exogenous information not determined by `h`, the premise
fails. For example, at identical `h`, suppose two legal menus both contain
`a,b`, but a desired ranking is `a>b` in `{a,b}` and `b>a` in `{a,b,c}`.
No fixed real scores `f(h,a),f(h,b)` with a menu-independent tie rule can
represent both strict rankings; a set-dependent scorer can. This is a
counterexample to extending the result to hidden/exogenous menu changes,
not evidence such menu effects are desirable in CBMJev. Exposing menu identity
as part of the state would restore the premise.

## Consequences for claims and experiments

1. Candidate-set attention is an **inductive-bias / finite-capacity / training
   hypothesis** here, not a generally necessary source of expressive power.
   The mathematically meaningful CBMJev distinction is conditional acquisition
   after observed responses, under costs and a genuinely visible state; the
   separate question is whether a NanoJev-inspired set head improves that
   controller in practice.
2. A positive `attention` versus `none` difference alone would confound set
   interaction with added capacity. Compare a parameter-matched nonlinear
   independent scorer, identical targets/history/menu and training budget, and
   a non-JEV sequential controller. If the gain vanishes, state that set
   attention is unnecessary in this tested regime.
3. STOP inside relative Choice softmax is only a ranking action. It does not
   provide calibrated absolute value of continuing. Compare the independent
   expected-utility objective and a separately calibrated STOP gate before
   making a value-of-information claim.
4. Since the current teacher is per-example softmax of realized error+cost,
   even a perfect population Choice fit need not select minimum conditional
   expected utility (T8). Do not use this representation argument to imply
   objective consistency or multi-step optimality.
5. If the action menu is capped, randomized, or changes with availability,
   include the public cap/availability/random seed in `h` or treat the menu as
   explicit input. Record which case each experiment implements.

The current CUB Choice and conditional-risk runs are pending. This note makes
no empirical claim about either arm.
