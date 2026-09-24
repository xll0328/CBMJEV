# Awesome-Jev scan: what transfers to CBMJev

Date: 2026-09-23. The request did not specify a URL and there are several
unrelated repositories named `awesome-jev`. I inspected the
[ckaraca list](https://github.com/ckaraca/awesome-jev),
[cobanov source-backed list](https://github.com/cobanov/awesome-jev), and
[valentynkit directory](https://github.com/valentynkit/awesome-jev-typesafe),
then followed their references to **official TypeSafe documentation**. These
lists are discovery aids, not evidence that any listed application works.

## The model-interface lessons

1. [Official speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
   recommends asking many independent questions about the *same state* in one
   request and discarding irrelevant answers in code. The
   [parallel-questions example](https://docs.typesafe.ai/cookbooks/parallel_questions)
   reports a large batching advantage for a document-dominated workload.
   Hence sequential concept queries do **not** automatically save Jev calls,
   latency, tokens, or encoder work. CBMJev needs explicit cost regimes:
   cached all-head automatic predictions, truly lazy response computation,
   and paid external/gold-label acquisition simulation. Never conflate them.
2. [Official model jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
   distinguishes relative Choice ranking from absolute per-candidate Boolean
   judgments and advises keeping arithmetic/invariants in code. For CBMJev,
   candidate ranking and STOP should be separate but coordinated: a complete
   set Choice head ranks feasible next concepts; a calibrated absolute
   continuation-versus-STOP risk gate decides whether *any* is worth asking.
   This is a hypothesis, not a vendor-guaranteed improvement. Our current
   `structured_choice` puts STOP into a single relative softmax and trains on
   softmaxed realized errors; it cannot be called an independently calibrated
   value-of-information gate.
3. [Official confidence guidance](https://docs.typesafe.ai/confidence)
   says Choice/Score confidence is derived from their output distribution,
   not an independent probability that the chosen action improves the task.
   We must not threshold Choice max probability as if it were expected CE
   reduction. Evaluate calibration of *task loss after acquiring versus
   stopping* under selected histories; set thresholds on held-out data.
4. [Official jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
   notes irrelevant state content, indirection, and structural inconsistency
   failure modes. Keep the question state compact and typed; compute budget,
   candidate legality, and costs deterministically. Probe candidate-order,
   paraphrase, irrelevant-description and menu-addition sensitivity. A model
   that merely benefits from longer prompts or leaks answer statistics through
   names does not demonstrate semantic transfer.
5. The [Awesome-Jev directory](https://github.com/valentynkit/awesome-jev-typesafe)
   notes the hosted Jev input is text/JSON, not direct image perception.
   A visual CBM therefore requires a separate image-to-concept responder.
   NanoJev is an independent open replica, not the unpublished official Jev
   architecture; the paper must retain that distinction.

## Consequence for the next method version

Build the smallest matched test, not an all-at-once new stack:

`observed typed concepts -> complete candidate-set ranker -> absolute
 continuation/STOP gate -> responder -> observed typed concepts`.

Run candidate-independent MLP, candidate-set Choice, and Choice + absolute
gate with the **same** responder, frozen task head, training histories,
budget, candidate menu, and supervision. Include `STOP`-within-Choice as the
current comparator. A gate is valuable only if it improves a real task-cost
frontier or selected-set reliability; threshold behavior alone is not a
contribution. Because an official Jev request may batch same-state questions,
measure any multi-request sequential overhead rather than assume speed.

The four-seed exploratory CUB two-query result in
`results/main/CUB_CONDITIONAL_SECOND_QUERY_20260923.md` suggests some
conditional CE signal after the first query, but little robust accuracy
signal. It justifies a controlled gate/ranker diagnostic; it does not justify
claiming a JEV-specific advantage yet.
