# JEV ecosystem scan — 2026-09-23

Scope: quick freshness scan for positioning CBMJev relative to the emerging
Jev/System-One-model ecosystem.  This is not a full citation audit.

## Primary anchor

- TypeSafe AI announced Jev on 2026-09-15 as its first "System One Model":
  a decision-oriented model class for fast, structured, probabilistic outputs
  rather than free-text generation.
  Source: <https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- The official framing emphasizes typed decisions from state/question pairs,
  with RLCD (Reinforcement Learning for Calibrated Decisions) as the training
  story and claims of high efficiency on System-One tasks.  For CBMJev, these
  are motivation and interface analogies, not evidence that the official Jev
  model solves concept acquisition.

## Open reproduction / community anchor

- NanoJev is the concrete open repository supplied by the user:
  <https://github.com/TianyuCodings/NanoJev>.
- Its public README currently frames NanoJev as a small open replica of Jev
  principles: dynamic candidate choices, boolean decisions, ordered score
  outputs, complete distributions, zero output decoding, and persistent serving.
- The README reports game/navigation demos and a Qwen3-0.6B-style backbone, but
  these are not CBM experiments.  Use NanoJev as an interface and training
  inspiration, not as a plug-in proof of CBMJev performance.

## Rapidly emerging related artifacts

The scan found several very recent community and preprint artifacts around
Jev-style typed decisions:

- `Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents`,
  arXiv 2609.23986.
- `Open-Jev Judgments on CallScreenBench: Calibrated One-Pass Scam Screening
  with a Small Language Model`, arXiv 2609.23959.
- `Calibrated Decisions at Scale: Converting Police Crash Narratives into
  Probabilistic Crash Variables with a System One Model (Jev)`, arXiv 2609.24052.
- Multiple community explainers and comparison pages appeared within days of
  the launch.  These are useful for zeitgeist/terminology but should not be used
  as authoritative technical claims without tracing back to code, docs, or
  papers.

## Implication for CBMJev positioning

CBMJev should not claim "first sequential CBM" or "first adaptive feature
acquisition."  The defensible angle is narrower and cleaner:

1. Import the JEV-style interface idea into CBMs: a state plus a bounded question
   over dynamic candidates returns calibrated typed decision values.
2. Apply it to concept acquisition, where the candidates are concept groups,
   `STOP`, or batch actions, and the state is acquired concept evidence.
3. Test whether value-style typed decisions outperform fixed/static/random
   concept schedules under matched responders and declared costs.
4. Explain boundary cases where dynamic choice does not help, especially under
   cheap shared encoders, weak responders, or noisy/imbalanced concepts.

## Wording guardrails

- Say "JEV-style typed decision interface" or "NanoJev-inspired dynamic
  candidate scoring" unless the implementation actually calls the official Jev
  API or faithfully reproduces NanoJev internals.
- Do not claim zero-shot / label-free training.  CBMJev uses public concept
  labels and automatic concept responders.
- Do not claim official Jev speedups or RLCD guarantees.  Any speed/cost result
  must come from CBMJev's own live profiling.
- Do not cite community SEO/explainer sites for core technical novelty when a
  primary source, code repo, or paper is available.

## Open follow-up before final submission

- Identify whether TypeSafe has released official technical docs/specs beyond
  the launch blog and API docs.
- Decide whether to cite NanoJev as software only, or whether a citable preprint
  appears.
- Add the three recent arXiv JEV-style papers to the related-work matrix after
  verifying their methods and relevance.
- Re-run this scan before arXiv/submission because the Jev ecosystem is moving
  daily.

## 2026-09-24 primary-source follow-up

- [TypeSafe's API schema](https://api.typesafe.ai/docs) exposes typed Choice,
  Noul, and Score questions. The [launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  describes a state/questions interface and RLCD, but does not publish a
  reproducible RLCD objective or a CBM/concept-acquisition experiment. The
  company's [workflow examples](https://evals.typesafe.ai/) combine narrow
  judgments with code-defined conditional routing; this supports a workflow
  analogy, not an algorithmic novelty or accuracy claim for CBMJev. The launch
  article also says its higher-cardinality Choice path first scores options
  independently and then makes an explicit choice. That public detail does
  not establish that CBMJev's learned cross-candidate attention head matches
  the proprietary Jev architecture; it is a distinct empirical ablation.
- NanoJev now documents an [RLCD-inspired sampled proper-reward comparison](https://github.com/TianyuCodings/NanoJev/blob/main/docs/RLCD_EXPERIMENT.md).
  Its authors explicitly distinguish their independent objective from the
  undisclosed official recipe and report no superiority over direct
  probability-loss controls. CBMJev must not cite this as a recovered Jev
  training algorithm or use "RLCD reproduction" for its own Choice head.
- The paper's CBM nearest-neighbor paragraph now cites primary AAAI,
  NeurIPS, TMLR, and arXiv sources for interactive intervention,
  autoregressive prediction, input-dependent sparse gating, and nested
  budget order. This closes a bibliography gap, not the unmeasured JEV gain.
