# Selective concept verification: development pipeline

**Status: implementation and source preparation, not a confirmed method gain.**
This is a separate, prospective experiment from the historical acquisition
results. It does not replace or relabel their negative findings.

The new question is narrower: after a cheap shared encoder has predicted all
initial concepts, can a policy improve a decision by checking **one** concept
with another fallible source, or should it stop?

## Information boundary

The deployment path is:

```text
text -> cheap encoder -> all initial hard concepts
                            |
                     STOP or verify j
                            |
                 actual second-source answer j
                            |
                    concept-only fusion
                            |
                 common hard-concept task head
```

The selector receives only initial hard concepts. The task head receives only
fused hard concepts: no raw text, hidden embeddings, source identity, or query
mask. An offline environment owns complete answer caches and action targets;
these are not selector inputs. Negative as well as positive action gains are
retained.

## Components and comparisons

| Component | Implementation |
| --- | --- |
| Typed verification | Frozen Qwen3-4B full-candidate contextual features and a concept-supervised scalar readout |
| Shared strong reference | The same backbone encodes a text once, followed by all concept heads |
| Initial source for appraisal | Concept-supervised DistilRoBERTa, one encoding and native-category heads |
| Fusion and task head | Concept-only confusion fusion; nested group-OOF states; one common task head |
| Selectors | Direct signed-gain regression, a shared-marginal joint answer model, its own product-of-marginals ablation, and independently fitted factorization |
| Controls | STOP, every fixed concept, fixed concept plus STOP, all-concept fusion, an independently fitted full-strong CBM, and explicitly nondeployable hindsight |

The source is **Jev-inspired**, not an official Jev call. A contextual candidate
readout is not an LM sequence likelihood. Expected value of information is a
standard decision rule, not claimed here as a newly discovered formula.

Source fitting uses only source-role concept annotations. Head fitting and
policy fitting have disjoint groups. Fusion selection uses concept labels,
not task gains. Source quality is checked against a prewritten gate before any
conditional source upgrade; task outcomes must not choose a new backbone.

The shared-encoder reference matters: predicting fewer concepts does not by
itself save computation. Without a measured, cache-consistent cost profile,
the evaluator emits **only zero-cost development diagnostics**, with `J=null`.
Cache replay time is never a deployment speed measurement.

## Data and task meaning

- CEBaB retains its family roles. Historical validation is seen development
  data; this release does not certify its official test as historically blind.
- The crowd-enVENT 2023 adapter preserves 21 author-rated ordinal appraisals
  and 13 native **elicited/prompted emotion categories**. The task is not
  clinical diagnosis, and its target is not an independent post-event rating.
  Inputs exclude prompting labels and demographic metadata. A fixed,
  label-independent native-emotion-term mask is applied to all texts.
- Appraisal roles separate connected components of author IDs and normalized
  original/final texts. The prospective split uses seed 20260930 and preserves
  a confirmation role that development tools refuse by default.

Obtain the corpus from its [official author resource](https://www.romanklinger.de/data-sets/)
and cite the [original study](https://aclanthology.org/2023.cl-1.1/).
The release documents research use but does not supply a named corpus license.
This repository does not redistribute it or grant permission to sublicense it.
Dataset and model terms are separate from the code's MIT license.

## Entry points

Run from the repository root in an environment with the optional Hugging Face
dependencies. These commands show interfaces; they do not download data or
launch training:

```bash
python scripts/prepare_appraisal_data.py --help
python scripts/train_initial_verification_source.py --help
python scripts/train_verification_sources.py --help
python scripts/evaluate_concept_verification.py --help
```

The intended order is data preparation, source extraction/training, source
quality and consistency checks, then development evaluation. Model revisions
are pinned; models must already exist locally. Feature caches, checkpoint
manifests, predictions, and row-level audit files are private runtime artifacts
and must not be committed. In particular, metadata can contain sample IDs and
supervision provenance even when it contains no text.

Core tests use synthetic fixtures only:

```bash
python -m unittest \
  tests_cbmjev.test_appraisal_data \
  tests_cbmjev.test_initial_verifier \
  tests_cbmjev.test_typed_verifier \
  tests_cbmjev.test_verification \
  tests_cbmjev.test_verification_experiment \
  tests_cbmjev.test_verification_learning \
  tests_cbmjev.test_verification_inputs \
  tests_cbmjev.test_concept_verification_evaluator
```

Passing implementation tests is not evidence of task improvement. Independent
confirmation, full cost comparisons, semantic-damage analysis, and the
predeclared seeds remain necessary before making a research claim.
