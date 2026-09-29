# CBMJEV reopened diagnostics (development evidence)

This page documents a bounded research diagnostic, **not** a positive-method result or a locked-test evaluation. The code and aggregated numbers are provided so that the observed limits of the current implementation are inspectable. Dataset files, model weights, sample-level responses, and the manuscript are not distributed here.

## Questions and results

| Question | Design | Observed development result | Supported conclusion |
| --- | --- | --- | --- |
| A. Is the task predictor too weak to reveal an adaptive-order gain? | CEBaB, three seeds, same hard responses, original head vs all-16-mask same-capacity MLP vs shrunk discrete head; 24 fitted orders with STOP vs full-depth empirical DP at six costs | At cost 0.03, MLP fixed-minus-dynamic `J` was `+0.005862 / +0.000703 / +0.003072`. Full-concept MLP accuracy was `.5018 / .5170 / .5252`, not consistently above the original head. | This predictor control does not establish a stable, substantial ranking gain. It is not a Bayes upper bound. |
| B. Does a larger concept menu yield useful conditional branching? | Official ASAP 18-aspect gold-concept simulation; three seeds, one partial-input task head, fitted fixed order with STOP, DIME Eq. 3 frozen-head adaptation, random and all-query controls | At budget 8, adaptive gained 1.38 / 0.14 / 0.02 accuracy points over the initial fitted order; a bounded stronger fixed-set follow-up reduced adaptive-minus-fixed `J` to `-.001619 / +.001619 / -.002834`, with all three paired family intervals crossing zero. At zero acquisition cost, all-18 outperformed adaptive in every seed. | Conditional action branching exists, but a robust task-risk/cost gain over strong fixed/all-query controls was not established. Gold concepts are not free deployable measurements. |
| C. Can a second typed source usefully verify concepts? | CEBaB original responder plus pinned Qwen3-0.6B hard-answer source; head-fit channel learning, policy-fit smoothing selection, held-out development fusion; actual singleton calls | On 2,576 labeled validation concepts, original/new source hard correctness was 2,236/637. The new source corrected 86 original errors but damaged 1,685 original-correct answers. Joint fusion changed concept NLL from `.479214` to `.478137`, but five-class task error remained `.490035`. | This particular backend/prompt does not support an actionable hard-only verification benefit. It does not rule out other backends or probability-enabled interfaces. |

All CEBaB validation results have been repeatedly inspected and are exploratory. ASAP official dev was used for development; no official test label was used to fit, select, or evaluate the reported methods. The stronger ASAP fixed-set comparison is explicitly **post-dev exploratory**, not independent confirmation. CEBaB seed 40/41/42 share the same family split, so seed variation is not equivalent to independent datasets. Paired intervals resample family/text components, not response masks as independent cases.

The CEBaB concept responder's existing cache is a mixed-role file. The A diagnostic's legacy loader structurally parses protected-role records before excluding them. Its training, tuning, empirical DP, and metrics use only `head_fit`, `policy_fit`, and `validation`; the accurate statement is *test not evaluated*, not *test file never opened*. The current second-source script filters roles before parsing sample outcomes. The public scripts reject protected roles in analysis.

## Data and reproduction entry points

CEBaB requires a compatible prepared dataset, seed-specific frozen model bundles and cached original responses. See [CEBaB adaptive controls](CEBAB_ADAPTIVE_CONTROLS.md) and each script's `--help`; do not mix seed-specific bundle/cache pairs. The A entry point is `scripts/evaluate_cebab_predictor_sufficiency.py --models ... --cache ... --prepared ... --out ...`. It trains at most eight predeclared MLP configurations inside `head_fit`. Its small direct-text reference is a fixed char-ngram TF-IDF linear classifier, not a strong text-model upper bound; complete gold annotations cover only 310 of 853 validation rows.

Download ASAP from the [official repository](https://github.com/Meituan-Dianping/asap) under its own license and point `--data` to its directory containing `train.csv`, `dev.csv`, and `test.csv`. `cbmjev/asap_data.py` validates the official 21-column CSV (ID, review, star, and 18 aspects), treats `-2` as the legitimate *not mentioned* category, and audits exact-text duplicate components. Run `scripts/evaluate_asap_gold_branch.py --data ... --out ... --seed 40` and likewise for seeds 41/42. The bounded post-dev fixed-set stress test is `scripts/evaluate_asap_stronger_fixed.py` with the same arguments. The DIME label denotes a frozen-head adaptation, **not** the original paper's joint/on-policy reproduction.

The C path requires the optional Hugging Face dependencies (`pip install -e '.[hf]'`) and an authorized local copy of pinned `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`. Use `scripts/evaluate_cebab_second_source.py --cache ... --prepared ... --model ... --out ... --roles head_fit,policy_fit,validation --batch-size 1 --device ...`, then `scripts/analyze_cebab_second_source.py --responses .../responses.jsonl --out .../fusion_report.json`. On the sampled source, batch 1 and batch 8 changed 1,065 of 19,344 hard answers; batch 8 must not be substituted for a sequential one-concept-at-a-time claim. A 100-review repeated singleton check reproduced all 100 answers. The measured singleton forward time was 509.23 s for 19,344 calls on one A100; this excludes model loading/tokenization and is not an end-to-end deployment comparison.

Focused protocol checks:

```bash
python -m unittest tests_cbmjev.test_cebab_predictor_sufficiency \
  tests_cbmjev.test_asap_gold_branch \
  tests_cbmjev.test_asap_stronger_fixed \
  tests_cbmjev.test_cebab_second_source -q
```

The present evidence does not justify claiming an ARR-ready new adaptive method, universal failure of adaptive CBMs, or a speedup from fewer logical concept questions.
