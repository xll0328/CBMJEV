# CEBaB adaptive-acquisition controls

This code release includes additional, **exploratory validation** controls for
the four-concept CEBaB setting. These scripts require a locally prepared
response cache and a compatible frozen model bundle. Neither data, model
weights, per-example traces, nor the manuscript are distributed here.

The controls ask whether adaptive concept order or an acquisition-value
estimator adds anything after stronger static and stopping alternatives are
fitted. They do not establish a new method's superiority, a locked-test result,
or a deployment speedup. All scripts reject or avoid test evaluation.

| Script | What it checks |
| --- | --- |
| `evaluate_cebab_ordered_empirical_dp.py` | Dynamic empirical DP versus all 24 fixed orders, under the same frozen head, terminal 0/1 loss, and declared query cost. The fixed order is selected on `policy_fit`. |
| `summarize_cebab_e0_pair_bootstrap.py` | Paired family-cluster descriptive uncertainty for the fixed-versus-dynamic STOP comparison from an existing E0 factorial run. It is not a seed-level interval or an independence certificate. |
| `analyze_cebab_repair_damage.py` | Offline repair/damage opportunities when another automatic concept response is revealed. Gold labels and hidden responses are used only for diagnosis, not for inference-time action selection. |
| `evaluate_cebab_e1_controls.py` | Frozen-head signed-value, uncertainty-only gate, and an entropy-cap *sensitivity*. The sensitivity is not a trained LAVOIR reproduction. |
| `train_evaluate_cebab_dime_adapter.py` | A concept-only, frozen-head DIME-style value adapter. It does not reproduce DIME's joint/on-policy training. |
| `train_evaluate_cebab_lavoir_adapter.py` | Equal-capacity uncapped and Gini-capped concept-only LAVOIR-style adapters trained on realized gold-probability gain. It does not reproduce LAVOIR's text encoder, answer simulator, or joint training. |

The DIME and LAVOIR names identify the methodological controls being adapted,
not affiliations or exact reproductions. See the [DIME paper](https://arxiv.org/abs/2306.03301)
and [LAVOIR preprint](https://arxiv.org/abs/2609.30706) for their original
objectives and implementations.

## Inputs and invocation

Each model-based script takes `--models`, `--cache`, and a **new or empty**
`--out` directory. Use the same seed's frozen bundle and response cache for
paired comparisons. Obtain and prepare CEBaB under its original conditions;
follow [data adapters](DATA_ADAPTERS.md) and the repository's CEBaB run setup.
Check each entry point's `--help` before running. For example:

```bash
python scripts/evaluate_cebab_ordered_empirical_dp.py \
  --models /path/to/seed40/models \
  --cache /path/to/seed40/cache \
  --out /path/to/new-output/e0-dp

python scripts/evaluate_cebab_e1_controls.py \
  --models /path/to/seed40/models \
  --cache /path/to/seed40/cache \
  --out /path/to/new-output/e1-controls

python scripts/train_evaluate_cebab_dime_adapter.py \
  --models /path/to/seed40/models \
  --cache /path/to/seed40/cache \
  --out /path/to/new-output/dime-adapter

python scripts/train_evaluate_cebab_lavoir_adapter.py \
  --models /path/to/seed40/models \
  --cache /path/to/seed40/cache \
  --out /path/to/new-output/lavoir-adapter
```

For the bootstrap helper, pass the output of the separate E0 order/STOP
factorial run with `--e0` and a fresh `--out`; it expects that run's
`report.json` and `validation_traces.jsonl`. The adapter scripts save reports
and local weights/traces in their output directories: **keep those outputs out
of public commits**. `--limit` and shortened epochs are pilot-only; they
must not be represented as the full validation result. The external objective
is `J_lambda = validation error + lambda * (mean queries / 4)` for the
predeclared lambda grid. Internal thresholds or penalties are selected using
`policy_fit` only. Response provenance and sample roles must be inspected
before interpreting a run; these utilities do not certify that independent
families or a blind test set exist.

## Minimal checks

```bash
python -m unittest discover -s tests_cbmjev -p 'test_cebab_*adapter.py' -q
python -m unittest discover -s tests_cbmjev -p 'test_cebab_ordered_empirical_dp.py' -q
python -m unittest discover -s tests_cbmjev -p 'test_cebab_e1_controls.py' -q
python -m unittest discover -s tests_cbmjev -p 'test_cebab_repair_damage.py' -q
python -m unittest discover -s tests_cbmjev -p 'test_cebab_e0_pair_bootstrap.py' -q
```

The tests use synthetic fixtures. Passing them is not evidence of a real-data
improvement. For trace metrics, sample/family estimands, and latency caveats,
see the [evaluation protocol](EVALUATION.md).
