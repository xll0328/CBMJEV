# CBMJEV

CBMJEV is a research prototype exploring sequential, adaptive concept acquisition for concept-based prediction. Instead of predicting every concept in one pass, a policy can choose which concept group to query next, update its state, and stop under a query budget.

## Current status

This repository is a curated research-code snapshot, not a finished benchmark or submission release. The available results are development/validation evidence only. They do not establish test-set superiority, generalization across datasets, or real-world latency/speedup. The current CUB analyses are mixed and should be read with their accompanying notes and protocols. No paper author list is finalized in this snapshot.

The repository intentionally excludes private server configuration, datasets, model checkpoints, and unpublished manuscript sources/PDFs. Download and prepare each dataset under its own license and official access rules.

## Layout

- `cbmjev/`: core method and evaluation package.
- `configs/`: portable experiment configurations.
- `scripts/`: training, evaluation, analysis, and figure-building entry points.
- `tests_cbmjev/`: unit and protocol tests.
- `docs/`, `research/`, `theory/`: method, experiment, research, and theory notes.
- `results/`: selected, explicitly labeled development/validation snapshots.
- `figures/editable/v4/`: editable PowerPoint figure source and PDF previews.

## Quick start

Python 3.9+ is required by the package metadata. Install the project and run the local tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests_cbmjev -v
```

Most experiments require dataset-specific preparation and a compatible PyTorch/CUDA installation. See `docs/DATA_ADAPTERS.md`, `docs/EXPERIMENTS.md`, and `docs/EVALUATION.md` before launching a run. Configurations may need local path overrides; do not commit machine-specific paths or credentials.

## Reproducibility and claims

Experiment notes distinguish exploratory, development/validation, and paper-evidence status. Preserve split boundaries and use training/validation data for all method and hyperparameter choices; reserve test sets for a predeclared final evaluation. Do not treat development snapshots as benchmark claims. See `docs/PROJECT_SCOPE_FREEZE_20260924.md` and `docs/RELEASE_CHECKLIST.md` for the current scope and release gates.

## License

No software license has been declared yet. Until a license is added, all rights are reserved; public visibility does not grant permission to reuse, modify, or redistribute this code.
