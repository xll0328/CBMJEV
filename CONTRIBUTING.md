# Contributing

Use a focused branch and describe the behavior your change affects. Keep
dataset access, prepared artifacts, checkpoints, API keys, and machine paths
out of commits. Never upload medical/personal data or credentials in issues.

Before submitting a change:

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests_cbmjev -q
python tools/verify_bundle.py
git diff --check
```

For changes to acquisition or evaluation, add focused tests for legal actions,
group budgets, information isolation, split roles, reproducible seeds, and the
distinction between replay cost and measured deployment cost. Do not remove
provenance/integrity checks to make incompatible artifacts load.

For numerical claims, retain a machine-readable source and distinguish
fixture tests, exploratory validation, and locked-test evaluation. A smoke
run is not paper evidence. For paper figures, preserve editable originals,
use vector PDF exports, and match manuscript notation.

Report bugs using the issue template with package versions and a minimal
synthetic reproducer. For sensitive reports, do not post secrets publicly;
use GitHub's private vulnerability reporting if enabled or contact a maintainer
privately through an established channel.
