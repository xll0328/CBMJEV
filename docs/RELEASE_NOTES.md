# Code-release preparation - 29 September 2026

This cleanup makes the existing research prototype easier to inspect and run.
It does not change model behavior, reported experimental values, or the scope
of the paper's scientific claims.

## Changes

- Reorganized the landing page around the actual framework, quick start,
  development evidence, editable figures, and citation metadata.
- Added the MIT software license, third-party/asset boundaries, contributor
  guidance, and issue templates. Full manuscript PDF and LaTeX source remain
  outside this public code release at the authors' request.
- Added a tested CPU smoke recipe and declared optional data/PDF dependencies.
- Repaired 32 old documentation links. Missing internal records are explicitly
  marked historical/non-distributed instead of silently inventing files.
- Added the 23 numerical theory checks and a CPU CI workflow. The CI definition
  uses read-only repository permissions and does not download datasets or run
  GPU experiments.

## Local verification

- Existing suite: 655 tests, 7 skipped, no failures (Python 3.9.6, PyTorch 2.8.0).
- Numerical theory checks: 8 original + 15 extension checks, all passed.
- Editable package installation passed in a fresh virtual environment sharing
  preinstalled numerical dependencies, after upgrading its old pip.
- Synthetic end-to-end CPU smoke passed with seed 17; no test split evaluated.
  This is a software check, not real-data or statistical-certification evidence.
- Markdown links/structured files, README local image targets, citation-format
  schema, YAML syntax, and whitespace checks passed.
- A targeted scan of tracked files, seven prior commits' reachable blobs,
  PPTX XML and PDF metadata found no common token/private-key signatures or
  local-machine paths. This is a bounded pattern scan, not a security guarantee.

The local checks above do not imply that GitHub-hosted CI has completed;
its status is shown separately on the Actions tab. Real-data re-training and
CUDA runs were not performed for this documentation/release cleanup.
