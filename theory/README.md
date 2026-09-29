# Theory and numerical checks

The complete manuscript and its LaTeX source are not distributed in this code
release. This directory includes numerical checks and earlier research notes,
such as [THEORY_APPENDIX.md](THEORY_APPENDIX.md). Historical notes preserve the
development process and do not override the current manuscript or establish
that every proposed result was used.

Two lightweight scripts use only the Python standard library:

```bash
python theory/check_theory_sanity.py
python theory/check_theory_extensions.py
```

They contain 8 and 15 checks respectively. They test finite constructions,
counterexamples, and numerical consistency of the bounds; successful checks
do not establish an unproved theorem or verify real-data assumptions.
