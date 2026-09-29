# Quick start

Start here for the current release. Older experiment contracts record planned
and development work; they are not a claim that every experiment is complete.

## 1. Install

Python 3.9+ is declared by the package. The release checks were run with Python
3.9.6, PyTorch 2.8.0, NumPy 2.0.2, and Pillow 11.3.0 on CPU. For a fresh
environment, Python 3.11 is a convenient choice; choose a PyTorch build matching
your platform before running CUDA experiments.

```bash
git clone https://github.com/xll0328/CBMJEV.git
cd CBMJEV
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m cbmjev doctor --check-device cpu
```

Optional dependencies are separate:

```bash
python -m pip install -e '.[data]'    # CEBaB Parquet acquisition/audit helpers
python -m pip install -e '.[vision]'  # torchvision image backbones
python -m pip install -e '.[hf]'      # optional Hugging Face responders
python -m pip install -e '.[paper]'   # plots and PDF figure inspection
python -m pip install -e '.[dev]'     # pytest, if preferred to unittest
```

These commands install packages, not datasets or pretrained weights. Model
downloads require the relevant explicit CLI options. The paper extra does not
install LaTeX, PowerPoint/LibreOffice, or the historical presentation SDK.

## 2. Run a small CPU example

```bash
cbmjev_smoke_parent=$(mktemp -d)
python -m cbmjev smoke --out "$cbmjev_smoke_parent/run" --seed 17
```

This creates synthetic fixtures and fits small models locally. The output
directory must be new. It exercises preparation, automatic concept responses,
controller/head fitting, validation replay, live execution, and summary export;
it is **not** a reproduction of CUB/CEBaB results or deployment performance.
Inspect `run/tables/` and the run's recorded configuration before interpreting
any metric. No real dataset or GPU is required.

## 3. Check the code and analytical examples

```bash
python -m unittest discover -s tests_cbmjev -q
python tools/verify_bundle.py
python theory/check_theory_sanity.py
python theory/check_theory_extensions.py
```

The numerical theory checks test finite constructions and bound consistency;
they are not substitutes for the assumptions and proofs in the manuscript.
Some optional-backbone tests skip when their dependencies are unavailable.
The bundled-link checker validates references and structured files, not the
truth of experimental claims.

## 4. Work with real data

Read [data preparation](DATA_ADAPTERS.md), [code map](CODE_MAP.md), and
[evaluation protocol](EVALUATION.md). Use official-format data obtained under
its access terms, an explicit source revision, a fresh output directory, and
the chosen seed. Inspect `python -m cbmjev prepare --help` before conversion.

For CUB, start from the `configs/cub_crossfit_*seed60.json` configurations and
the cross-fit commands listed by `python -m cbmjev --help`. Prepared data,
fold-specific responders, response caches, and policy targets are generated
artifacts, not files bundled in this repository. Follow their recorded
dependency order and keep training roles disjoint. The release includes
selected aggregate results, not every original run directory or checkpoint;
re-running the paper's real-data experiments requires obtaining data and
training the corresponding systems.

Do not substitute fixture scores for real-data results, tune on the test set,
or interpret fewer revealed groups as lower measured latency. The current
paper reports development-validation comparisons, including negative results.

## 5. Read, build, or extend

- [Numerical theory checks](../theory/README.md)
- [Results and their scope](../results/README.md)
- [Editable figures and tool requirements](FIGURES.md)
- [Contributing guide](../CONTRIBUTING.md)

The full manuscript PDF and LaTeX source are deliberately not distributed in
this code release. The authors will add a paper link after publication.
