# License scope and third-party notices

The root [MIT License](LICENSE) covers the original CBMJEV software and its
associated usage documentation. It does not grant a new license to datasets,
pretrained weights, third-party software, or the manuscript and figure assets.

## Manuscript and figures

The complete article text, compiled manuscript, and LaTeX source are not part
of this code release. The released figure assets under `figures/` are not
relicensed under the software MIT license. The authors will
choose the manuscript's distribution license when submitting to arXiv.
Original figure-generation and numerical-check Python/JavaScript code remains
under the repository's MIT license. Do not infer an arXiv license before one
has actually been selected.

No complete manuscript or third-party conference-template bundle is included
in this repository release. Existing research-note TeX files are not the full
paper source and should not be mistaken for a submission package.

## Data, backbones, and optional tools

- CUB-200-2011, CEBaB, Derm7pt, and ISIC-derived data are **not distributed**.
  Obtain permitted data separately, follow the original terms, and do not
  upload patient information, raw examples, access credentials, or images here.
- PyTorch, torchvision, Hugging Face packages, downloaded model weights, and
  TeX/LibreOffice distributions retain their own licenses. Installing an extra
  does not change those terms or grant access to gated assets.
- The historical editable-figure builder uses an external presentation SDK
  and runtime. That runtime is not bundled or licensed by this repository;
  see the [figure reproduction guide](docs/FIGURES.md). The released PPTX files
  can be edited independently with a compatible presentation application.
- The project is JEV-inspired, not an official JEV or NanoJev product. No
  upstream endorsement or independently verified benefit is implied.
