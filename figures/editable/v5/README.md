# CBMJev paper figures, readability revision

`CBMJev_Figures.pptx` contains all five native editable figures. Slide 1 raises
the smallest text from 8.5 pt to 9.5 pt, widens the action scorer, and shortens
the legal-action and acquisition-loop labels. The measurement, evidence,
legal-action, STOP and task-prediction paths retain their original semantics.
Slides 2–5 retain their data and layout. The existing v4 files remain intact.

Each slide uses three explicit Arial sizes. The framework uses 13, 10.5 and
9.5 pt. PDF exports embed Arial and contain no raster images. Masks illustrate
legal action sets; they are not experimental trajectories. The training strip
describes the public-label, cross-fitted-response route used for CUB.

## Source and reproduction

For the public release, start with the [figure reproduction guide](../../../docs/FIGURES.md).
The existing PPTX is independently editable; rebuilding it from JavaScript
requires the original external SDK/runtime and lab-workspace paths described
below, which are not bundled or installed by the Python paper extra.

The builder is `scripts/figures/build_editable_paper_figures.mjs`; version v5
and later selects `scripts/figures/framework_layout_v5.mjs`. Earlier versions
retain the v3 layout module. Set the presentation runtime variables documented
in the v4 README and build a fresh version. Export with the supplied headless
LibreOffice, setting `FONTCONFIG_FILE` to the generated version-specific
`fonts.conf`. Crop and verify with
`scripts/figures/export_editable_figure_pdfs.py`.

This revision used the presentation runtime's bundled headless LibreOffice.
Native PPTX finalization
used the bundled Python; PDF inspection used the existing system Python with
PyMuPDF and pikepdf. Microsoft PowerPoint desktop was not available.

`figure_manifest.json` records source hashes and raw plotted points.
`figure_validation.json` records editability, font sizes, embedded fonts,
clipping and vector-content checks. PNG previews are inspection copies only.
