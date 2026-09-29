# Editing and reproducing figures

The [v5 figure bundle](../figures/editable/v5/) contains native editable PPTX,
vector PDFs, and PNG previews. Open `CBMJev_Figures.pptx` in a compatible
presentation application to edit labels, arrows, and shapes. PNG previews are
not the editable source. Use exported vector PDF when inserting a figure into
LaTeX, and check font embedding and crop boundaries after export.

The manuscript itself is not part of this code release. Older editable-bundle
previews can differ in framing from the current manuscript; preserve the
actual values, source captions, and validation-only scope when exporting.

## Two different reproduction paths

1. **Read or edit released figures:** use the included PPTX/PDF directly. No
   presentation SDK is needed to read the PDF or edit the existing PPTX.
2. **Rerun the historical JavaScript builder:**
   `scripts/figures/build_editable_paper_figures.mjs` requires an external
   `@oai/artifact-tool` runtime, presentation-skill utilities, and a compatible
   headless LibreOffice/font environment. These are not installed by pip and
   are not distributed here. Its original paths target the lab workspace's
   `paper/cvpr2027/` tree, which is not distributed in this code release.
   That builder is retained as provenance, not advertised as a one-command
   standalone public reproduction. Do not replace missing runtimes with
   fabricated artifacts or silently change the numerical data.

Python plots and PDF inspection use `pip install -e '.[paper]'`.
`scripts/figures/export_editable_figure_pdfs.py --help` describes the crop and
validation interface; this extra includes PyMuPDF and pikepdf, not LibreOffice.

Current mathematical results and terminology are authoritative in the paper.
Use the existing palette, Arial figure font, and three-size hierarchy; do not
invent notation or slogans when editing.
