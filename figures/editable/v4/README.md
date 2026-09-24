# Revised CBMJev framework

Open `CBMJev_Figures.pptx`: slide 1 is the redesigned framework; slides 2–5
retain the existing empirical and analytical figure content. Text, panels,
model glyphs, masks, connectors, plot curves and error bars are editable native
PowerPoint objects. Plots use vector marks rather than Excel chart workbooks.

Figure 1 now prioritizes the method: measurement → acquired evidence → legal
candidate actions and action scoring → acquire again or STOP → task prediction.
The training-target strip is explicitly the CUB cross-fitting route. The
observed mask and candidate masks illustrate legal action sets; they are not
empirical trajectories or additional experimental results.

The revision was informed by a visual review of top-conference figure examples.
Third-party gallery images and reference assets are not included in this
repository; none is embedded in this deck.

The manuscript draft used `v4/measurement_boundary_editable.pdf` for Figure 1,
with a method-oriented caption. Other figures still use their v2 PDFs. Arial is
explicitly set and embedded. Each slide uses exactly three font sizes. PDF
exports contain no raster images; previews are for inspection only.

## Regenerate

The framework layout is in `scripts/figures/framework_layout_v3.mjs`. Its name
refers to the layout revision, while v4 is the finalized render revision.
The deck builder is `scripts/figures/build_editable_paper_figures.mjs` and the
PDF crop/verification script is `scripts/figures/export_editable_figure_pdfs.py`.

Use a fresh output version, for example `v5`, with the supplied presentation
runtime configured via `RUNTIME_NODE_MODULES`, `PRESENTATION_SKILL_DIR` and
`RUNTIME_PYTHON`. Run the Node builder, wait for successful completion, and
export with the bundled LibreOffice. Set `FONTCONFIG_FILE` to that version's
generated `.build/v5/fonts.conf` before export. Then run the PDF verification
script against the generated output directory.

Source paths and plotted data are retained in `figure_manifest.json`.
`figure_validation.json` verifies native slide objects, explicit font sizes,
embedded Arial, unclipped PDF text and vector-only content. The finalized PPTX
has been opened/exported in bundled LibreOffice; Microsoft PowerPoint desktop
is not installed on this Mac.
