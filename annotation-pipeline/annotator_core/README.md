# annotator_core — the shared annotation stages

Screenshots in (`<data>/<App>/*.png`), MedSPOT-format click-grounding annotations out. This package holds the
five-stage pipeline (propose → describe → ground → verify → refine), selection and tiering, the exporter, the QC
overlays, the keyboard grading tool (`tools/review_server.py`) and the export checker (`hpc/verify_annotations.py`).
The model calls go through a provider-neutral `Generator` seam; `../openai_pipeline` is the driver that was used to
build the released dataset.

## Tests

    python -m pytest -q                    # no GPU, no network
    python -m annotator --help

Dependencies: Pillow, httpx, pytest (Python 3.11+).

## Coordinate spaces and provider seams

A model box is read in the space of *the image as sent*: `thousand` (Qwen, 0-1000 of that image, the default
everywhere) or `pixels` (OpenAI's convention). `geometry.CoordSpace(window, shown_size, mode)` maps either back to
frame pixels; `RunConfig.coord_mode` picks the mode, `provenance.coord_mode` records it, and `reset --parse-only`
rebuilds the same space. Record fields named `bbox_1000` hold the model-space box whatever the mode. Every `Call`
carries `kind` (propose | describe | ground | verify | refine_ground | probe) so another provider's client can pick an
output schema and effort per stage, `Reply.usage` carries its token accounting, and `RunConfig.generators` routes a
kind to a different generator.
