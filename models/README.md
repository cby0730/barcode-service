# Models

Model weights are not part of this repository and are git-ignored; only this file is tracked.
Place the ONNX files here (or point `BARCODE_MODELS_DIR` elsewhere). Docker Compose mounts this
directory read-only at `/app/models`.

Default file names:

- `yolov8s-barcode.onnx` (detector, `BARCODE_DETECTOR_MODEL`)
- `pan-x2.onnx` (x2 super-resolution, `BARCODE_UPSCALER_MODEL`)

How to get them:

- `scripts/download_models.py` fetches the recommended pretrained weights (pinned revisions,
  sha256-checked) into `models/pretrained/`.
- `scripts/export_models.py` converts them to the ONNX files above.
- To use your own models, see [docs/custom-models.md](../docs/custom-models.md) for the tensor contracts.

Each model carries its own license; check it before use or redistribution.
