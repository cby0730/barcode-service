"""Offline export of pretrained weights to the ONNX files the service loads.

Needs torch, ultralytics, super-image and onnx (the ``export`` dependency group), which are
never installed in the service image. Run it with an interpreter that has them, e.g.
``uv run --group export python scripts/export_models.py ...``.

Output: <models-dir>/yolov8s-barcode.onnx and <models-dir>/pan-x2.onnx
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export_detector(weights: Path, out: Path, imgsz: int, opset: int) -> None:
    import tempfile

    from ultralytics import YOLO

    out.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics writes the ONNX next to the .pt, so export from a scratch copy; this keeps
    # the source directory untouched (it may be read-only or shared).
    with tempfile.TemporaryDirectory(dir=out.parent) as work:
        local = Path(work) / "detector.pt"
        shutil.copyfile(weights, local)
        # Static batch of 1: the service runs one letterboxed image at a time.
        exported = Path(
            YOLO(str(local)).export(
                format="onnx", imgsz=imgsz, opset=opset, dynamic=False, simplify=False, batch=1
            )
        )
        shutil.move(exported, out)


def export_upscaler(pan_dir: Path, out: Path, scale: int, opset: int) -> None:
    import torch
    from super_image import PanModel

    # Expects config.json and pytorch_model_{scale}x.pt, the layout super-image reads.
    model = PanModel.from_pretrained(str(pan_dir), scale=scale).eval()
    dummy = torch.rand(1, 3, 64, 64)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["output"],
        # Height and width must stay dynamic: the upscaler runs on arbitrary crops.
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "out_height", 3: "out_width"},
        },
        opset_version=opset,
        dynamo=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--yolo-weights", type=Path, required=True, help="YOLOv8 .pt file")
    parser.add_argument("--pan-dir", type=Path, required=True, help="dir with PAN config+weights")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args(argv)

    detector_out = args.models_dir / "yolov8s-barcode.onnx"
    upscaler_out = args.models_dir / f"pan-x{args.scale}.onnx"
    export_detector(args.yolo_weights, detector_out, args.imgsz, args.opset)
    print(f"wrote {detector_out}")
    export_upscaler(args.pan_dir, upscaler_out, args.scale, args.opset)
    print(f"wrote {upscaler_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
