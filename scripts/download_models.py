"""Download the recommended pretrained weights into ``models/`` (pinned revisions, sha256-checked).

Weights are NOT part of this repository. Convert them to ONNX afterwards with
``scripts/export_models.py``.

Usage: uv run --group models python scripts/download_models.py [--models-dir models]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    repo: str
    revision: str  # full commit sha, so later upstream changes cannot alter what we fetch
    filename: str
    sha256: str
    dest: str  # relative to the models dir
    license: str


ASSETS = [
    Asset(
        repo="Piero2411/YOLOV8s-Barcode-Detection",
        revision="59098fa84b8258fdffdc632e36482dcbf78fc5f4",
        filename="YOLOV8s_Barcode_Detection.pt",
        sha256="316ded312281da5d4de06c36c66fdc682bd1c2052689008237baf22eb8e4f5ed",
        dest="pretrained/yolov8s-barcode.pt",
        license="AGPL-3.0",
    ),
    Asset(
        repo="eugenesiow/pan",
        revision="d42ed37983a3f7307737948c0dec05b60c3ea137",
        filename="pytorch_model_2x.pt",
        sha256="8fc18d9e39da7734661b8db9f1b024f874d5ba74ef39b5963f9f534443245beb",
        dest="pretrained/pan/pytorch_model_2x.pt",
        license="Apache-2.0",
    ),
    Asset(
        repo="eugenesiow/pan",
        revision="d42ed37983a3f7307737948c0dec05b60c3ea137",
        filename="config.json",
        sha256="272f261f7f08f3d382ee340f9d1d1e1e739edfabc07466962ec803ee33e0097d",
        dest="pretrained/pan/config.json",
        license="Apache-2.0",
    ),
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(asset: Asset, models_dir: Path, timeout: float) -> None:
    from huggingface_hub import hf_hub_download

    dest = models_dir / asset.dest
    if dest.is_file() and sha256_of(dest) == asset.sha256:
        print(f"ok (cached)  {dest}")
        return

    cached = Path(
        hf_hub_download(
            asset.repo,
            asset.filename,
            revision=asset.revision,
            cache_dir=models_dir / ".hf-cache",
            etag_timeout=timeout,
        )
    )
    actual = sha256_of(cached)
    if actual != asset.sha256:
        raise ValueError(f"sha256 mismatch for {asset.repo}/{asset.filename}: got {actual}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    print(f"ok           {dest}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--timeout", type=float, default=30.0, help="network timeout in seconds")
    args = parser.parse_args(argv)

    print("Downloading third-party weights; they are not part of this repository.")
    for repo, lic in sorted({(a.repo, a.license) for a in ASSETS}):
        print(f"  {repo}: license {lic}")
    print("Check each license before using or redistributing the weights or derived ONNX files.")

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("error: huggingface_hub is missing; run with `uv run --group models ...`.")
        return 1

    for asset in ASSETS:
        try:
            fetch(asset, args.models_dir, args.timeout)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - surface any network/hub failure readably
            print(
                f"error: could not download {asset.repo}/{asset.filename} "
                f"({type(exc).__name__}: {exc}).\n"
                "Check network/proxy settings, or fetch the file manually and place it at "
                f"{args.models_dir / asset.dest}.",
                file=sys.stderr,
            )
            return 1
    print(f"Done. Next: python scripts/export_models.py --models-dir {args.models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
