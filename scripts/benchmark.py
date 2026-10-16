"""Strategy and backend benchmark on a deterministic synthetic dataset.

Barcodes are rendered with zxing-cpp, placed on a canvas and degraded with fixed seeds.
Every candidate strategy is run on every image with the real models, and the script reports
hit rate and latency per strategy and per backend, plus the hit rate each strategy adds over
the ones before it in the default order.

The data is synthetic: it shows relative differences between strategies and backends, not
hit rates on real photographs.

    uv run --no-sync python scripts/benchmark.py --per-cell 6 --out tmp/bench.md
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from barcode_service.pipeline import DEFAULT_STRATEGIES, Strategy

CANVAS_W, CANVAS_H = 640, 480
KINDS = ("ean13", "code128", "qr", "multi")
_FORMATS = {
    "ean13": zxingcpp.BarcodeFormat.EAN13,
    "code128": zxingcpp.BarcodeFormat.Code128,
    "qr": zxingcpp.BarcodeFormat.QRCode,
}
_ALNUM = np.array(list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"))


# Every strategy worth measuring, independent of what the pipeline ships as default, so the
# default list can be trimmed without losing the evidence for the strategies that were dropped.
CANDIDATES = (
    Strategy("clahe", ("clahe",)),
    Strategy("sauvola", ("sauvola",)),
    Strategy("detect", ("detect",)),
    Strategy("detect+clahe", ("detect", "clahe")),
    Strategy("detect+sauvola", ("detect", "sauvola")),
    Strategy("upscale", ("upscale",)),
    Strategy("detect+upscale", ("detect", "upscale")),
    Strategy("detect+upscale+clahe", ("detect", "upscale", "clahe")),
    Strategy("detect+upscale+sauvola", ("detect", "upscale", "sauvola")),
)

# The order the pipeline shipped with before it was trimmed. The marginal-gain column is
# computed over this order so the table keeps showing why the dropped strategies were dropped.
SEED_ORDER = (
    "clahe",
    "sauvola",
    "detect",
    "detect+clahe",
    "detect+sauvola",
    "detect+upscale",
    "detect+upscale+clahe",
    "detect+upscale+sauvola",
)

# ---------------------------------------------------------------- synthetic data


def _ean13_text(rng: np.random.Generator) -> str:
    digits = [int(d) for d in rng.integers(0, 10, 12)]
    check = (10 - sum(d * (3 if i % 2 else 1) for i, d in enumerate(digits)) % 10) % 10
    return "".join(map(str, [*digits, check]))


def _text_for(kind: str, rng: np.random.Generator) -> str:
    if kind == "ean13":
        return _ean13_text(rng)
    if kind == "code128":
        return "".join(rng.choice(_ALNUM, int(rng.integers(8, 13))))
    return f"https://example.com/item/{int(rng.integers(0, 10**6))}"


def make_symbol(kind: str, text: str, scale: int = 4) -> np.ndarray:
    """Render a barcode as a grayscale image (black on white, with quiet zone)."""
    barcode = zxingcpp.create_barcode(text, _FORMATS[kind])
    return np.array(zxingcpp.write_barcode_to_image(barcode, scale=scale, add_quiet_zones=True))


def make_canvas(symbols: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Place symbols at random positions on a light background; returns BGR."""
    bg = int(rng.integers(215, 256))
    canvas = np.full((CANVAS_H, CANVAS_W), bg, dtype=np.uint8)
    # One slot for a single symbol, a 2x2 grid for several (three symbols use three slots).
    if len(symbols) == 1:
        slots = [(0, 0, CANVAS_W, CANVAS_H)]
    else:
        sw, sh = CANVAS_W // 2, CANVAS_H // 2
        slots = [(0, 0, sw, sh), (sw, 0, sw, sh), (0, sh, sw, sh)]
    for sym, (sx, sy, sw, sh) in zip(symbols, slots, strict=False):
        h, w = sym.shape
        target_w = sw * float(rng.uniform(0.5, 0.85))
        scale = min(target_w / w, sh * 0.85 / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(sym, (nw, nh), interpolation=cv2.INTER_AREA)
        x = sx + int(rng.integers(0, max(1, sw - nw)))
        y = sy + int(rng.integers(0, max(1, sh - nh)))
        canvas[y : y + nh, x : x + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), float(rng.uniform(1.5, 2.5)))


def _downscale(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = float(rng.uniform(0.3, 0.45))
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return small


def _noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noisy = img.astype(np.float32) + rng.normal(0.0, float(rng.uniform(25, 45)), img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _rotate(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    angle = float(rng.uniform(10, 25)) * float(rng.choice([-1, 1]))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=(235, 235, 235))


def _low_contrast(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Squeeze the range into a narrow band around a random mid-grey; a little noise on top.
    factor = float(rng.uniform(0.08, 0.15))
    mid = float(rng.uniform(100, 160))
    out = (img.astype(np.float32) - 127.5) * factor + mid
    out += rng.normal(0.0, 1.5, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    quality = int(rng.integers(8, 20))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


DEGRADATIONS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "clean": lambda img, rng: img,
    "blur": _blur,
    "downscale": _downscale,
    "noise": _noise,
    "rotation": _rotate,
    "low_contrast": _low_contrast,
    "jpeg": _jpeg,
}


@dataclass(frozen=True)
class Sample:
    kind: str
    degradation: str
    image: np.ndarray
    expected: frozenset[str]


def build_dataset(per_cell: int, seed: int = 20240601) -> list[Sample]:
    """Deterministic dataset: every (kind, degradation) cell holds ``per_cell`` images."""
    samples: list[Sample] = []
    for ki, kind in enumerate(KINDS):
        for di, (deg_name, deg) in enumerate(DEGRADATIONS.items()):
            for i in range(per_cell):
                rng = np.random.default_rng([seed, ki, di, i])
                sub_kinds = ["ean13", "code128", "qr"] if kind == "multi" else [kind]
                texts = [_text_for(k, rng) for k in sub_kinds]
                symbols = [make_symbol(k, t) for k, t in zip(sub_kinds, texts, strict=True)]
                img = deg(make_canvas(symbols, rng), rng)
                samples.append(Sample(kind, deg_name, img, frozenset(texts)))
    return samples


# ---------------------------------------------------------------- measurement


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def _stats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values)
    return float(arr.mean()), float(np.percentile(arr, 95))


def _hit(found: list, expected: frozenset[str]) -> bool:
    """At least one expected text decoded; this matches the pipeline's first-hit-wins exit."""
    return bool(expected & {b.text for b in found})


def _all_found(found: list, expected: frozenset[str]) -> bool:
    return expected <= {b.text for b in found}


@dataclass
class BackendResult:
    backend: str
    names: list[str]  # candidate strategies, raw first
    hits: np.ndarray  # (n_samples, n_names) bool
    times: np.ndarray  # (n_samples, n_names) seconds
    e2e_hit: np.ndarray
    e2e_wrong: np.ndarray
    e2e_all: np.ndarray
    e2e_time: np.ndarray
    e2e_strategy: list[str]
    detector_ms: float
    upscaler_ms: float


def run_backend(backend: str, samples: list[Sample], models_dir: Path) -> BackendResult:
    from barcode_service.decoders import PyzbarDecoder, ZxingDecoder, decode_all
    from barcode_service.detector import OnnxDetector
    from barcode_service.pipeline import RAW_STRATEGY, Pipeline
    from barcode_service.preprocess import resize_max_side
    from barcode_service.runtime import create_session
    from barcode_service.upscaler import OnnxUpscaler

    detector = OnnxDetector(create_session(backend, models_dir / "yolov8s-barcode.onnx"))
    upscaler = OnnxUpscaler(create_session(backend, models_dir / "pan-x2.onnx"))
    decoders = [ZxingDecoder(), PyzbarDecoder()]

    candidates = list(CANDIDATES)
    pipe = Pipeline(decoders, detector, upscaler, strategies=candidates)
    default_pipe = Pipeline(decoders, detector, upscaler)
    names = [RAW_STRATEGY, *(s.name for s in candidates)]

    # Warm-up so first-call allocation and graph optimisation do not skew the first samples.
    for s in samples[:2]:
        pipe._apply(s.image, ("detect", "upscale"))

    n = len(samples)
    hits = np.zeros((n, len(names)), dtype=bool)
    times = np.zeros((n, len(names)))
    e2e_hit = np.zeros(n, dtype=bool)
    e2e_wrong = np.zeros(n, dtype=bool)
    e2e_all = np.zeros(n, dtype=bool)
    e2e_time = np.zeros(n)
    e2e_strategy: list[str] = []

    for i, s in enumerate(samples):
        t0 = time.perf_counter()
        found = decode_all(decoders, s.image)
        times[i, 0] = time.perf_counter() - t0
        hits[i, 0] = _hit(found, s.expected)

        base = resize_max_side(s.image, 800)
        for j, strategy in enumerate(candidates, start=1):
            t0 = time.perf_counter()
            # _apply is private on purpose: Pipeline.decode would stop at "raw" before
            # reaching the strategy, and a standalone measurement needs the strategy alone.
            found = decode_all(decoders, pipe._apply(base, strategy.steps))
            times[i, j] = time.perf_counter() - t0
            hits[i, j] = _hit(found, s.expected)

        t0 = time.perf_counter()
        result = default_pipe.decode(s.image)
        e2e_time[i] = time.perf_counter() - t0
        e2e_hit[i] = _hit(result, s.expected)
        e2e_all[i] = _all_found(result, s.expected)
        e2e_wrong[i] = any(b.text not in s.expected for b in result)
        e2e_strategy.append(result[0].strategy if result else "(none)")

    # Per-call model latency on a typical full-canvas input and on a typical detector crop.
    probe = samples[0].image
    crop = probe[: CANVAS_H // 2, : CANVAS_W // 2]
    detector_ms = _time_calls(lambda: detector.detect(probe))
    upscaler_ms = _time_calls(lambda: upscaler.upscale(crop))
    return BackendResult(
        backend,
        names,
        hits,
        times,
        e2e_hit,
        e2e_wrong,
        e2e_all,
        e2e_time,
        e2e_strategy,
        detector_ms,
        upscaler_ms,
    )


def _time_calls(fn: Callable[[], object], repeat: int = 20) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return _ms((time.perf_counter() - t0) / repeat)


# ---------------------------------------------------------------- reporting


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def strategy_table(res: BackendResult) -> str:
    """Per-strategy standalone hit rate and latency, plus marginal gain in seed order."""
    n = res.hits.shape[0]
    order = ["raw", *SEED_ORDER]
    covered = np.zeros(n, dtype=bool)
    marginal: dict[str, int] = {}
    cumulative: dict[str, float] = {}
    for name in order:
        col = res.hits[:, res.names.index(name)]
        marginal[name] = int((col & ~covered).sum())
        covered |= col
        cumulative[name] = float(covered.mean())

    rows = []
    for j, name in enumerate(res.names):
        mean, p95 = _stats(list(res.times[:, j] * 1000))
        in_default = name in order
        rows.append(
            [
                f"`{name}`",
                _pct(float(res.hits[:, j].mean())),
                f"{mean:.1f}",
                f"{p95:.1f}",
                f"+{marginal[name]} ({_pct(marginal[name] / n)})"
                if in_default
                else "not in seed order",
                _pct(cumulative[name]) if in_default else "-",
            ]
        )
    header = [
        "strategy",
        "standalone hit",
        "mean ms",
        "p95 ms",
        "marginal over previous",
        "cumulative hit",
    ]
    return _table(header, rows)


def _cumulative(res: BackendResult, names: list[str]) -> float:
    cols = [res.names.index(n) for n in ["raw", *names]]
    return float(res.hits[:, cols].any(axis=1).mean())


def greedy_order(res: BackendResult) -> list[tuple[str, int]]:
    """Order strategies by marginal gain, breaking ties by lower mean latency."""
    n = res.hits.shape[0]
    covered = res.hits[:, 0].copy()
    remaining = list(range(1, len(res.names)))
    out: list[tuple[str, int]] = []
    while remaining:
        gains = {j: int((res.hits[:, j] & ~covered).sum()) for j in remaining}
        best = min(remaining, key=lambda j: (-gains[j], res.times[:, j].mean()))
        out.append((res.names[best], gains[best]))
        covered |= res.hits[:, best]
        remaining.remove(best)
    assert covered.sum() <= n
    return out


def degradation_table(res: BackendResult, samples: list[Sample]) -> str:
    rows = []
    for deg in DEGRADATIONS:
        idx = [i for i, s in enumerate(samples) if s.degradation == deg]
        rows.append(
            [
                deg,
                str(len(idx)),
                _pct(float(res.hits[idx, 0].mean())),
                _pct(float(res.e2e_hit[idx].mean())),
            ]
        )
    return _table(["degradation", "images", "raw only", "default pipeline"], rows)


def kind_table(res: BackendResult, samples: list[Sample]) -> str:
    rows = []
    for kind in KINDS:
        idx = [i for i, s in enumerate(samples) if s.kind == kind]
        rows.append(
            [
                kind,
                str(len(idx)),
                _pct(float(res.hits[idx, 0].mean())),
                _pct(float(res.e2e_hit[idx].mean())),
                _pct(float(res.e2e_all[idx].mean())),
            ]
        )
    return _table(
        ["kind", "images", "raw only", "default pipeline", "default: all codes found"], rows
    )


def backend_summary(results: list[BackendResult]) -> str:
    rows = []
    for r in results:
        mean, p95 = _stats(list(r.e2e_time * 1000))
        rows.append(
            [
                f"`{r.backend}`",
                _pct(float(r.e2e_hit.mean())),
                _pct(float(r.e2e_wrong.mean())),
                f"{mean:.1f}",
                f"{p95:.1f}",
                f"{r.detector_ms:.1f}",
                f"{r.upscaler_ms:.1f}",
            ]
        )
    header = [
        "backend",
        "default pipeline hit",
        "wrong text",
        "mean ms",
        "p95 ms",
        "detector ms/call",
        "upscaler ms/call (320x240)",
    ]
    return _table(header, rows)


def environment() -> str:
    import os

    import onnxruntime as ort

    gpu = "n/a"
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.split("\n")
        gpu = f"{out[0].strip()} (x{len([g for g in out if g.strip()])})" if out[0] else "n/a"
    except (OSError, subprocess.SubprocessError):
        pass
    return (
        f"- CPU count: {os.cpu_count()}\n- GPU: {gpu}\n"
        f"- Python {platform.python_version()}, onnxruntime {ort.__version__} "
        f"(providers: {', '.join(ort.get_available_providers())})"
    )


def render(results: list[BackendResult], samples: list[Sample], skipped: dict[str, str]) -> str:
    cells = len(KINDS) * len(DEGRADATIONS)
    per_cell = len(samples) // cells
    parts = [
        "## Synthetic benchmark",
        f"Dataset: {len(samples)} images ({len(KINDS)} kinds x {len(DEGRADATIONS)} degradations "
        f"x {per_cell} per cell), {CANVAS_W}x{CANVAS_H} canvas, fixed seeds. "
        "Synthetic data only: relative differences, not real-photo hit rates. "
        "A hit means at least one expected text was decoded (the pipeline stops at the first hit); "
        "the by-kind table also gives the rate at which all codes of an image were found.",
        environment(),
    ]
    for backend, reason in skipped.items():
        parts.append(f"- `{backend}` skipped: {reason}")
    parts += ["", "### Backends (end to end, default strategies)", backend_summary(results)]
    for r in results:
        winners = Counter(r.e2e_strategy)
        parts += [
            "",
            f"### Strategies on `{r.backend}` (each run standalone; latency includes decoding)",
            strategy_table(r),
            "",
            "Cumulative hit of the shipped default (raw + "
            + ", ".join(f"`{s.name}`" for s in DEFAULT_STRATEGIES)
            + f"): {_pct(_cumulative(r, [s.name for s in DEFAULT_STRATEGIES]))}; "
            f"of all candidates together: {_pct(_cumulative(r, list(r.names[1:])))}",
            "",
            "Greedy order by marginal gain (images newly solved): "
            + ", ".join(f"`{n}` +{g}" for n, g in greedy_order(r)),
            "",
            "First-hit strategy in the default pipeline: "
            + ", ".join(f"`{k}` {v}" for k, v in winners.most_common()),
            "",
            f"By degradation on `{r.backend}`:",
            degradation_table(r, samples),
            "",
            f"By kind on `{r.backend}`:",
            kind_table(r, samples),
        ]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-cell", type=int, default=6, help="images per (kind, degradation)")
    parser.add_argument("--seed", type=int, default=20240601)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--backends",
        default="ort-cpu,openvino,ort-cuda",
        help="comma-separated; unavailable ones are skipped and reported",
    )
    parser.add_argument("--out", type=Path, help="also write the markdown here")
    args = parser.parse_args(argv)

    from barcode_service.types import BackendUnavailableError

    samples = build_dataset(args.per_cell, args.seed)
    print(f"built {len(samples)} images", file=sys.stderr)

    results: list[BackendResult] = []
    skipped: dict[str, str] = {}
    for backend in (b.strip() for b in args.backends.split(",") if b.strip()):
        print(f"running {backend} ...", file=sys.stderr)
        try:
            results.append(run_backend(backend, samples, args.models_dir))
        except BackendUnavailableError as exc:
            skipped[backend] = str(exc).split(";")[0]
    if not results:
        print("no backend could run: " + "; ".join(skipped.values()), file=sys.stderr)
        return 1

    report = render(results, samples, skipped)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
