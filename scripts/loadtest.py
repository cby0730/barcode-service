"""Load test for POST /v1/decode using httpx.AsyncClient.

    # fixed number of requests
    uv run --no-sync python scripts/loadtest.py http://127.0.0.1:8080 -c 8 -n 400
    # fixed duration, harder image (needs detector + upscaler to decode)
    uv run --no-sync python scripts/loadtest.py http://127.0.0.1:8080 -c 8 -d 20 --image hard
    # overload: far more concurrency than BARCODE_MAX_CONCURRENCY, expect 503 + Retry-After
    uv run --no-sync python scripts/loadtest.py http://127.0.0.1:8080 --overload

Reports RPS, latency percentiles, error rate and a status-code histogram.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import httpx
import numpy as np
import zxingcpp


def _render(text: str, fmt: zxingcpp.BarcodeFormat) -> np.ndarray:
    barcode = zxingcpp.create_barcode(text, fmt)
    return np.array(zxingcpp.write_barcode_to_image(barcode, scale=4, add_quiet_zones=True))


def make_payload(kind: str) -> bytes:
    """PNG bytes. 'easy' decodes on the raw path; 'hard' only decodes via detect + upscale."""
    gray = _render("5901234123457", zxingcpp.BarcodeFormat.EAN13)
    canvas = np.full((480, 640), 235, np.uint8)
    h, w = gray.shape
    if kind == "hard":
        # Shrunk to 30%: too small for the raw pass, but the x2 upscaler recovers it. The factor
        # was found empirically with the shipped models; change it if they change.
        canvas[100 : 100 + h, 10 : 10 + w] = gray
        canvas = cv2.resize(canvas, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    else:
        canvas[100 : 100 + h, 150 : 150 + w] = gray
    return cv2.imencode(".png", canvas)[1].tobytes()


@dataclass
class Stats:
    latencies: list[float] = field(default_factory=list)  # seconds, all completed requests
    statuses: Counter = field(default_factory=Counter)  # HTTP code or exception name
    retry_after: int = 0  # 503 responses carrying a Retry-After header
    wrong: int = 0  # 200 responses with an unexpected body


async def _worker(
    client: httpx.AsyncClient, body: dict, stats: Stats, stop: asyncio.Event, budget: list[int]
) -> None:
    while not stop.is_set():
        if budget[0] == 0:
            return
        if budget[0] > 0:
            budget[0] -= 1
        t0 = time.perf_counter()
        try:
            r = await client.post("/v1/decode", json=body)
        except httpx.HTTPError as exc:
            stats.statuses[type(exc).__name__] += 1
            continue
        stats.latencies.append(time.perf_counter() - t0)
        stats.statuses[r.status_code] += 1
        if r.status_code == 503 and "retry-after" in r.headers:
            stats.retry_after += 1
        elif r.status_code == 200 and not r.json().get("barcodes"):
            stats.wrong += 1


async def run_load(
    url: str,
    concurrency: int,
    requests: int | None,
    duration: float | None,
    image: str,
    keepalive: bool = True,
) -> tuple[Stats, float]:
    """Run until ``requests`` are sent or ``duration`` seconds pass (whichever is set)."""
    body = {"image": base64.b64encode(make_payload(image)).decode()}
    stats, stop = Stats(), asyncio.Event()
    budget = [requests if requests is not None else -1]  # -1 = unlimited
    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency if keepalive else 0
    )
    async with httpx.AsyncClient(base_url=url, timeout=30, limits=limits) as client:
        # One request first so lazy initialisation does not count against the measurement.
        await client.post("/v1/decode", json=body)
        t0 = time.perf_counter()
        tasks = [
            asyncio.create_task(_worker(client, body, stats, stop, budget))
            for _ in range(concurrency)
        ]
        if duration is not None:
            await asyncio.sleep(duration)
            stop.set()
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0
    return stats, elapsed


def summarise(stats: Stats, elapsed: float) -> dict[str, float]:
    total = sum(stats.statuses.values())
    ok = stats.statuses.get(200, 0)
    lat = np.asarray(stats.latencies) * 1000 if stats.latencies else np.zeros(1)
    return {
        "total": total,
        "rps": total / elapsed,
        "ok_rps": ok / elapsed,
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "p99": float(np.percentile(lat, 99)),
        "error_rate": (total - ok) / total if total else 0.0,
    }


def print_report(label: str, stats: Stats, elapsed: float) -> dict[str, float]:
    s = summarise(stats, elapsed)
    print(f"== {label}")
    print(
        f"requests {s['total']:.0f} in {elapsed:.1f}s | RPS {s['rps']:.1f} "
        f"(200 only: {s['ok_rps']:.1f}) | p50 {s['p50']:.1f} ms | p95 {s['p95']:.1f} ms "
        f"| p99 {s['p99']:.1f} ms | error rate {100 * s['error_rate']:.1f}%"
    )
    print(
        "status histogram: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(stats.statuses.items(), key=str))
    )
    if stats.statuses.get(503):
        print(f"503 responses with Retry-After: {stats.retry_after}/{stats.statuses[503]}")
    if stats.wrong:
        print(f"200 responses without a barcode: {stats.wrong}")
    return s


async def overload(url: str, concurrency: int, duration: float, image: str) -> int:
    """Saturate the service and require 503 + Retry-After rather than timeouts or crashes."""
    stats, elapsed = await run_load(url, concurrency, None, duration, image)
    print_report(f"overload ({concurrency} concurrent for {duration:.0f}s)", stats, elapsed)
    n503 = stats.statuses.get(503, 0)
    transport_errors = {k: v for k, v in stats.statuses.items() if isinstance(k, str)}
    bad_codes = {
        k: v for k, v in stats.statuses.items() if k not in (200, 503) and isinstance(k, int)
    }
    failures = []
    if not n503:
        failures.append("no 503 seen; raise --concurrency or lower BARCODE_MAX_CONCURRENCY")
    if n503 != stats.retry_after:
        failures.append("some 503 responses lack Retry-After")
    if transport_errors:
        failures.append(f"timeouts/transport errors: {transport_errors}")
    if bad_codes:
        failures.append(f"unexpected status codes: {bad_codes}")
    if not stats.statuses.get(200):
        failures.append("no request succeeded; service did not keep serving under overload")
    for f in failures:
        print(f"FAIL: {f}")
    print(
        "overload check: "
        + ("FAILED" if failures else "ok (saturation returned 503 + Retry-After)")
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8080")
    parser.add_argument("-c", "--concurrency", type=int, default=8)
    parser.add_argument("-n", "--requests", type=int, help="total requests (default 200)")
    parser.add_argument("-d", "--duration", type=float, help="seconds; overrides --requests")
    parser.add_argument("--image", choices=["easy", "hard"], default="easy")
    parser.add_argument(
        "--no-keepalive",
        action="store_true",
        help="new connection per request; spreads load across workers (keep-alive pins a "
        "connection to one worker process)",
    )
    parser.add_argument("--overload", action="store_true", help="saturation check (503 expected)")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    if args.overload:
        c = args.concurrency if args.concurrency > 8 else 64
        return asyncio.run(overload(url, c, args.duration or 10, args.image))

    requests = None if args.duration else (args.requests or 200)
    stats, elapsed = asyncio.run(
        run_load(url, args.concurrency, requests, args.duration, args.image, not args.no_keepalive)
    )
    print_report(
        f"c={args.concurrency} image={args.image} keepalive={not args.no_keepalive}", stats, elapsed
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
