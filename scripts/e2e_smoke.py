"""End-to-end smoke test against a running service.

    uv run --no-sync python scripts/e2e_smoke.py http://127.0.0.1:8080

Posts synthetic barcodes as JSON (base64) and multipart, and checks /health, /ready and the
error codes. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time

import cv2
import httpx
import numpy as np
import zxingcpp

CASES = {
    "EAN13": ("5901234123457", zxingcpp.BarcodeFormat.EAN13),
    "CODE128": ("SMOKE-0042", zxingcpp.BarcodeFormat.Code128),
    "QRCODE": ("https://example.com/smoke", zxingcpp.BarcodeFormat.QRCode),
}


def render_png(text: str, fmt: zxingcpp.BarcodeFormat) -> bytes:
    barcode = zxingcpp.create_barcode(text, fmt)
    gray = np.array(zxingcpp.write_barcode_to_image(barcode, scale=4, add_quiet_zones=True))
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    return buf.tobytes()


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        self.failures += not ok


def wait_ready(client: httpx.Client, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if client.get("/ready").status_code == 200:
                return True
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    return False


def run(base_url: str, wait: float) -> int:
    c = Checker()
    with httpx.Client(base_url=base_url, timeout=30) as client:
        if not wait_ready(client, wait):
            print(f"service at {base_url} did not become ready within {wait:.0f}s")
            return 1

        r = client.get("/health")
        c.check("GET /health", r.status_code == 200 and r.json() == {"status": "ok"})
        r = client.get("/ready")
        c.check("GET /ready", r.status_code == 200, r.text)

        for fmt_name, (text, fmt) in CASES.items():
            png = render_png(text, fmt)
            r = client.post("/v1/decode", json={"image": base64.b64encode(png).decode()})
            got = [(b["text"], b["format"]) for b in r.json().get("barcodes", [])]
            c.check(f"json {fmt_name}", r.status_code == 200 and (text, fmt_name) in got, str(got))

            r = client.post("/v1/decode", files={"file": ("a.png", png, "image/png")})
            got = [(b["text"], b["format"]) for b in r.json().get("barcodes", [])]
            c.check(f"multipart {fmt_name}", r.status_code == 200 and (text, fmt_name) in got)

        blank = cv2.imencode(".png", np.full((200, 300), 255, np.uint8))[1].tobytes()
        r = client.post("/v1/decode", files={"file": ("b.png", blank, "image/png")})
        c.check("blank image gives empty list", r.status_code == 200 and r.json()["barcodes"] == [])

        r = client.post("/v1/decode", json={"image": "!!! not base64 !!!"})
        c.check("invalid base64 -> 400", r.status_code == 400, str(r.status_code))
        r = client.post("/v1/decode", files={"file": ("x.png", b"not an image", "image/png")})
        c.check("non-image bytes -> 400", r.status_code == 400, str(r.status_code))
        r = client.post("/v1/decode", json={"wrong": 1})
        c.check("missing field -> 422", r.status_code == 422, str(r.status_code))

        # 64 MiB of zeros exceeds any sane default limit; the server must reject it early.
        huge = b"\0" * (64 * 1024 * 1024)
        r = client.post("/v1/decode", files={"file": ("big.png", huge, "image/png")})
        c.check("oversized upload -> 413", r.status_code == 413, str(r.status_code))

    print(f"\n{c.failures} failure(s)")
    return 1 if c.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8080")
    parser.add_argument("--wait", type=float, default=60, help="seconds to wait for /ready")
    args = parser.parse_args()
    return run(args.url.rstrip("/"), args.wait)


if __name__ == "__main__":
    sys.exit(main())
