"""Load shedding and request isolation under parallel calls."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tests.api.conftest import BlockingPipeline, EchoPipeline, b64, png_bytes


def test_saturation_returns_503_with_retry_after(make_client):
    pipeline = BlockingPipeline()
    client = make_client(pipeline, max_concurrency=1)
    body = {"image": b64(png_bytes())}

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/v1/decode", json=body)
        assert pipeline.entered.wait(timeout=10)
        rejected = client.post("/v1/decode", json=body)
        pipeline.release.set()
        assert first.result(timeout=10).status_code == 200

    assert rejected.status_code == 503
    assert "retry-after" in rejected.headers
    # The slot is released afterwards, so the service recovers.
    assert client.post("/v1/decode", json=body).status_code == 200


def test_parallel_requests_get_their_own_results(make_client):
    client = make_client(EchoPipeline(delay=0.01), max_concurrency=64)

    def call(i: int) -> tuple[int, str]:
        # Alternate transports so both code paths are exercised concurrently.
        if i % 2:
            resp = client.post("/v1/decode", json={"image": b64(png_bytes(i))})
        else:
            resp = client.post("/v1/decode", files={"file": ("a.png", png_bytes(i), "image/png")})
        assert resp.status_code == 200
        return i, resp.json()["barcodes"][0]["text"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(call, range(32)))

    assert results == [(i, str(i)) for i in range(32)]
