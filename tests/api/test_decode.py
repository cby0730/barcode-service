"""POST /v1/decode behaviour and error mapping."""

from __future__ import annotations

import logging

from tests.api.conftest import (
    BrokenPipeline,
    EchoPipeline,
    EmptyPipeline,
    b64,
    noisy_png,
    png_bytes,
)


def test_json_returns_barcodes(make_client):
    client = make_client(EchoPipeline())
    resp = client.post("/v1/decode", json={"image": b64(png_bytes(42))})
    assert resp.status_code == 200
    assert resp.json() == {"barcodes": [{"text": "42", "format": "QRCode", "strategy": "raw"}]}


def test_data_uri_accepted(make_client):
    client = make_client(EchoPipeline())
    uri = "data:image/png;base64," + b64(png_bytes(7))
    resp = client.post("/v1/decode", json={"image": uri})
    assert resp.status_code == 200
    assert resp.json()["barcodes"][0]["text"] == "7"


def test_empty_result_is_200_with_empty_list(make_client):
    client = make_client(EmptyPipeline())
    resp = client.post("/v1/decode", json={"image": b64(png_bytes())})
    assert resp.status_code == 200
    assert resp.json() == {"barcodes": []}


def test_multipart_upload(make_client):
    client = make_client(EchoPipeline())
    files = {"file": ("a.png", png_bytes(99), "image/png")}
    resp = client.post("/v1/decode", files=files)
    assert resp.status_code == 200
    assert resp.json()["barcodes"][0]["text"] == "99"


def test_multipart_without_file_field_is_422(make_client):
    client = make_client(EchoPipeline())
    resp = client.post("/v1/decode", files={"other": ("a.png", png_bytes(), "image/png")})
    assert resp.status_code == 422


def test_invalid_base64_is_400(make_client):
    client = make_client(EchoPipeline())
    resp = client.post("/v1/decode", json={"image": "!!! not base64 !!!"})
    assert resp.status_code == 400


def test_non_image_bytes_are_400(make_client):
    client = make_client(EchoPipeline())
    assert client.post("/v1/decode", json={"image": b64(b"hello world")}).status_code == 400
    files = {"file": ("a.txt", b"hello world", "text/plain")}
    assert client.post("/v1/decode", files=files).status_code == 400


def test_missing_image_field_is_422(make_client):
    client = make_client(EchoPipeline())
    assert client.post("/v1/decode", json={"nope": 1}).status_code == 422


def test_unsupported_content_type_is_415(make_client):
    client = make_client(EchoPipeline())
    resp = client.post("/v1/decode", content=b"x", headers={"content-type": "text/plain"})
    assert resp.status_code == 415


def test_oversize_json_is_413(make_client):
    client = make_client(EchoPipeline(), max_image_bytes=1000)
    resp = client.post("/v1/decode", json={"image": b64(noisy_png(64))})
    assert resp.status_code == 413


def test_oversize_multipart_is_413(make_client):
    client = make_client(EchoPipeline(), max_image_bytes=1000)
    resp = client.post("/v1/decode", files={"file": ("a.png", noisy_png(64), "image/png")})
    assert resp.status_code == 413


def test_too_many_pixels_is_413(make_client):
    client = make_client(EchoPipeline(), max_image_pixels=10)
    resp = client.post("/v1/decode", json={"image": b64(png_bytes(size=8))})
    assert resp.status_code == 413


def test_decode_without_pipeline_is_503(make_client):
    client = make_client(None)
    resp = client.post("/v1/decode", json={"image": b64(png_bytes())})
    assert resp.status_code == 503


def test_unexpected_error_is_generic_500_and_logged(make_client, caplog):
    client = make_client(BrokenPipeline())
    with caplog.at_level(logging.ERROR):
        resp = client.post("/v1/decode", json={"image": b64(png_bytes())})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "internal server error"}
    assert "secret internal detail" not in resp.text
    assert any(r.exc_info for r in caplog.records)
