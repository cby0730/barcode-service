import numpy as np
import pytest
import zxingcpp

from barcode_service import decoders
from barcode_service.decoders import PyzbarDecoder, ZxingDecoder, decode_all
from barcode_service.types import Barcode, Decoder
from tests.conftest import (
    CODE128_TEXT,
    EAN13_TEXT,
    QR_TEXT,
    blur,
    downscale,
    gaussian_noise,
    jpeg_compress,
    low_contrast,
    make_barcode,
    rotate,
)

BOTH = [PyzbarDecoder(), ZxingDecoder()]


@pytest.mark.parametrize("decoder", BOTH, ids=lambda d: d.name)
class TestBothDecoders:
    def test_satisfies_protocol(self, decoder):
        assert isinstance(decoder, Decoder)

    def test_ean13(self, decoder, ean13_img):
        assert decoder.decode(ean13_img) == [Barcode(EAN13_TEXT, "EAN13", "")]

    def test_code128(self, decoder, code128_img):
        assert decoder.decode(code128_img) == [Barcode(CODE128_TEXT, "CODE128", "")]

    def test_qr(self, decoder, qr_img):
        assert decoder.decode(qr_img) == [Barcode(QR_TEXT, "QRCODE", "")]

    def test_grayscale_input(self, decoder, ean13_img):
        gray = ean13_img[:, :, 0]
        assert [b.text for b in decoder.decode(gray)] == [EAN13_TEXT]

    def test_multi_barcode(self, decoder, multi_barcode_img):
        found = {(b.text, b.format) for b in decoder.decode(multi_barcode_img)}
        assert found == {(EAN13_TEXT, "EAN13"), (QR_TEXT, "QRCODE")}

    def test_blank_image(self, decoder):
        assert decoder.decode(np.full((200, 300, 3), 255, np.uint8)) == []

    def test_noise_only_image(self, decoder):
        assert decoder.decode(gaussian_noise(np.full((200, 300, 3), 128, np.uint8), 40)) == []

    @pytest.mark.parametrize(
        "degrade",
        [
            lambda im: blur(im, 1.0),
            lambda im: downscale(im, 0.7),
            lambda im: gaussian_noise(im, 8),
            lambda im: low_contrast(im, 0.5),
            lambda im: jpeg_compress(im, 50),
        ],
        ids=["blur", "downscale", "noise", "low_contrast", "jpeg"],
    )
    def test_mild_degradation_still_decodes(self, decoder, degrade):
        img = make_barcode("qr", QR_TEXT, scale=6)
        assert [b.text for b in decoder.decode(degrade(img))] == [QR_TEXT]


def test_zxing_handles_rotation():
    img = make_barcode("qr", QR_TEXT, scale=6)
    assert [b.text for b in ZxingDecoder().decode(rotate(img, 15))] == [QR_TEXT]


@pytest.mark.parametrize("decoder", BOTH, ids=lambda d: d.name)
def test_non_utf8_payload_does_not_crash(decoder):
    # Raw bytes that are not valid UTF-8.
    barcode = zxingcpp.create_barcode(
        bytes([0xFF, 0xFE, 0x80, 0x41]), zxingcpp.BarcodeFormat.QRCode
    )
    gray = np.array(zxingcpp.write_barcode_to_image(barcode, scale=6))
    img = np.pad(gray, 20, constant_values=255)
    results = decoder.decode(img)
    assert len(results) == 1
    assert isinstance(results[0].text, str) and results[0].format == "QRCODE"


class _Fake:
    def __init__(self, name, result=None, error=None):
        self.name, self._result, self._error = name, result or [], error
        self.calls = 0

    def decode(self, img):
        self.calls += 1
        if self._error:
            raise self._error
        return list(self._result)


def test_decode_all_first_hit_wins():
    hit = [Barcode("a", "QRCODE", "")]
    first, second = _Fake("first"), _Fake("second", hit)
    third = _Fake("third", [Barcode("b", "QRCODE", "")])
    assert decode_all([first, second, third], np.zeros((4, 4), np.uint8)) == hit
    assert third.calls == 0


def test_decode_all_skips_failing_decoder_and_logs(caplog):
    hit = [Barcode("a", "QRCODE", "")]
    bad = _Fake("bad", error=RuntimeError("boom"))
    with caplog.at_level("DEBUG", logger=decoders.__name__):
        assert decode_all([bad, _Fake("ok", hit)], np.zeros((4, 4), np.uint8)) == hit
    assert "bad" in caplog.text


def test_decode_all_empty():
    assert decode_all([], np.zeros((4, 4), np.uint8)) == []
    assert decode_all([_Fake("x")], np.zeros((4, 4), np.uint8)) == []
