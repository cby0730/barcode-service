import dataclasses

import pytest

from barcode_service import types


def test_value_objects_are_frozen():
    b = types.Barcode("123", "EAN13", "raw")
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.text = "x"  # type: ignore[misc]
    assert types.Box(0, 0, 1, 1, 0.9).score == 0.9


def test_exception_hierarchy():
    assert issubclass(types.ImageDecodeError, ValueError)
    assert issubclass(types.ImageTooLargeError, ValueError)
    assert issubclass(types.BackendUnavailableError, RuntimeError)
    assert issubclass(types.ModelNotFoundError, FileNotFoundError)


def test_protocols_are_runtime_checkable():
    class D:
        name = "d"

        def decode(self, img):
            return []

    assert isinstance(D(), types.Decoder)
    assert not isinstance(object(), types.Decoder)
