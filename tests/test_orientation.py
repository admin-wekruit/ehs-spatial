"""Semantic orientation recovery: the VLM's call rotates pixels in place;
provider failure leaves the file untouched (the gravity tripwire guards)."""

from pathlib import Path

import pytest
from PIL import Image

from ehs_spatial.orientation import OrientationCall, ensure_upright


class FakeAdapter:
    def __init__(self, rotation):
        self.rotation = rotation
        self.calls = 0

    def _create(self, op, **kwargs):
        self.calls += 1
        return op

    def _parse(self, response, model, op):
        return OrientationCall(rotate_clockwise_deg=self.rotation), None


class BrokenAdapter:
    def _create(self, *a, **k):
        raise RuntimeError("provider down")


@pytest.fixture
def portrait_stored_landscape(tmp_path):
    # tall scene stored sideways (300 wide x 200 high), no EXIF
    path = tmp_path / "img.jpg"
    Image.new("RGB", (300, 200), (90, 90, 90)).save(path)
    return path


def test_rotation_applied_in_place(portrait_stored_landscape):
    adapter = FakeAdapter(90)
    applied = ensure_upright([str(portrait_stored_landscape)], adapter)
    assert applied == [90]
    with Image.open(portrait_stored_landscape) as image:
        assert image.size == (200, 300)  # upright now


def test_upright_left_untouched(portrait_stored_landscape):
    before = portrait_stored_landscape.read_bytes()
    applied = ensure_upright([str(portrait_stored_landscape)], FakeAdapter(0))
    assert applied == [0]
    assert portrait_stored_landscape.read_bytes() == before


def test_provider_failure_is_fail_soft(portrait_stored_landscape):
    before = portrait_stored_landscape.read_bytes()
    applied = ensure_upright([str(portrait_stored_landscape)], BrokenAdapter())
    assert applied == [0]
    assert portrait_stored_landscape.read_bytes() == before
