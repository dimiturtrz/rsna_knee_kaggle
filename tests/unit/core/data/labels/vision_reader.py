"""The transport half of the probe: PNG encoding, and which reader a `--model` value selects.

Neither `answer` is tested here — one needs a server and the other nine gigabytes of weights, which is
an integration concern. What is testable without either is everything a wrong answer could hide
behind: that the bytes handed to a model really are the montage, and that naming a model reaches the
transport it was meant to.
"""

import struct
import zlib

import numpy as np

from core.data.labels.vision_reader import (
    PNG_MAGIC,
    SERVED,
    LocalReader,
    PngImage,
    Reader,
    ServedReader,
)

MEDGEMMA = "google/medgemma-4b-it"


def test_encoded() -> None:
    """A greyscale array round-trips: magic, declared size, and the pixels back out of the deflate."""
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    encoded = PngImage(image).encoded()
    assert encoded.startswith(PNG_MAGIC)
    width, height = struct.unpack(">II", encoded[16:24])
    assert (width, height) == (4, 3)
    start = encoded.index(b"IDAT") + 4
    payload = zlib.decompress(encoded[start : start + int.from_bytes(encoded[start - 8 : start - 4], "big")])
    assert payload == b"".join(b"\x00" + row.tobytes() for row in image)


def test_chunk() -> None:
    """A chunk is its length, its tag, its payload and the CRC over tag-plus-payload."""
    framed = PngImage(np.zeros((1, 1), dtype=np.uint8)).chunk(b"IEND", b"")
    assert framed == struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))


def test_url() -> None:
    """The montage is handed over as a base64 PNG data URL."""
    url = PngImage(np.arange(12, dtype=np.uint8).reshape(3, 4)).url()
    assert url.startswith("data:image/png;base64,")
    assert len(url) > len("data:image/png;base64,")


def test_for_model() -> None:
    """`served` reaches the endpoint; anything else is a repo id to load off disk."""
    assert isinstance(Reader.for_model(SERVED, "cpu"), ServedReader)
    assert isinstance(Reader.for_model(MEDGEMMA, "cpu"), LocalReader)


def test_served_reader_label() -> None:
    """The served reader records the model the endpoint reports, so two runs cannot be merged."""
    assert ServedReader(model="a/b").label == "a/b"


def test_local_reader_label() -> None:
    """The local reader records the repo id its weights came from."""
    assert LocalReader(MEDGEMMA, "cpu").label == MEDGEMMA
