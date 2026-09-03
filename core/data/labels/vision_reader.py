"""How a montage reaches a vision model and how its raw text comes back.

Two transports, and the split is not an abstraction for its own sake: the model already on the card is
reachable only over HTTP and belongs to somebody else's service, while the models the owner asked to
be re-run exist as weights on disk and have to be loaded into whatever VRAM is left. Same prompt, same
montage, same parse — only the transport differs, so a disagreement between two readers is a fact
about the models.
"""

import base64
import json
import logging
import struct
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override
from urllib.request import Request, urlopen

import numpy as np
from jaxtyping import UInt8

from core.data.labels.vision_prompt import AclPrompt

logger = logging.getLogger(__name__)

ENDPOINT = "http://localhost:8001/v1/chat/completions"
"""The vLLM server already resident on the card.

Deliberate: the 5090 holds one model at a time and this container occupies 30.7 of 32.6 GiB, so a
second model would mean evicting somebody else's service to answer a question that can be answered
for free. It is also an INDEPENDENCE win rather than a compromise — the first run used a 4B, a 7B and
another 7B, and this is a 27B from a different generation."""

MODEL = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
"""What that endpoint serves. Read off `/v1/models` rather than assumed."""

SERVED = "served"
"""The `--model` value that reaches the resident endpoint instead of loading weights."""

LOCAL_DTYPE = "bfloat16"
"""What locally loaded weights are cast to. The owner's standing rule is bf16 for training and fp16 for
serving; this is neither — it is a zero-shot READ, and bf16 is what these checkpoints were released in,
so it is the cast that adds no error of its own."""

TIMEOUT = 180.0
"""Seconds for one call. A 27B answering ~40 tokens over a twelve-tile montage is seconds; this is a
bound that catches a wedged server without turning a slow queue into a failed run."""

MAX_TOKENS = 200

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_GREY_8BIT = (8, 0, 0, 0, 0)
"""Bit depth, colour type (0 = greyscale) and the three fixed zeros of a PNG header.

The encoder is written here in fifteen lines of stdlib rather than pulled in with an image library:
the cache is already 8-bit greyscale, so the whole job is a zlib deflate and three CRCs, and adding a
dependency to this project's declared set to do it would be the larger change."""

DEFLATE_LEVEL = 6


@dataclass(frozen=True)
class PngImage:
    """One greyscale montage, encodable without an image library."""

    pixels: UInt8[np.ndarray, "rows cols"]

    def encoded(self) -> bytes:
        """The array as a PNG.

        Each scanline is prefixed with a zero filter byte, the lot is deflated, and the three mandatory
        chunks are framed with their CRCs. This is the whole of the PNG spec that a greyscale image needs.
        """
        height, width = self.pixels.shape
        header = struct.pack(">II5B", width, height, *PNG_GREY_8BIT)
        scanlines = b"".join(b"\x00" + row.tobytes() for row in self.pixels)
        return (
            PNG_MAGIC
            + self.chunk(b"IHDR", header)
            + self.chunk(b"IDAT", zlib.compress(scanlines, DEFLATE_LEVEL))
            + self.chunk(b"IEND", b"")
        )

    def chunk(self, tag: bytes, payload: bytes) -> bytes:
        """One length-prefixed, CRC-suffixed PNG chunk."""
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    def url(self) -> str:
        """The montage as the `image_url` an OpenAI-compatible server accepts."""
        return "data:image/png;base64," + base64.b64encode(self.encoded()).decode("ascii")


class Reader(ABC):
    """A vision model that can be asked one montage-shaped question."""

    @classmethod
    def for_model(cls, model: str, device: str) -> "Reader":
        """The resident endpoint when asked for `served`, otherwise the named weights off disk."""
        return ServedReader() if model == SERVED else LocalReader(model, device)

    @abstractmethod
    def answer(self, prompt: AclPrompt, grid: UInt8[np.ndarray, "rows cols"]) -> str:
        """One montage's raw text."""

    @property
    @abstractmethod
    def label(self) -> str:
        """What answered, recorded beside every reading so two runs can never be confused."""


class ServedReader(Reader):
    """A model already resident behind an OpenAI-compatible endpoint."""

    def __init__(self, endpoint: str = ENDPOINT, model: str = MODEL) -> None:
        self.endpoint = endpoint
        self.model = model

    @property
    @override
    def label(self) -> str:
        """The served model's name, as the endpoint reports it."""
        return self.model

    @override
    def answer(self, prompt: AclPrompt, grid: UInt8[np.ndarray, "rows cols"]) -> str:  # devtools-ignore: test-mirror
        """One montage's raw text from the local server.

        `enable_thinking` is forced off: a reasoning template spends the token budget on a preamble and
        emits no JSON, which would land every study on the middle rung for a reason that has nothing to
        do with the image.
        """
        body = json.dumps(
            {
                "model": self.model,
                "messages": prompt.turns({"type": "image_url", "image_url": {"url": PngImage(grid).url()}}),
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        request = Request(self.endpoint, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read())
        return str(payload["choices"][0]["message"]["content"])


class LocalReader(Reader):
    """Weights loaded from disk — the arm that answers "run medgemma and Lingshu yourself".

    GREEDY, AND THAT IS THE MEASUREMENT. `do_sample` off and no temperature: the question is what the
    model believes about the image, and sampling would let a run's spread stand in for the model's.

    The montage is greyscale and every processor in this family expects three channels, so it is
    broadcast rather than colour-mapped — a colour map would put information in the image that the
    scanner never recorded.
    """

    def __init__(self, repo: str, device: str) -> None:
        self.repo = repo
        self.device = device
        self._loaded: tuple[object, object] | None = None

    @property
    @override
    def label(self) -> str:
        """The repo id the weights came from."""
        return self.repo

    def parts(self) -> tuple[object, object]:  # devtools-ignore: test-mirror
        """The processor and model, loaded once and kept.

        Lazy because constructing the reader is how a caller names a model, and naming one should not
        cost twenty seconds and nine gigabytes before the split has even been read.
        """
        if self._loaded is None:
            from transformers import (  # noqa: PLC0415 - a 9 GB import behind a flag
                AutoModelForImageTextToText,
                AutoProcessor,
            )

            model = AutoModelForImageTextToText.from_pretrained(self.repo, dtype=LOCAL_DTYPE)
            self._loaded = (AutoProcessor.from_pretrained(self.repo), model.to(self.device).eval())
        return self._loaded

    @override
    def answer(self, prompt: AclPrompt, grid: UInt8[np.ndarray, "rows cols"]) -> str:  # devtools-ignore: test-mirror
        """One montage's raw text, generated locally."""
        import torch  # noqa: PLC0415 - only the local backend needs it

        processor, model = self.parts()
        turns = prompt.turns({"type": "image"})
        text = processor.apply_chat_template(turns, add_generation_prompt=True, tokenize=False)  # type: ignore[attr-defined]
        inputs = processor(text=[text], images=[np.repeat(grid[:, :, None], 3, axis=2)], return_tensors="pt")  # type: ignore[operator]
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            produced = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)  # type: ignore[attr-defined]
        fresh = produced[:, inputs["input_ids"].shape[1] :]
        return str(processor.batch_decode(fresh, skip_special_tokens=True)[0])  # type: ignore[attr-defined]
