"""The locator's shapes, its class merge, the argmax it reads out, and the slice corpus it is fit on."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from kneemri.models.backbone import ENCODERS
from kneemri.models.locator import CLASSES, Locator, SliceCorpus
from kneemri.models.variants import SMALL

SIDE = 56
CORPUS_SIDE = 28
CPU = torch.device("cpu")
RATES = (0.1, 1.0)

torch.manual_seed(0)


def test_forward(tiny_locator: Locator) -> None:
    logits = tiny_locator.forward(torch.randint(0, 256, (2, SIDE, SIDE), dtype=torch.uint8))
    assert logits.shape == (2, len(CLASSES), SIDE, SIDE)


def test_predict(tiny_locator: Locator) -> None:
    drawn = tiny_locator.predict(torch.zeros(1, SIDE, SIDE))
    assert drawn.shape == (1, SIDE, SIDE)
    assert int(drawn.max()) < len(CLASSES)


def test_merged() -> None:
    assert Locator.merged(torch.arange(7)).tolist() == [0, 1, 2, 3, 3, 4, 4]


def test_groups(tiny_locator: Locator) -> None:
    groups = tiny_locator.groups(*RATES)
    assert [group["lr"] for group in groups] == list(RATES)
    assert groups[1]["params"] == list(tiny_locator.head.parameters())


def test_pretrained(tiny_locator: Locator) -> None:
    with patch.object(type(ENCODERS[SMALL]), "load", return_value=tiny_locator.trunk):
        built = Locator.pretrained(ENCODERS[SMALL])
    assert built.trunk is tiny_locator.trunk
    assert built.head[0].in_channels == ENCODERS[SMALL].dim * len(built.layers)


def test_read(slice_directory: Path) -> None:
    assert SliceCorpus.read(slice_directory).images.shape == (8, 2, CORPUS_SIDE, CORPUS_SIDE)


def test_indices(slice_corpus: SliceCorpus) -> None:
    assert slice_corpus.indices("val").tolist() == [4, 5]


def test_batch(slice_corpus: SliceCorpus) -> None:
    corpus = slice_corpus
    slices, labels = corpus.batch(np.array([3, 1]), np.array([1, 0]), CPU)
    assert torch.equal(slices[0], torch.from_numpy(corpus.images[1, 0]))
    assert torch.equal(slices[1], torch.from_numpy(corpus.images[3, 1]))
    assert int(labels.max()) < len(CLASSES)
