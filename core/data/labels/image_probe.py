"""Ask a vision model to read ACL tears off the pixels: `python -m core.data.labels.image_probe`.

WHY THIS EXISTS, AND WHY IT IS A REPLICATION RATHER THAN A FIRST LOOK. The peer lane reported that
zero-shot VLMs cannot separate a torn ACL from an intact one on this corpus, on **two studies** — one
positive, one negative, three models, every answer "torn". Two studies is one coin flip per model, and
a verdict that would close the whole vision-audit arm cannot rest on it. This is the independent
second run the owner asked for: same question, a different model, a different montage, a different
prompt, and a sample large enough to put a confusion matrix under the claim.

WHAT IS DELIBERATELY NOT SHARED WITH THE FIRST RUN. The montage code and the prompt are written here
from scratch. If both lanes made the same slice-selection mistake, a shared bug would read as a shared
finding — which is the failure mode this replication is for. Two concrete divergences: the first run
picked six consecutive slices (4-9) and this one takes **all twelve**, so slice choice is not a free
parameter that both lanes could get wrong the same way; and the answer here is a GRADED score rather
than a word, so the reading survives a model that is merely miscalibrated rather than blind.

THE SPLIT IS FROZEN AND ONLY HALF OF IT IS LOOKED AT. `research/splits/2026-09-03_vlm_filter_split58.csv`
was frozen before any model output existed. This reads the **tune** half only. The holdout half stays
untouched so that anything built on the back of this can still be tested.

THE THREE NUMBERS THAT DECIDE IT, and they are reported whatever the verdict:

  1. The confusion matrix against the competition's own ACL column on the annotated studies.
  2. The BASE RATE. A model that answers "torn" every time scores the positive rate by construction,
     and on a near-balanced set that is close to half. Accuracy alone cannot tell the two apart, so
     the call rate is printed beside it and AUC is computed over the graded score.
  3. The DEGENERATE-OUTPUT rate — how often two different studies come back with the same text. The
     first run saw near-identical wording across studies on two models, which says the answer is not
     a function of the image at all. That is a stronger refutation than a bad matrix, and a different
     one, so it is measured rather than inferred.

WHAT THIS CANNOT SAY. It reads ONE finding on ONE slot with ONE prompt, so a null here is a fact about
zero-shot ACL reading off a sagittal fluid-sensitive montage, not about vision models on knee MRI, and
certainly not about the separate text task of filtering the 108 reader false positives — nothing here
bears on that. It is also zero-shot: it prices what a model knows already, never what a fine-tune
could learn.

IT WRITES ITS OWN KEYED FILE AND TOUCHES NO LABEL TABLE. `train.csv`, `report_labels_v2.csv` and
`overcall_audit.csv` are append-only and are read here, never written.
"""

import json
import logging
import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import UInt8

from core import cli
from core.data.labels.vision_prompt import ACL, CALLED_POSITIVE, GRADES, JSON_OBJECT, SCORE_OF, AclPrompt
from core.data.labels.vision_reader import SERVED, Reader
from core.findings import STUDY

logger = logging.getLogger(__name__)

SPLIT_COLUMN = "split"
TUNE = "tune"
"""The half of the frozen split that may be looked at. The other half is not read by this module."""

CLIP_PERCENTILES = (2.0, 98.0)
"""Where each slice's window is taken from, when one is asked for.

MIN-MAX IS A NULL OPERATION HERE, AND WAS RUN AS ONE. The cache already stores every slice spanning
roughly 0-255, so rescaling to the endpoints moved 0.0 pixels and three models returned byte-identical
answers — which read as a cleared suspect and was not one. Endpoints are not the distribution: two
studies both spanning 0-255 still differ in where their tissue sits inside that range. Clipping at
percentiles is what equalises them, and it is the only version of this flag whose null means anything.
"""

DEGENERATE_SIMILARITY = 0.95
"""How alike two answers must be before they count as the same text. Not 1.0: the first run's
degenerate pairs differed by a word or two while making identical claims, and an exact-match test
would have scored that as two distinct readings."""


@dataclass(frozen=True)
class Montage:
    """How one study's slot becomes a single image, and the only place slice choice is decided.

    TAKING EVERY SLICE IS THE DEFAULT, AND THE FIRST RUN USED IT. A probe that selects a sub-band is
    making a radiological judgement on the model's behalf, and if it selects the wrong band the null it
    produces is the selection's, not the model's. Twelve tiles at 336 px is 1.3 megapixels, which the
    server handles, so there is no reason to spend that decision by default.

    `band` and `stretch` exist because the first four models all read AUC ~0.5 at that default, and a
    montage is a suspect as much as a model is: twelve tiles reach the vision tower at roughly 224 px
    each, the ACL is a thin band across two or three of them, and slot 0 varies more between studies in
    sequence, window and crop than the ACL varies within. Cutting to the notch band and stretching each
    slice to its own range removes both suspects, so a null that survives them is the model's.
    """

    slot: int = 0
    columns: int = 4
    band: tuple[int, int] | None = None
    stretch: bool = False

    def slices_of(
        self, volume: UInt8[np.ndarray, "slots slices height width"]
    ) -> UInt8[np.ndarray, "slices height width"]:
        """The slot's slices, cut to the band if one was asked for and stretched if contrast was."""
        slices = np.asarray(volume[self.slot])
        if self.band is not None:
            slices = slices[self.band[0] : self.band[1]]
        if not self.stretch:
            return slices
        flat = slices.reshape(len(slices), -1)
        floor = np.percentile(flat, CLIP_PERCENTILES[0], axis=1)[:, None, None].astype(np.float32)
        ceiling = np.percentile(flat, CLIP_PERCENTILES[1], axis=1)[:, None, None].astype(np.float32)
        span = np.maximum(ceiling - floor, 1.0)
        return np.asarray(((slices - floor) * (255.0 / span)).clip(0, 255).astype(np.uint8))

    def tile(self, volume: UInt8[np.ndarray, "slots slices height width"]) -> UInt8[np.ndarray, "rows cols"]:
        """This study's slot laid out as a grid, in slice order, reading left to right then down."""
        slices = self.slices_of(volume)
        rows = -(-len(slices) // self.columns)
        blank = np.zeros_like(slices[:1])
        padded = np.concatenate([slices, np.repeat(blank, rows * self.columns - len(slices), axis=0)])
        grid = padded.reshape(rows, self.columns, *slices.shape[1:])
        return np.asarray(grid.transpose(0, 2, 1, 3).reshape(rows * slices.shape[1], self.columns * slices.shape[2]))


@dataclass(frozen=True)
class Reading:
    """One study's answer: the grade, the score it becomes, and the raw text it was parsed from.

    The raw text is kept because the degenerate-output rate is a fact about the TEXT, not about the
    grade — two studies can land on the same grade honestly, and only identical prose says the model
    stopped looking.
    """

    study: str
    grade: str
    because: str
    raw: str

    @classmethod
    def parse(cls, study: str, answer: str) -> "Reading":
        """A model answer read into a grade, with anything unrecognised falling to `possible`.

        Falling to the middle rung rather than raising: a parse failure is not evidence either way, and
        scoring it as a negative would flatter a model that failed to answer.
        """
        match = JSON_OBJECT.search(answer)
        spoken: dict[str, object] = {}
        if match is not None:
            try:
                spoken = dict(json.loads(match.group()))
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("study %s: answer was not JSON", study)
        word = str(spoken.get("acl", "")).strip().lower()
        return cls(study, word if word in GRADES else "possible", str(spoken.get("because", "")), answer)

    @property
    def score(self) -> float:
        """The grade on the 0-1 scale AUC is computed over."""
        return SCORE_OF[self.grade]


@dataclass(frozen=True)
class ProbeResult:
    """Every reading, and the three numbers that decide the question.

    They live together because reporting any one of them alone is how this measurement goes wrong: a
    matrix without the call rate cannot be told from a constant answer, and a call rate without the
    degenerate rate cannot be told from a model that is looking but miscalibrated.
    """

    readings: tuple[Reading, ...]
    truth: tuple[int, ...]

    def matrix(self) -> dict[str, int]:
        """True/false positives and negatives at `CALLED_POSITIVE`."""
        called = [reading.score >= CALLED_POSITIVE for reading in self.readings]
        pairs = list(zip(called, self.truth, strict=True))
        return {
            "true_positive": sum(call and label == 1 for call, label in pairs),
            "false_positive": sum(call and label == 0 for call, label in pairs),
            "false_negative": sum(not call and label == 1 for call, label in pairs),
            "true_negative": sum(not call and label == 0 for call, label in pairs),
        }

    def call_rate(self) -> float:
        """How often the model said `torn`. The number that exposes a constant answer."""
        return float(np.mean([reading.score >= CALLED_POSITIVE for reading in self.readings]))

    def base_rate(self) -> float:
        """How often the truth is positive. A constant `torn` scores exactly this by construction."""
        return float(np.mean(self.truth))

    def auc(self) -> float:
        """Rank AUC over the graded score, ties counted at half — the reading that survives miscalibration.

        Written out rather than imported so a tie, which a three-rung vocabulary produces constantly,
        is handled the way the metric defines it rather than however a library happens to.
        """
        scores = [reading.score for reading in self.readings]
        positive = [score for score, label in zip(scores, self.truth, strict=True) if label == 1]
        negative = [score for score, label in zip(scores, self.truth, strict=True) if label == 0]
        if not positive or not negative:
            return float("nan")
        wins = sum((one > zero) + 0.5 * (one == zero) for one in positive for zero in negative)
        return float(wins / (len(positive) * len(negative)))

    def degenerate_rate(self) -> float:
        """Share of study PAIRS whose answers are the same text to within `DEGENERATE_SIMILARITY`.

        Over pairs rather than over studies, because "the model says the same thing about everything"
        is a statement about the relation between answers and has no per-study form.
        """
        texts = [reading.because.strip().lower() for reading in self.readings]
        pairs = [(one, two) for index, one in enumerate(texts) for two in texts[index + 1 :]]
        if not pairs:
            return float("nan")
        alike = sum(SequenceMatcher(None, one, two).ratio() >= DEGENERATE_SIMILARITY for one, two in pairs)
        return float(alike / len(pairs))

    def frame(self) -> pd.DataFrame:
        """Every reading as a row, so the run can be re-cut without querying the model again."""
        return pd.DataFrame(
            [
                {
                    STUDY: reading.study,
                    "grade": reading.grade,
                    "score": reading.score,
                    "acl": label,
                    "because": reading.because,
                    "raw": reading.raw,
                }
                for reading, label in zip(self.readings, self.truth, strict=True)
            ]
        )

    def report(self) -> str:
        """The three numbers, in the order that makes them readable, as one block of text."""
        matrix = self.matrix()
        counts = " ".join(f"{name}={count}" for name, count in matrix.items())
        return (
            f"n={len(self.readings)}  base_rate={self.base_rate():.3f}  call_rate={self.call_rate():.3f}\n"
            f"{counts}\n"
            f"auc={self.auc():.3f}  degenerate_pair_rate={self.degenerate_rate():.3f}"
        )


@dataclass(frozen=True)
class ProbeCache:
    """The pixel cache and the study order that indexes it — the two paths that always travel together."""

    images: Path
    index: Path

    def positions(self) -> dict[str, int]:
        """Where each study sits along the cache's first axis."""
        return {str(study): position for position, study in enumerate(json.loads(self.index.read_text()))}

    def volumes(self) -> UInt8[np.ndarray, "studies slots slices height width"]:
        """The cache itself, memmapped — 35 GB that must never be read into memory whole."""
        return np.load(self.images, mmap_mode="r")


@dataclass(frozen=True)
class FrozenSplit:
    """The study-level split that was frozen before any model output existed."""

    path: Path

    def tune(self) -> list[str]:
        """The half of the split this module is allowed to look at."""
        table = pd.read_csv(self.path)
        return [str(study) for study in table.loc[table[SPLIT_COLUMN] == TUNE, STUDY]]


class VisionProbe:
    """Montage in, graded reading out, nothing leaving the machine."""

    def __init__(self, reader: Reader, montage: Montage | None = None, prompt: AclPrompt | None = None) -> None:
        self.reader = reader
        self.montage = montage or Montage()
        self.prompt = prompt or AclPrompt()

    def read(  # devtools-ignore: test-mirror
        self, study: str, volume: UInt8[np.ndarray, "slots slices height width"]
    ) -> Reading:
        """One study, graded."""
        return Reading.parse(study, self.reader.answer(self.prompt, self.montage.tile(volume)))

    def over(  # devtools-ignore: test-mirror
        self, studies: list[str], cache: ProbeCache, truth: pd.Series, seed: int
    ) -> ProbeResult:
        """Every study read in a shuffled order, against its ACL label.

        THE ORDER IS SHUFFLED ON PURPOSE. The cache is in split order, so reading it straight through
        presents the model with whatever ordering the corpus happens to carry; a seeded shuffle means a
        drift in the server over the run cannot line up with the labels.
        """
        positions = cache.positions()
        ordered = sorted(studies, key=lambda study: random.Random(f"{seed}:{study}").random())
        volumes = cache.volumes()
        readings: list[Reading] = []
        for count, study in enumerate(ordered, start=1):
            readings.append(self.read(study, np.asarray(volumes[positions[study]])))
            logger.info("read %d of %d: %s", count, len(ordered), readings[-1].grade)
        return ProbeResult(tuple(readings), tuple(int(truth[reading.study]) for reading in readings))


def main() -> None:
    """Read the tune half of the frozen split and print the matrix, the base rate and the degenerate rate."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--split", type=Path, required=True, help="the frozen study-level split")
    command.parser.add_argument("--labels", type=Path, required=True, help="train.csv, with its ACL column")
    command.parser.add_argument("--images", type=Path, required=True, help="the pixel cache .npy")
    command.parser.add_argument("--index", type=Path, required=True, help="the cache's study-order .json")
    command.parser.add_argument("--out", type=Path, required=True, help="where the per-study readings are written")
    command.parser.add_argument("--seed", type=int, default=0, help="seeds the presentation order only")
    command.parser.add_argument("--model", default=SERVED, help=f"'{SERVED}' for the endpoint, else a HF repo id")
    command.parser.add_argument("--device", default="cpu", help="where locally loaded weights go")
    command.parser.add_argument("--band", type=int, nargs=2, help="first and last+1 slice, instead of all twelve")
    command.parser.add_argument("--columns", type=int, default=4, help="tiles across the montage")
    command.parser.add_argument("--stretch", action="store_true", help="rescale each slice to its own full range")
    options = command.parsed()

    annotated = pd.read_csv(options.labels).dropna(subset=[ACL]).set_index(STUDY)[ACL]
    studies = [study for study in FrozenSplit(options.split).tune() if study in annotated.index]
    reader = Reader.for_model(options.model, options.device)
    montage = Montage(columns=options.columns, band=options.band and tuple(options.band), stretch=options.stretch)
    probe = VisionProbe(reader, montage)
    result = probe.over(studies, ProbeCache(images=options.images, index=options.index), annotated, options.seed)
    result.frame().assign(model=reader.label).to_csv(options.out, index=False)
    logger.info("wrote %s\nmodel=%s\n%s", options.out, reader.label, result.report())


if __name__ == "__main__":
    main()
