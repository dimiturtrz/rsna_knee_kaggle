"""Serve the mattiaangeli DINOv3 fold checkpoints through THEIR OWN kernel code.

`python -m kneemri.evaluation.vendor_folds --name dinov3_f0 --like holdsolo --folds 0`

WHY IT IS NOT `kneemri.inference.foreign_member`. That module rebuilds a published member from fields
carried in the checkpoint (`arch`, `res`, `lab`), which works for archives whose architecture is a timm
name plus a linear head. These folds are not that: their model is CODE — a slot-token wrapper around a
ViT, a pool zoo, an optional depth-compress stem — defined in the kernel that trained them and nowhere
in the `.pt`. Rebuilding it here would be hand-vendoring someone else's architecture and pricing the
result as if it were theirs.

WHY IT IS NOT `kneemri.evaluation.foreign_member` EITHER. That one imports a vendor script by path,
which is the right shape and the wrong file: it is hard-wired to raptor's module-level API, and this
vendor ships a KERNEL, not a library — its module level asserts `/kaggle/input` exists and then runs a
full submission. So the adaptation here is a marker-sliced `exec` of the three regions that define the
serving path (constants, decode, model+predict), with `CKPT`/`SERIES_ROOT`/`DEV` injected in place of
the Kaggle discovery block, and their own driver loop replaced by ours over the studies of one of our
holdout records. Their pixels, their crop, their slot table, their weights; our study list, our ruler.

BOUNDARIES ARE MARKERS, NEVER LINE NUMBERS. The checkout is a pull of a live kernel and will move under
us; a slice by line number would silently take the wrong code, which is the failure this project pays
for most. A missing marker refuses.

WHAT IT DOES NOT DO. It never touches the region before `import timm` — that half of the file is their
legacy pipeline, training included, and executing it would spend the card on someone else's run.

TRUST DELTA, STATED. Their loader is `torch.load(..., weights_only=False)` because the checkpoints carry
a `cfg` dict; our own published-archive loader deliberately uses `weights_only=True`. This module runs
their line, so the checkout and the weights are trusted code. That is a deliberate, named exception.

THE FETCH, WHICH IS THE PIN (a kernel has no commit; the version is the last-run stamp):

    kaggle kernels pull mattiaangeli/bend-the-knee-to-dinov3-ensembled -p external/dinov3folds
    kaggle datasets download mattiaangeli/knee-mri-fold-weights -p $DATA/external/kaggle_foreign --unzip

Neither costs a submission slot. Licence: the weights are DINOv3-derived and ship under the DINOv3
License (Meta Platforms, 2025) with attribution — the dataset's CC0 metadata is not the operative
licence, and citing CC0 for them would be wrong.
"""

import csv
import importlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.runs.holdout_record import FOREIGN, UNKNOWN_EPOCH, HoldoutRecord

logger = logging.getLogger(__name__)

Records = Mapping[str, Sequence[Mapping[str, object]]]
"""Their `by`: the series rows of each study keyed by study UID, as `build_study` reads them.

`Fat_Suppression` MUST arrive as an int. Their slot table is `('Sagittal', 1)` and their selector is
`rows.Fat_Suppression == fs`, so a CSV read that leaves it `'1'` matches nothing, every slot is absent,
every study returns an empty mask, and the member reads as a coin flip rather than as a broken feed —
the exact shape of defect this repo keeps paying for. `SeriesTable.records` coerces it and nothing else."""

EXTERNAL = Path("external/dinov3folds")
"""Their kernel checkout, relative to the repo root — gitignored, per the external-deps rule."""

SCRIPT = "bend_the_knee_to_dinov3_ensembled.py"
"""The code cells of their pulled notebook, concatenated in order (`nbformat` is not a dependency here;
the extraction is `json.loads(...)["cells"]`). A notebook cannot be sliced by marker; a script can."""

WEIGHTS = "kaggle_foreign/mattiaangeli_dinov3"
"""Where their fold checkpoints sit under the external data root."""

SEGMENTS = (
    ("import timm\n", "def _find_dir("),
    ("def ordered_files(", "sub_df = pd.read_csv("),
    ("N_SLOT_TYPES, MASK_IDX = (6, 0)", "# Macro ROC-AUC depends on ordering"),
)
"""The three executable regions, in order: serving constants · DICOM decode · model, weights, predict.

What sits BETWEEN them is what this module replaces. Between one and two: `/kaggle/input` discovery and
the asserts that go with it. Between two and three: their competition-CSV read, which decides WHICH
studies get served — the one decision that must be ours, since the whole point is a record on our
holdout."""

PRELUDE = ("warnings", "os", "math", "time", "gc", "cv2", "pydicom", "numpy", "pandas")
"""Modules their cells assume are already imported, because the cells that imported them are in the
legacy half we refuse to execute."""

ALIASES = {"numpy": "np", "pandas": "pd"}
"""Their cells spell two of those by the conventional short name."""

FALLBACK_SHARE = 0.02
"""The share of studies allowed to fail before the record is refused rather than written.

A study whose every slot is missing comes back all-NaN and would otherwise be written as 0.5 on twelve
findings, which reads as a weak member rather than an unread one. Two percent of 909 is eighteen —
a bad day on the disk, and below the arm floor."""

CHUNK = 24
"""Studies decoded before a forward pass. Theirs is 48 across four worker processes; ours is serial, so
a smaller block bounds the `uint8` buffer (24 x 6 x 16 x 336^2 is about 780 MB) without costing throughput."""

FALLBACK_PROBABILITY = 0.5
RANK_FLOOR = 1
"""A one-study rank span would divide by zero; their own combination clamps the same way."""


@dataclass(frozen=True)
class SeriesTable:
    """The series metadata their decode reads, as a file on disk."""

    path: Path

    def records(self, studies: Sequence[str]) -> Records:
        """The rows of that table, grouped by study and restricted to the studies we are serving."""
        grouped: dict[str, list[Mapping[str, object]]] = {}
        wanted = set(studies)
        with self.path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                study = row["StudyInstanceUID"]
                if study not in wanted:
                    continue
                typed: dict[str, object] = dict(row)
                typed["Fat_Suppression"] = int(row["Fat_Suppression"])
                grouped.setdefault(study, []).append(typed)
        return grouped


@dataclass(frozen=True)
class VendorScript:
    """Their kernel as text, sliced by the markers that bound its serving path."""

    path: Path

    def segment(self, opening: str, closing: str) -> str:
        """The source between two markers, refusing rather than guessing when either has moved."""
        text = self.path.read_text(encoding="utf-8")
        start = text.find(opening)
        if start < 0:
            message = f"marker not in {self.path}: {opening!r}"
            raise ValueError(message)
        stop = text.find(closing, start)
        if stop < 0:
            message = f"closing marker not after {opening!r} in {self.path}: {closing!r}"
            raise ValueError(message)
        return text[start:stop]


@dataclass(frozen=True)
class VendorSpace:
    """The namespace their serving code runs in, with our paths where their Kaggle discovery was."""

    script: VendorScript
    checkpoints: Path
    series_root: Path
    device: str

    def built(self) -> dict[str, object]:
        """Execute the three segments and hand back the live namespace (their globals, our roots)."""
        space: dict[str, object] = {"__name__": "vendor_dinov3_folds"}
        for name in PRELUDE:
            space[ALIASES.get(name, name)] = importlib.import_module(name)
        space["Path"] = Path
        constants, decode, serving = (self.script.segment(*markers) for markers in SEGMENTS)
        exec(compile(constants, str(self.script.path), "exec"), space)
        space["CKPT"] = self.checkpoints
        space["SERIES_ROOT"] = self.series_root
        space["DEV"] = self.device
        exec(compile(decode, str(self.script.path), "exec"), space)
        exec(compile(serving, str(self.script.path), "exec"), space)
        labels = list(cast("Sequence[str]", space["LABELS"]))
        if labels != list(TARGETS):
            message = f"label order differs: theirs {labels} against ours {list(TARGETS)}"
            raise ValueError(message)
        return space


@dataclass(frozen=True)
class VendorFolds:
    """Their fold models, served over studies we choose."""

    space: dict[str, object]

    def kept(self, folds: Sequence[int]) -> int:
        """Narrow the loaded models to the folds asked for; returns how many are left.

        A solo instrument prices ONE foreign trunk in our kernel — five folds rank-averaged is a
        different entry answering a different question, and the two are one flag apart."""
        models = list(cast("Sequence[object]", self.space["models"]))
        if not folds:
            return len(models)
        outside = [fold for fold in folds if not 0 <= fold < len(models)]
        if outside:
            message = f"no such fold {outside} among {len(models)} checkpoints"
            raise IndexError(message)
        self.space["models"] = [models[fold] for fold in folds]
        return len(folds)

    @staticmethod
    def combined(
        raw: Float[NDArray[np.float32], "folds studies labels"],
    ) -> Float[NDArray[np.float64], "studies labels"]:
        """One number per study and finding: their probability for a single fold, their rank mean for several.

        Macro AUC is ordering, so a fold's probability SCALE would otherwise dominate a plain mean —
        their own combination ranks each fold within the served set first. For one fold the transform is
        monotone and therefore free on AUC, so the probabilities are kept instead: they are what a later
        blend with our members expects."""
        if raw.shape[0] == 1:
            return raw[0].astype(np.float64)
        readable = np.isfinite(raw).all(axis=(0, 2))
        mean = np.zeros(raw.shape[1:], np.float64)
        for fold in raw:
            ordinal = fold[readable].argsort(0).argsort(0).astype(np.float64)
            mean[readable] += ordinal / max(int(readable.sum()) - 1, RANK_FLOOR)
        mean /= raw.shape[0]
        mean[~readable] = np.nan
        return mean

    def predicted(
        self, studies: Sequence[str], records: Records
    ) -> tuple[Float[NDArray[np.float64], "studies labels"], int]:
        """Probabilities for each study, and how many of them could not be read.

        The count is returned rather than logged and dropped: a 0.5 row is indistinguishable from an
        uncertain model in every number computed after this point."""
        space = self.space
        build_study = cast("object", space["build_study"])
        predict = cast("object", space["predict"])
        slots, slices, size = (int(cast("int", space[key])) for key in ("N_SLOT", "N_SLICE", "SIZE"))
        folds = len(cast("Sequence[object]", space["models"]))
        raw = np.full((folds, len(studies), len(TARGETS)), np.nan, np.float32)
        for start in range(0, len(studies), CHUNK):
            block = studies[start : start + CHUNK]
            images = np.zeros((len(block), slots, slices, size, size), np.uint8)
            masks = np.zeros((len(block), slots), np.uint8)
            for index, study in enumerate(block):
                try:
                    _, volume, mask = build_study((index, study, list(records.get(study, []))))  # type: ignore[operator]
                except (KeyError, ValueError, RuntimeError, OSError) as failure:
                    logger.warning("study %s failed to decode (%s)", study[:16], type(failure).__name__)
                    continue
                images[index], masks[index] = volume, mask
            raw[:, start : start + len(block)] = predict(images, masks)  # type: ignore[operator]
            logger.info("%d/%d studies", min(start + CHUNK, len(studies)), len(studies))
        combined = self.combined(raw)
        readable = np.isfinite(combined).all(axis=1)
        probabilities = np.full((len(studies), len(TARGETS)), FALLBACK_PROBABILITY, np.float64)
        probabilities[readable] = combined[readable]
        return probabilities, int((~readable).sum())

    def record(
        self,
        destination: Path,
        studies: Sequence[str],
        truth: Float[NDArray[np.float32], "studies labels"],
        records: Records,
    ) -> int:
        """Write this vendor member's holdout record, or refuse if too much of it is fallback."""
        probabilities, fallbacks = self.predicted(studies, records)
        share = fallbacks / len(studies) if studies else 1.0
        if share > FALLBACK_SHARE:
            message = f"{fallbacks} of {len(studies)} studies failed to decode ({share:.1%}) — refusing to write"
            raise RuntimeError(message)
        logger.info("%d of %d studies fell back (%.1f%%)", fallbacks, len(studies), 100 * share)
        HoldoutRecord(destination).write(list(studies), probabilities, truth, UNKNOWN_EPOCH, source=FOREIGN)
        return fallbacks


def main() -> None:  # devtools-ignore: test-mirror
    """Serve the vendor folds over the studies of an existing record, and write one beside it.

    Unmirrored deliberately: every line of it is path resolution against the real data root and the
    gitignored vendor checkout, which is precisely what a unit test may not touch. The pieces it wires
    are each covered; the wiring is proven by the smoke (`--limit 8`)."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--name", required=True, help="the member name to write the record under")
    command.parser.add_argument("--like", required=True, help="a run whose holdout record names the studies")
    command.parser.add_argument("--folds", default="", help="comma-separated fold indices; empty means all")
    command.parser.add_argument("--device", default="cuda", help="the device their models are served on")
    command.parser.add_argument("--limit", type=int, default=0, help="serve only the first N studies (a smoke)")
    arguments = command.parsed()

    root = command.rooted(arguments)
    models, raw_root, external = root.models(), root.raw(), root.kind("external")
    reading = HoldoutRecord(models / f"{arguments.like}.pt").read()
    if reading is None:
        message = f"no holdout record for {arguments.like} — nothing names the studies to score"
        raise FileNotFoundError(message)

    studies, truth = list(reading.studies), reading.truth
    if arguments.limit:
        studies, truth = studies[: arguments.limit], truth[: arguments.limit]

    space = VendorSpace(
        script=VendorScript(EXTERNAL / SCRIPT),
        checkpoints=external / WEIGHTS,
        series_root=raw_root / "train_series",
        device=arguments.device,
    ).built()
    member = VendorFolds(space)
    chosen = [int(part) for part in arguments.folds.split(",") if part.strip()]
    logger.info("serving %d fold(s) over %d studies", member.kept(chosen), len(studies))

    records = SeriesTable(raw_root / "train_series.csv").records(studies)
    fallbacks = member.record(models / f"{arguments.name}.pt", studies, truth, records)
    logger.info("wrote %s over %d studies, %d unreadable", arguments.name, len(studies), fallbacks)


if __name__ == "__main__":
    main()
