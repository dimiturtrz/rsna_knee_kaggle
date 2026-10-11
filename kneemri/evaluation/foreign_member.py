"""Serve a FOREIGN checkpoint through ITS OWN decode, and record the result as one of our members.

`python -m kneemri.evaluation.foreign_member --weights raptor_ft_coatnet_v4_full.pt --name raptor_v4`

WHY THIS EXISTS, IN ONE NUMBER. The same published checkpoint reads **0.9167** through their decode and
**0.8819** through our best cache (`336px_20sl_0.1-0.9band`) — a **-0.0348** shortfall that is entirely
the feed, since the weights are byte-identical on both sides. Every foreign member this project has
priced was priced through that shortfall, so every one of them was priced against a crippled version of
itself. This module removes the only difference: their DICOM reader, their slot table, their mm-based
crop, their window count, their normalisation.

WHAT IT BUYS, AND WHY IT IS WORTH A MODULE. `2026-08-27_the-pool-is-82-percent-one-number.md` measures
81.7% of what separates our 105 members as ONE quality scalar — the pool has almost nothing left to
trade, and the under-searched axis is a genuinely different representation. A CoAtNet with per-finding
attention over 42 windows is that: not our trunk, not our decode, not our head. Whether it seats is a
question only a record on our holdout can answer, and this is what writes one.

IT VENDORS NOTHING. Their script is fetched under `external/` (gitignored) and imported BY PATH, never
copied. The two functions in it that know about Kaggle — `find_test_root` and `find_weight_file` — are
the two this module does not call; everything else takes plain arguments, so passing our paths is the
whole of the adaptation. The fetch, which is the pin (a Kaggle kernel has no commit to name, so the
version is the last-run stamp the list command prints — 2026-08-23 04:03:58 UTC, 85 votes, when this
was written):

    kaggle kernels pull dreaddevelopment/knee-mri-twelve-findings-from-a-single-model -p external/raptor
    kaggle datasets download dreaddevelopment/raptor-knee-widedense -p $DATA/external/raptor --unzip

Neither costs a submission slot. The weights are theirs and stay out of the repo with every other
dataset; `--weights` names the file inside that download.

THE COLUMN ORDER IS CHECKED, NOT ASSUMED. Their `LAB` and our `core.findings.TARGETS` are currently the
same twelve strings in the same order, which is exactly the coincidence that gets silently broken by
someone else's edit. `Decode.checked` refuses a mismatch rather than writing a member that is wrong on
every finding instead of loudly wrong on one.

AND THE DENOMINATOR IS REPORTED. Their `main` catches a per-study failure and leaves 0.5 in the row,
which scores as a coin flip and looks like a weak member rather than a broken read. Here a fallback is
counted, logged, and above `FALLBACK_SHARE` refuses to write at all — a rate computed over the studies
that happened to decode is not a rate over the holdout.
"""

import csv
import importlib.util
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import STUDY, TARGETS
from kneemri.runs.holdout_record import FOREIGN, UNKNOWN_EPOCH, HoldoutRecord

logger = logging.getLogger(__name__)

Records = Mapping[str, Sequence[Mapping[str, object]]]
"""Their `ser_records`: the series rows of each study, keyed by study UID, as `build_study` reads them.

READ WITH `csv`, NOT PANDAS, AND THAT IS WHAT MAKES THIS TYPE HONEST. A `groupby(...).to_dict("records")`
is keyed by whatever pandas inferred and rowed by `Hashable` — an all-digit UID comes back an integer
while their reader indexes it as a string — so a pandas seam here would need either a laundering step
that cannot be typed without `Any` or a cast that asserts what it does not check. `csv.DictReader` hands
back `str` keys by construction, which is the same guarantee with nothing to enforce.

`Mapping`/`Sequence` rather than `dict`/`list` because a mapping's VALUE is covariant where a dict's is
not: the rows are `dict[str, str]` off the reader, and this is what lets them arrive unconverted."""

EXTERNAL = Path("external/raptor")
"""Where their checkout sits, relative to the repo root — gitignored, per the external-deps rule."""

SCRIPT = "knee-mri-twelve-findings-from-a-single-model.py"
"""Their published inference script, imported by path. A hyphenated stem cannot be `import`ed by name,
which is the reason this goes through `importlib.util.spec_from_file_location` rather than a plain
import — not a preference."""

FALLBACK_SHARE = 0.02
"""The share of studies allowed to fail decoding before the record is refused rather than written.

Their loop answers a failed study with 0.5 on every finding, which is not an error anywhere downstream:
it is a member that reads slightly weak. Two percent of 909 studies is eighteen, enough to be a bad day
on the disk and not enough to move an AUC past the 0.0062 arm floor. Above it the number would be about
the reader rather than the weights, and a member priced on that is the harness bug this project keeps
paying for."""


@dataclass(frozen=True)
class Decode:
    """Their inference path, loaded from the checkout — the module, not a copy of what is in it."""

    script: Path

    def loaded(self) -> ModuleType:
        """Import their script by path and hand back the module.

        By PATH because the file's stem is hyphenated and cannot be spelled as an import, and by
        `spec_from_file_location` rather than an `exec` of the source because their module has a
        `if __name__ == "__main__"` guard that only a real module name honours.
        """
        specification = importlib.util.spec_from_file_location("raptor_decode", self.script)
        if specification is None or specification.loader is None:
            message = f"not an importable module: {self.script}"
            raise FileNotFoundError(message)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    @staticmethod
    def checked(module: ModuleType) -> ModuleType:
        """Refuse a module whose label order is not ours — the one mismatch nothing downstream can see.

        A member fitted under one order and served under another is wrong on all twelve findings and
        still writes a well-formed file that scores near chance, which reads as a weak member. Their
        `LAB` and our `TARGETS` agree today; this is what notices when an update to either ends that.
        """
        theirs = list(module.LAB)
        if theirs != list(TARGETS):
            message = f"label order differs: theirs {theirs} against ours {list(TARGETS)}"
            raise ValueError(message)
        return module


@dataclass(frozen=True)
class ForeignMember:
    """One foreign checkpoint, served through its own decode over studies we choose."""

    module: ModuleType
    weights: Path
    series_root: Path
    device: str = "cuda"

    def predicted(
        self, studies: list[str], records: Records
    ) -> tuple[Float[NDArray[np.float64], "studies labels"], int]:
        """Probabilities for each study, and how many of them fell back to 0.5.

        The fallback COUNT is returned rather than logged and dropped, because a caller that cannot see
        it cannot tell a weak member from an unread one — and 0.5 rows are indistinguishable from a
        genuinely uncertain model in every number computed after this point.
        """
        module = Decode.checked(self.module)
        try:
            reader = module._make_reader()  # noqa: SLF001 - their module's own entry point, named private
        except ImportError as missing:
            # THEIR READER OPENS `import pydicom, cv2`, and cv2 was not in this venv. Unhandled, that
            # surfaces as a bare ModuleNotFoundError from inside someone else's file — after the DICOM
            # tree and a 292 MB checkpoint are already on the way. The install is one line and it is
            # cheaper to say it here than to find it in a log.
            message = f"their decode needs a library this venv lacks ({missing.name}) — `uv sync --extra foreign`"
            raise ImportError(message) from missing
        arm = module.ARMS[0]
        model, resolution = module.load_model(str(self.weights), arm["arch"], arm["res"], self.device)
        probabilities = np.full((len(studies), len(TARGETS)), 0.5, np.float64)
        fallbacks = 0
        for position, study in enumerate(studies):
            try:
                volume, mask = module.build_study(study, records, str(self.series_root), reader)
                windows = module.eval_windows(volume, mask, k=module.K_EVAL, res=resolution, norm=module.NORM)
                probabilities[position] = module.infer_probs(model, windows, self.device)
            except (KeyError, ValueError, RuntimeError, OSError) as failure:
                fallbacks += 1
                logger.warning("study %s fell back to 0.5 (%s: %s)", study[:16], type(failure).__name__, failure)
            if (position + 1) % 100 == 0:
                logger.info("%d/%d studies", position + 1, len(studies))
        return probabilities, fallbacks

    def record(
        self,
        destination: Path,
        studies: list[str],
        truth: Float[NDArray[np.float32], "studies labels"],
        records: Records,
    ) -> int:
        """Write this checkpoint's holdout record, or refuse if too much of it is fallback. Returns the count.

        `UNKNOWN_EPOCH` because a foreign checkpoint has no epoch of OURS, and a plausible integer is
        what `holdout_record` already documents as the mis-join waiting to happen. `FOREIGN` as the
        source for the same reason it distinguishes `DERIVED`: nothing downstream can tell by looking
        that this array came through someone else's reader.
        """
        probabilities, fallbacks = self.predicted(studies, records)
        share = fallbacks / len(studies) if studies else 1.0
        if share > FALLBACK_SHARE:
            message = f"{fallbacks} of {len(studies)} studies failed to decode ({share:.1%}) — refusing to write"
            raise RuntimeError(message)
        logger.info("%d of %d studies fell back (%.1f%%)", fallbacks, len(studies), 100 * share)
        HoldoutRecord(destination).write(studies, probabilities, truth, UNKNOWN_EPOCH, source=FOREIGN)
        return fallbacks


def main() -> None:
    """Serve a foreign checkpoint over the studies of an existing record, and write one beside it."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--weights", required=True, help="a checkpoint under the external checkout")
    command.parser.add_argument("--name", required=True, help="the member name to write the record under")
    command.parser.add_argument("--like", required=True, help="a run whose .holdout.npz names the studies")
    command.parser.add_argument("--device", default="cuda", help="the device their model is served on")
    arguments = command.parsed()

    root = command.rooted(arguments)
    processed, raw = root.processed(), root.raw()
    reading = HoldoutRecord(processed / f"{arguments.like}.pt").required()

    grouped: dict[str, list[Mapping[str, object]]] = {}
    with (raw / "train_series.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row[STUDY], []).append(row)

    member = ForeignMember(
        module=Decode(EXTERNAL / SCRIPT).loaded(),
        weights=Path(arguments.weights),
        series_root=raw / "train_series",
        device=arguments.device,
    )
    fallbacks = member.record(processed / f"{arguments.name}.pt", reading.studies, reading.truth, grouped)
    logger.info("wrote %s over %d studies, %d fallback", arguments.name, len(reading.studies), fallbacks)


if __name__ == "__main__":
    main()
