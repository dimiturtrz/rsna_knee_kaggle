"""What each member we own loses under a real distribution shift: `kneemri.evaluation.foreign_readout`.

`python -m kneemri.evaluation.foreign_readout --checkpoints a.pt b.pt --labels labels.csv
--foreign D:/kaggle/input/raptor-corpus`

WHY. Every offline number this project owns is read on 909 studies from OUR corpus, through OUR decode,
under OUR label table — the same three things every candidate was fitted under, so an offline seat ADD
asks only "do you add on studies you were built to fit?". Sixteen candidates spanning every axis we own
have priced negative on that ruler while both seats that ever moved the board were out-of-distribution
members. Nothing here has ever measured generalisation. `core.data.corpus.foreign_cache` holds a second
decode of the SAME studies under the SAME labels, so scoring a member on it moves the pixel path alone.

HOW BIG A SHIFT IT IS, MEASURED BEFORE THIS RAN. D205 compared every one of our 120 (slot, slice) planes
against every one of his 44 on 200 studies, against two controls on the same planes: our own 20-slice
decode against our 12-slice one reads median correlation 0.9999, an unrelated patient reads 0.3958, and
his decode against ours reads 0.6517 — **40% of the way from chance to identity**. A Spearman lift of
+0.028 puts ~8% of that gap on windowing and the remaining ~92% on geometry. So this readout has real
dynamic range and it prices SPAN, ORIENTATION and SERIES SELECTION, not interpolation: resampling is
provably ~0 of the difference, which is what licenses the resample below.

HIS 44 IS SLOTTED TOO, WHICH IS THE WHOLE REASON THIS MODULE IS NOT A ONE-LINER. `all_masks.npy` fills at
1.000 on his columns 0-11, 0.973 on 12-21, 1.000 on 22-29, 0.742 on 30-35, 1.000 on 36-43 — five blocks
with two partial ones, which is a slot scheme, not a flat stack. Dumping all 44 into one of our six slots
would have measured "a member that read six acquisitions now reads one", a loss with no decode in it at
all, and reported it as a pixel result.

WHICH BLOCK IS WHICH SLOT. D206 §3 answered it as a per-study argmax and then as its distribution over
200 studies, which is the only form that can show constancy: four of his five blocks land on one of our
slots on 79-93% of studies, and the fifth splits between two of ours that are near-duplicates of each
other. `BLOCKS` below carries the resulting permutation. Both schemes are the same SIZE — his live-block
count is median 5 and so is our live-slot count — so the head is never handed one sequence where it
expects six, and the slot loss this module was originally built to control for is not there to price.

WHAT THE MIDDLE NUMBER IS NOW. With the full splice `kept` serves our own pixels through the SHIFTED code
path, so `slot_cost` must read ~0.0000 and is a plumbing self-check rather than a confound: it is the
same members, the same pixels and the same windows as `native`, reaching the score through `predict`
instead of through the cache. Anything but ~0 means the shifted path is broken and `decode_cost` is not
about pixels at all — which is the one failure that would otherwise be invisible, since a broken slab
still produces a plausible number. `decode_cost` is the pixel result and `drop` equals it.

WHAT THE PIXEL NUMBER PRICES, WORDED BEFORE IT EXISTS. Not sequence selection: the two schemes correspond.
Not resampling: D205's positive control puts our own decode at twenty slices against our own at twelve at
0.9999, so changing slice count is free on this instrument. It prices SPAN and SLICE PLACEMENT inside
corresponding acquisitions — his span runs 20-50% wider at the median (D206 §5), and on a tenth of
studies his two coronal-side blocks cover three to four times our extent. The ledger line therefore reads
"decode + a 20-50% span extension, larger and more variable on the coronal-side blocks", and it reads
that whichever way the number falls.

WHY THE CACHE IS BYPASSED ON THE SHIFTED SIDES. `HoldoutPredictions.cached` answers from
`PredictionCache` and then from `TrainerRecord`, and NEITHER keys on the pixels — they key on the
checkpoint, the window, the precision and the studies, all identical across the two paths. A shifted pass
routed through it would return the NATIVE probabilities in milliseconds, with a log line saying it read
from the run's own holdout record, and every drop would come back at exactly 0.0000.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.data.corpus.foreign_cache import ForeignCache
from core.data.labels.targets import Targets
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.evaluation.foreign_blocks import BLOCKS, KEPT, MappedSlab
from kneemri.evaluation.foreign_slots import SlotsKept
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.inference.norm_adaptation import ADAPT_LR, ADAPT_STEPS, CardinalityAwareLN
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.models.backbone import ENCODERS
from kneemri.models.model import StudyModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShiftedReading:
    """One member scored three ways on one holdout: as the ledger has it, slot-stripped, and shifted."""

    name: str
    native: float
    kept: float
    foreign: float

    @property
    def slot_cost(self) -> float:
        """Our own pixels through the shifted path: ~0 when every slot is in `KEPT`, else the withheld slots' price."""
        return self.kept - self.native

    @property
    def decode_cost(self) -> float:
        """What HIS pixel path costs once that is already paid — the only pixel number in this module."""
        return self.foreign - self.kept

    @property
    def drop(self) -> float:
        """What a deployment on his path would pay in total. Not evidence about pixels; see `decode_cost`."""
        return self.foreign - self.native

    def logged(self) -> str:
        """One line, with the control beside the delta rather than folded into it."""
        return (
            f"{self.name}: native {self.native:.4f} kept {self.kept:.4f} foreign {self.foreign:.4f} "
            f"| slots {self.slot_cost:+.4f} decode {self.decode_cost:+.4f} total {self.drop:+.4f}"
        )


@dataclass(frozen=True)
class ScoredPass:
    """One pass's mean label score beside the per-study probabilities it was computed from.

    The two travel together because the mean is a POPULATION statistic — an AUC has no per-study value —
    so a subset reading cannot be recovered from it afterwards. Handing back only the float is what makes
    every later split another run on the card.
    """

    mean: float
    probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class PerStudyDump:
    """Every member's raw per-study probabilities on both shifted passes, so a SUBSET costs no card.

    The metric is an AUC and an AUC has no per-study value, so every question of the form *does this
    member's toll concentrate on the studies where X* is a re-score over a SUBSET of the same
    probabilities. Written once they turn each such question into arithmetic; unwritten, each one is
    another readout on the card with the encoders reloaded.

    The first buyer is the padding hypothesis (D269 §c). `crop_coverage` reads only ~40% of series wide
    enough to fill a 160 mm field, so `crop160_span`'s lower toll is either concentrated in the studies
    that fill it — which makes it an EXTENT effect — or it is flat across both subsets, which makes the
    extent story wrong whatever the pooled number says. That split is a subset AUC over these arrays and
    nothing else.

    The native pass is deliberately absent: it comes from the cache, which does not key on the pixels, so
    a native column here would be the one array in the file that is not about the decode being priced.
    """

    path: Path
    studies: list[str]
    passes: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def add(self, name: str, kept: NDArray[np.float64], foreign: NDArray[np.float64]) -> None:
        """One member's two shifted passes, keyed by member and pass so a reader needs no column order."""
        self.passes[f"{name}::kept"] = kept
        self.passes[f"{name}::foreign"] = foreign

    def written(self, truth: NDArray[np.float32]) -> None:
        """The file, with the study ids and the truth beside the probabilities.

        All three travel together because a subset reading joins on the study and re-scores against the
        truth: a file holding probabilities alone would need a join nobody could check afterwards.
        """
        # The ignore is the stub's `allow_pickle` keyword: any typed `**arrays` reads as an attempt to
        # pass an array there. The values are arrays and nothing else.
        arrays: dict[str, NDArray[np.float64] | NDArray[np.float32] | NDArray[np.str_]] = {
            "studies": np.array(self.studies),
            "truth": truth,
            **self.passes,
        }
        np.savez_compressed(self.path, **arrays)  # pyrefly: ignore[bad-argument-type]
        logger.info("per-study probabilities for %d members -> %s", len(self.passes) // 2, self.path)


@dataclass(frozen=True)
class ShiftPair:
    """The two shifted slabs for one member — never one without the other, since one alone has no control."""

    ours: HoldoutSlab
    his: HoldoutSlab


class ForeignReadout:
    """Scores one member on our pixels, on our pixels stripped to the corresponded slots, and on his."""

    def __init__(self, predictions: HoldoutPredictions, scores: LabelScores, dump: PerStudyDump | None = None) -> None:
        self.predictions = predictions
        self.scores = scores
        self.dump = dump

    def rebuilt(self, member: Member) -> StudyModel:  # devtools-ignore: test-mirror (needs an encoder download)
        """The member's own network. Both shifted passes miss every cache, so it is loaded once."""
        encoder = ENCODERS[member.variant]
        return self.predictions.load(member.file, encoder.load(), encoder.dim, member.slot_count)

    def scored(  # devtools-ignore: test-mirror (one line over `rebuilt`, which needs an encoder download)
        self, member: Member, slab: HoldoutSlab, truth: NDArray[np.float32], *, overlap: bool
    ) -> ScoredPass:
        """This member's score over the studies this slab serves, WITH the probabilities it was read off.

        Goes straight to `predict`: `cached` keys on the checkpoint and the studies and NOT on the
        pixels, so it would answer a shifted pass with the native probabilities and report exactly zero.

        Returns both halves rather than the mean alone because the mean is a population statistic and
        cannot be split afterwards; see `PerStudyDump` for what the array buys.
        """
        window = member.window(slab.depth, overlap=overlap)
        probabilities = self.predictions.predict(self.rebuilt(member), slab, window, member.precision)
        return ScoredPass(self.scores.mean(probabilities, truth), probabilities)

    def read(
        self,
        member: Member,
        native: float,
        pair: ShiftPair,
        truth: NDArray[np.float32],
        *,
        overlap: bool = False,
    ) -> ShiftedReading:
        """The three numbers for one member, in the order they get cheaper to explain."""
        kept = self.scored(member, pair.ours, truth, overlap=overlap)
        foreign = self.scored(member, pair.his, truth, overlap=overlap)
        if self.dump is not None:
            self.dump.add(member.file.name, kept.probabilities, foreign.probabilities)
        reading = ShiftedReading(member.file.name, native, kept.mean, foreign.mean)
        logger.info("%s", reading.logged())
        return reading


def main() -> None:
    """Price every member's generalisation gap under a second decode of the same studies."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="members sharing ONE split")
    command.parser.add_argument("--labels", type=Path, required=True)
    command.parser.add_argument("--foreign", type=Path, required=True, help="the directory holding his decode")
    command.parser.add_argument("--fold", type=int, default=0)
    command.parser.add_argument("--folds", type=int, default=5)
    command.parser.add_argument("--split-seed", type=int, default=2026)
    command.parser.add_argument("--overlap", action="store_true", help="every window start, not just disjoint ones")
    command.parser.add_argument("--device", default=VisibleCards().device().type, choices=("cuda", "cpu"))
    command.parser.add_argument(
        "--per-study",
        type=Path,
        help="also write the raw per-study probabilities of both shifted passes here (.npz), so a later "
        "SUBSET reading — coverage, vendor, anything — is arithmetic instead of another run on the card",
    )
    command.parser.add_argument(
        "--adapt",
        action="store_true",
        help="tune each study's LayerNorm affines to it before reading it (CARVE), then undo it",
    )
    # THE BUDGET IS A FLAG BECAUSE THE DEFAULT ONE CANNOT MOVE A READING. Adam displaces a parameter by
    # about the learning rate per step whatever the gradient, so the shipped 2 steps at 1e-5 move
    # LayerNorm affines of order 1.0 by ~2e-5 — a 0.002% perturbation. Measured 2026-09-09: against a
    # same-hour control the mean decode cost moved by 0.000033, three orders inside the arm's own NULL
    # band. That is not a reading of CARVE, it is a reading of the budget, and a constant cannot be
    # swept. M274 asks the researcher what the paper's optimiser and step count actually are; until it
    # answers, these two flags are how the question gets asked of the card instead.
    command.parser.add_argument(
        "--adapt-lr", type=float, default=ADAPT_LR, help="learning rate for the per-study adapt steps"
    )
    command.parser.add_argument(
        "--adapt-steps", type=int, default=ADAPT_STEPS, help="gradient steps taken per study before reading it"
    )
    arguments = command.parsed()

    root = command.rooted(arguments)
    adaptation = CardinalityAwareLN(lr=arguments.adapt_lr, steps=arguments.adapt_steps) if arguments.adapt else None
    if adaptation is not None:
        logger.info(
            "adapting: %d step(s) at lr %g, so |dtheta| ~ %g",
            adaptation.steps,
            adaptation.lr,
            adaptation.steps * adaptation.lr,
        )
    predictions = HoldoutPredictions(torch.device(arguments.device), adaptation=adaptation)
    scores = LabelScores()

    readings = MemberReadings(Member.held_out_from(arguments.fold, arguments.checkpoints), root.processed())
    corpus, reference = readings.reference()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.train_table()), reference), arguments.fold)
    truth = Targets.load(arguments.labels, root.train_table(), reference).scores[holdout]

    # His pixels for exactly the holdout studies, in the reference order, so his rows and the truth
    # vector line up by construction rather than by a join nobody checks.
    held_out = [reference[index] for index in holdout]
    foreign, _ = ForeignCache(arguments.foreign).load(held_out)
    positions = np.arange(len(held_out), dtype=np.int64)
    logger.info("his decode: %d studies, %d blocks spliced into slots %s", len(held_out), len(BLOCKS), list(KEPT))

    # The native side is the ledger number and is allowed to come from the cache; the two shifted sides
    # are not, because nothing in this key mentions the pixels.
    cache = PredictionCache(root.processed() / "holdout_predictions", held_out, predictions.views, refresh=False)

    results: list[ShiftedReading] = []
    dump = PerStudyDump(arguments.per_study, held_out) if arguments.per_study else None
    readout = ForeignReadout(predictions, scores, dump)
    with Wakelock():
        for member, slab, held in readings.slabs(holdout, reference, corpus):
            window = member.window(held.slices, overlap=arguments.overlap)
            native = scores.mean(predictions.cached(member, slab, cache, window), truth)
            pair = ShiftPair(
                SlotsKept(held, holdout, KEPT),
                MappedSlab(foreign, positions, member.slot_count, held.slices, BLOCKS),
            )
            results.append(readout.read(member, native, pair, truth, overlap=arguments.overlap))
    if dump is not None:
        dump.written(truth)

    command.report("\n".join(reading.logged() for reading in results))


if __name__ == "__main__":
    main()
