"""foreignseat — the champion entry with one seat added: a FOREIGN family, five folds, someone else's decode.

WHAT IS BEING PRICED. Not a reseating and not a reweighting of anything that already sits. The
champion's six members keep their weights, the raptor member keeps the 1/3 it shipped at, the vote is
the same rank-space join. The one change is a NEW SEAT: mattiaangeli's five released DINOv3 fold
checkpoints, served through their own lifted program, entering at the project's standard seat share.

WHY THIS IS MECHANISM AND NOT SEATING. The member axis was closed by seven board flats, and every one
of those flats reseated members drawn from the SAME family — our own recipes, our own decode, our own
geometry. This seat is the first that is architecturally foreign end to end: ViT-small DINOv3 at 336 px
over 16 slices with a cond-token slot embedding and an xcodex pool, none of which any seated member has.

THE OFFLINE NUMBER, AND WHY IT IS A FLOOR. Their five folds were served on our 909 holdout studies
fold-matched — every study scored by the ONE fold that never saw it, zero contaminated cells, their
split being md5(Report_text)[:8] % 5, which partitions our corpus too. Pooled 0.8812. As seat eight on
greedyseat_v1's seven (board 0.936, the largest entry all of whose members can be priced): the roster
of seven reads 0.8763, and with the foreign column 0.8854 — +0.0091, band [+0.0080, +0.0103], which
excludes zero and clears the pre-registered 0.0045 bar by two. It is a FLOOR twice over: the offline
reading is ONE fold per study where the shipped form rank-means five, and the roster it was measured on
scores 0.936 where this entry's scores 0.944.

WHAT COULD NOT BE MEASURED, SAID PLAINLY. The champion's own seven cannot be priced offline: five of
its members are --fold -1 and hold nothing out, so seat_value sees two of seven. The +0.0091 is honest
about greedyseat_v1 and is being transferred to a roster it was not measured on. That transfer is this
entry's risk, and it is named before the number rather than after it.

THE SEAT SHARE IS THE CONVENTION, NOT A FIT. RaptorSeat.fleet already says what a new foreign member is
worth on arrival: one seat of the widened roster, 1 / (6 + n_foreign). With two foreign members that is
0.125, and the two shares already there are scaled by 7/8 so nothing else moves. There is no offline
pairing to fit a weight on, and fitting one on the board costs a slot per value, so it is named in
advance and left alone.

LICENCE, FOUR FACTS, EACH FROM ITS PRIMARY SOURCE. Their weights sit in the public dataset
mattiaangeli/knee-mri-fold-weights, whose Kaggle metadata reads licenses: [{name: CC0-1.0}] — MOUNTED
here, never redistributed. The DINOv3 lvd1689m weights beneath them are under Meta's DINOv3 Licence,
which grants redistribution of derivative works royalty-free and asks two things in return, both done:
this kernel is Built with DINOv3, and a copy of the Agreement travels with the materials. Competition
Section 2.6b permits external models that are "publicly available and equally accessible to use by all
Participants ... at no cost", which a public CC0 Kaggle dataset is; Section 2.5 explicitly contemplates
"pretrained models with an incompatible license". Their notebook's own licence field is unreadable
through the API and does not arise: their code is lifted into our process, not republished.

WHAT WOULD REFUTE IT.

- 0.950 and above — the foreign seat transfers, the +0.0091 was a floor as claimed, and the next
  question is the census: which OTHER released families are mountable, ranked by distance.
- 0.944-0.949 — the seat is real but smaller than the greedyseat measurement said, which prices the
  transfer itself: a gain measured on a 0.936 roster shrinks on a 0.944 one.
- 0.938-0.943 — the gain does not survive the transfer to a roster of dark members, and the reading to
  take is about seat_value's blindness to five of seven champion seats, not about the family.
- below 0.938 — the served rank-mean is worse than the fold-matched floor predicted, which would mean
  their five folds disagree in a way our own five-fold members do not, and that is measurable offline
  without a slot.
"""

# >>> KERNEL-BOOTSTRAP
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

SERIES = {"train_series", "test_series", "train_images", "test_images"}
"""Competition mounts holding one directory per study. `rglob` walks every one of them."""

KIT_DEPTH = 5
"""How deep under /kaggle/input a mounted kit has ever sat, plus room. A bound rather than a guess: it
turns an unbounded walk into a bounded one even if a mount arrives with a name this does not know."""


def find_kit() -> Path:
    """The directory to put on `sys.path`, found WITHOUT walking the competition data.

    The obvious spelling — `Path("/kaggle/input").rglob("kneemri/__init__.py")` — cost **248 seconds** on
    2026-08-14, four minutes before a single line of ours ran. It descends into the competition mount,
    which holds one directory per study and one per series beneath that, and stats every DICOM on the
    way. The efficiency board scores total runtime, so this was our largest single expense and it was
    spent looking for a file we already knew the neighbourhood of.

    Pruning by name is what removes the cost; the depth bound is what keeps a differently-named mount
    from restoring it silently.
    """
    root = Path("/kaggle/input")
    for parent, directories, _ in os.walk(root):
        here = Path(parent)
        depth = len(here.relative_to(root).parts)
        directories[:] = [] if depth >= KIT_DEPTH else [name for name in directories if name not in SERIES]
        if here.name == "kneemri" and (here / "__init__.py").exists():
            return here.parent
        archive = here / "kneemri_src.zip"
        if archive.exists():
            zipfile.ZipFile(archive).extractall("/tmp/kit")  # our own kit, built by build_kit.py
            return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/__init__.py")))).parent.parent
    raise FileNotFoundError("no kneemri package under /kaggle/input")


def install_wheels(kit: Path) -> None:
    """Install the kit's wheels offline, before anything imports our package.

    The runtime probe checked the libraries the PIPELINE needs (pydicom, timm, transformers…) and found
    them all present — but not the ones our own modules import. `jaxtyping` is not in Kaggle's image and
    `kneemri.models.model` imports it at module top, so the kit needs wheels after all. Probe what you import,
    not only what you plan to use.

    Names what it installs, because a kit shipped without the wheel it needed fails much later, inside
    an import, in a log nobody reads until the slot is spent.
    """
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if not wheels:
        logging.getLogger(__name__).info("no wheels in the kit — assuming every import is already in the image")
        return
    logging.getLogger(__name__).info(f"installing {[wheel.name for wheel in wheels]}")
    subprocess.run(  # noqa: S603 - pip installing wheels we mounted ourselves, fixed argv, no shell
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
        check=True,
    )


def boot() -> Path:
    """Everything a kernel does before its first line of ours can run, as one call.

    Fifty-three kernels opened `main` with these same four statements in this same order, because the
    order is forced: nothing may import `kneemri` until the wheels are installed, and nothing may find
    the wheels until the kit is found. Four lines that MUST NOT be reordered are a step, not a preamble,
    and a step copied fifty-three times is fifty-three places for the order to be got wrong once.

    Returns the kit so a kernel that needs a path into it has one; most only needed it for `sys.path`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))
    return kit


# <<< KERNEL-BOOTSTRAP

FOREIGN_WEIGHT = 6.0 / 13.0
"""The DINOv3 family at SIX seats of the thirteen, which is where its offline curve stops climbing.

Named after a number for once, and the number is `seat_weight` over the seven `greedyseat_v1` members
with `foreign_dinov3_matched` as the eighth: +0.0091 at one seat, +0.0218 at six, and flat-to-falling
above. A weight is a fraction of a final table and a seat is a repetition inside a vote, and the two
are the same quantity: `w` copies beside `n` members is a fraction `w / (n + w)`. At n = 7 - our six
plus the raptor, which is what this kernel serves beside the family - six copies are 6/13 = 0.46.

WHY NOT ONE SEAT, WHICH IS WHAT EVERY OTHER FOREIGN ENTRY SHIPPED. Because one seat was never chosen;
it was the arithmetic of appending. The curve above is the first time the dial has been priced at all
offline, and it says the appending convention was giving away 0.013 of the seat's value. Split-half
answers the obvious objection - the weight chosen on 454 studies earns +0.0232 on the other 455, and
+0.0191 the other way round, against +0.0091 for one seat on the same studies - so the weight survives
being chosen out of sample, and the halves' disagreement (6x against 9x) is why the LOWER of the two
ships rather than the maximum.

WHAT THE OFFLINE NUMBER IS NOT. It is a FLOOR: fold-matched scores each of our 909 studies by the one
of their five folds that never saw it, while the entry serves the rank-mean of all five on a test split
none of them trained on. And it is priced on the `greedyseat_v1` seven, not on the champion's six, five
of which are `--fold -1` fits with no holdout record at all. The board is the only ruler for the served
form, which is what this slot buys.
"""

RAPTOR_WEIGHT = (1.0 - FOREIGN_WEIGHT) * (1.0 / 3.0)
"""What raptorweight_v1 shipped, of what the DINOv3 seat leaves. The raptor DIAL is untouched at 1/3."""

OUR_WEIGHT = (1.0 - FOREIGN_WEIGHT) * (2.0 / 3.0)
"""The champion's own share of the remainder. Reported, never passed - `combined` gives ours the rest."""

MOVED = "calib_span_ff"
"""THE ONE MEMBER THAT MOVES, and the whole design of this entry is the choice of which one.

The six span THREE readings as fitted: `final_ws_cf`, `grade_ladder_cf` and `through_dense_cf` at 20
slices on band 0.1-0.9; `calib_span_ff` alone at 12 slices on 0.1-0.9; `merged_blend_cf` and
`lr4e-5_ws_cf` at 12 slices on 0.2-0.8. The decode bill is the number of DISTINCT readings, so the group
that costs a whole decode to serve ONE member is the cheapest one to retire — and it retires by moving
that member onto the 12-slice band its two nearest neighbours already share, which is one axis (the
band) and not two. Every other pairing moves the depth as well.

So this entry disturbs a SIXTH of the roster where `onegroup_v1` disturbed five sixths, and it is that
ratio, not a new mechanism, that it is shipped to price.
"""

SERVED_SLICES = 12
"""The depth `calib_span_ff` already reads at, so the move is band-only.

THE BASELINE FOR THIS ENTRY IS 0.942 IF THE SEAT IS FLAT, AND WHATEVER 56060097 SCORED IF IT IS NOT.
`onegroup_v1` served these same six as ONE read group and paid a measured -0.002 (0.942 against the
champion's 0.944) to go from three decodes to one, projecting 7.6 h -> 2.15 h. Two decodes is the
intermediate nobody has priced: scaled off the foreignseat v2 kernel log, ours cost ~4.3 h at two groups
against ~2.15 h at one and ~7.6 h at three, and beside raptor (~2.6 h) and the foreign folds (~1.1 h)
that is ~8.0 h of a 9 h session -- inside it with an hour of margin, where three groups reads ~11.3 h and
is refused outright. What this slot buys is how much of the -0.002 comes back when only one member is
re-read.

UNPRICED OFFLINE, AND SAID HERE RATHER THAN IMPLIED: no `calib_span_ff` record exists at 12 slices on
band 0.2-0.8, so this move has no holdout twin. The -0.002 it is trying to recover was measured on the
FULL collapse, and a sixth of a collapse is not arithmetically a sixth of its cost.
"""

SERVED_BAND = (0.2, 0.8)
"""The band `merged_blend_cf` and `lr4e-5_ws_cf` are FITTED at, which is why they are the destination.

The alternative direction -- moving those two onto 0.1-0.9 -- reaches two groups just as well and costs
two re-read members instead of one. Their 0.12-0.88 is theirs, applied inside their lifted program and
untouched by any of this.
"""

DINOV3 = "Built with DINOv3"
"""Meta's DINOv3 Licence asks for this line prominently displayed; the Agreement ships in the kit."""


def main() -> None:
    boot()

    import time  # noqa: PLC0415 - nothing above the bootstrap block is importable

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot
    import torch  # noqa: PLC0415 - same

    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.device import VisibleCards  # noqa: PLC0415 - same
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415 - same
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - same
    from kneemri.inference.ensemble import ReadGroups, Serving, SharedRead  # noqa: PLC0415 - same
    from kneemri.inference.vote import RankAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.foreign_entry import CHECKPOINTS, SOURCE, ForeignEntry  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_run import WORKING, RaptorEntry  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415 - same

    logger = logging.getLogger(__name__)
    logger.info(DINOV3)
    started = time.time()
    mounts = Mounts()
    device = str(VisibleCards().device())
    competition = mounts.one("sample_submission.csv", "competition data")

    # All three producers write a file called submission.csv. Each runs into its own scratch directory;
    # whichever ran last would otherwise be the file Kaggle scores.
    raptor_scratch = WORKING / "raptor"
    raptor_scratch.mkdir(parents=True, exist_ok=True)
    member = RaptorMember.loaded(RaptorEntry.weights(mounts=mounts), device)
    raptor = RaptorEntry(competition, working=raptor_scratch).run(member, device)
    # Dropping the reference is not enough on its own: torch keeps the freed blocks in its caching
    # allocator, and five ViT-smalls plus six of ours is exactly the margin an OOM would eat.
    del member
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    foreign_scratch = WORKING / "foreign"
    foreign_scratch.mkdir(parents=True, exist_ok=True)
    theirs = ForeignEntry(
        competition=competition,
        kernel=mounts.one(SOURCE, "their serving program, carried in our kit") / SOURCE,
        checkpoints=mounts.one("m_f0.pt", f"the {CHECKPOINTS} fold checkpoints"),
        # Their program builds a study's path as SERIES_ROOT / study / series, so the root is the
        # test_series DIRECTORY and not the competition mount. Version 1 passed the mount, every path
        # under it missed, build_study returned its zero-filled default for all three visible studies,
        # and Submission.check refused the constant table - which is the refusal doing its job.
        series_root=competition / "test_series",
        working=foreign_scratch,
    ).run()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    # What the two foreign passes cost, handed to the six, so the budget is the SESSION's remaining time
    # and not a fresh nine hours. Without it every producer aims at the same ceiling and the kernel is
    # killed at the real one with no submission written at all.
    spent = time.time() - started
    logger.info("the two foreign passes cost %.0fs; the six get what is left", spent)
    # RankAverage rather than the champion's LogitAverage, because the seat curve was measured under both
    # and rank wins at every weight (+0.0091 against +0.0053 at one seat, +0.0218 against +0.0184 at six).
    # It is the entry's SECOND change and the reason a LogitAverage twin at the same weight is the
    # pre-registered follow-up: a flat here has to be attributable to one dial or the other.
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(
            vote=RankAverage(),
            budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0)),
            group=ReadGroups({MOVED: SharedRead(SERVED_SLICES, SERVED_BAND)}),
        ),
    ).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    blended = RaptorSeat.combined(ours, [raptor, theirs], [RAPTOR_WEIGHT, FOREIGN_WEIGHT])
    blended.to_csv(WORKING / "submission.csv", index=False)
    logger.info("ours %.4f  raptor %.4f  foreign %.4f", OUR_WEIGHT, RAPTOR_WEIGHT, FOREIGN_WEIGHT)


if __name__ == "__main__":
    main()
