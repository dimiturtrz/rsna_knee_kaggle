"""monotone — the champion's exact entry with one strictly increasing map applied to every column.

**WHAT THIS MEASURES, and it is not a method.** It measures THE METRIC. `comp/README.md` states the
score is the unweighted mean of twelve per-label ROC AUCs, and the same file records, at its line 279,
that this has never been verified: the competition's evaluation page is JavaScript-only and the API does
not serve the formula. Every number this project has produced in six weeks — every offline holdout AUC,
every seat value, every `board_transfer` slope — assumes a rank statistic. Nothing has ever tested it.

**THE TEST.** Take `raptorweight_v1`, the pinned champion, byte for byte: the same six members in the
same order, the same `overlap=False`, the same `LogitAverage`, the same raptor foreign seat at
`FOREIGN_WEIGHT = 1/3`, the same kit. Then apply `x -> x**3` to each of the twelve prediction columns
after the blend. Cubing is strictly increasing over the whole real line, so it preserves the order of
every pair within every column exactly and creates no new ties — while changing the VALUES beyond
recognition (0.9 -> 0.729, 0.5 -> 0.125, 0.1 -> 0.001).

A rank statistic cannot see this. A calibration-sensitive statistic cannot miss it.

**PRE-REGISTERED, and the branches are about the ruler rather than about us.**

- **exactly 0.944** — the metric is rank-based per column, as documented. Every monotone per-column
  recalibration is INERT, which closes calibration, temperature scaling, per-label thresholding and
  probability shaping as a family, permanently, for one slot and no card time. It also retires a
  standing suspicion about the four combiner nulls: they are not calibration artefacts.
- **anything else** — the metric is NOT purely rank-based, and this project has been optimising the
  wrong quantity for six weeks. Calibration becomes an untouched lever on a board where every axis we
  have priced ourselves reads flat, and the size of the move here is the first measurement of how much
  it is worth. A large DROP is the expected shape (cubing pushes mass toward zero, which a log-loss or
  Brier term punishes) and is the informative outcome, not a failure.

**WHY THIS IS WORTH A SLOT TONIGHT.** The member axis is closed (seven board flats), the vote-rule axis
read flat at the shipped seating hours ago (`aucvote_v1`, 0.943), and slice coverage closed on two arms
already sitting on disk. There is nothing mechanism-backed to ship, and the standing rule is that a slot
then goes to an INSTRUMENT. This is the cheapest instrument the project can build — no card time, no new
member, no packaging risk beyond one line — and it is the only one whose subject is the ruler rather
than the pool.

**WHAT IT DOES NOT DO.** It cannot improve the score. Under the documented metric it must tie exactly,
and under any other it will almost certainly lose. That is the point: a tie is the finding, and a
non-tie is a bigger one.
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

FOREIGN_WEIGHT = 1.0 / 3.0
"""The foreign side's share of the rank blend. `raptorseat_v1` shipped 1/7 = 0.1429 as a convention about
SEAT COUNT; this is a quality-weighted share of two SYSTEMS, named before the number and never fitted.

Below the bootstrap block rather than above it: the block opens with imports, and a constant in front of
them makes every one of them E402. The kernel that RAN carried it above and the ruff gate caught it after
the fact — the number is identical either way.
"""


def main() -> None:
    boot()

    import time  # noqa: PLC0415 - the bootstrap block above is byte-identical across kernels

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot
    import torch  # noqa: PLC0415 - same

    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.device import VisibleCards  # noqa: PLC0415 - same
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415 - same
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - same
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_run import WORKING, RaptorEntry  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415 - same
    from core.findings import TARGETS  # noqa: PLC0415 - same

    # The raptor member runs FIRST and into a scratch directory, because both producers write a file
    # called `submission.csv` and the one that runs last is the one Kaggle would otherwise score.
    scratch = WORKING / "raptor"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mounts = Mounts()
    device = str(VisibleCards().device())
    member = RaptorMember.loaded(RaptorEntry.weights(mounts=mounts), device)
    theirs = RaptorEntry(mounts.one("sample_submission.csv", "competition data"), working=scratch).run(member, device)
    # The six that follow load onto the same T4. Dropping the reference is not enough on its own —
    # torch keeps the freed blocks in its caching allocator, and a seventh model's worth of them is
    # exactly the margin an OOM would eat, in a kernel whose whole point is running both.
    del member
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
    # What the raptor pass cost, handed to the six so their nine-hour budget is the SESSION's and not
    # a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at the
    # real one with no submission — a budget defeated by its own assumption that it starts at zero.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the raptor pass cost {spent:.0f}s; the six get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left)).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    blended = RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs)
    # THE ONE CHANGE. Strictly increasing over the whole real line, so every within-column ordering and
    # every existing tie survives untouched while the values do not. Under a per-column rank statistic
    # the written file must score identically to the champion's; under anything calibration-sensitive
    # it cannot.
    for target in TARGETS:
        blended[target] = blended[target].astype("float64") ** 3
    blended.to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
