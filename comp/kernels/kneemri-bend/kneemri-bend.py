"""bend — our champion six and mattiaangeli's whole published program, rank-averaged equal weight.

**TWO SYSTEMS, BOTH BOARD-KNOWN.** Ours reads 0.944 (`raptorweight_v1`, 55875622). His
`bend-the-knee-to-the-dinosaurs-all-public` reads 0.939 (kernel 133147282, run 2026-09-06 16:32Z).
Neither number is inferred. What has never been measured is the two of them together, and every
cross-family addition of a >=0.93 solo this project has made has converted.

**VERBATIM, AND WHY THAT IS THE POINT.** `foreignseat_v1` lifted 45% of ONE of his three stages and
read -0.013. His program is a 24-member hashed-fold rank mean, then a `_micro` fold blend, then a
RadImageNet dual-head pass that `os.replace`s the result — three stages that each rewrite the same
file, none of which survives being sliced. So it runs as HIS FILE, as a subprocess, with one constant
changed. The only thing this kernel contributes to his side is the clock.

**ORDER.** Ours runs first and its table is moved to `ours.csv`. His stages 2 and 3 open
`/kaggle/working/submission.csv` by absolute path and cannot be pointed elsewhere, so he is given that
path outright; the blend is written back over it at the end. Running ours first also means the
known-good 0.944 side is complete before his begins, and his budget is the MEASURED remainder rather
than a split guessed in advance — our runtime on the hidden split has never been observed, so any
pre-computed split would be a number nobody has.

**EQUAL WEIGHT, NOT A FITTED ONE.** `RaptorSeat.combined(ours, [his], [0.5])` — the same rank-space
join by study the champion uses, at a half. The weight axis is closed: `raptorweight_v1` moved the
foreign share 1/7 -> 1/3 and the board did not resolve it. Two systems five thousandths apart get a
half each, and no slot is spent looking for a better one.

PRE-REGISTRATION, written before the number, against the champion's 0.944.

- **0.947 or above** — the cross-family blend converts. Re-pin `comp/champion.json` and every later
  entry is built on this pair.
- **0.944-0.946** — the two families overlap more than their scores suggest. Reweight 0.6/0.4 toward
  him at the last mark, since his side is the one this project has never fitted anything to.
- **below 0.944** — his program did not produce his output here. That is a mounting or budget failure
  and NOT a reading about diversity; `run_check` on the log is what says which, and this entry is void
  until it does.

**NO OFFLINE ORACLE EXISTS FOR THIS ENTRY.** His rad stage has no holdout path, so there is nothing to
score it against before the slot is spent. The four assertions available are that his program ran to
completion, that it banked 24 of 24 members, that all three Rad lines printed, and that the table it
left holds no nulls. Three of those are read from the log by `run_check`; the fourth is `BendEntry.read`
refusing here, before the blend.
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

FOREIGN_WEIGHT = 0.5
"""His share of the rank blend. Two systems five thousandths apart on the board get a half each."""

HIS_MARGIN = 30.0 * 60.0
"""Seconds held back from his budget beyond what our side measured.

He surrenders members gracefully when his own `TIME_BUDGET` runs out, but the surrender is not free —
it happens between members, and the last one has to finish. Half an hour is one of his heavier members
plus the rad pass's own write. Without it the kernel is killed at Kaggle's real ceiling instead, and a
killed kernel has no submission at all: the failure mode this margin buys off is total, not partial."""

HIS_FLOOR = 45.0 * 60.0
"""Below this, his program is not started at all.

A quarter of an hour of a three-stage program is stage one truncated to a handful of members and no
rad pass — a foreign table that is not the thing measured at 0.939, blended at half weight into a
system that was measured at 0.944. Shipping ours alone is the strictly better entry, and it is what
happens here."""

PROGRAM = "all_public.py"


def main() -> None:
    boot()

    import time  # noqa: PLC0415 - the bootstrap block above is byte-identical across kernels

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot

    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - same
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.bend_entry import BendEntry  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_run import WORKING  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415 - same

    log = logging.getLogger(__name__)
    started = time.time()
    source = Mounts().one(PROGRAM, "mattiaangeli's published program") / PROGRAM

    # OURS FIRST, and its table moved off the path his stages own. Ours is the side with a board
    # number under this exact configuration; whatever happens to his, this file is a 0.944 entry.
    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS)),
    ).run()
    ours = pd.read_csv(WORKING / "submission.csv")
    ours.to_csv(WORKING / "ours.csv", index=False)
    (WORKING / "submission.csv").unlink()

    spent = time.time() - started
    left = KERNEL_SECONDS - spent - HIS_MARGIN
    log.info(f"our six cost {spent:.0f}s; his program gets {left:.0f}s of a {KERNEL_SECONDS:.0f}s session")
    if left < HIS_FLOOR:
        log.info("under his floor — shipping our six alone rather than a truncated foreign side")
        ours.to_csv(WORKING / "submission.csv", index=False)
        return

    # His side is the one that can fail, so it is the one wrapped: a crash, a missing table or a null
    # leaves the champion's own table in place and the entry still scores.
    try:
        theirs = BendEntry(source=source, working=WORKING, seconds=left).run()
    except Exception:
        log.exception("his program did not produce a table — shipping our six alone")
        ours.to_csv(WORKING / "submission.csv", index=False)
        return

    blended = RaptorSeat.combined(ours, [theirs], [FOREIGN_WEIGHT])
    log.info(f"blended {len(ours)} studies at {FOREIGN_WEIGHT} to his side")
    blended.to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
