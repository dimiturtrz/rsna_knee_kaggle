"""rankvote — the champion seating combined by RankAverage, the third of four rules at the shipped six.

WHY A SLOT GOES TO A VOTE RULE AND NOT TO A SEATING. Seven board entries have now reseated the same
DINO-family pool - reweight, add, swap, count, precision, lens, distance - and every one of them read
flat inside 0.937-0.944. The member axis is closed. What has never been priced on the board is the
COMBINER: `comp/board.jsonl` holds ninety-three rows and not one of them changes the rule by which the
seats are combined. `aucvote_v1` (LabelQualityRank, 2026-09-02 17:29Z) and `kneemri-stackvote`
(StackedLogit) are the first two readings of that axis. This is the third, and with the champion's own
LogitAverage it completes the set: four rules, one seating, everything else held.

WHAT IT IS. `raptorweight_v1`'s exact six in the exact order (final_ws_cf, calib_span_ff,
grade_ladder_cf, merged_blend_cf, lr4e-5_ws_cf, through_dense_cf), overlap=False, the raptor foreign
seat at FOREIGN_WEIGHT=1/3, the same kit. One argument differs from the champion: the vote is
`RankAverage` rather than `LogitAverage`.

WHAT IT MEASURED OFFLINE, STATED BEFORE THE READ. On the five fold-0 twins over 909 holdout studies:
LogitAverage 0.8745, RankAverage 0.8742, StackedLogit 0.8755. RankAverage is -0.0003 - INSIDE the
0.0062 noise floor and an order of magnitude under it. At the +0.98 board-per-holdout-point slope that
is -0.0003 on the board, far under BOARD_RESOLVABLE = 0.0015.

SO THIS IS AN INSTRUMENT AND NOT A CANDIDATE, and it is pre-registered to read flat. It is worth a
slot for one reason only: the champion is six-of-six `--fold -1`, so no offline ruler can see the
seating that actually ships, and the four offline readings above were all taken on fold-0 twins. If
the three rules read flat on the BOARD at the shipped seating, the combiner axis closes with a
board-level measurement rather than an inference from twins - and closing it is what licenses spending
the remaining slots elsewhere.

WHERE RANK DIFFERS FROM LOGIT, mechanically: rank averaging is invariant to each member's calibration
and throws away the margin. Our members are calibrated differently on purpose (calib_span_ff carries
--calibrated-targets, the others do not), so rank is the rule that should benefit most if
miscalibration is costing the logit average anything. It reading flat is therefore also a statement
that miscalibration is NOT what the pool is losing to.

PRE-REGISTERED against the champion 0.944 and BOARD_RESOLVABLE 0.0015:
  >= 0.9455 - the offline twins mispriced the rule; calibration spread IS costing the logit average at
    the shipped seating, and the combiner is a live lever after all.
  0.9425-0.9455 - FLAT, the expected branch. With aucvote and stackvote it closes the combiner axis on
    the board, at the seating that ships.
  <= 0.940 - discarding the margin costs, which prices what the logit average's magnitudes are worth
    and argues against every rank-space rule, LabelQualityRank included.
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


def main() -> None:
    boot()

    from kneemri.inference.vote import RankAverage  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.kaggle.seated_entry import SeatedEntry  # noqa: PLC0415 - same

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    SeatedEntry(
        names=chosen,
        vote=RankAverage(),
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
