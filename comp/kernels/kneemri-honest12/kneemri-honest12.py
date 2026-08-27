"""honest12 — TWELVE distinct recipes, seated by the honest greedy rather than by lineage.

WHAT IS NEW HERE. Every entry this project has shipped seated members it already trusted and then
varied one of them. This one asks the seating question directly: `record_mix --seats honest --vote
logit` runs a greedy refitted on half the 909 shared holdout studies and scored on the other half,
both directions, over the 89 records that share a ruler. That curve peaks at TWELVE seats / 0.8751
held-out. The shipped champion's five-twin shadow, on the same studies under the same vote, reads
0.8701. +0.0050 offline, and at the measured +0.74 board points per holdout point that is +0.0037
board — 0.941 -> ~0.945 if the transfer holds.

WHAT ARGUES AGAINST IT, STATED FIRST. `calibmix` seated ELEVEN and scored 0.941, exactly what six
seats scored, and read as a flat count axis. That reading does not survive resolving its names: its
eleven are five recipes with four of them entered TWICE under a different fit, so offline it collapses
onto the same five twins the champion holds. It measured fit-duplication, and fit-duplication is flat.
These twelve are twelve DISTINCT recipes, which is the axis calibmix never varied.

THE SECOND ARGUMENT AGAINST IT is real and unresolved: seven of these twelve have never been on the
board in any entry, and one of them (`mae_matched_s2026`, seat nine) reads 0.6735 solo. The greedy
wants it for disagreement, not for quality, and this project has never shipped a member that weak. If
this entry lands FLAT rather than up, that seat is the first suspect and the honest curve's own dip at
nine seats (0.8735 against 0.8736 at eight) already names it.

SEATING, in the greedy's own order. Seats 2 and 12 are full-fit; the rest are the fold-0 holdout twins
the curve was measured on, which is what makes the offline number a statement about THIS tuple rather
than about its lineage.
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

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    chosen = (
        "calib_dense_hold_s2026",
        "merged_blend_cf",
        "finding_calib_dense_s2026",
        "blendhold_s2026",
        "margin_seat_s2026",
        "calib_span_s2026",
        "through_dense_s2026",
        "small_batch6_s2026",
        "mae_matched_s2026",
        "base224_s2026",
        "calib_span_hold_s2026",
        "merged_dense_cf",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
