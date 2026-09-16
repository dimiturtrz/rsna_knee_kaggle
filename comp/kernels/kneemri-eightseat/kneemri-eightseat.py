"""eightseat — `foldseven`'s seven seats plus ONE more, chosen to be worth almost nothing. An INSTRUMENT.

It is not an attempt on the champion and its offline number is deliberately indistinguishable from
`foldseven_v1`'s. What it measures is the BOARD's own scatter at a fixed ruler reading.

WHY THIS IS THE QUESTION. `pricetop_v1` reads offline 0.8757 -> board 0.938 and `foldseven_v1` reads
0.8767 -> 0.944: one offline point apart, SIX board points apart. The board blur `board_transfer`
reports is 0.0020, so that gap is three times the amount the board is supposed to be unable to
certify, and it is wider than the entire offline span the top stratum is fitted over. Until it is
known whether that six-point gap is the SEATS or the BOARD, no arm in this band can be priced
through the transfer curve at all -- and more pair points cannot settle it, because a slope fitted
through 0.006 of residual needs about ten of them at one slot each.

WHAT IS HELD STILL. The eighth seat is `labelwise_depth24_s2026`. `seat_add` prices it onto these
exact seven at **+0.0005 ledger75** (0.9097 -> 0.9102, relation EXACT over 909 rows) -- BELOW the
+0.0015 bar, and below the board's own 0.0020 blur. The seat is chosen for being NEGLIGIBLE: among
the shipped members not already seated it is the one whose add sits closest to zero from above. A
criterion that MINIMISES the add cannot inflate this entry's offline, which is why no bias-corrected
number is owed beside the raw one here (knee-jsgb bites on searches that maximise).

The roster's best unseated member, `labelwise_depth24_negpull_s2026` (0.8666, add +0.0006), would
have served identically and is on no shelf -- shipping it meant a member-dataset re-upload, which is
wall-clock this slot did not have. The two adds differ by 0.0001; nothing in the reading turns on
which one took the seat.

PRE-REGISTRATION. The quantity read is the board delta against `foldseven_v1`'s 0.944:

- **0.943-0.945** -- the board is reproducible at a fixed ruler reading. The pricetop/foldseven gap
  is then about WHICH seats are taken, not noise, and seat identity re-opens as an axis carrying
  ~0.006 -- larger than anything the ledger can currently see.
- **0.937-0.940** -- the board's scatter at fixed offline IS ~0.006. `board_transfer` is then
  unusable in this band at any n, and every arm priced through it needs re-reading against a ruler
  that has been shown not to order two entries. The more likely branch, and the more valuable one.
- **0.946+** -- a +0.0006 seat bought a champion-beating move, which refutes the add's own arithmetic
  rather than confirming the seat. Re-read `seat_add`'s row set before believing it.

Check `run_check`'s census FIRST: 8 of 8 loaded or this reads nothing.
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
        "calib_span_s2026",
        "stack_increments_s2026",
        "through_dense_s2026",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
        "labelwise_depth24_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
