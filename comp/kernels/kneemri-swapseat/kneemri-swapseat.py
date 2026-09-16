"""swapseat — `foldseven_v1`'s seats with ONE exchanged for a member of another family at the SAME
ensemble reading. An INSTRUMENT, and the third leg of a triplet.

`through_dense_s2026` (solo 0.8864) leaves, `clean_calib_drop40_long_s2026` (solo 0.8977) takes the
seat. `seat_swap` prices the exchange at **swap d proxy +0.0001** over 909 rows -- the ensemble ruler
does not move -- while solo moves +0.0113 and the family changes. Its D43 verdict is NO-GO on gain,
which is the requirement here and not an objection: a zero-gain swap is the only kind that can ask
this question.

THE TRIPLET, and why no one of these reads without the other two. `pricetop_v1` (0.8757 -> 0.938) and
`foldseven_v1` (0.8767 -> 0.944) sit one offline point and six board points apart. Six points is 3x
the board blur and wider than the whole offline span board_transfer's top stratum is fitted over, so
until that residual is assigned, no arm in this band can be priced through the transfer curve at all.
Three entries decompose it, each holding the ruler still and varying exactly one thing:

  foldseven-rerun   same seats, same reading      -> pipeline run-to-run scatter (never measured)
  eightseat         +1 seat, +0.0005 reading      -> response to a sub-blur ruler move
  swapseat          different seats, +0.0001      -> SEAT IDENTITY at a fixed reading   <- this one

PRE-REGISTRATION, read against `foldseven_v1`'s 0.944 and only AFTER the rerun fixes the error bar:

- **within the rerun's scatter of 0.944** -- seat identity carries nothing at a fixed reading. The
  pricetop/foldseven six points are then NOT about which seats were taken, and the remaining suspect
  is the reading itself: two entries the holdout calls equal are not equal on the board, which makes
  the holdout the thing to fix rather than the seating.
- **outside it, either direction** -- seat identity carries ~0.006 at a fixed reading, which is
  larger than anything the ledger can currently see and larger than most closures in this repo were
  decided on. That re-opens seating as an axis and, more seriously, means a matched pair of entries
  can differ by more than the effect any recent arm was trying to measure.

A move in OUR favour here is not a rank result and must not be pinned as one: the entry was chosen
for a zero ensemble delta, so an upward board move is evidence about the ruler, not about the seats
being better.

Check `run_check`'s census FIRST: 7 of 7 loaded or this reads nothing.
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
        "clean_calib_drop40_long_s2026",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
