"""Does more data license a longer anneal, or is the refit's gain independent of schedule length?

WHY THIS ENTRY EXISTS. The full-corpus refit (`--fold -1`) is the only positive BOARD term this repo
owns. Read twice, it is +0.003 on `labelwise_depth24_drop40_long` and +0.008 on
`labelwise_ext_drop40_long` — and those two twins' offline75 are TIED at 0.9053 and 0.9045, inside the
0.0062 paired floor, so the two readings sit at the SAME abscissa and their spread is board scatter
rather than a shape. Read that way the term is flat at about +0.0055, and no refit of anything we own
reaches the pinned champion's 0.946 by itself.

So the refit is not a road; it is a step with one unspent compound. Refitting adds 909 studies, +26%,
and more data is exactly what licenses a LONGER schedule. This member is that compound and nothing
else: `labelwise_depth24_drop40_long_ff`'s line with `--epochs 36` substituted for 24, every other flag
byte for byte. Its control is already on the board at 0.944 (`ffsolo_v1`, ref 56296698), so the
comparison costs no second slot.

WHAT THE HONEST SPLIT SAYS, AND WHY IT DOES NOT SETTLE THIS. The one 36-epoch point this project owns,
`labelwise_dense_drop40_long36_s2026`, reads 0.9030 against its 24-epoch twin's 0.9048: **-0.0018,
inside the 0.0062 floor**, which is NO READING rather than a null. It was also taken on the 3498-study
training split — the regime where a longer schedule overfits soonest. The refit relaxes that by 26%.

THE MEMBER. DARK by construction (knee-s8kim): it holds no study back, so no ledger row can price it
and none is owed. Its only ruler is the board.

PRE-REGISTERED, on the SCHEDULE rather than on the member:

  * **board >= 0.947** — anneal length compounds with the refit, this is the shipping solo, and the
    schedule axis re-opens on the full corpus where it was only ever closed on the honest split.
  * **board 0.944-0.946** — the longer anneal buys nothing the refit did not already have. The
    schedule axis closes on the corpus side too, and future refits stay at 24 epochs for 1.5x less card.
  * **board <= 0.943** — a longer schedule COSTS on the full corpus. That prices every future refit at
    24 epochs and is the most useful of the three readings, because it is the one that changes a recipe.

WHAT IT COSTS: 2.4 h of card, measured off the arm's own 238 s/epoch, and no decode. bead knee-lz8pd.
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

    from kneemri.inference.budget import (  # noqa: PLC0415 - the kit is on sys.path only after boot
        KERNEL_SECONDS,
        Budget,
    )
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # A set of ONE, and it has to be: the reading is about SCHEDULE LENGTH on one member, and a second
    # name would add a seating effect on top of the term this entry exists to measure.
    chosen = ("depth24_ff_long36",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
