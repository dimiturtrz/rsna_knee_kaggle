"""v4solo — `labels_v4hybrid_s2026` ALONE. One member, one label table, no vote and no foreign side.

**AN INSTRUMENT, AND THE MESSAGE SAYS SO.** No slot goes to a seating, selection, weighting or
re-lensing operation; the member axis is closed. This is none of them. What changes between this entry
and its control is neither the architecture nor the seat but the TABLE the member was graded by, and the
board is the only ruler that can read it.

**WHY IT CANNOT BE READ OFFLINE.** `labels_v4hybrid` is `band_dense`'s recipe with one flag moved:
`--labels blend_hybrid_fx_v4b`. Every offline ruler we own is a v2 table, so scoring it there charges
every place v4b DISAGREES with v2 as an error — it reads 0.8524 against band_dense's 0.8578, and that
-0.0054 is a disagreement rate, not a loss. The other ruler is the gold-58 census, where v4b leads v2 by
+0.0337 — but that census shares an origin with the table it grades, so it cannot settle it either. Two
rulers, one blind and one interested. The test corpus is the third and the only independent one.

**THE CONTROL IS `holdsolo_v1`** (ref on the board, 0.8634 offline, **0.928**), the same trunk at the
same size in the same solo shape, graded by v2. A solo is what makes the comparison clean: a seat can be
diluted — `dark6` reads its seventh seat as near-inert — so a flat in a vote is ambiguous between "the
table does nothing" and "an eighth voice does nothing". Undivided, the table reads at full size.

PRE-REGISTRATION, written before the number, against holdsolo's 0.928:

- **0.931 or above** — the v4b table is a DATA lever, and the six-seat retrain is then on the TABLE and
  not on the teacher. The mechanism it would be riding is the gold-58 Lateral Meniscus column, where v4b
  leads v2 by +0.082; that is what the retrain would be buying, and it is what to check first if the
  entry clears the bar for some other reason.
- **0.927 to 0.930** — flat. The label axis closes for SEATS. It stays open for compose only, where a
  differently-graded member is a diversity term rather than a better one.
- **0.926 or below** — v4b grading HURTS generalisation, and v2 stands as the shipped table. That is a
  real finding on a table our own gold census ranks first, and it would say the census measures
  agreement with its own lineage rather than quality.

The refutation this cannot make: it cannot say whether the table helps IN AN ENSEMBLE. A solo that reads
flat and a seat that adds are consistent — that is what a diversity term looks like — and nothing here
prices it.

**THE MOUNT.** `labels_v4hybrid_s2026` lives on `kneemri-members-2`, merged in by
`MemberPackage.spanning`. A census reading `1 of 1` is the proof the member loaded; `0 of 1` produces no
submission at all rather than a quiet wrong number, which is the one respect in which a solo is safer to
read than a seat. `run_check` on the log before the slot either way.
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

    # A set of ONE. `LogitAverage` over a single member is the identity, named rather than special-cased
    # so the vote is the champion's own convention here too — what differs from `holdsolo` is the TABLE
    # the member was graded by, and nothing else in this file.
    chosen = ("labels_v4hybrid_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
