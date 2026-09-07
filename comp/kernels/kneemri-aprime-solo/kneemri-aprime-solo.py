"""A' — his recipe, his corpus, his labels, OUR fold held out — served solo on his own decode.

**WHAT IS BEING PRICED, AND WHY IT IS NOT A SEAT.** `raptor_ft_repl2x2_a_swa.pt` is the raptor training
script run under `foreign_run` at its own TYPICAL RUN settings with one change: `--folds 1 --fold 0
--fold_file external/raptor/fold0_909.json`, so it holds out our fold-0 909 and trains on the remaining
~3,511 rows of his 4,349. It is the first foreign-family checkpoint in this project with an HONEST
offline number, because every released one was fitted with his default `--folds 0` and has therefore
seen our whole holdout. The researcher's seat test reads **−0.0001**: as a MEMBER it is worth nothing,
and this entry does not propose it as one. It is an INSTRUMENT.

**THE QUESTION IT ANSWERS.** His training table (`labels_llm_soft`) calls positives at roughly HALF our
rate on eleven of twelve columns and DOUBLE on Synovitis. That is a training-side difference between his
program and ours which nothing on the board has ever priced, and it cancels exactly in every offline
comparison we can run. A board number for a model trained on his labels, against a board we know well,
is the only ruler that sees it.

**THE DENOMINATOR IS 0.8328, AND IT CARRIES NO FEED TERM.** A' reads **0.8307** at its best epoch and
**0.8328** SWA (12/9/14) on the OOF-896 under `report_labels_v2.csv`. This kernel serves through
`RaptorFeed`, which is their decode reimplemented from their published prose spec and held to EQUALITY
rather than agreement — 200 studies decode byte-identically to theirs — so the pixels this model sees
here are the pixels it trained on. `board − 0.8328` is a clean ruler-transfer reading, comparable to
`foreignsolo`'s +0.001 and our own +0.054.

**NO FOREIGN CODE RUNS HERE.** `external/raptor` is a notebook export with no LICENSE file and cannot
enter a kit. `kneemri.evaluation.raptor_feed` and `raptor_model` are ours; the checkpoint is ours,
mounted from the private `dimiturnt/kneemri-aprime`.

PRE-REGISTRATION, written before the number, against the 0.8328 denominator and the family's known
transfer.

- **0.920 or above** — our own ruler UNDER-prices models trained on his label table: the transfer is far
  above the family's +0.001 and the label axis is live. Next arm is one of ours retrained on a blend of
  the two tables, not another seating.
- **0.885 to 0.920** — the label choice is AUC-neutral at the board. The axis closes, and it closes for
  our own relabelling work too, which is the expensive half of what this buys.
- **0.885 or below** — his table is FURTHER from the hosts' truth than ours is. Our labels are the better
  asset and the raptor family's board strength comes from its decode and its capacity, not its targets.

**WHAT THIS ENTRY CANNOT DO.** It cannot move us up. A' is a weak member by measurement and this is one
member served alone; the champion is untouched. The slot buys the label axis, and the axis is what the
message says it buys.
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

WEIGHTS = "raptor_ft_repl2x2_a_swa.pt"
"""A', averaged over epochs 12/9/14 — the arm this entry serves, and the only foreign-family checkpoint
this project owns an honest offline number for."""


def main() -> None:
    boot()

    from core.mounts import Mounts  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.device import VisibleCards  # noqa: PLC0415
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415
    from kneemri.kaggle.raptor_run import RaptorEntry  # noqa: PLC0415

    mounts = Mounts()
    device = str(VisibleCards().device())
    member = RaptorMember.loaded(RaptorEntry.weights(WEIGHTS, mounts), device)
    entry = RaptorEntry(mounts.one("sample_submission.csv", "competition data"))
    entry.run(member, device)


if __name__ == "__main__":
    main()
