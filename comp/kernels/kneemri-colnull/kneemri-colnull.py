"""colnull — every finding column a constant. The metric's FORM, read directly off the board.

**WHAT IS BEING MEASURED.** Not a method, not a member, not a seating: the shape of the scoring function.
`comp/README.md` has recorded since the first week that the formula is unread — the evaluation page is
JavaScript-only and the API does not serve it — and every per-label and per-column number this project has
priced rests on the assumption that the score is an unweighted mean of twelve per-column AUCs.

**THE MECHANISM, and why this entry needs no model at all.** The metric is per-column and rank-based:
`kneemri-monotone` shipped a monotone transform of the champion's table and read the champion's exact 0.944.
A column holding one constant therefore has an AUC of exactly 0.5, whatever the constant is — no seed, no
variance, no calibration. Fill EVERY column with a constant and a mean-of-per-column-AUCs metric must return
exactly 0.500. Nothing else in this project can produce a number that is known in advance to three decimals,
which is what makes this a measurement of the formula rather than of us.

It is the companion to `kneemri-colmcl`, which damages ONE column and reads that column's share. This one
reads the form; that one reads the weight. Neither is readable without the other: a single-column drop can
be explained by a weight or by a different board-side AUC, and this entry removes the second explanation by
pinning the intercept.

PRE-REGISTRATION, written before the number.

- **exactly 0.500** — the score is a mean of per-column AUCs with no offset and no non-AUC term. Every
  per-column reading in `interpretations/` is priced on the right formula, and `colmcl`'s drop is then a
  clean read of MCL's weight.
- **0.495-0.505 but not 0.500** — a mean of per-column AUCs with a tie-handling or a per-column
  prevalence weighting that a constant column does not resolve to exactly a half. Small, but it means the
  formula carries a term nothing here has modelled.
- **anything else** — the score is NOT a mean of per-column AUCs. Every per-label plan this project holds is
  priced on a formula that does not hold, and the ledger's per-column arithmetic has to be re-derived before
  another arm runs on the label axis. This is the branch worth a slot.

**COST, STATED HONESTLY.** This entry scores about 0.5 — the worst row this project will ever have put on
the board, on purpose. It buys the metric's form for one slot, no card time, and no members: it does not
load a checkpoint, which is also why `run_check`'s member census does not apply to it. It is an INSTRUMENT.

The refutation this cannot make: it says nothing about which columns are weighted, only whether the total is
a plain mean of per-column AUCs. That is `colmcl`'s question.
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

CONSTANT = 0.5
"""What every finding column is filled with. Any single value gives a column an AUC of exactly 0.5 under a
rank metric; a half is chosen so a reader of the file can see at a glance that it is not a score."""

WORKING = Path("/kaggle/working")
"""Written out rather than imported from `kneemri.kaggle.kernel_run`. That module reaches the whole
serving stack to hand back a constant path, and this entry loads nothing, so importing it would buy
runtime on the efficiency board for no reason.

Version 1 failed on that import with `ModuleNotFoundError: jaxtyping`, and the cause was NOT the import:
`main` called `find_kit` directly and skipped `install_wheels`, so the kit's own wheels — jaxtyping among
them — were never installed. `boot()` is the canonical four-step order and is what this now calls. The
lesson is the one the bootstrap module already states: copy the step, do not re-spell it."""


def main() -> None:
    boot()

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot

    from core.findings import TARGETS  # noqa: PLC0415 - same
    from core.mounts import Mounts  # noqa: PLC0415 - same

    # The sample submission, not our own test frame: this entry deliberately loads no model, so the ONLY
    # thing it can be sure of is the row set the competition itself asked for.
    # `one` answers the DIRECTORY holding the name, not the file — every other kernel hands that directory
    # to `RaptorEntry`, which appends the filename itself. Version 3 read the directory and died on
    # `IsADirectoryError`. Reading a helper's return TYPE is cheaper than a kernel version.
    entry = pd.read_csv(Mounts().one("sample_submission.csv", "competition data") / "sample_submission.csv")
    for target in TARGETS:
        entry[target] = CONSTANT
    logging.getLogger(__name__).info(f"{len(entry)} studies, {len(TARGETS)} columns, every one at {CONSTANT}")
    entry.to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
