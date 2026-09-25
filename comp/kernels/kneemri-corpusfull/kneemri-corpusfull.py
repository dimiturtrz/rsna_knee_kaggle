"""corpusfull - the external-corpus dose taken to the TOP of its curve: six seats of six.

WHY THIS POINT AND NOT ANOTHER. The dose curve already has three readings on this board, and the grep
that found them is what killed the two cheaper candidates for this slot. The champion seats ONE
external member (`merged_blend_cf`) and reads 0.944. `spanseat_v1` seats TWO (`merged_blend_cf` plus
`merged_span_ff`) and reads 0.944 - so the 1 -> 2 step is already measured, and shipping it again would
have been a repeat wearing an instrument's label. `corpusdose_v1` seats FOUR and is still pending at the
time of writing. What the curve has never had is a point where the dose is SATURATED, and that is the
only place left where it can still bend.

  external seats   1        2        4          6
  board            0.944    0.944    pending    this entry

WHAT MOVES, AND THE CONFOUND, stated first because it is the reason this entry is an instrument and not
a bid. `corpusdose_v1` could hold structure while it raised the dose: it took the pool's whole-stack
near-twins and left the pixel-distant (`through_dense_cf`) and objective-distant (`grade_ladder_cf`)
seats standing. Reaching six of six cannot do that. No `merged_*` member exists at either of those two
corners, so the last two seats go to `merged_corpus_s2026` (0.8545) and `merged_blend_s2026` (0.8587,
the fold-0 sibling of a seat that stays), which are external-corpus members but NOT twins of what they
replace. So TWO quantities move together:
the corpus dose rises 4 -> 6, and the pool loses its two most structurally distant seats. Every branch
below is written knowing that, and the low branch cannot separate them.

Held at the champion exactly: count (six plus the foreign seat), vote (LogitAverage), overlap=False,
serving precision, the raptor seat and its 1/3 weight, the kit. It cannot be scored offline - the
`--fold -1` fits make every holdout study in-sample for part of the pool - so the board is its ruler.

PRE-REGISTERED, against the champion 0.944 and BOARD_RESOLVABLE 0.0015:
  >= 0.9455  THE DOSE COMPOUNDS LATE. Flat from one to two, then paying only once most of the pool has
             seen the corpus, which would be an argument about the pool rather than about any member -
             and it would make the fastMRI and MRNet corpora the plan rather than a side errand.
  0.9455 > s > 0.9425  FLAT, and with 1 and 2 already flat this is the branch that CLOSES the
             external-corpus SEAT axis: one, two, four and six external seats all read the same, so the
             corpus a member trained on is not what its seat is worth. It would not close external data
             as such - a trunk continued on it is a different mechanism, and `ortho_dense` is the arm
             that asks that - but it would retire seat-counting over the `merged_*` family for good.
  <= 0.9425  THE TWO DISTANT SEATS WERE CARRYING SOMETHING. Read this against `corpusdose_v1` before
             reading it against the champion: if four external seats held 0.944 and six drop below it,
             the loss is the structure that left, not the corpus that arrived, and the next question is
             which of `grade_ladder_cf` and `through_dense_cf` was doing the work.
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

    from kneemri.kaggle.seated_entry import SeatedEntry  # noqa: PLC0415 - the kit is on sys.path only after boot

    chosen = (
        "merged_span_cf",
        "merged_span_ff",
        "merged_corpus_s2026",
        "merged_blend_cf",
        "merged_dense_cf",
        "merged_blend_s2026",
    )
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
