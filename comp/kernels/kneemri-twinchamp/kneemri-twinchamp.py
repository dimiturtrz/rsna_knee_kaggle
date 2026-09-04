"""The CLEAN fold-0 half of the `fiveship_v1` pair — the refit constant, measured without a dirty half.

**WHAT IT REPLACES.** `twinfive_v1` was built as the fold-0 half of `fiveship_v1` and it is not one.
`fiveship_v1` seats calib_span_ff, grade_ladder_cf, merged_blend_cf, lr4e-5_ws_cf, through_dense_cf.
`twinfive_v1` seats the fold-0 twins of four of those and **merged_blend_cf itself** — the full-fit
member, unchanged, in the half that is supposed to be the fold-0 control. So the +0.006 the pair reports
(0.935 -> 0.941) is the price of refitting FOUR seats, measured against a baseline that was already
one-fifth refit. This entry seats the fold-0 twin of every one of `fiveship_v1`'s five, so the delta it
completes is the price of refitting all five and nothing else. The seating is COPIED from `fiveship_v1`,
not selected — no greedy ran, nothing was swapped, reweighted or re-lensed. Not a seating operation.

**AND IT FIXES A CONTAMINATED NUMBER IN THE RECORD.** `twinfive_v1`'s row carries `offline: 0.8705` with
no exclusion. It cannot: merged_blend_cf was fitted at `--fold -1` and has seen all 909 holdout studies,
so that figure is partly in-sample. It is the `raptor_solo` defect on a row the refit constant rests on.
This entry's offline is honest and was measured BEFORE the slot with `record_mix` on the same 909 shared
studies, LogitAverage: **0.8700**. Against the contaminated 0.8705 the contamination bought 0.0005,
which is itself worth knowing — a full-fit seat is nearly free on the ruler and evidently is not on the
board.

**WHY THE CONSTANT MATTERS ENOUGH TO SPEND A SLOT ON.** It is measured twice and the two readings differ
six-fold: `twinfive`/`fiveship` +0.006 at five seats, `foldfour`/`fitfour` +0.001 at four. It licenses
roughly thirty hours of card spent refitting a fleet and it is the final term of the only arithmetic
path this project has to 0.953. One of the two readings now has a dirty half. Resolving that is worth
more than any reseating, and reseating is banned anyway.

**PRE-REGISTERED, written before the score.** Read the DELTA `fiveship_v1` (0.941) minus this entry.
The honest offline-to-board ladder maps this entry's 0.8700 to **0.932-0.934**.

- **delta >= +0.008** — refitting five seats is worth more than the dirty pair reported, and the
  under-estimate is confirmed with a size. The fleet-refit step of the 0.953 path is licensed at a
  LARGER number than the arithmetic currently assumes, and the card hours are the best-priced ones open.
- **delta +0.004 to +0.007** — replicates the dirty pair's +0.006 within the 0.0040 single-pair floor.
  The constant stands, `foldfour`'s +0.001 becomes the outlier, and the seat-count dependence is real.
- **delta +0.001 to +0.003** — agrees with `foldfour` instead. The +0.006 was the contamination reading
  backwards, the constant is small, and the path's arithmetic must be rewritten at the smaller number
  BEFORE any card is spent refitting anything. That is the outcome this instrument is most valuable for.
- **delta <= 0.000** — full-fit is not a board term at all, thirty hours of card is saved, and the
  champion's margin over the honest band is something other than refit.

**WHAT IT CANNOT DO.** Two board reads on one hidden test set cannot separate "refitting" from "these
five members' refit twins", and the pair is at five seats while the champion sits at seven. It also does
not price the champion: `raptorweight_v1` adds `final_ws_cf` (no fold-0 twin exists anywhere on disk —
a card arm, not an omission) and foreign `raptor_v4_swa`, which is excluded here deliberately because
`foreign_overlap` measured it 98.6% in-sample on our studies and it would make the offline dishonest.
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

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - the kit is on sys.path only after boot

    # The champion's own seating, fold-0 twins. Order copied from comp/champion.json. final_ws has no
    # fold-0 twin on disk and raptor_v4_swa is 98.6% in-sample, so both are absent by argument.
    chosen = (
        "calib_span_s2026",
        "grade_ladder_s2026",
        "merged_blend_s2026",
        "lr4e-5_ws_s2026",
        "through_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
