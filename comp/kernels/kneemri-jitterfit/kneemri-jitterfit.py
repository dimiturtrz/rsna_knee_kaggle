"""dark6's identical six under RIGID TEST-TIME VIEWS — the last serving axis with no reading anywhere.

**WHAT MOVES AND WHAT IS HELD.** The seats are `dark6`'s six all-data members, in `dark6`'s order, under
`dark6`'s `RankAverage` and `overlap=False`. The only change is `Serving.views`: a second view of every
study at -6 degrees, averaged with the identity view before the vote. Nothing is retrained; the members
are byte-identical to the ones that read 0.937.

**WHY THIS AXIS IS OPEN AT ALL.** `votefit` logged `serving: vote=LogitAverage views=identity only
(default)` — and that default is what every board entry on the CURRENT roster has run. Views were only
ever attempted on rosters nobody would ship now: `kneemri-views` (a two-member 0.919 pair) and
`kneemri-topviews` / `kneemri-rateviews` (fold-0 members from before the all-data roster existed). The
first of those came back COMPLETE WITH A BLANK PUBLIC SCORE, killed by an OOM inside the warp — the
defect `WARP_CHUNK = 512` fixed. So this axis has no board number and no offline number on any roster
that matters. It is not a knob measured flat; it is a knob never measured.

**TWO VIEWS, NOT THREE, AND THE REASON IS THE CEILING.** `SpatialViews()` defaults to three degrees and
multiplies the forward pass by three. Six members at three views is eighteen member-passes against a
nine-hour ceiling that two members at three views once failed to clear. Two views is the smallest step
that asks the question, and it costs a factor of two.

**THE ONE WAY THIS ENTRY CAN LIE, AND HOW IT IS CHECKED.** `Ensemble` gives up views FIRST when its
budget projects an overrun. A jitter entry that runs long therefore degrades silently into `dark6` and
reads 0.937 for a reason that has nothing to do with jitter. This kernel is RUN FREE before it is ever
submitted, and its log must contain no `giving up test-time views` line. Kernel runs cost no slot; only
submissions do.

**NO FLIPS.** `test_time.py` states it and it is load-bearing here: a horizontal flip undoes the
laterality normalisation that five of the twelve targets depend on. Rotation and shift preserve it.

PRE-REGISTRATION, written before the number, against `dark6`'s 0.937.

  * **>= 0.940** — the members are brittle to small rigid pose changes and averaging over them recovers
    real signal. Then the third view is licensed (and worth its runtime), and every future entry serves
    under views by default.
  * **0.935-0.939 FLAT** — the members are already pose-robust, which is what whole-stack training on a
    registered corpus ought to buy. Combined with `votefit`, BOTH serving axes are then closed on board
    numbers rather than on the holdout, and nothing serving-shaped remains.
  * **<= 0.934** — the warp is destroying information rather than averaging over nuisance, most likely
    at the slice edges where interpolation invents tissue. That would retire TTA on this pool for good.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# >>> KERNEL-BOOTSTRAP
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


# <<< KERNEL-BOOTSTRAP


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - kit is on sys.path only after install_wheels
    from kneemri.inference.test_time import SpatialViews  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    chosen = (
        "final_ws_e12",
        "calib_span_ff",
        "grade_ladder_ff",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    views = SpatialViews(degrees=(0.0, -6.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(views=views)).run()


if __name__ == "__main__":
    main()
