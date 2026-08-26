"""Three champion seats replaced by their own blend-supervised twins - the dose, at its midpoint.

**WHAT MOVES: SUPERVISION, THREE TIMES, AND NOTHING ELSE.** This is `kneemri-blendseat` - the champion
at 0.939 - with three of its six seats replaced by members whose recipes are argv-identical apart from
the label table and the corpus they read:

    lr4e-5_ws_ff   -> blend_ws
    calib_span_ff  -> blend_calib
    grade_ladder_ff -> blend_ladder

Each entering member is the outgoing one's own twin fitted on `merged_blend_v1.csv` over the merged
corpus. The vote rule, the count, the kit and the serving config are the champion's, untouched. Three
seats move and every one of them moves along the SAME axis, which is what makes this a dose reading
rather than a roster reshuffle.

**WHY A MIDPOINT EXISTS AT ALL.** `blendsix` builds five such members sequentially at ~1.2 h each, so
the six-seat roster it argues for does not exist for seven hours. Reading the arm once would spend a
slot on one endpoint; reading it twice puts a point at dose 3 and a point at dose 6 against
`blendseat`'s dose 1, and the SHAPE across the three is worth more than either endpoint alone.

**WHAT THE AXIS HAS BEHIND IT, at its real strength.** `blendlabels` (one member, fold-0): +0.0034
mean, BELOW the 0.0062 paired floor, and +0.0070 on its own label-bottlenecked six. `merged_blend`
(one member): 0.8587 holdout, seat value +0.0017. `blendseat` (one seat, on the board): 0.939 against
votefit's 0.938, one tick, noise. None of that is a result. What raises the prior here is external and
independent: competition forum topic 735304 holds five self-reported single-model board scores, and
the two highest 5-fold entries both carry the author's note "LLM output labels only". The frontier
says supervision, and this is the dose of it.

**THE COST OF THE DEVIATION, STATED.** The merged scheme is decoded at 12 slices / 0.2-0.8 only, so
the entering members cannot reproduce the dense (20-slice) geometry two of the outgoing seats carry.
The roster therefore loses geometric spread at the same time it gains supervision, and that is a live
suspect if this reads at or below 0.935 - `member_distance` over the three new members answers it with
no card and no slot.

**BLIND BY CONSTRUCTION.** All three entering members are `--fold -1`, as are five of the six seats
they join. This entry has no offline number of any kind and none can be computed for it. It ships
against the board alone.

PRE-REGISTRATION, written before the number, against `blendseat` 0.939 (dose 1):

  * **>= 0.947.** The dose is the lever and it already saturates at three. Slot 5 goes to dose 6 to
    find the knee, and every card hour after that goes to better label TABLES.
  * **0.943-0.946.** Real, under the 1.5% house bar as a single step, and composing. Dose 6 becomes
    the reading that matters.
  * **0.940-0.942.** Directional, inside the scatter our nine flat 6-7 member entries already span.
    The dose is not the lever; supervision QUALITY might still be, and that is a different table.
  * **0.936-0.939.** FLAT at three. Dose 6 still earns slot 5 - a linear dose effect reads flat at the
    midpoint - but the prior on it drops hard.
  * **<= 0.935.** Suspect the harmonised 12/0.2-0.8 geometry BEFORE the supervision. Three of the
    champion's seats carried geometric spread this roster does not.
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

    # Three seats replaced by their own blend-supervised twins. The order is the champion's, unchanged,
    # so the diff against `kneemri-blendseat` is three lines and each one is the same swap.
    chosen = (
        "final_ws_e12",
        "blend_calib",
        "blend_ladder",
        "merged_blend_s2026",
        "blend_ws",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
