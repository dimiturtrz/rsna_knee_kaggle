"""Every seat blend-supervised - the dose at its far end, and the roster `blendsix` was written for.

**WHAT MOVES: SUPERVISION, SIX TIMES.** The champion at 0.939 carries exactly ONE seat fitted on the
three-reader blend table over the merged corpus (`merged_blend_s2026`). This entry carries six. The
five entering members are `blendsix`'s output - `blend_ws`, `blend_calib`, `blend_ladder`,
`blend_through`, `blend_base` - each argv-identical to a member this project already ships apart from
the label table, the corpus and `--fold -1`. Count, vote rule, kit and serving config are the
champion's, untouched.

**THIS IS THE THIRD POINT ON A DOSE CURVE, NOT A ONE-OFF.** Three board entries now hold count, vote
and serving fixed and move only how many seats saw the blend supervision:

    dose 1  blendseat_v1   0.939
    dose 3  blendthree_v1  (slot 4, ~06:00Z)
    dose 6  this entry

Read the SHAPE, not this number alone. The all-data axis was read exactly this way and came back
real-but-sub-linear (one seat +0.001, the next five +0.003 together), and a dose that saturates is a
different instruction from one that scales.

**WHAT RAISES THE PRIOR, and it is external.** Competition forum topic 735304 holds five self-reported
single-model board scores; the two highest 5-fold entries both carry their author's note "LLM output
labels only". Topic 735826 put the bottleneck at labels first and crop geometry second, explicitly NOT
at backbone capacity. Our own evidence for the axis is real and small - `blendlabels` +0.0034 mean
(below the 0.0062 floor) and +0.0070 on its own label-bottlenecked six, `merged_blend` seat value
+0.0017, `blendseat` one board tick. Nothing there is a result; all of it points one way, and it is
the only axis on this board that does.

**THE COST OF THE DEVIATION, STATED.** The merged scheme is decoded at 12 slices / 0.2-0.8 only, so
none of the entering members reproduce the dense 20-slice geometry three champion seats carry. This
roster trades ALL of its geometric spread for supervision, which is a stronger version of the same
trade `blendthree` makes and the first suspect if this reads low.

**BLIND BY CONSTRUCTION, AND MORE SO THAN ANY PIN BEFORE IT.** Five of the six seats are `--fold -1`
and the sixth is fold-0 on a corpus the other five also saw, so no pool here is seat-scoreable,
curve-scoreable or distance-scoreable. There is no offline number for this entry and none can be
computed. It ships against the board alone.

PRE-REGISTRATION, written before the number, against `blendseat` 0.939 (dose 1) AND against whatever
`blendthree` reads at dose 3:

  * **>= 0.947.** The dose is the lever, clears the house bar as a single step, and every remaining
    card hour goes to better label TABLES rather than more seats of this one.
  * **0.943-0.946.** Real, under the bar as a step, composing. Combined with dose 3 this gives the
    curve's slope and the next question is where it flattens.
  * **0.940-0.942.** Directional, inside the scatter our nine flat 6-7 member entries already span.
    The DOSE is not the lever; supervision QUALITY might still be, and that is a different table.
  * **0.936-0.939.** FLAT at six. Six blend seats buy what one bought, which is nothing measurable.
    The label axis closes at the DOSE term and stays open at the TABLE term.
  * **<= 0.935.** Two suspects in order: the harmonised geometry above, then five members fitted on
    one finer table agreeing with each other more than the champion's five did. `member_distance`
    over the five answers both with no card. Only after both come back clean does this indict the
    blend supervision itself.
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

    # Every seat blend-supervised. `merged_blend_s2026` keeps its place as the one that was already
    # there; the other five are blendsix's output, in the order that arm builds them.
    chosen = (
        "blend_ws",
        "blend_calib",
        "blend_ladder",
        "merged_blend_s2026",
        "blend_through",
        "blend_base",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
