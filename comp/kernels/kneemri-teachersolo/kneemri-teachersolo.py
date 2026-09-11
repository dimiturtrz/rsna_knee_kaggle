"""`oof_teacher_s2026` ALONE -- the #1 solo of 296, and the pair the board record does not have.

**WHY THIS SLOT, AND IT IS AN INSTRUMENT, NOT A CHAMPION ATTEMPT.** Nothing mechanism-backed is ready
to beat 0.944 at this mark, so per the pacing rule the slot buys a MEASUREMENT and says so.

D372 (researcher, tonight) interrogated `board_transfer` and the answer invalidates how every arm in
this project has been priced:

  - the +1.55 this repo has quoted all week is STALE. `board_transfer.py:24,34` says so in its own
    docstring ("EVERY NUMBER IN THIS DOCSTRING IS HISTORY"). The live whole-record figure is **+1.04**
    over 45 ensemble entries with both numbers / 828 orderable pairs.
  - the live transfer SATURATES: >=0.8668 offline gives +0.99 [+0.94, +1.05] (45 pairs); >=0.8695
    gives +0.82 [+0.71, +0.93] (4 pairs, the instrument tags it DIRECTION ONLY).
  - and at the OPERATING POINT there is nothing to fit at all. All 21 ensemble rows at board >=0.940
    carry `offline: None` -- their ff seats were fitted at `--fold -1` and have seen every holdout
    study. The highest rows holding BOTH numbers are 0.935-0.937, seven of them, spanning 0.0046 of
    offline. Any slope quoted at 0.944 is an extrapolation past the last resolved pair by the full
    width of the band.

So the record cannot price the leader gap, and the reason is a HOLE in the record: no honest, fold-0,
high-offline member has ever been given a board reading on its own. `oof_teacher_s2026` is exactly
that member -- **#1 of 296 ruler-consistent honestly-held solos @holdout909 midpoint, at 0.8685**
(`comp/members2.roster`, packaged in `kneemri-members-2` 2026-09-11T20:28Z). This slot fills the hole.

**PRE-REGISTRATION, written before the score.** Anchor: `stackfit_s2026`, the record's cleanest
resolved solo-side pair, 0.8631 offline -> 0.933 board. `oof_teacher` is +0.0054 of offline above it.
The three live regimes therefore make three SEPARATED predictions, and the separation (0.006) clears
D368's 0.003 estimability floor, which is what makes this a measurement rather than a hope:

  - **>= 0.941** -- consistent with the stale +1.55. The old constant survives at the top and every
    target computed with it (0.8696) stands. I do not expect this.
  - **0.937-0.940** -- consistent with the mid regime +0.99. The required solo is ~0.8732 and
    `distill_stack_max`'s best branch does not reach it. The distill-cross route is priced as short.
  - **0.934-0.936** -- consistent with the top-band saturation +0.82 or below. Solo quality is a
    spent lever at this board position and the remaining card time belongs to the label-corrected
    family, which is what the gold ruler and the T1 corrected ruler both order above the champion.
  - **<= 0.933** -- at or below a member 0.0054 of offline BELOW it. Then `holdout909` does not order
    members at the top of its own range, and the offline ruler stops being an argument for anything.

**WHAT IT IS NOT.** It is not an attempt on 0.944. A single fold-0 solo has never come near the
champion band; `holdsolo` read 0.928 from 0.8634 and `solo_v1` 0.868 from 0.8518. Reading this row as
a failed champion attempt would be reading the instrument backwards.

Partner rows for the transfer read: `holdsolo` (0.8634 -> 0.928), `stackfit` (0.8631 -> 0.933),
`solo_v1` (0.8518 -> 0.868), `ortho_solo_v1` (0.8452 -> 0.907).
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

    # The #1 solo of 296 @holdout909 midpoint, 0.8685. See the module docstring for the branches.
    chosen = ("oof_teacher_s2026",)
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
