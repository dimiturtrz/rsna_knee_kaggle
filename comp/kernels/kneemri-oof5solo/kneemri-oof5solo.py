"""oof5solo — `seat_oof5_distill_s2026` ALONE. One member, no vote, no weight, no foreign side.

**AN INSTRUMENT, AND THE MESSAGE SAYS SO.** The member axis is closed and no slot goes to a seating,
selection, weighting or re-lensing operation. This is none of them: a SET OF ONE the board has never
held, reading whether a training-side treatment survives the move to a corpus that is not ours.

**WHAT IT IS, AND WHY IT REPLACES `oofsolo`.** `oofsolo` shipped the same treatment from a LEAKY
teacher: each of the four `oof_fold_k` blocks was fitted at `--fold k`, which trains on fold 0 — the
fold its student was scored on. The board caught it without knowing why (0.926 from an offline read
ABOVE `holdsolo`'s 0.928 → 0.928). This member removes the leak by removing the HOLDOUT instead of
weakening the teachers: it fits at `--fold -1` against `teacher_oof5.npz`, five REAL out-of-fold blocks
covering all 4,406 studies, each predicting only the fold it held out. There is no evaluation set left
for a memorised label to leak into, by construction.

**AND IT HAS NO OFFLINE NUMBER, ON PURPOSE.** `--fold -1` holds nothing out, so this entry's score IS
its first reading. That is not a gap in the pre-registration; it is why the slot is worth spending.

**WHAT THE OTHER HALF ALREADY SAID.** `seat_clean_distill` attacked the same leak from the opposite
side — keep the holdout, weaken the teachers with `--exclude-fold 0` — and read **0.8634**, matching its
leaky twin's 0.8633 out of teachers priced −0.0067 weaker on their own holdouts. So offline, removing
the leak cost −0.0001 and the +0.0107 distillation term is a treatment effect rather than memorised
labels. This entry asks whether that term is also a BOARD term.

PRE-REGISTRATION, written before the number. Its control on the board is `holdsolo` at **0.928** — the
best clean solo row we own — and the comparison is deliberately against that rather than against
`oofsolo`'s 0.926, because 0.926 is the number the leak produced and re-using it would make the leak the
baseline.

- **0.933 or above** — the distillation term transfers at full size to a corpus that is not ours, and it
  is the per-seat term `retrain_six_distill` applies six times. The retrain is licensed BY THE SOLO.
- **0.929 to 0.932** — the term transfers, damped. Re-price the retrain against the SLOPE (board delta ÷
  offline delta), never against the offline +0.0107; that slope is what the 8.5 h actually buys.
- **0.925 to 0.928** — flat against `holdsolo`. The term is holdout-specific and the retrain's ≤0.946
  band is the live one; its named successor is a PER-SEAT teacher rather than a re-run.
- **below 0.925** — worse than the leaky entry. Then the suspect is not the leak at all but distillation
  onto a `--fold -1` fit: with no holdout, nothing stopped the student from over-following the teacher,
  and `--distill-weight 0.7` was chosen on held runs. Do not spend the retrain until that is priced.

The refutation this cannot make: it cannot say whether the term survives ENSEMBLING. A solo cannot be
diluted, which is the point, but it also cannot show redundancy — `retrain_six_distill`'s partial-swap
entry is what reads that, and the two are read together at two dilutions of one term.
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

    # A set of ONE. `LogitAverage` over a single member is the identity, and it is named rather than
    # special-cased so the vote is the champion's own convention here too — the difference between this
    # entry and `oofsolo` is the MEMBER, and nothing else.
    chosen = ("seat_oof5_distill_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because the seated entries subtract and this file is read beside them.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
