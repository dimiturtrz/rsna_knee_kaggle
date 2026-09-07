"""oofsolo — `oof_teacher_s2026` ALONE. One member, no vote, no weight, no foreign side.

**AN INSTRUMENT, AND THE MESSAGE SAYS SO.** The member axis is closed and no slot goes to a seating,
selection, weighting or re-lensing operation. This is none of them: the entry is a SET OF ONE that the
board has never held, and what it measures is not who sits where but whether a single training-side
treatment survives the move to a test corpus that is not ours.

**WHAT IT PRICES THAT `oofseat` CANNOT.** `oofseat` (ref 56064173) seats the same member as one of seven
in a `LogitAverage`. A seat can be diluted — `dark6` already reads the seventh seat as near-inert
(knee-uco3) — so a flat there is ambiguous between "the distilled member adds nothing on test" and "an
eighth voice adds nothing to a seven-way vote, whoever it is". **A solo cannot be diluted.** It reads the
distillation term at full size, undivided, and that is the number that licenses seven hours of card.

**AND IT ARRIVES FIRST.** `oofseat` is the champion six plus one, served as several read groups; the
`onegroup` instrument's own projection for the champion six is **7.6 h**, so its score cannot land before
~06:40Z. This entry loads ONE member at ONE decode corner — a single read group — and scores in a small
fraction of that. The instrument that answers the bigger question is also the one that answers sooner.

**THE MOUNT.** `oof_teacher_s2026` lives on `kneemri-members-2`, the second weights dataset merged in by
`MemberPackage.spanning`. Here that path is not a side-effect, it is the whole load: a census reading
`1 of 1` is the proof, and `0 of 1` produces no submission at all rather than a quiet wrong number — which
is the one respect in which a solo is SAFER to read than a seat. `run_check` on the log before the slot
is recorded either way.

PRE-REGISTRATION, written before the number. Offline the member reads **0.8681** against `band_dense`'s
0.8578, +0.0107 and the highest solo this project owns. Using board(ensemble) ~ best single + 0.015 and
the champion's 0.944:

- **0.930 or above** — the distillation term transfers at full size to a corpus that is not ours. The
  six-seat retrain is licensed BY THE SOLO, and that licence holds even if `oofseat` reads flat, because
  a seat can be diluted and a solo cannot.
- **0.920 to 0.930** — the term transfers partially. Re-price the retrain against the shrunk term rather
  than the offline one; the pass-through slope, not the offline delta, is what the seven hours buys.
- **below 0.920** — the +0.0107 is holdout-specific. The retrain is NOT licensed, and the first suspect is
  the teacher's own fold structure: an OOF posterior fitted on our corpus can encode our corpus.

The refutation this cannot make: it cannot say whether the term SURVIVES ensembling. A solo that transfers
and a seat that does not would mean the gain is real and redundant with the six — which is exactly the
pair of readings `oofseat` and this entry make together, at two dilutions of one term.
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

    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # A set of ONE. `LogitAverage` over a single member is the identity, and it is named rather than
    # special-cased so the vote is the champion's own convention here too — the difference between this
    # entry and `oofseat` is the SET, and nothing else.
    chosen = ("oof_teacher_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
