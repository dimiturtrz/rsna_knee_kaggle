"""The FOLD-0 half of a matched pair — four seats, fitted on 3,497 studies.

**THE PAIR, AND WHY IT IS THE ONLY THING THIS SLOT BUYS.** `kneemri-fitfour` serves the SAME four seats fitted the
other way. Nothing else moves: same seat identities, same count, same vote, same kit, same decode. The
single free variable is the FOLD the members were fitted at, so the difference between the two board
scores is the price of full-corpus fitting and nothing else.

**WHY THE PROJECT NEEDS THAT NUMBER.** The record already holds one such pair: `twinfive_v1` (fold 0)
read 0.935 and `fiveship_v1` (full-fit, same five seats) read 0.941, a delta of **+0.006**. That single
pair is currently the whole evidence base for the claim that full-corpus fitting is worth six
thousandths of board, and that claim is load-bearing: it is the final step of the only arithmetic path
this project has to the owner's 0.953 target, and it licenses roughly thirty hours of card spent
refitting a fleet at `--fold -1`. One pair is an anecdote. A second, at a different seat count with a
different seating, is what makes it a constant.

**PRE-REGISTRATION, written before either score.** Read the DELTA (full-fit minus fold-0), never either
score alone. The board quantises to 0.001 and the resolution floor on a single pair is 0.0040.

- **delta >= +0.004** — full-fit is a real and repeatable board term. The `twinfive`/`fiveship` +0.006 is
  a constant rather than a coincidence, and the refit-the-fleet step of the 0.953 path is licensed.
- **delta +0.001 to +0.003** — directional and below the floor. The +0.006 was partly seat-count or
  seating luck; the fleet refit is still plausible but may no longer be assumed to pay +0.006, and the
  path's arithmetic has to be rewritten at the smaller number before any card is spent on it.
- **delta 0.000 or negative** — the +0.006 does not replicate. Full-fit is NOT the explanation for the
  champion band, the last step of the 0.953 path is dead as written, and thirty hours of card is saved.
  That is the outcome this instrument is most valuable for.

**WHAT IT CANNOT DO.** Two board reads on one hidden test set cannot separate "full-fit" from "these
four members' full-fit twins". It also cannot say whether the term keeps paying at seven seats, where
the champion sits — the pair is at four. Both limits are the price of a matched contrast that exists at
all; no larger matched pair is buildable, because only these four seats have twins on both sides.

**THIS HALF ALSO CARRIES AN HONEST OFFLINE NUMBER,** which its twin cannot. Every seat here is a fold-0
fit, so all 909 holdout studies are out-of-sample for the whole pool and the entry's offline score may
be entered beside the ledger's. That makes it a third honest offline-to-board anchor as well as half of
the pair, at no extra cost.
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

    # The four seats that have full-fit twins, fitted at fold 0. Matched against kneemri-fitfour.
    chosen = (
        "calib_dense_hold_s2026",
        "calib_span_s2026",
        "through_dense_s2026",
        "small_batch6_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
