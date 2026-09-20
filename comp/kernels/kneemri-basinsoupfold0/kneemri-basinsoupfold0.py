"""INSTRUMENT, not a candidate: what does the FOLD PENALTY cost on the board?

WHAT THIS IS. `basinsoup_k3_s2026` is the uniform weight average of three 4-epoch seed-children forked
off `labelwise_ext_drop40_long_ema_s2026` at `--fold 0` with `--init-scope all`. It reads 0.9089
ledger75 offline. `basinsoupff_v1` (board 0.946, ref 56406877) is the SAME souping operation carried out
on the SAME recipe's full-data refit. Single checkpoint, single forward pass, same serving line, same kit.

WHY THIS IS AN INSTRUMENT AND SAYS SO. It cannot be expected to beat the pin: it trained on two thirds of
the corpus while every solo this project has ever shipped is `--fold -1`. It proposes no new mechanism.
Its job is to measure a CONSTANT this project has used implicitly in every pricing argument it has ever
made and has never once measured: how much board a fold-0 member gives up for the third of the corpus it
did not see.

WHY THAT CONSTANT IS WORTH A SLOT. Every offline number on this shelf is fold-0 ledger75, and every board
candidate is `--fold -1`. The solo board-transfer (knee-0dfys: slope +1.071, intercept -0.0307, resid sd
0.0019) was fitted on full-data members only, so applying it to a fold-0 score answers "what would a
FULL-DATA member with this offline number board?" -- 0.9089 maps to 0.9425. The fold-0 member must board
at or below that, and the shortfall IS the penalty. Measure it once and every fold-0 number on the shelf
becomes a board prediction without a slot, forever. Refuse to measure it and the shelf stays unpriceable
in the only currency that counts, which is the state it has been in all week.

WHAT IT SHARES WITH TONIGHT'S OTHER SLOT, and this is deliberate. `kneemri-basinsoupffk3` ships the K=3
average on the FULL-DATA line against the K=4 that boarded 0.946. This entry and that one differ from
`basinsoupff_v1` in different single terms -- this one in the FOLD, that one in K. If the K entry reads
0.946, K is inert and this entry's shortfall is the fold penalty alone. If it does not, the K term is
known from it and subtracts out of this one. Two slots, two terms, neither confounded with the other.

PRE-REGISTERED, against the 0.9425 the transfer predicts for a full-data member of this offline score:

  * **0.942-0.943** - the fold penalty is at or under the board's own 0.001 resolution. The transfer
    applies to fold-0 numbers directly, and the entire fold-0 shelf becomes board-predictive tonight.
    This is the branch worth the slot.
  * **0.935-0.941** - a penalty of 0.002-0.008: real, measurable, and subtractable. The shelf still
    prices, with a constant carried.
  * **0.934 or below** - a penalty of 0.009 or more, which is LARGER than the 0.011 gap to first place.
    Fold-0 ledger75 would then be unable to rank board candidates at all, and every arm this project has
    priced on it -- including the souping family -- has been priced on a ruler that cannot reach the
    board. That is the most consequential of the three and the reason not to guess this number.

WHAT IT DOES NOT DO. It does not attack the 0.011 gap, and it is not expected to approach the champion.
The pacing rule's instruction when nothing mechanism-backed can beat the pin is to spend the slot on a
declared instrument rather than a reseating. This is that instrument, and the submission message says so.
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

    # A set of ONE. `LogitAverage` over a single member is the identity, named rather than
    # special-cased so the vote stays the champion's convention here too. The member is DARK:
    # it holds out nothing, so no offline number for it exists or can be made, and the board
    # number this entry returns is the only reading it will ever have.
    chosen = ("basinsoup_k3_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
