"""The FIRST CNN board point this project owns: one resnet50d member, alone, no ensembling.

THE QUESTION THIS ANSWERS, AND WHY IT COMES BEFORE ANY CNN SEATING. On 2026-09-10
`kneemri.evaluation.diversity_slope` fitted, over 300 random subsets of 216 members, what a unit of
member decorrelation buys in ensemble margin once member QUALITY is controlled on both ends: a partial
slope of +0.1328 per unit, p 2.6e-35. Diversity is real. But the span the pool reaches WITHOUT paying
for it in quality is only 0.0718 wide, worth +0.0095 — everything more decorrelated than that is
decorrelated BECAUSE it is bad. So a foreign architecture family is worth card only if it is both
different AND good, and the term nobody has measured is the second one: this project has never put a
CNN on the board alone.

WHAT IT IS PRE-REGISTERED AGAINST. Five fold-0 solos already sit on the board with their offline
midpoint beside them, and they line up:

    ortho_dense_s2026   (DINOv3 ViT)  offline 0.8452 -> board 0.907
    labels_v4hybrid     (DINOv2)      offline 0.8524 -> board 0.922
    calib_span          (DINOv2)      offline 0.8545 -> board 0.921
    calib_dense_hold    (DINOv2)      offline 0.8634 -> board 0.928
    oof_teacher         (DINOv2)      offline 0.8685 -> board 0.926

`resnet_224_short_s2026` reads offline 0.8527 midpoint / 0.8842 ledger75, which puts it within 0.0003
of `labels_v4hybrid` and 0.0018 under `calib_span`. If a CNN transfers to the hidden test set the way
our transformers do, this lands 0.920-0.923. The branches were fixed before the slot was spent:

  * **>= 0.918** — the family transfers at least as well as DINOv2 per offline point. Its decorrelation
    is the GOOD kind, and the slope licenses spending card on a full CNN family rather than another
    variation of the recipe we already own.
  * **0.900-0.918** — it transfers like the DINOv3 point did, a little worse per offline point.
    Directional: the family is real and is not better, and a CNN seat stays a coin flip.
  * **< 0.900** — the offline number does not transfer for this family at all. Then the CNN's apparent
    diversity is the bad kind the slope's own reachable-span line warns about, and the cross-family
    thesis loses its best candidate rather than gaining evidence.

THIS IS NOT A SEATING, AND THAT IS DELIBERATE. The obvious move was `kneemri-cnnseat` — the champion
seven with this member added — and it is built and green. It was not shipped, because the board has
already answered that operation: `fleettwo_v1` (2026-08-29) put a CNN, `raptor_ft_effv2l480`, into the
champion seven and read 0.942 against the champion's 0.944, and the paired offline interval prices
adding this member at +0.0009, a sixth of that interval's own width. A seat is a question about the
MIXTURE; the mixture cannot be read until the family's own quality is on the board.

Our weights, our data layer, our submission path. The member was fitted by `kneemri.training.train` on
the report-derived labels, selected on a holdout split by REPORT-TEXT HASH so the template reports that
recur across studies cannot straddle the divide. The same fingerprint check guards it as guards every
published member: a member run under a different reading than it was fitted under loads cleanly and
predicts from the wrong image, and nothing else in the run would say so.

Everything below the bootstrap lives in `kneemri.kaggle.kernel_run`. Only the bootstrap — finding the
mounted kit and installing its wheels — has to be here, because it runs BEFORE the package it is
finding can be imported.
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

    # A set of ONE. `LogitAverage` over a single member is the identity, named rather than special-cased
    # so the vote is the champion's own convention here too — what differs from `v4solo` is the
    # ARCHITECTURE FAMILY of the member, and nothing else in this file.
    chosen = ("resnet_224_short_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
