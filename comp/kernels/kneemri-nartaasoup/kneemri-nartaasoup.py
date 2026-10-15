"""nartaasoup — basinsoupff's operation with ONE change: a teacher term from nartaa's CoAtNet-384 (knee-p844l).

WHAT THIS IS. `nartaa_soup_k4_s2026` is the uniform weight average of the shipping solo
`labelwise_ext_drop40_long_ff` (extffsolo_v1, board 0.945) and three 4-epoch seed-children (801/802/803)
forked off it at `--fold -1`, exactly as `basinsoup_ff_k4_s2026` (basinsoupff_v1, board 0.946) - except
each child also fits nartaa's soft readings at distill weight 0.5, the share nartaa holds in the champion
blend. A SINGLE checkpoint, a SINGLE forward pass: if it reads above 0.946, the public pile's strongest
reader has been folded into a member we own, rather than ridden as a foreign producer.

PRE-REGISTERED, board solo against basinsoupff_v1 0.946: >= 0.948 the teacher transfers; 0.946-0.947
null; <= 0.945 the pseudo-label axis closes for this teacher. The teacher trained on all data, so its
readings of our train studies are partly memorised and no offline ruler can price the child; the board
is the only reading this entry will ever have.
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
    chosen = ("nartaa_soup_k4_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
