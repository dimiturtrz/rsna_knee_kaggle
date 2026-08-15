"""The bootstrap every submission kernel opens with, in ONE place, spliced into each by `build_kernels`.

A Kaggle notebook is a single pasted file that runs before our package exists on it: it has to find the
mounted kit and install its wheels with nothing but the standard library. So the bootstrap cannot be
imported — it must be copied — and copying is what it did, into seven kernels.

That cost real time twice. The 248-second `rglob` this file's `find_kit` exists to avoid had to be
fixed in seven places, and `install_wheels` still sits on disk in three variants whose behaviour agrees
and whose docstrings do not. A submission runs the notebook, not this repo, so a kernel that missed a
fix is a slot spent on code we had already corrected — which has happened.

The region between the markers below is the canonical text. `python -m comp.build_kernels` writes it
into the matching region of every kernel that carries the markers, and `--check` fails when a kernel's
copy has drifted, so the copy is a build product rather than a thing anyone maintains.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# The imports sit ABOVE the region on purpose: every kernel already opens with these five, and a
# region that carried its own imports would splice a second copy into each of them. Here they are what
# makes this file a real module — checked by ruff and pyrefly like everything else, which the pasted
# copies never were.

# >>> KERNEL-BOOTSTRAP
SERIES = {"train_series", "test_series", "train_images", "test_images"}
"""Competition mounts holding one directory per study. `rglob` walks every one of them."""

KIT_DEPTH = 5
"""How deep under /kaggle/input a mounted kit has ever sat, plus room. A bound rather than a guess: it
turns an unbounded walk into a bounded one even if a mount arrives with a name this does not know."""


def find_kit() -> Path:
    """The directory to put on `sys.path`, found WITHOUT walking the competition data.

    The obvious spelling — `Path("/kaggle/input").rglob("kneemri/model.py")` — cost **248 seconds** on
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
        if here.name == "kneemri" and (here / "model.py").exists():
            return here.parent
        archive = here / "kneemri_src.zip"
        if archive.exists():
            zipfile.ZipFile(archive).extractall("/tmp/kit")  # our own kit, built by build_kit.py
            return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/model.py")))).parent.parent
    raise FileNotFoundError("no kneemri package under /kaggle/input")


def install_wheels(kit: Path) -> None:
    """Install the kit's wheels offline, before anything imports our package.

    The runtime probe checked the libraries the PIPELINE needs (pydicom, timm, transformers…) and found
    them all present — but not the ones our own modules import. `jaxtyping` is not in Kaggle's image and
    `kneemri.model` imports it at module top, so the kit needs wheels after all. Probe what you import,
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
