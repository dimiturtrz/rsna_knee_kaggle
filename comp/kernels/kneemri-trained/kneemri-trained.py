"""Submission from OUR OWN trained members: the strongest five of the all-ours package.

ALL SIX, INCLUDING THE DINOv3 ONE. This ran five for a week because the v3 member's weights were not
mounted — and the code to find them (`Encoder.locates`, which tells `vits16` from `vitb16`) was already
written. The missing piece was one line of `dataset_sources`, not the "open work" the docstring claimed:
excluding the member was faster than mounting it, and that is the only reason it was excluded twice.

The cost of including it is honest and separate: `lora_r8_v3small_224` reads 224 px, which is its own
pixel group, and a group costs a FULL EXTRA DECODE of the split (`predict_groups` calls the decoder once
per group). Whether our weakest member at 0.8217 earns that is a measurement nobody has made.

Our weights, our data layer, our submission path. The members were fitted by `kneemri.train` on the
report-derived labels, selected on a holdout split by REPORT-TEXT HASH so the template reports that
recur across studies cannot straddle the divide.

The same fingerprint check guards these members as guards the published ones: a member run under a
different reading than it was fitted under loads cleanly and predicts from the wrong image, and nothing
else in the run would say so.

Everything below the bootstrap lives in `kneemri.kernel_run`, shared with the efficiency kernel. Only
the bootstrap — finding the mounted kit and installing its wheels — has to be here, because it runs
BEFORE the package it is finding can be imported.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")


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
            zipfile.ZipFile(archive).extractall("/tmp/kit")
            return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/model.py")))).parent.parent
    raise FileNotFoundError("no kneemri package under /kaggle/input")


def install_wheels(kit: Path) -> None:
    """jaxtyping is absent from Kaggle's image and our modules import it at module top."""
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if wheels:
        subprocess.run(  # noqa: S603 - pip installing wheels we mounted ourselves, fixed argv, no shell
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
            check=True,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.kernel_run import KernelRun  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs

    KernelRun().run()


if __name__ == "__main__":
    main()
