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
import subprocess
import sys
import zipfile
from pathlib import Path

WORKING = Path("/kaggle/working")


def find_kit() -> Path:
    """The directory to put on `sys.path`, found by looking for the package rather than the wrapper.

    Mounts nest by kind (`/kaggle/input/{competitions,datasets,models}/…`) and a tree uploaded with
    `--dir-mode zip` is expanded with its wrapping directory REMOVED. Three runs were spent on that;
    the rule is to search for the artefact you need, never the container assumed to hold it.
    """
    marker = next(iter(sorted(Path("/kaggle/input").rglob("kneemri/model.py"))), None)
    if marker is not None:
        return marker.parent.parent
    archive = next(iter(sorted(Path("/kaggle/input").rglob("kneemri_src.zip"))), None)
    if archive is None:
        raise FileNotFoundError("no kneemri package under /kaggle/input")
    zipfile.ZipFile(archive).extractall("/tmp/kit")
    return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/model.py")))).parent.parent


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
