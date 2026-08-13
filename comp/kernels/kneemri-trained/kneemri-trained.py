"""Submission from OUR OWN trained members: the strongest five of the all-ours package.

FIVE AND NOT ALL SIX, and the reason is a slot. The package's sixth member is `lora_r8_v3small_224`,
which is fitted on DINOv3 - and this kernel mounts DINOv2 small and base only, with internet disabled,
because wiring a v3 member through to a submission is still open work (knee-98j). Running all members
would reach for weights that are not there and fail the run, which is how the 0.500 entry was spent.
`strongest(5)` drops exactly that member, which is also our weakest by 0.024.

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

    KernelRun(members=5).run()


if __name__ == "__main__":
    main()
