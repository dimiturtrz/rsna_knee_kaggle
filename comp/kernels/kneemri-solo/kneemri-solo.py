"""Submission from ONE model: the single strongest member, with no ensembling at all.

THE QUESTION THIS ANSWERS. Every entry so far has been a rank-average of several members, and fold and
seed ensembling are now out on purpose: they buy variance reduction rather than signal, and each
member costs a full run on 80% of the corpus. What nobody here has measured is the PRICE of dropping
them - if one model loses little, the whole ensembling apparatus can go and the GPU it consumes moves
to mechanisms. `strongest(1)` picks by recorded holdout, which is fold 2 at 0.8518.

Read it against the three-member entry pushed alongside it: same members, same reading, same kit, so
the only difference is how many models vote.


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
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
            check=True,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.kernel_run import KernelRun

    # ONE model, no ensemble — see the module docstring.
    KernelRun(members=1).run()


if __name__ == "__main__":
    main()
