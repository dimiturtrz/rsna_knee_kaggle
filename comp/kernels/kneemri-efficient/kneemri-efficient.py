"""The Efficiency Prize entry: two members, no overlapping windows.

Identical to the accuracy kernel except for two arguments, which is the whole point of putting the work
in `kneemri.kernel_run`.

WHY TWO AND NOT ONE, which is what this kernel shipped until 2026-08-13. It said "our own member curve is
flat" and cited a SEED ensemble's holdout, where members two and three bought +0.0018 and +0.0007. The
leaderboard has since answered the same question directly, and the curve is not flat at the bottom:

    entry                        members  passes/study    LB
    solo, our strongest member       1         10        0.868
    whole-stack                      2          2        0.882
    whole-stack                      3          3        0.882
    ours v2, windowed                5         50        0.888

One member costs 0.020 against five. TWO whole-stack members cost 0.006 for a twenty-fifth of the
forward passes, and a third buys nothing on top. So two is the entry, and the file's own justification
was arguing for a configuration its ledger had already refuted.

WHY IT IS THIS CHEAP is measured rather than hoped: a whole-stack member's slot embedding is already the
elementwise max over every slice, so it reads ONE window and the ten-fold window TTA multiplier does not
apply to it at all — worth +0.0146 to a windowed member and exactly 0.0000 to this one. The accuracy
work and the efficiency work want the same member, which is not what the trade-off framing assumed.

The pair is chosen by `strongest(2)`, which ranks by each member's own recorded holdout; that is not
necessarily the same pair as the 0.882 entry above, only the same cost class.

It does not compete with the accuracy entry: 347 of 897 teams carry an efficiency score strictly below
their public best, so a cheap submission adds a candidate rather than displacing one.

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

    KernelRun(members=2, overlap=False).run()


if __name__ == "__main__":
    main()
