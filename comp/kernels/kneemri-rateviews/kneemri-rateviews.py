"""The same upgraded pair as `kneemri-ratepair`, served under TWO rigid views instead of one.

One variable against `kneemri-ratepair`: identical members, identical weights, identical decode, and
only `Serving.views` moves. The views axis is the last test-time axis this project has not spent —
overlap TTA buys a whole-stack member nothing (knee-eeo), and rigid jitter is the one thing left that
gives the encoder a genuinely different sampling grid of the same anatomy.

IT HAS NEVER BEEN MEASURED, and the reason is a defect rather than a result: `kneemri-views` returned
COMPLETE with a BLANK public score on 2026-08-17 (knee-c9q4). It was the only arm that tripled decode,
so a hidden-set timeout is the reading that fits — the visible three studies cannot project a cost the
full set pays. Two things changed since. The kit carries the afford-sheds-views mitigation, so the
worst case now degrades to the identity view rather than returning blank; and this entry runs TWO
views, not three, which is the cheapest form of the question that can still answer it.

EXPECT A FEW THOUSANDTHS, said before measuring. Training already augments with rotation of exactly
this width, so the model has been taught the invariance this exploits — the honest prior is well under
the +0.0146 that slice-coverage TTA bought a windowed member, and probably under the board's own
resolution. What the slot really buys is the CELL: an untested axis stays untested until an entry
comes back with a number in it.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# >>> KERNEL-BOOTSTRAP
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


# <<< KERNEL-BOOTSTRAP


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - kit is on sys.path only after install_wheels

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.test_time import SpatialViews  # noqa: PLC0415 - same

    chosen = ("calibrated_targets_s2026", "lr4e-5_ws_s2026")
    views = SpatialViews(degrees=(0.0, -6.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(views=views)).run()


if __name__ == "__main__":
    main()
