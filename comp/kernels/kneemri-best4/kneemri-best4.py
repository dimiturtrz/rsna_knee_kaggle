"""The champion four with its 392 px member replaced by the first member that reads a WIDER BAND.

`kneemri-solotop` scored **0.925** with res392 + lr1e-4 + calibrated + lr4e-5, and it is the entry every
number below is measured against. This is that four with ONE member swapped: `res392_ws_s2026` (0.8523
solo) out, `band_dense_s2026` (0.8578) in.

WHAT THE SWAP CHANGES, AND WHAT IT DOES NOT. The count is the same, and so is the number of decodes:
res392 was the only member of the 448 px group in this four, and band_dense is the only member of the
336 px / 20 slice / 0.10-0.90 group, so both configurations decode the test set exactly twice
(`Member.pixel_group` puts the band and the slice count in the key). Runtime, memory and the mixing rule
are unchanged. One member moved, and nothing else did.

WHY THIS MEMBER. Every entry this competition has scored for us reads the same 0.20-0.80 sagittal band,
which is a reading axis the whole pool has held FIXED while sweeping rate, resolution, targets and
depth. `band_dense` is the first member fitted outside it — 0.10 to 0.90 at 20 slices — and it is also
our best single member on the fold-0 holdout by 0.0053 over the next one. The two facts are not
independent: widening the band is what produced the score.

THE OFFLINE NUMBER, STATED HONESTLY. Rank-averaged over the 909-study fold-0 holdout this four reads
**0.8648** against the champion four's **0.8612** — +0.0036, which is UNDER this project's 0.0062 noise
floor and is directional only. It is worth a slot for a recorded reason rather than a hoped one: our
holdout under-prices ensemble diversity by about 2.4x (`kneemri-onedecode` lost 0.0059 offline and
0.014 on the board), and a member that varies the one reading axis nothing else varies is diversity in
the exact sense that measurement was about.

PRE-REGISTERED. **0.929.** The three clean single-member swaps this board has given us each moved by
about the displaced-vs-candidate SOLO gap (+0.0052 -> +0.006, +0.0002 -> +0.002, and solotop's own
+0.0031 -> +0.001), and this swap's gap is +0.0055. A score at or below 0.925 says the band member's
solo strength does not survive being mixed with three members that all read the centre — which would be
the first time the solo gap has failed to carry, and would point at the band being redundant with what
the other three already see rather than at the member being weak.
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

    chosen = ("band_dense_s2026", "band_span_s2026", "depth_regions_narrow_s2026", "calibrated_targets_s2026")
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
