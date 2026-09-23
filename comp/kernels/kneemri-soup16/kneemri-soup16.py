"""Does serving SHALLOWER than the fitted depth cost board on a lone member? The shallow half of deepsolo.

`basinsoupff_v1` is `basinsoup_ff_k4_s2026` served at its fitted 20 slices and boards 0.946, tying the
champion as ONE checkpoint. This entry is that member byte for byte, served at 16. The band stays the
fitted 0.1-0.9, so density alone moves. `deepsolo_v1` read the other side: 20 -> 32 cost -0.004 on the
board while the offline ladder (band_dense_s2026 0.8859 / 0.8877 / 0.8888 ledger75 at 12 / 20 / 24,
band held) projected a gain, so on this axis the board is the only ruler that has been right.

INSTRUMENT, not a champion attempt. It prices the decode knob knee-uy8z0 names: a fifth fewer DICOM
reads for the solo, on both boards, with the standing entry untouched.
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

SERVED_SLICES = 16
"""The one variable. The member was fitted at 20 and is served here at 16, band unchanged.

The shallow side of the ladder `deepsolo_v1` read on the deep side: 20 -> 32 cost -0.004 on the board,
against an offline ladder that projected +0.0024, so the board has never agreed with the offline depth
curve and the shallow rung has never been read on it. member_curve prices 16 at -0.0008 offline, an
UPPER bound, since its window also narrows the band. 16 is the knee knee-uy8z0 named: 20 -> 16 drops a
fifth of the DICOM decodes, the I/O-bound half of a four-CPU Kaggle host.

INSTRUMENT, pre-registered against basinsoupff_v1 (0.946, this member at its fitted 20):
>= 0.946  the shallow rung is free, and the solo buys a fifth of its decode for nothing.
<= 0.944  the fitted depth is the optimum from BOTH sides, and serving-time depth closes for solos.
"""

SERVED_BAND = (0.1, 0.9)
"""The member's own fitted band, held so DENSITY is the only thing that moves.

`SharedRead` spreads its indices across the band it is given (`SlotReader.band_indices`). Moving the
band as well is what charged band loss to depth in the retracted knee-uy8z0 ladder, and the offline
ladder this entry extrapolates from (0.8859 / 0.8877 / 0.8888) was itself measured band-held.
"""


def main() -> None:
    boot()

    from kneemri.inference.budget import (  # noqa: PLC0415 - the kit is on sys.path only after boot
        KERNEL_SECONDS,
        Budget,
    )
    from kneemri.inference.ensemble import Serving, SharedRead  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # A set of ONE, identical to `basinsoupff_v1`'s. `LogitAverage` over a single member is the identity,
    # named rather than special-cased so the vote stays the champion's convention here too.
    chosen = ("basinsoup_ff_k4_s2026",)

    # `SharedRead` over a set of one is not a collapse — there is nothing to collapse — it is the only
    # surface this repo has for serving a fitted member at another depth, and the kernel and
    # `holdout_backfill --served-slices` reach the same object so a regroup has one definition.
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(
            vote=LogitAverage(),
            budget=Budget(seconds=KERNEL_SECONDS),
            group=SharedRead(SERVED_SLICES, SERVED_BAND),
        ),
    ).run()


if __name__ == "__main__":
    main()
