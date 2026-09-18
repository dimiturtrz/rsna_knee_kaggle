"""Is the full-corpus refit an ADDITIVE term, or a CEILING the refit converges to?

WHY THIS ENTRY EXISTS. The refit (`--fold -1 --ema`) is the only positive BOARD term this repo owns:
`interpretations/kneemri/2026-09-18_state_of_the_levers.md` shows 62 ensemble attempts of five or more
members have never beaten the best of seventeen solos, while every tick this project holds came from a
refit. Two readings of that term are live and nothing shipped so far can separate them.

  ADDITIVE - a refit adds about +0.005 to whatever its honest twin scores.
  CEILING  - refits converge near 0.9445 regardless of where the twin started (knee-33iwi).

WHY THE TWO EXISTING READINGS CANNOT DECIDE IT. Both refit twins we have shipped sit at the SAME
abscissa: `labelwise_ext_drop40_long` at offline75 0.9045 and `labelwise_depth24_drop40_long` at 0.9053,
tied inside the 0.0062 paired floor. Their refits read +0.008 and +0.003. Two points at one abscissa
measure the term's VALUE and say nothing about its SHAPE, which is the thing in dispute.

WHAT THIS MEMBER IS. `calib_span_ff` is `calib_span_s2026`'s flags plus `--fold -1 --ema`
(`comp/members.roster:390`; the pairing is the corrected one at roster:489-492, which replaced an earlier
mis-identified twin). `calib_span_s2026` scored **board 0.921** solo as `spansolo_v1` at offline75
**0.8895** - 0.016 below the other two twins, far outside any floor. It is the first point that can see
the shape. The checkpoint has existed since 2026-08-24; this entry costs no card at all.

PRE-REGISTERED, and they are 18 ticks apart:

  * **board near 0.926** (additive, 0.921 + 0.005) - the term is a genuine addition. An honest-base gain
    is then directly bankable at +0.005 on top, and raising the honest base becomes the schedule.
  * **board near 0.9445** (ceiling) - the refit lever is EXHAUSTED. No refit of anything we own reaches
    the pinned 0.946 champion, the remaining eleven ticks to first place must come from supervision
    quality or slice-to-study aggregation rather than from the fit, and the refits still queued behind
    this one are predictable nulls whose slots are freed.
  * **anything else** - the term has a shape neither reading predicted, which is itself the finding.

The board's scatter among offline-tied entries is sd 1.9 ticks, so this separates at about 9 sigma. The
comparable test on `headlr_ship_ff` separates at 3 ticks, about 1.5 sigma; same slot, six times less
signal, which is why this one is spent first.

THIS IS AN INSTRUMENT, NOT A CANDIDATE. P(beats the champion) is about 0.02. It does not displace the
pinned entry. bead knee-uh1j5.
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

    # A set of ONE, and it has to be: the reading is about the REFIT TERM on one member, and a second
    # name would add a seating effect on top of the term this entry exists to measure.
    chosen = ("calib_span_ff",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
