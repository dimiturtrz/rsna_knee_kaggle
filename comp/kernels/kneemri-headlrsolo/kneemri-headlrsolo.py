"""Does the top of the solo transfer line hold, at the highest offline75 this project owns?

WHY THIS ENTRY EXISTS. Every plan here multiplies an offline gain by a TRANSFER SLOPE to predict a
board number, and the line doing that multiplying is fitted on eight solo rows whose top point is
`labelwise_depth24_drop40_long_s2026` at offline75 0.9053 -> board 0.941. The band we ship in is
therefore anchored by ONE observation, and every extrapolation above it is unconstrained. This member
sits at **0.9059**, above that anchor, so it is the first reading that tests the line at its own top
rather than inside its body.

THE MEMBER, AND WHAT IT IS NOT. `headlr_ship_s2026` is one variable off that anchor: `--head-lr` 2e-4
rather than 1e-3, the knob knee-xuarh pre-registered. It read **0.8702 midpoint / 0.9059 ledger75**,
peak epoch 17 of 24, converged, 909 studies. Against the anchor's 0.8679 / 0.9053 that is **+0.0023
midpoint and +0.0006 ledger75**, against a paired floor of 0.0062 — **DIRECTIONAL, not a result.**
Compressed at the top band's ledger75 slope of +1.09 the expected board delta is ~+0.0007, BELOW the
competition's own 0.001 reporting resolution. So this entry does not claim an improvement and cannot
demonstrate one: a knob that is invisible to the board is being shipped because the member is now the
highest point on our own ruler, not because the knob worked.

THIS IS AN INSTRUMENT AND IS SUBMITTED AS ONE. It is not a champion attempt. The pinned champion is
`tripleseat_v2` at 0.946 and this is predicted well below it; the standing entry is not at risk,
because the board keeps the best.

THE LINE HAS ALREADY MISSED ONCE AT ITS TOP, WHICH IS WHY THIS ENTRY IS WORTH A SLOT.
`extsolo_v1` (ref 56315030, 2026-09-17) shipped `labelwise_ext_drop40_long_s2026`, offline75 0.9045 —
the second-highest ever shipped. The eight-row line predicted 0.9403, 2 sd [0.9371, 0.9435]; it read
**0.937**, below the band at -2.1 sd, after three consecutive thousandth-exact calls lower down. One
point outside a 2 sd band does not break a ruler, but it is the FIRST point taken near the top and it
came in LOW. So the top of this line is now one hit (0.9053 -> 0.941) and one miss (0.9045 -> 0.937)
at essentially the same abscissa, and no plan can extrapolate it until a third point says which.

THE PREDICTION, fitted on SOLO ROWS ONLY, n=9 with that point in. Seat count is the board's strongest
covariate, so the whole-record slope is the wrong line for a set of one. OLS over the nine
single-member rows on a clean scope carrying an `offline75`:
`board = -0.0705 + 1.1164 * offline75`, residual sd **0.0019** (was 0.0016 at n=8 — the miss widened
it). At offline75 0.9059 it predicts **0.9408**, 2 sd band **[0.9370, 0.9446]**. That point is an
UPPER BOUND rather than a centre: the only two readings near it disagree by 0.004, which is twice the
residual sd, so the honest expectation spans the whole band.

PRE-REGISTERED, on the LINE rather than on the knob — the knob is board-invisible by construction:

  * **board 0.940-0.942** — MODAL. The line holds at its top, `extsolo_v1` was scatter, and the
    0.9053 anchor is joined by a second point. Extrapolation into our own band keeps its licence and
    the knob is confirmed free-and-directional.
  * **board >= 0.943** — the line UNDER-predicts at the top and `extsolo_v1`'s low read was the
    outlier. The gap to first place is then SMALLER than the +0.012 currently quoted off the ledger75
    ruler, and offline gains above 0.905 are worth more than the slope says.
  * **board <= 0.939** — two of two at the top read low, and that is no longer scatter: the line
    BENDS DOWN above ~0.904. Offline gains stop buying board where we actually ship, every plan
    priced off this slope over-predicts, and the gap to first place is larger than +0.012. It is the
    single most useful thing this slot can return, and it would retire the whole family of plans that
    end "raise offline75 by X and collect 1.1X of board".

WHAT IT COSTS: no GPU. The checkpoint was fitted by an arm that has already finished and its encoder
already mounts in every entry we ship.
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

    # A set of ONE, and it has to be: the reading is about a member, not about a seating. A second
    # name would need a selection step, and the holdout is the only thing available to select on.
    chosen = ("headlr_ship_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
