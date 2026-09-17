"""Is the full-fit premium a constant, or a property of the member it was measured on?

WHY THIS ENTRY EXISTS. Every plan that ends "refit at `--fold -1` and ship it" multiplies by a REFIT
TERM, and that term has exactly one measurement behind it. `l75solo_v1` shipped
`labelwise_depth24_drop40_long_s2026` held out (board 0.941) and `ffsolo_v1` shipped the same recipe
and seed refit (`_ff`, board 0.944, ref 56296698). One member, one pair, **+0.003** — and `l75solo`'s
own pre-registration said that anything inside +/-0.004 was inside 2 sd, so it closed by calling the
premium FREE rather than by resolving it. A single unresolved pair is not something to multiply by.

THE SECOND PAIR, AND IT IS ALREADY HALF-BUILT. `extffsolo_v1` (ref 56312235) is
`labelwise_ext_drop40_long_ff` at board **0.945**. Its held-out twin,
`labelwise_ext_drop40_long_s2026`, has been seated in NINE board rows and never shipped alone. This
entry is the missing half, and it costs no GPU: the checkpoint has been on the shelf since 2026-09-15
and its encoder already mounts in all nine.

THE PREDICTION, fitted on SOLO ROWS ONLY. Seat count is the board's strongest covariate, so neither
the whole-record nor the top-band slope is the right line for a set of one. `comp/board.jsonl` now
holds eight single-member rows on a clean scope carrying an `offline75` — seven, plus `l75solo_v1`
itself, which is what makes this fit tighter than the one `l75solo` was priced against:

    ortho_dense_s2026 0.8748 -> 0.907      resnet_224_long_s2026 0.8764 -> 0.907
    resnet_224_short_s2026 0.8842 -> 0.915 labels_v4hybrid_s2026 0.8874 -> 0.922
    calib_span_s2026 0.8895 -> 0.921       calib_dense_hold_s2026 0.8949 -> 0.928
    stackfit_s2026 0.8962 -> 0.933         labelwise_depth24_drop40_long_s2026 0.9053 -> 0.941

OLS over those eight: `board = -0.1131 + 1.165 * offline75`, **residual sd 0.0016**, n=8. At offline75
0.9045 this member predicts **0.9403**, 2 sd band **[0.9371, 0.9435]**. The board reports at 0.001.

PRE-REGISTERED, on the REFIT TERM `0.945 - board`, against the +0.003 the first pair read:

  * **board 0.939-0.942** — term +0.003 to +0.006. The two pairs AGREE inside scatter and the premium
    is a constant near +0.004. Refit-and-ship may be planned as a step with a known size, which is
    what the head_lr path (knee-xuarh, knee-h25fx) needs before it spends a card on a refit.
  * **board >= 0.943** — term <= +0.002. The premium is zero twice over, and `l75solo`'s "holding 909
    studies out is FREE" graduates from one reading to two. The held-out ledger row costs nothing and
    should be kept on every deliverable.
  * **board <= 0.938** — term >= +0.007, more than twice the first pair's. The premium is then
    MEMBER-SPECIFIC, not a constant, and no plan may multiply by it; a refit becomes a measurement
    that must be taken per member.

Every branch also gives the eight-row solo transfer its second out-of-sample test, and the first one
taken below the point that anchors its top end.

THIS IS AN INSTRUMENT AND IS SUBMITTED AS ONE. The modal branch sits six thousandths under the 0.946
pinned in `comp/champion.json`; a single held-out member is not offered as a way past it. The slot
buys the constant, not the rank.

Our weights, our data layer, our submission path. Everything below the bootstrap lives in
`kneemri.kaggle.kernel_run`.
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

    # A set of ONE. `LogitAverage` over a single member is the identity, named rather than
    # special-cased so the vote stays the champion's convention here too. One name is also what
    # makes the reading clean: a set of two would need a selection step, and the holdout is the
    # only thing available to select on.
    chosen = ("labelwise_ext_drop40_long_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
