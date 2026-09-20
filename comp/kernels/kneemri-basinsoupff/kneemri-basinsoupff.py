"""Weight souping on the shipping solo: the one weight-space operation this project has measured POSITIVE.

WHAT THIS IS. `basinsoup_ff_k4_s2026` is the uniform weight average of FOUR checkpoints - the shipping
solo `labelwise_ext_drop40_long_ff` (= `extffsolo_v1`, board 0.945) and three 4-epoch seed-children
forked off it at `--fold -1` with `--init-scope all`, seeds 301/302/303. It is a SINGLE checkpoint with
a SINGLE forward pass. Not an ensemble: the averaging happened in weight space, once, on disk.

WHY THIS OPERATION AND NOT A NEW RECIPE. basinsoup is the only weight-space op this project has ever
measured positive: +0.0016 ledger75, parent 0.9073 -> basinsoup_k3 0.9089, on the fold-0 line.
`basinsoup2` then established why it pays and why it cannot be pushed - every child of a converged EMA
parent lands BELOW it, so the average can only interpolate between one good point and worse ones, and
+0.0016 is the reading of the least-travelled children available. Doubling the cycle bought real child
divergence and the product FELL. So 4 epochs at backbone-lr 2e-5 is the measured optimum of this op
rather than a knob with a sweep in it, and it is copied here unchanged.

WHY K=4 AND NOT K=3, DECIDED BEFORE EITHER PRODUCT EXISTED. Both are full-data fits, so neither has an
offline ruler and the choice could otherwise be made by looking at nothing. K=3 averages three children
that basinsoup2 showed all land BELOW a converged EMA parent - its expected position is the mean of
three worse points. K=4 includes the parent, so it is a quarter-weight perturbation of a checkpoint
already known to board 0.945: the basin's centre by construction, and the only member of the pot with a
long fit behind it. K=4 is the anchored candidate and it is what ships.

WHY THE PARENT IS THE SHIPPING ENTRY RATHER THAN THE BEST OFFLINE MEMBER. Every other route priced this
week proposes a NEW member of a cluster whose `_ff` refits all board 0.943-0.945 (knee-xjxdn: the top
four recipes sit inside the 0.0053 paired floor and none has produced 0.946). This entry proposes no new
member at all - it applies a measured +delta ON TOP of the 0.945 entry already held.

WHY THE BOARD IS THE ONLY RULER. Every checkpoint in the pot is `--fold -1`: full data, nothing held
out. A holdout record for any of them would be a number about training data, so the arm deliberately
ships no `holdout_backfill` and no `ledger75` tail. That omission is the reading, not a gap.

PRE-REGISTERED IN THE ARM HEADER, before any child existed, against `extffsolo_v1`'s 0.945:

  * **0.947 or above** - the op transfers to the full-data line. A new champion and the best solo this
    project owns, from zero new recipe and one hour of card.
  * **0.946** - a tie at the board's own 0.001 resolution, and a tie with the eight-seat champion from
    ONE member. Keep the standing entry; the op is real but under the blur.
  * **0.945 or below** - weight souping does not transfer to the full-data fit, and the family closes on
    its FOURTH and last explanation.

HONEST PRICE. The measured effect is +0.0016 ledger75 on the fold-0 line, which maps to ~+0.0017 board
at the top band's +1.09 transfer - UNDER the 0.0022 top-band residual sd. So the modal outcome of this
slot is a tie, and the pre-registration says so. It is worth the slot because the op is cheap, already
paid for in card, and applies to whatever member ships next; it is not worth a second slot to re-read.

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
    # special-cased so the vote stays the champion's convention here too. The member is DARK:
    # it holds out nothing, so no offline number for it exists or can be made, and the board
    # number this entry returns is the only reading it will ever have.
    chosen = ("basinsoup_ff_k4_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
