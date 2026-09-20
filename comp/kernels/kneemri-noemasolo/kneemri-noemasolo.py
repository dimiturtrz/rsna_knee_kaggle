"""Does `--ema` cost a SOLO what the matched groups say it does?

WHAT THIS MEASURES, STATED FIRST. This is `extffsolo_v1`'s member with ONE flag removed and nothing
else moved - `labelwise_ext_drop40_long_noema_ff` is `labelwise_ext_drop40_long_ff`'s byte-identical
command line without `--ema`. The control is not an offline number and cannot be: both members fit
`--fold -1`, so neither holds a study back and no ledger row can price either. The control is the
STANDING BOARD ROW of `extffsolo_v1` at 0.945 (ref 56312235). One variable, one ruler.

WHY THE FLAG WAS THERE, AND WHY IT IS NOT ANYMORE. `labelwise_ext_drop40_long_ff.sh` states its own
reason: "--fold -1 --ema - the champion's own fit, matched so the new seat is shaped like the ones
beside it rather than like our honest pool." That is an ENSEMBLE-SHAPE argument. It buys nothing about
the member and everything about its six neighbours. The object that ships is a SOLO. There are no
neighbours, so the argument is spent - and this is the first entry to test what it was costing.

THE MECHANISM, WHICH IS WHAT LICENSES THE SPEND. `kneemri/models/ema.py` states its own purpose in its
docstring: "An EMA is not a better model so much as a less arbitrary one - it stops the answer depending
on exactly which step training happened to stop at." It damps the 0.0062 three-seed wobble. It buys
VARIANCE, not mean. We are 0.011 board behind first, which is 4.9 sd of the top-band `ledger75` residual
(sd 0.0022) - real, and not reachable by scatter. A competitor that is behind wants the upper tail; the
damped middle is exactly what it is paying to avoid.

THE NUMBER, AND ITS HONEST WEIGHT. `matched_axes` over 264 recipe-carrying members reads `ema False >
True` at **+0.0049**, n=2, 2/2 agreeing, sd 0.0017. One of the two groups IS this member's own corner:

| pair | ema False | ema True | delta |
|---|---|---|---|
| `labelwise_dense_drop40_long_s2026` vs `chainfour_drop40_long_s2026` | 0.9048 | 0.8982 | +0.0066 |
| `probe224_grand_noema_s2026` vs `probe224_grand_s2026` | 0.8808 | 0.8776 | +0.0032 |

BOTH pairs are single-seed and the pooled mean is UNDER the 0.0062 seed spread, so **it is DIRECTIONAL
ONLY and cannot be told from seed luck.** This entry is not justified by +0.0049 and does not claim it.
It is justified by the mechanism above plus the fact that this is the single contrast, out of every one
the project owns, where the shipping member sits on the disfavoured side - `backbone_lr` 4e-05,
`calibrated_targets`, `epochs` 24, `image_size` 336, `mixup` 0.0, `slice_pooling` labelwise,
`hold_annotations` and `slot_dropout` 0.40 are all already at their favoured value.

WHY THERE IS NO FREE VERSION. `WeightAverage.state_dict` returns the shadow and the live weights are
never written, so the saved `model` tensor IS the average. The ema-off member cannot be lifted out of
the existing checkpoint; it had to be trained, and it was, for ~1.7 h on one card.

THE SCHEDULE IS KNOWN-GOOD WITHOUT AN AVERAGE. The risk a full fit carries is that with no holdout
there is nothing to early-stop on, so a run without an average could land on an arbitrary final step.
It cannot here: `--epochs 24` is fixed, and the fold-0 twin `labelwise_ext_drop40_long_s2026` runs this
exact recipe with `ema False`, returns CONVERGED under `convergence_check`, and reads ledger75 0.9045
over 909 rows. The anneal, not the average, is what picks the endpoint.

`submit_check` refuses this on SET equality with `extffsolo_v1` and is RIGHT to - the member NAME
differs but the roster shape does not, and the variable here is the training flag, not the seating. It
ships with `--anyway` for that reason, stated again in the submission message.

PRE-REGISTERED against the standing 0.945:

  * **>= +0.005 (0.950 or above)** - the flag was costing the solo what the matched groups say, and the
    ensemble-shape justification was paying for a neighbour set this entry does not have. Then `--ema`
    comes off every solo full fit.
  * **flat (within 0.004, which is 2 sd of the top-band residual)** - EMA is free on a solo. That
    RETIRES the axis rather than leaving it open, and it is the reading the +0.0049 sitting under the
    seed floor already warned about.
  * **<= -0.005 (0.940 or below)** - the average IS load-bearing on a full fit with nothing held out,
    the two fold-0 groups do not transfer to `--fold -1`, and the honest-pool readings stop being
    evidence about the objects that ship. That is the sharpest of the three.

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
    chosen = ("labelwise_ext_drop40_long_noema_ff",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
