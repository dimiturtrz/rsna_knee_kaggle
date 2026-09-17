"""Does training on the whole corpus make a BETTER MEMBER? The only shape that can see it.

WHAT IS UNMEASURED, STATED FIRST. `--fold -1` holds nothing out, so a full-corpus refit can never carry
a ledger row: the board is its only ruler. knee-sexix closed the lever on a three-point dose series -
`tripleseat_v2` (no refit seats) 0.946, `fullseat_v1` (one) 0.946, `fullseat2_v1` (two) 0.946 - and that
closure is correct about what it measured and blind to what this entry asks. All three are seven- and
eight-seat entries, and the board's own seat-count curve saturates there: best board by seat count runs
1 -> 0.933, 3 -> 0.939, 6 -> 0.944, 7 -> 0.945, 8 -> 0.946, and turns over at 9. A member-level gain
cannot show at the flat end of a saturating curve. Full-corpus training has never been put on a member.

THE MEMBER. `labelwise_depth24_drop40_long_ff` is the `--fold -1 --ema` refit of the best recipe this
project owns, 24/24 epochs, already mounted on `dimiturnt/kneemri-members-2`. It trains on 4,407 studies
against its fold-0 twin's 3,498, which is +26% - the largest data change available without new data.

THE REFERENCE, AND IT IS A PREDICTION RATHER THAN A PAIR. Its fold-0 twin
`labelwise_depth24_drop40_long_s2026` reads ledger75 0.9053 (909 rows, epoch 18). `stackfit_s2026` reads
0.8962 on the same ruler and is on the board at 0.933 solo (`matchpair_v1`, ref 56150916). At the
ledger75 transfer slope of +1.09 the twin therefore projects a solo board of ~0.943, and the refit is
predicted at that number plus whatever the extra fifth of the corpus is worth to one model.

PRE-REGISTERED, against that ~0.943 and the top band's residual sd of 0.0022:

  * **0.948+** - the extra data is worth more to a member than the entire seven-seat ensemble is worth
    over its best member (0.946 - 0.943 = +0.003). Every future member is refit at `--fold -1` before it
    is seated, and knee-sexix's closure is re-scoped to "invisible in a saturated ensemble" rather than
    "worth nothing".
  * **0.9455-0.9479** - a real member-level gain, clear of 2 sd, that the dose series could not see.
    Same consequence, smaller: refit the champion's seats and re-ship the seating.
  * **0.9405-0.9454** - FLAT on the prediction. More data does not make a better member either, and the
    data axis closes at every level, not just inside an ensemble. This is the modal branch.
  * **below 0.9405** - either the refit is WORSE than its fold-0 twin, which would make `--ema` or the
    absent early-stop the suspect rather than the data, or the +1.09 transfer is wrong at solo size. The
    first suspect is then the transfer, because `matchpair_v1` is its only solo anchor.

THIS IS AN INSTRUMENT AND IS SUBMITTED AS ONE. A single member cannot beat a 0.946 ensemble and is not
offered as one - the modal branch is sixteen thousandths under the pin. It buys the answer to the one
axis this project has not measured at the resolution that could see it, and it costs no GPU: the
checkpoint has been on the shelf since 2026-09-15.

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
    chosen = ("labelwise_depth24_drop40_long_ff",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
