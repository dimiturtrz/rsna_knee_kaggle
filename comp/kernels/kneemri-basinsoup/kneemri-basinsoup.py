"""Does a weight soup's OFFLINE gain reach the BOARD, and at what slope?

WHAT THIS MEASURES, STATED FIRST. `basinsoup_k3_s2026` is the uniform weight average of three children
re-annealed off ONE fine-tuned parent (`labelwise_ext_drop40_long_ema_s2026`, ledger75 0.9073) and it
reads ledger75 0.9089 over 909 held-out studies - +0.0016, the first positive weight souping has
produced in this project. That is an OFFLINE number. Nothing in the record says what fraction of it
survives the trip to the board, because no soup has ever been shipped.

THIS IS AN INSTRUMENT, NOT A CHALLENGER, AND THE ARITHMETIC SAYS SO BEFORE THE SLOT IS SPENT. The
member is FOLD 0: it trained on two thirds of the corpus, while the pinned champion `tripleseat_v2`
(0.946) and every solo above 0.941 on this board is a `--fold -1` full fit. Five fold-0 solos carry
BOTH an `offline75` and a board row, and they are the matched population for this entry:

| member | offline75 | board |
|---|---|---|
| `headlr_ship_s2026` | 0.9059 | 0.937 |
| `labelwise_depth24_drop40_long_s2026` | 0.9053 | 0.941 |
| `labelwise_ext_drop40_long_s2026` | 0.9045 | 0.937 |
| `stackfit_s2026` | 0.8962 | 0.933 |
| `calib_dense_hold_s2026` | 0.8949 | 0.928 |

OLS over those five reads `board = 0.1907 + 0.826 x offline75`, residual sd 0.0025, n=5. It is a WEAK
line - five points, and its own scatter is larger than the 0.0022 the ensemble scope's top band fits to
- and it is quoted here as the entry's pre-registration, never as a constant to reuse. On it this
member predicts **0.9415**, four board ticks under the champion. So this entry CANNOT win and is not
offered as a win. Protecting the standing 0.946 costs nothing here: a losing row does not unseat a pin.

WHY IT IS WORTH A SLOT ANYWAY. The three matched fold-0 / `_ff` pairs the same table holds -
`labelwise_ext_drop40_long` +0.008, `labelwise_depth24_drop40_long` +0.003, `headlr_ship` +0.001,
median +0.003 - say a fold-0 soup that cleared the 0.9117 bar would still land near 0.944 and need a
full-data re-fit to pass the champion. That re-fit costs about two hours of card and, being `--fold -1`,
holds nothing back, so it can be priced by NOTHING except a board slot. Committing that card and that
second slot to a soup whose transfer slope is unmeasured is the spend this entry retires. It buys the
slope with a member already trained, already packaged and already uploaded.

THE SHELF. `basinsoup_k3_s2026` lives on `kneemri-members-3`, a third weights dataset opened the same
day at 171 MB, because shelf 1 is 156 members and ~13.8 GB and `kaggle datasets version` re-pushes a
whole shelf. The mount is declared in `dataset_sources`; `KernelRun` DROPS a member it cannot mount and
keeps going, so `run_check` on this kernel's own log is the gate before the slot is counted.

PRE-REGISTERED against the fold-0 solo line's 0.9415, with its own residual sd 0.0025 as the ruler:

  * **>= 0.944 (+0.0025, 1 sd above)** - souping transfers BETTER than the line, the +0.0016 offline is
    not the whole gain, and an `_ff` re-fit of the winning soup is licensed on the spot.
  * **0.939..0.944 (within 1 sd)** - souping transfers AS the line says. Then the decision is pure
    arithmetic and needs no further slot: a soup must clear offline75 0.9120 before an `_ff` re-fit can
    beat 0.946, which is exactly the bar `basinsoup2` was pre-registered against this afternoon.
  * **<= 0.939 (-0.0025)** - the offline gain does NOT reach the board. Weight souping then closes on
    its TRANSFER rather than on its mechanism, and the card that would have gone to re-fits goes
    elsewhere. That is the outcome this entry most wants to know and the reason it is an instrument.

`board blur` is 0.0010 - the competition reports at three decimals - so every threshold above is stated
in units the board can actually resolve.
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
    # special-cased so the vote stays the champion's convention here too. Unlike the `_ff` solos
    # beside it this member is NOT dark: it is fold 0 and holds 909 studies out, which is what makes
    # it an instrument - the offline number and the board number are both readable and the pair is
    # the measurement.
    chosen = ("basinsoup_k3_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
