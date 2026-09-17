"""Is the full-corpus refit term real, or was it one lucky pair? The second matched pair.

WHAT THIS MEASURES, STATED FIRST. `ffsolo_v1` (0.944) and `l75solo_v1` (0.941) are the same recipe
differing only in whether the holdout fifth was trained on, and the +0.003 between them is the ONLY
measurement this project has of what a `--fold -1 --ema` whole-corpus refit is worth to a LONE member.
That single number now sits inside the inversion that prices every open arm - it is what says a solo
needs +0.0021 offline75 to beat the 0.946 champion and +0.0089 to reach 0.955. A load-bearing n=1.

WHY THE ENSEMBLE ROWS DO NOT ALREADY ANSWER IT. `tripleseat_v2` / `fullseat_v1` / `fullseat2_v1` read
0.946 / 0.946 / 0.946 over 0 / 1 / 2 refit seats, which bounds the term under the board's 0.001
resolution at a one-of-eight share. That is a different population and it is silent about the solo end
by construction: the refit recovers information seven neighbours already supply between them, so a flat
dose series is what a REAL solo term predicts, not evidence against it.

THE MEMBER. `labelwise_ext_drop40_long_ff` is byte-identical to `labelwise_ext_drop40_long_s2026` apart
from `--fold -1 --ema` and `--out` - the diff was read, not assumed. Already mounted on
`dimiturnt/kneemri-members-2`. Its fold-0 twin reads ledger75 **0.9045** over 909 rows, CONVERGED.

PRE-REGISTERED, from the solo-only transfer fit re-derived the same day
(`board = -0.1131 + 1.165 * offline75`, n=8, residual sd 0.0017). The twin projects a solo board of
**0.941** with no refit term, the same rounded point the 0.9053 twin projected before `ffsolo_v1` ran:

  * **0.944 or above** - the refit term REPRODUCES at n=2. The +0.003 stays in the inversion and every
    future solo is refit before it is shipped.
  * **0.941 or below** - the term was n=1 scatter. It comes out, and a solo beat of the champion goes
    from +0.0021 to +0.0047 above the best member we own, which is a materially harder target.
  * **0.942-0.943** - ambiguous on a board reporting at 0.001, leaning small. Record it; quote no term.

THIS IS AN INSTRUMENT AND IS SUBMITTED AS ONE. The modal branch is 0.944, two thousandths under the
pinned 0.946, and one member is not offered as a challenger to an eight-seat entry. It costs no GPU -
the checkpoint has been on the shelf since 2026-09-15 - and it removes the n=1 caveat from the number
that every open lever is now priced against.

RE-OPENING A DECLINE, ON ITS OWN TERMS. `comp/members.declines` declined this checkpoint as a SEAT,
because "more data per seat" is measured flat at two doses, and invited re-opening "by naming a
mechanism that is not more data per seat". This is not a seat. It is the solo end.

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
    chosen = ("labelwise_ext_drop40_long_ff",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
