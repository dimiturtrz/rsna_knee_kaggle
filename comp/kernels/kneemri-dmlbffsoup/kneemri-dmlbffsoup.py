"""ff-of-soup: the _ff operation applied to a SOUP, the order this project had never run.

CANDIDATE, not an instrument. It is offered against the pinned champion.

THE TWO ORDERS, AND WHY ONLY ONE HAD BEEN TRIED
  `basinsoup_ff_k4_s2026` -- the standing 0.946 entry -- is SOUP-OF-FF: three seed-children were each
  refitted on the whole corpus, then averaged. This member is FF-OF-SOUP: the average was taken first,
  on the fold-0 line, and the average was then given the corpus. Same two operations, opposite order,
  and the second order had no checkpoint on disk before 2026-09-20.

THE BASE, AND WHY IT WAS ONLY PICKABLE TODAY
  `basinsoup_dmlb_k4_s2026`, 0.9092 ledger75 -- the highest fold-0 number this project owns, and 0.0041
  above the base the next-ranked refit arm (knee-9grwy) would have used. Until 2026-09-20T23:36Z there
  was no licence to rank a fold-0 base against the board at all: knee-xjxdn had recorded fold-0 ledger75
  as ANTI-correlating with the _ff board, on three points spanning 0.0033 -- under the 0.0053 paired
  floor. Slot 56407707 measured the matched pair that bead never had: `basinsoup_k3_s2026` (fold-0,
  0.9089) boards 0.943 against its _ff sibling `basinsoup_ff_k3_s2026`'s 0.945. THE FOLD PENALTY IS
  0.002, in the ordinary direction, and a fold-0 number is board-predictive at a -0.002 offset. That
  reading is what chose this base over knee-9grwy's.

THE ARITHMETIC THAT PRICED IT
  Solo board-transfer fit (n=10 single-member entries, slope +1.071, intercept -0.0307, resid sd 0.0021)
  puts 0.9092 at fit 0.9430. The _ff refit term measured across four bases is +0.001 / +0.003 / +0.004 /
  +0.007, median ~0.004, giving ~0.947 against a 0.946 champion. P(clears) ~0.25, and the arm cost 18
  minutes of card because the operation is four epochs, not twenty-four.

WHAT THE RECIPE IS
  Four epochs at backbone-lr 2e-5 with --init-scope all on --fold -1, --ema, batch 4 x accumulate 3 at
  --vram-gb 22. Those epoch and rate values are the ones souping measured POSITIVE; `basinsoup2` read
  8 epochs BELOW the 4-epoch pair. A 24-epoch fit was rejected deliberately: it would wash the soup out
  and spend 3 h re-running an operation already priced on four bases. No distillation term -- the dml_b
  peer supervision is already in the soup's weights, and a teacher pass would change two variables.

WHAT IT IS NOT
  Not an ensemble. ONE checkpoint, ONE forward pass, no new recipe and no new seat.

WHAT MAKES IT BETTER-EVIDENCED THAN THE THING IT CHALLENGES
  It carries a real INFERENCE WITNESS: `witness_check --device cuda` reads max|diff| 0.00446 against the
  0.0312 bound, 5 of 12 labels carrying it. Every weight soup in this project -- the standing champion
  included -- can only ABSTAIN on that gate, because a witness is written at FIT time and a soup is
  never fitted (knee-cw5f3). This member was fitted, so for the first time in this family the shipped
  weights are the verified ones.

WHAT NO RULER CAN SAY ABOUT IT
  It fits --fold -1, so it holds nothing back and no ledger row can price it (knee-s8kim). The board is
  its only ruler, which is true of every _ff this project ships.

PRE-REGISTERED BRANCHES (knee-17la8)
  >= 0.947  The op is REAL and the order matters: averaging before the corpus beats averaging after it.
            Round 2 forks seed-children off THIS checkpoint and the champion pin moves.
  0.944-0.946  A DRAW inside board resolution. ff-of-soup is an equivalent route, not a lever, and the
            weight-space family is closed in both orders -- nothing further is owed it.
  <= 0.943  The order matters the OTHER way: soup-of-ff stays the only weight-space op that pays, and
            the 0.0041 higher base bought nothing, which would also say the 0.002 fold-penalty offset
            does not survive being applied to a base outside the family it was measured on.
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
    chosen = ("basinsoup_dmlb_ff",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
