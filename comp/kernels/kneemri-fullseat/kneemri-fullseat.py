"""fullseat — tripleseat_v2 with ONE seat refit on the WHOLE corpus: the DATA axis, not a reseat.

MECHANISM entry (knee-sexix). Serving is the champion's, untouched: `LogitAverage` over seven,
`overlap=False`, raptor at 1/3, kit `c3effa61`. One variable: `labelwise_ext_drop40_long_s2026` (fold 0,
trained on ~3500 studies, 0.9045 ledger75, the best honest solo owned) out, and its byte-identical recipe
refit at `--fold -1 --ema`, `labelwise_ext_drop40_long_ff` (all ~4400 studies, 24 of 24 epochs), in.
Same recipe, same seat, same count: the moved term is the ~fifth of the corpus the seat never saw, plus
the EMA the champion's other full-fit seats already carry.

Not a seat-identity move. Two single-seat swaps in a row (tripleseat -> doseseat, warmseat) read the
seat axis saturated at board resolution; this holds the seat's recipe and moves its DATA, the lever the
honest harness could never ask for because a `--fold -1` member has no holdout row.

UNPRICEABLE OFFLINE BY CONSTRUCTION: `--fold -1` holds nothing back. Its twin is the seat it replaces.
Candidates it was chosen over, priced on 896 A'-OOF rows, twin n=6, ledger75, 200 bootstraps
(`<scratchpad>/price.py`): raptor weight 1/3 -> 0 reads +0.0016 [-0.0003, +0.0038] against the only
honest foreign proxy (A', solo 0.8787, weaker than the shipped raptor) while `raptor_v4_swa` is in-sample
and reads monotone in weight, so no honest offline read of the weight exists; the board pair
honestfar/honestsix already says raptor at 1/3 carries +0.003. RankAverage -0.0003 [-0.0007, +0.0002],
MedianLogit +0.0002 [-0.0008, +0.0011]; the combiner closed on the board (stackvote_v1).

PRE-REGISTRATION, against tripleseat_v2's 0.946 and `BOARD_RESOLVABLE = 0.0015`:

- **0.948+** — full-corpus data moves the entry at ONE seat of seven. Re-pin; refit the other two drop40
  seats (`labelwise_depth24`, `labelwise_dense`) at `--fold -1 --ema` and ship all three.
- **0.947** — directional, one tick. Do not re-pin on it alone; the three-seat refit is the confirm.
- **0.946** — FLAT: one seat's extra fifth of data is under board resolution at a 2/3 share over seven.
  Does NOT close full-fit of the pool; a three-seat refit is the dose that could resolve it.
- **0.945** — one tick down: noise-level; the fold-0 member's diversity against the full-fit seats was
  worth a tick. Check `run_check` census first.
- **Below 0.945** — the full-fit refit hurts (EMA or fit-all overshoot on this recipe); stop full-fitting
  drop40 recipes and keep fold-0 bases.
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

FOREIGN_WEIGHT = 1.0 / 3.0
"""The foreign side's share of the rank blend. `raptorseat_v1` shipped 1/7 = 0.1429 as a convention about
SEAT COUNT; this is a quality-weighted share of two SYSTEMS, named before the number and never fitted.

Below the bootstrap block rather than above it: the block opens with imports, and a constant in front of
them makes every one of them E402. The kernel that RAN carried it above and the ruff gate caught it after
the fact — the number is identical either way.
"""


def main() -> None:
    boot()

    import time  # noqa: PLC0415 - the bootstrap block above is byte-identical across kernels

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot
    import torch  # noqa: PLC0415 - same

    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.device import VisibleCards  # noqa: PLC0415 - same
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415 - same
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - same
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_run import WORKING, RaptorEntry  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415 - same

    # The raptor member runs FIRST and into a scratch directory, because both producers write a file
    # called `submission.csv` and the one that runs last is the one Kaggle would otherwise score.
    scratch = WORKING / "raptor"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mounts = Mounts()
    device = str(VisibleCards().device())
    member = RaptorMember.loaded(RaptorEntry.weights(mounts=mounts), device)
    theirs = RaptorEntry(mounts.one("sample_submission.csv", "competition data"), working=scratch).run(member, device)
    # The seven that follow load onto the same T4. Dropping the reference is not enough on its own —
    # torch keeps the freed blocks in its caching allocator, and a seventh model's worth of them is
    # exactly the margin an OOM would eat, in a kernel whose whole point is running both.
    del member
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "stack_increments_ff",
        "through_dense_cf",
        # The three drop40 bases live on the SECOND shelf (`dimiturnt/kneemri-members-2`).
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_ff",
        "labelwise_dense_drop40_long_s2026",
    )
    # `labelwise_ext_drop40_long_s2026` out, its `--fold -1 --ema` refit in: same recipe, whole corpus.
    # What the raptor pass cost, handed to the seven so their nine-hour budget is the SESSION's and not
    # a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at the
    # real one with no submission — a budget defeated by its own assumption that it starts at zero.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the raptor pass cost {spent:.0f}s; the seven get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left)).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
