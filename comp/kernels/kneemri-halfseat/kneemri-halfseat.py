"""The champion seven's exact roster, with the foreign seat weighted 1/2 instead of 1/7.

**THE ONE VARIABLE `onepair` LEFT UNSEPARATED.** Three scored entries share one ruler and read as a
clean story until you look at what else moved with them:

| entry | our members | foreign weight | board |
|---|---|---|---|
| `calibsix` | 6 | 0 | 0.941 |
| `raptorseat_v1` | 6 | **1/7** | **0.943** |
| `onepair_v1` | **1** | **1/2** | 0.936 |

`onepair` was read as *distance is worth more per seat than count* — two seats buying 65% of what seven
buy. That reading is sound, but it is not the only one its number supports: `onepair` differs from
`raptorseat` in TWO ways at once, dropping five of our members AND raising the foreign share from 1/7 to
1/2. A blend that is simply UNDER-weighting its most distant member would produce the same table.

This entry changes exactly one of the two. Same six members, same foreign member, same rank-space join,
same seating code — `RaptorSeat(weight=FOREIGN_SHARE)` where the champion used the default. Its delta
against 0.943 is the foreign weight and nothing else.

**IT IS AN INSTRUMENT AND A CANDIDATE AT ONCE.** If 1/7 under-weights a member that is measured at 0.923
solo and is the only one in the roster trained on a different architecture, then the champion is leaving
score on the table for free — no new member, no new decode, one constant. That is the cheapest possible
upgrade if it exists, and a null closes a whole axis. Either way one slot answers it.

**WHY 1/2 AND NOT A SWEEP.** A sweep costs one slot per value, and the arms are `--fold -1` so no
offline pairing can fit the weight (the reason `RaptorSeat`'s docstring gives for 1/7 in the first
place). 1/2 is not a tuned value: it is the weight `onepair` already spent a slot measuring, so the two
entries together read as one grid rather than two guesses — and it is the far end of the plausible
range, which is what makes a null at this size informative about everything between.

PRE-REGISTRATION, written before the number, against `BOARD_RESOLVABLE = 0.0015` and the pinned champion
`raptorseat_v1` = 0.943.

- **0.9445 or above** — 1/7 UNDER-weights the foreign member. The champion's seating rule is costing
  score, `onepair`'s strength was partly its weight rather than purely its distance, and the next
  foreign seat should be weighted by distance rather than by head-count. Re-pin.
- **0.9415-0.9444** — flat. The blend is INSENSITIVE to foreign weight anywhere between 1/7 and 1/2,
  which is itself a result: rank-space averaging is robust over that range, so seat COUNT is the lever
  and weight is not. Closes the weight axis and strengthens `onepair`'s distance reading by elimination.
- **Below 0.9415** — 1/2 is too much foreign. 1/7 was right or generous, and `onepair` reached 0.936
  DESPITE a half-weighted foreign member rather than because of it — which makes distance-per-seat
  stronger than measured, not weaker, and says our six carry the quality the blend is built on.

**WHAT THIS ENTRY CANNOT DO.** It cannot price the weight for a DIFFERENT foreign member. `v4_full_swa`
is board-measured at 0.923 solo, which is close to our own members' quality; a weaker or stronger foreign
arm would have its own optimum and this says nothing about where. It also cannot separate weight from
member for the fleet — `fleettwo` seats two foreign members and this seats one.
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

FOREIGN_SHARE = 0.5
"""The one value that differs from the pinned champion. See the docstring: it is `onepair`'s weight,
reused so the two entries read as one grid, not a fresh guess."""


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
    # The six that follow load onto the same T4. Dropping the reference is not enough on its own —
    # torch keeps the freed blocks in its caching allocator, and a seventh model's worth of them is
    # exactly the margin an OOM would eat, in a kernel whose whole point is running both.
    del member
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    # What the raptor pass cost, handed to the six so their nine-hour budget is the SESSION's and not
    # a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at the
    # real one with no submission — a budget defeated by its own assumption that it starts at zero.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the raptor pass cost {spent:.0f}s; the six get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left)).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_SHARE).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
