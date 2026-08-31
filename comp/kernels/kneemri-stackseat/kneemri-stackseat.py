"""stackseat — the champion, with its ONE fold-0 seat replaced by the pool's best solo, refit blind.

**WHAT IS BEING PRICED.** Not a permutation and not a weight. `raptorweight_v1` (0.944) seats six of ours
plus the foreign table at 1/3, and five of those six are `--fold -1 --ema` — fitted on all 4,406 studies.
The sixth, `merged_blend_cf`, is fold 0: it was fitted on 3,497 and never saw the 909 it holds out. That
seat is paying a fit penalty the other five do not, and the measured blind-vs-fold-0 board gap is roughly
0.008 — larger than any seating move this project has ever read. `stack_increments_ff` takes that seat.

**WHY THAT MEMBER.** `stack_increments_s2026` is the highest solo the project owns (0.8642), the only one
that distils from `raptor_teacher.npz`, and it carries the blend-three corpus — the two levers that made
the pool's best members. It has never been shippable into the champion precisely BECAUSE it is fold 0:
its quality edge over the incumbent is smaller than the fit penalty it would pay walking in. So the arm
`stack_increments_ff` refit it with one moved flag, `--fold -1 --ema`, corpus and teacher and distill
weight and calibrated targets and band and slices all held. The penalty is gone and the quality delta is
the only thing this swap moves.

**WHY IT IS NOT ANOTHER FLAT RESEATING.** knee-urvf states the standing result: every operation on the
champion's SEATING is measured and inert — adding a distant seat read 0.943/0.943/0.942, swapping a seat
read 0.944 dead flat (`holdswap_v1`), and three offline candidates priced +0.0002/+0.0005/+0.0000 on one
day. The public frontier independently confirms it: their three-FAMILY panel buys +0.001 over its best
single. This entry does NOT vary who sits or how many; it varies HOW ONE SEAT WAS FIT, which is the one
axis the flat readings never touched and the only free term knee-urvf leaves open (member quality).

**NO OFFLINE NUMBER EXISTS FOR THE NEW MEMBER, by construction.** It has seen every holdout study, so
nothing may enter the ledger as its score, `seat_value` cannot price it, and no paired floor applies. The
board is its only instrument. That is the cost of the lane and it is stated here rather than worked around.

**EVERYTHING ELSE IS THE CHAMPION, UNCHANGED.** Same count, same order, same `overlap=False`, same
`LogitAverage`, same kit, same raptor checkpoint at weight 1/3, same rank-space join by study.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.946 and above** — the fit penalty was the binding constraint, and the entry clears the top-ten
  floor. Re-pin `comp/champion.json`, then refit every remaining fold-0 member in the pool blind before
  any other axis gets a slot.
- **0.9455 to 0.9459** — a real gain over the champion, one resolution step clear of it. Member quality
  IS the free term knee-urvf leaves open, and the blind refit is how it is spent. Re-pin and continue.
- **0.9425 to 0.9455** — FLAT, the band `BOARD_RESOLVABLE = 0.0015` cannot separate from 0.944. Then the
  ~0.008 blind-vs-fold-0 gap does NOT transfer to a single seat inside a six-member average: the other
  five absorb it. The fit axis joins count and weight as a closed dimension, knee-urvf's free term is
  spent, and the remaining slots go to the KIND of member (knee-ombx, external data) rather than its
  quality.
- **Below 0.9425** — the incumbent `merged_blend_cf` was carrying something the higher solo does not.
  That would be the first evidence in this project that a seat's VALUE and its SOLO disagree in the
  direction that matters. Read `record_mix` on the pair before touching the roster.
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
        "stack_increments_ff",
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

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
