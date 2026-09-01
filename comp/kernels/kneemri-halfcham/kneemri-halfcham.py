"""halfcham — the champion's exact seating served in FORCED float16. Nothing is reseated.

**WHY THIS AND NOT ANOTHER SEATING.** Six board rows now say the seating is not a lever on this pool:
`raptorweight_v1` moved the foreign SHARE by 2.3x for one unresolvable tick; `vjepaseat_v1` ADDED the
most architecturally distant member the project owns for -0.001; `holdswap_v1` EXCHANGED a seat for the
best-priced alternative a fold-0 swap table could name, for exactly 0.000; `dark7_v1` added a seventh
dark member for a flat 0.937; `spanseat_v1` moved the corpus axis for 0.944; `stackseat_v1` moved HOW a
seat was fitted for exactly 0.000. Reweight, add, swap, count, kind, fit — six operations on WHO SITS,
six flats. What is left is what a member SEES, what it IS, and HOW IT IS RUN.

**THE AXIS.** `Serving` carries four knobs no member can answer, and three have now been asked on the
board: the vote (`votefit_v1`, 2026-08-25, LogitAverage over RankAverage, 0.938 and best at the time —
which is why the champion carries LogitAverage and why re-asking it would be a backwards repeat), the
views (`champviews_v1`, submitted 19:52Z today, still scoring), and the budget (implicit in every entry).
The fourth, PRECISION, has never been varied on the board. This entry varies it and nothing else.

**WHAT FORCING FP16 ACTUALLY DOES, AND THE TRAP IT HAS ALREADY SPRUNG ONCE.** `Serving.precision` is
None by default, which serves each member in the arithmetic it was FITTED in. Forcing `torch.float16`
makes every member read in half precision regardless. The trap is documented on the field itself: an
earlier loadcheck wrapped `predict_member` in an OUTER autocast to compare three precisions, the inner
autocast that `arithmetic` decides carried an explicit dtype and overrode it, so all three arms ran the
same arithmetic and reported max|diff| of exactly 0.0 twelve times — which read as "fp16 is free" and was
really a knob that was not connected. This path is the connected one (`Ensemble.arithmetic` at
`ensemble.py:306` reads `self.serving.precision or member.precision`, and `ensemble.py:396` opens the
autocast with it), but the failure mode is a submission IDENTICAL to the champion's, so the FIRST thing
to check on the kernel log is that the members actually served in half and not in their fitted dtype.
An entry that reads exactly 0.944 is ambiguous between "fp16 costs nothing" and "the knob did not fire",
and only the log separates them.

**THE SECOND THING THIS BUYS IS RUNTIME**, which the efficiency board scores. Half precision across six
members plus the raptor pass should cut the session's wall-clock, and `Ensemble.afford` gives up the
weakest member rather than the entry when a ceiling is threatened — so a faster run also widens the
margin against the `farseat_v1` defect, where a silently smaller entry read as a clean flat for five days.

PRE-REGISTRATION, written before the number, against the matched control `raptorweight_v1` = 0.944 and
`BOARD_RESOLVABLE` = 0.0015.

- **0.9455 or above** — half precision is not merely free but favourable, which would mean the members'
  fitted arithmetic is not the one that serves this pool best. Keep fp16 on every later entry and re-read
  the witness bound, since `witness_check`'s eps was calibrated against fitted-precision members.
- **0.9425 to 0.9455 exclusive** — FLAT, and the useful branch. fp16 serving costs nothing measurable on
  the board, which licenses it as the DEFAULT for every future entry on runtime grounds alone, and closes
  the last unasked `Serving` knob. Only valid if the log confirms the members served in half.
- **0.942 or below** — half precision COSTS. Check the log for non-finite fallbacks before believing it:
  `predict_member` has a documented fallback path, and a member that fell back is a member served in a
  precision this entry did not choose. Only after that comes back clean does a negative read as a fact
  about fp16 rather than about one unstable member.

**WHAT NO BRANCH LICENSES.** Nothing here says anything about fp16 TRAINING (the owner's standing recipe
is bf16 train / fp16 serve, and this asks only the serving half), about bf16 serving (a different dtype
with a different exponent range, untested), or about fp16 on a member whose fitted precision is already
half — where there is no variable to move and the entry would be the champion under another name.
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
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left, precision=torch.float16)
    ).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
