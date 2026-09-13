"""tripleseat — the champion seven, plus the three best honest solos this project owns.

**WHAT IS BEING PRICED. MEMBER QUALITY AT COUNT 1, WHICH IS THE ONLY THING THAT HAS EVER MOVED THIS
BOARD 0.005+** (knee-0v8yz). Not a reseating: the champion's six sit in the champion's own order, at the
champion's own `LogitAverage`, with the raptor seat at the champion's own 1/3. Three rows are added and
nothing else in the file differs.

**WHERE THE THREE CAME FROM.** Every twelve-epoch member of the `labelwise` family peaks at epoch 7 to 10
of 12 and then declines -- depth24 0.8998 @e8, extended 0.8986 @e8, drop 0.8984 @e10, dense 0.8964 @e7
(ledger75). A budget wall peaks on the LAST epoch; peaking at 7 of 12 and declining for five is a missing
regulariser. Two flags answer it -- `--slot-dropout 0.40` (the operating point legislated in knee-yotj8,
not a swept knob) and `--epochs 12 -> 24` -- and on three of three bases the pair cleared a
pre-registered `>= +0.0040`:

    labelwise_depth24   0.8998 -> labelwise_depth24_drop40_long   0.9053  (+0.0055, knee-9bnkc)
    labelwise_extended  0.8986 -> labelwise_ext_drop40_long       0.9045  (+0.0059, knee-cgt9m)
    labelwise_dense     0.8964 -> labelwise_dense_drop40_long     0.9048  (+0.0084, knee-vs06j)

All three CONVERGED under `convergence_check`. They are now the three best honest solos the project owns,
and no member of the champion is within 0.005 of any of them on this ruler.

**AND THEY ARE ADDITIVE, WHICH IS THE PART THAT IS NOT OBVIOUS.** Three upgraded members of one family
could easily be three copies of one improvement. Measured 2026-09-13 on the full 909, `LogitAverage`, no
selection, against the honest core six at 0.9050:

    + depth_up                              0.9070   +0.0020
    + ext_up                                0.9067   +0.0017
    + depth_up + ext_up                     0.9081   +0.0031
    + all three                             0.9090   +0.0040

Near-additive at two and still climbing at three, so each upgraded base is a fresh SEAT rather than a
copy of one already held. `+0.0040` is the seat floor this project has used all along, and it is CLEARED
by the trio and by no single member of it -- which is precisely why all three ship and not the best one.

**EINSTEIN GATE.** `grep -icE "labelwise_depth24_drop40_long|labelwise_ext_drop40_long|labelwise_dense_drop40_long"`
over `comp/board.jsonl` returns 0: no submission in the competition's history carries any of the three.
`submit_check` runs on the set before the slot is counted.

**WHAT THE OFFLINE NUMBER IS AND IS NOT.** The champion's six are `--fold -1` and unscoreable offline, so
`+0.0040` is measured on the honest core six, a PROXY pool, not on the entry that ships. At
`board_transfer +1.135` a proxy gain of that size is ~0.0045 of board. That makes this a CHAMPION-BEATER
by intent rather than an instrument -- but the transfer is the assumption, and the pre-registration below
is written so that a flat still buys a fact.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.947 and above** -- the proxy transferred and the labelwise budget axis is live. Re-pin
  `comp/champion.json`; the fourth base (knee-fnhbs, on the card as this ships) and the operator pair on
  the NON-labelwise families take the next slots ahead of everything else.
- **0.9455 to 0.9469** -- a real gain, one resolution step clear of the champion. Re-pin, and widen the
  pair before widening anything else.
- **0.9425 to 0.9454** -- FLAT. Then the honest pool's best three solos add nothing to a champion whose
  own members cannot be scored against them, and the reading is about the PROXY, not about the members:
  it would be the first direct evidence that offline gains on the honest six do not reach the champion's
  six, which retires the proxy rather than the members. Check `run_check`'s census FIRST -- a dropped
  member and a null member look identical from the board.
- **Below 0.9425** -- three seats of one family actively dilute the champion. That closes family-block
  seating, an axis nothing here has yet tested at count 3.
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
        # THE ONLY LINES THAT DIFFER FROM THE CHAMPION. They sit last so a diff of the two kernels is
        # three added rows: the six above are in the champion's own order, unchanged, and nothing about
        # how they are weighted or joined moves. All three live on the SECOND shelf
        # (`dimiturnt/kneemri-members-2`), which is why this kernel mounts one dataset the champion does
        # not. They ship TOGETHER because the +0.0040 seat floor is cleared by the trio and by none of
        # them alone — shipping the best one would be a different, weaker experiment.
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
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
