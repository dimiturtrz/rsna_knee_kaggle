"""oofseat — the champion, plus ONE new member: `oof_teacher_s2026`, the best solo this project owns.

**WHAT IS BEING PRICED, and why it is not a reseating.** The member axis is closed — seven board entries
read flat while WHO sits was permuted, and no slot goes to a reordering of the existing pool. This is not
that. `oof_teacher_s2026` did not exist when any of those entries was built: it is `band_dense`'s recipe
trained under a distillation loss against our own out-of-fold posterior, and it reads **0.8681 offline
against band_dense's 0.8578**, +0.0107 and the highest solo in the project's history. A member that is new
and better is the one thing a seating entry was never testing.

**WHY THIS SLOT, TONIGHT, RATHER THAN AFTER THE RETRAINS.** The distillation reading licenses a seven-hour
plan: retrain all six champion seats under the same teacher, on the arithmetic that a uniform loss-side
treatment passes through to the board at 92-147% (measured, this project) and 0.944 + 0.0107 lands at
0.954-0.960 — the leader. That plan spends the card for a night. Before it does, the board should say
whether the 0.8681 is real on TEST, where the training corpus is not our corpus. One seat is the cheapest
possible instance of the treatment: if the member cannot move a seven-way vote at all, six retrained seats
are unlikely to move it either, and the night is better spent elsewhere.

**MECHANISM, not weighting.** The seat enters at the champion's own convention — one member of seven in
`LogitAverage`, no weight fitted, no dial moved. Everything else is `raptorweight` unchanged: the same six,
the same order, the same `overlap=False`, the same raptor checkpoint at `FOREIGN_WEIGHT`, the same kit.
The single difference from the pinned champion is one additional name in `chosen`, which is what makes the
read attributable.

**THE MOUNT IS ALSO NEW, and that is a second thing this entry proves.** The shipped weights dataset holds
127 members and about 10 GB, so a member trained after the last big push could never reach the board
without re-pushing all of it — every offline gain since was unshippable. `oof_teacher_s2026` is the first
member served from a SECOND dataset (`kneemri-members-2`) mounted beside the first, merged by
`MemberPackage.spanning`. A kernel log census reading `7 of 7` is the proof that path works; `6 of 7` means
the second root never mounted and the score is `raptorweight`'s, not this entry's. Read the log first.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.946 or above** — the 0.8681 is real on test and the seat retrains are licensed by the BOARD, not
  only by the holdout. Re-pin `comp/champion.json` and spend the night on the six-seat retrain.
- **0.9455 down to 0.9425** — a gain above the resolution floor but under the leader's margin. The
  member is real and partly redundant with the six it votes beside; the retrain is still the right spend,
  because a retrain changes the seats themselves rather than adding an eighth voice.
- **0.942-0.945** — FLAT. The member is real offline and redundant with the seats in the vote. That does
  NOT refute the retrain — an added seat and a replaced seat are different operations — but it does say
  the +0.0107 will not arrive as a sum, and the retrain's own branch table should be re-priced first.
- **below 0.942** — read the kernel log BEFORE the score. A dropped member is silent everywhere else:
  `farseat_v1` shipped six of seven named members, read the six-member score exactly, and that flat was
  written down as a finding for five days. `7 of 7` in the census, or the number means nothing.

The refutation this cannot make: it cannot separate "the distilled member adds nothing on test" from "an
EIGHTH voice adds nothing to a seven-way vote, whoever it is". `dark6` already reads the seventh seat as
near-inert (knee-uco3). Only a seat REPLACEMENT — the retrain — separates those two, which is why a flat
result here re-prices the retrain rather than cancelling it.
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
        # The seventh, and the only name here that is NOT on the first weights dataset. It is served
        # from `kneemri-members-2`, merged in by `MemberPackage.spanning`; if that root fails to mount
        # the census reads 6 of 7 and this entry IS `raptorweight`.
        #
        # `final_ws_df` is the whole-corpus calibrated-distill seat — 336 px / 20 slices / band 0.1-0.9,
        # --ema, --calibrated-targets, --distill-weight 0.7 on teacher_oof5.npz, fitted at `--fold -1`.
        # It has no offline number and can never have one; the board is its only ruler. THE MECHANISM
        # THIS ENTRY BUYS is calibrated distillation inside the entry rather than beside it: the six
        # parents are all uncalibrated, and D29 measured calibrated targets at +0.014 macro on the
        # foreign-distill corner with the gain concentrated on PFOA/MCL/Fracture/Effusion/LatOA.
        # It ships ONLY if `clean_calib_s2026` (its two-knob fold-0 stand-in) clears +0.005 L75mid over
        # `seat_clean_distill_s2026` AND the researcher's seat read clears +0.002 out of sample on 20
        # halves. Otherwise colsyn ships and this kernel stays staged, unspent.
        "final_ws_df",
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
