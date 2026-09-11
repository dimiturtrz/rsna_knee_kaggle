"""corrseat — the SAME vacated seat as tonight's `kneemri-goldseat`, filled by the other operation. This
entry exists to separate two things that entry cannot separate on its own.

**WHAT IS BEING PRICED.** An OPERATION, not a member and not a ruler. Both entries start from the same
finding: the blind adjudication of 469 holdout cells says `report_labels_v2` is wrong on 89.1% of the
cells where our table dissents from three independent report extractors. There are two ways to act on
that, and they act on ALMOST THE SAME CELLS:

- **MASK** (`discord_mask_s2026`, shipped earlier tonight) — drop supervision on the 7,369 cells four
  extractors disagree on.
- **SUBSTITUTE** (`v2_corrected_s2026`, this entry) — keep every cell, and replace v2's value on the
  2,315 cells where v2 is the LONE dissenter against three agreeing extractors.

2,301 of those 2,315 substitution cells (99.4%) sit INSIDE the 7,369-cell mask. So this is not cell
diversity and is not sold as it: it is the same finding wearing two hats, and the only difference between
the two entries is whether a suspect cell is deleted or corrected. The 5,068 cells the mask holds and the
substitution does not are priced at ~0.07 v2-wrong — near the corpus base rate, i.e. mostly innocent
supervision the mask throws away.

**THE TWO RULERS DISAGREE, WHICH IS WHY THIS IS WORTH A SLOT.** On the gold cells the MASK ranks first of
everything on disk (AUC-vs-verdict 0.6982) and the SUBSTITUTION ranks fourth (0.6272, against a 0.6008
control and a 0.932 constant-0 accuracy floor). On the SELECTOR audit the substitution's cells are the
sharp ones: lone-dissent precision 0.727 at recall 0.533 (~20x lift over the 3.7% base), against 0.156 for
the four-way discordance set the mask uses. One ruler says mask, the other says substitute, and no offline
measurement we own breaks the tie — the board is the only instrument that can. Shipping only the mask
would leave that permanently unresolved and would let the mask's board number be read as the label
finding's number.

**WHAT IS NOT CLAIMED.** Not that this beats the mask; the gold ruler says the opposite, and a 0.070 AUC
gap at 32 affirmed cells is not readable anyway. Not that it beats the champion. Not that its Contusion
number (0.883) is a member-quality argument — it is one column on one split. This is an INSTRUMENT: it
buys the sign of `mask − substitute` on the competition's own targets, on identical seats, weights, order,
`overlap=False`, `LogitAverage`, raptor checkpoint at 1/3 and rank-space join.

And one live caution against this entry's own operation, recorded before the number: on the 28 cells where
v2 and all four extractors said PRESENT and four corrected-family members said ABSENT, blind adjudication
sided with v2 on all 28. That sample is selected for member-v2 disagreement and cannot be turned into a
rate, but it is the first cell-level sign that the corrected chain deletes some TRUE positives, and it
rhymes with the 31 over-corrections found in v3. If this entry loses to the mask, that is the first place
to look.

**THE SWAP.** `lr4e-5_ws_cf` out, `v2_corrected_s2026` in — the same seat `kneemri-goldseat` vacated, for
the same reason: the v2 ruler ranked `lr4e-5_ws` JOINT-BEST of the whole pool, and on the adjudicated
cells it comes LAST of the champion's six at 0.5781, below the 0.6008 control.

**THE MOUNT.** `v2_corrected_s2026` lives on `dimiturnt/kneemri-members-2`, already mounted and already
uploaded — it needs no encoder the champion does not hold (same 12-slice recipe family, fold 0) and no
dataset push. `run_check` is run on this kernel's own log before the slot is counted and the census must
read 7 of 7: `farseat_v1` is what a silent member drop costs, and a dropped member and a harmful member
look identical from the board.

PRE-REGISTRATION, written before the number, against the champion's 0.944, `BOARD_RESOLVABLE = 0.0015`,
and tonight's `kneemri-goldseat` score G.

- **Above G by more than 0.0015** — the SUBSTITUTION is the operation, and the gold ruler's ranking column
  does NOT pick operations even though it picks members. Next corrected table substitutes rather than
  masks, and the 5,068-cell tail stays supervised.
- **Within 0.0015 of G** — mask and substitute are the same intervention at the board's resolution, which
  is what the 99.4% cell overlap predicts. Then stop buying seats on this axis: both are the same finding,
  it is worth about one seat, and the next spend goes to the corpus-scale rescore that CHANGES which cells
  are suspect rather than re-arguing what to do with these ones.
- **Below G by more than 0.0015** — the mask's extra 5,068 cells are not innocent after all; deleting them
  helps more than correcting the sharp ones. That re-prices the ~0.07 figure and says the discordance set
  carries label noise the lone-dissent selector cannot see.
- **Both at or below 0.9425** — the vacated seat, not the operation, is the story: `lr4e-5_ws` was carrying
  something both rulers are blind to. Check `run_check`'s census FIRST, then re-run as an ADDITION.

---- the parent kernel's own reading, kept because this entry inherits its seventh seat ----

cnnseat — the champion seven, plus the first non-transformer member this project has ever fitted.

**WHAT IS BEING PRICED.** An ARCHITECTURE FAMILY, not a seating. Every one of the 122 members on the
shipped roster is a vision transformer or a hybrid; twelve of the thirteen registry encoders are. The one
convolutional line, `radnet121`, closed EMPTY at 0.6716 — and that number is a reading about RadImageNet's
weights, not about convolution. No ImageNet CNN had ever been fitted here, so "CNNs are weak on this task"
was a belief the ledger could not support in either direction. `resnet_224_short` makes it a number:
resnet50d at 224 px, 2.5D through-plane RGB, max over 20 slices, full-trunk finetune at 2e-4, twelve
epochs — the budget every other seat on the ledger was fitted under. It peaked at **0.8527 on the 909 at
epoch 8** and converged, which TIES the best single member the project owns (`res392_ws` 0.8523).

**WHY THIS IS NOT AN EIGHTH FLAT RESEATING.** knee-urvf's standing result is that every operation on the
champion's SEATING is inert — seven board flats, and every one of them a permutation, a weight, a count or
a refit WITHIN the ViT family. This entry does not move who sits or how they are weighted. It adds one
seat from a family the board has never held: the only prior non-ViT seat is `raptor_v4_swa`, which is
FOREIGN (someone else's checkpoint, mounted, not ours) and is already in this entry unchanged at 1/3.

**THE OFFLINE RULER SAYS IT CANNOT TELL, AND THAT IS THE POINT.** `seat_value --against honest12 --ruler
ledger75` reads **+0.0007** as a thirteenth seat and **+0.0020** as the best swap (for `mae_matched`) —
both under the +0.0040 under which this ruler has never been shown to pick a board tick. So no score is
claimed from it. What the member axes DO say is that it is the odd one out: nearest neighbour 0.8581
against the roster's own 0.8956 boundary, js 0.0188, maha 4.654 — the DISTINCT cell, not the NOISY one.
D277's finding is the reason that disagreement is worth a slot rather than a shrug: method ranking does
not reliably survive a cohort change, our holdout ordering is an in-distribution statement about 909
studies, and the two members that have actually MOVED this board were both out-of-distribution. An
in-distribution rank ruler is the exact instrument that cannot answer this question.

**IT IS ALSO THE MOUNT'S FIRST EXECUTION.** `dimiturnt/resnet50d-224` has never run on Kaggle, and neither
has any convolutional encoder path in the kit. `farseat_v1` is what that costs when it is not checked: a
COMPLETE kernel, a real `submission.csv`, a board score, and one member silently dropped for five days.
`run_check` is run on this kernel's own log before the slot is counted, and the census must read 7 of 7.

**EVERYTHING ELSE IS THE CHAMPION, UNCHANGED.** Same six, same order, same `overlap=False`, same
`LogitAverage`, same raptor checkpoint at weight 1/3, same rank-space join by study.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.946 and above** — architecture family is a live axis and the offline ruler is blind to it exactly as
  D277 predicts. Re-pin `comp/champion.json`, and the 48-epoch arm plus a second CNN family (convnextv2,
  already on the registry) take the next slots ahead of everything else.
- **0.9455 to 0.9459** — a real gain, one resolution step clear. The family is worth a seat; continue with
  the long arm before widening.
- **0.9425 to 0.9455** — FLAT. Then the offline +0.0007 transferred honestly and a tied-solo CNN adds
  nothing a ViT of the same solo would not. This does NOT close the family: `resnet_224_long` (48 epochs,
  his recipe's real ask) is on the card as this ships, and a member fitted to a quarter of its budget is
  the weakest possible test of the axis. What it DOES close is the cheap version of the question.
- **Below 0.9425** — the seat is actively harmful, which for an equal-weight average of seven would be the
  first such reading this project has taken. Check `run_check`'s census FIRST: a dropped member and a
  harmful member look identical from the board.
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
        # THE ONLY LINE THAT DIFFERS FROM THE CHAMPION IS A SWAP, NOT AN ADDITION -- and it is the SAME
        # seat `kneemri-goldseat` vacated earlier tonight, filled by the other operation on the same
        # finding. `lr4e-5_ws_cf` stood here; `v2_corrected_s2026` takes its seat. Seat count, order, vote
        # and foreign weight are unchanged, so this entry differs from `stackseat_v1` in exactly one
        # member, and from `kneemri-goldseat` in exactly one member.
        #
        # WHY THE SEAT WAS VACATED. 469 holdout cells were hand-adjudicated blind (ten chunks, key sealed
        # until the last verdict was on disk). Scored with `research/scripts/adjudicated_rescore.py`,
        # `lr4e-5_ws` is the WORST of the champion's six on the cut-free column -- AUC-vs-verdict 0.5781
        # against a 0.6008 control -- and it is the member the v2 ruler ranked joint-best of the entire
        # pool. The member most aligned with v2's error distribution wins on the v2 ruler and loses on
        # gold, which is also why three differently-seated entries all read 0.944: the seat search was
        # optimising v2-alignment, and every v2-aligned set ties.
        #
        # WHY THIS MEMBER RATHER THAN THE MASK. `v2_corrected` SUBSTITUTES on the 2,315 cells where v2 is
        # the lone dissenter against three agreeing extractors; `discord_mask` DELETES supervision on the
        # 7,369 cells those extractors disagree on. 2,301 of the 2,315 (99.4%) sit inside the 7,369, so
        # these are the same cells under two operations, not two findings. The selector audit prices the
        # substitution's cells at precision 0.727 / recall 0.533 (~20x over a 3.7% base) against 0.156 for
        # the mask's set; the gold ruler prices the MASK first (0.6982) and this member fourth (0.6272,
        # control 0.6008, constant-0 accuracy floor 0.932). The two rulers disagree about the OPERATION
        # and nothing offline breaks the tie, so the board is asked. This seat is bought as an instrument
        # for the sign of `mask - substitute`, not as a claim that it beats the mask.
        #
        # It lives on the SECOND shelf (`dimiturnt/kneemri-members-2`), already uploaded, and needs no
        # encoder the champion does not already mount -- same 12-slice recipe family, fold 0.
        "v2_corrected_s2026",
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
