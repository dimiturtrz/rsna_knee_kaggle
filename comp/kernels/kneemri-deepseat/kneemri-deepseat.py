"""deepseat — goldseat's swap, PLUS the one member the offline pool does not price down.

**THE OTHER END OF A DISAGREEMENT THIS PROJECT IS NOT RESOLVING.** D367 ranked six candidates two ways
and the rankings came back near-anti-aligned: by marginal gain `labelwise_depth24` is the only positive
on either ruler (+0.0012 v2 / +0.0011 T1) and is the second-MOST correlated member; by decorrelation
`v5blend_span` leads (0.849 / 0.837) and loses on both rulers. `kneemri-twoshape` ships the
decorrelation end. This entry ships the GAIN end. Neither is a bet on which ranking is right — the
board is being asked, because after eight entries reading exactly 0.9440 it is the only ruler left that
has seen the competition's distribution.

**THE SEATS.** goldseat's swap is kept (`lr4e-5_ws_cf` out at gold AUC-vs-verdict 0.5781,
`discord_mask_s2026` in at 0.6982) and `labelwise_depth24_s2026` is ADDED as a seventh, making seven of
ours plus the raptor seat at weight 1/3. It is the only member on the shelf that is not a label-table
variant: 24 slices, labelwise slice-pooling, plain v2 labels, solo 0.8617 @holdout909 midpoint at epoch
8 of 12, converged.

**THE WEAKER PRIOR, STATED.** The record says the ensemble saturates at six seats and degrades past
them, so an ADDITION is the less likely of the two moves. It is taken because that saturation was
measured entirely within one recipe family, and this member is the only candidate outside it.

---

Inherited from `goldseat`, whose swap this entry keeps:

**WHAT IS BEING PRICED.** A RULER, by way of a member. Every seating decision this project has ever made
was made on a holdout scored against `report_labels_v2`. On 2026-09-11 a blind hand-adjudication of 469
holdout cells finished — ten chunks, key sealed until the last verdict was on disk — and it says v2 is
wrong on 89.1% of the cells where our table dissents from three independent report extractors, and wrong
on 277 of 277 tear-column dissent cells. The consequence measured today is that the v2 ruler penalises a
member roughly in proportion to how much of v2's error the member unlearned. Ranking the champion's own
six on the adjudicated cells reorders them, and it puts `lr4e-5_ws` — the member the v2 ruler ranked
JOINT-BEST of the whole pool — last, at AUC-vs-verdict 0.5781 against a 0.6008 control.

**THE SWAP.** `lr4e-5_ws_cf` out, `discord_mask_s2026` in. Nothing else moves: same seat count, same
order, same `overlap=False`, same `LogitAverage`, same raptor checkpoint at weight 1/3, same rank-space
join. `discord_mask` was KILLED on the v2 ruler at −0.0417 for masking the 7,369 cells four extractors
disagree on — the ruler scoring removed BIAS as removed supervision. On the gold cells it ranks first of
everything on disk: **AUC-vs-verdict 0.6982, balanced accuracy 0.649, specificity 0.860.**

**WHAT IS NOT CLAIMED.** Its raw accuracy lead (0.832) OVERSTATES it: the 469 are 93% negative, so a
constant-0 predictor scores 0.932 and accuracy at a fixed cut on this split is largely a specificity
statement. Its sensitivity cost is UNREADABLE at 32 affirmed cells — 0.438 [0.28, 0.61] against the
control's 0.531 [0.36, 0.69], intervals overlapping. The cut-free AUC lead of +0.097 is the entire basis
for this seat, and the tear-split numbers are specificity-only by construction (zero affirmed cells).

**WHY IT IS WORTH A SLOT AND A RESEATING IS NOT.** All 119 scored rows say the ensemble axis is spent:
best solo 0.933, best ensemble 0.944, saturating at six seats and degrading past them, and nothing has
ever exceeded 0.944 across 6-, 7- and 8-seat entries two weeks apart. The 0.010 gap to the leader is the
same size as the ensemble's entire lifetime contribution, so no seating closes it — only a better MEMBER
does. This is the first entry whose member was chosen by a ruler that is not v2, and `discord_mask` never
trained on the cells the other six overfit, which is the diversity argument stated as a mechanism rather
than as a distance.

**THE MOUNT.** `discord_mask_s2026` lives on `dimiturnt/kneemri-members-2`, already mounted by this
kernel's parent, and needs no encoder the champion does not hold — same 12-slice recipe family, fold 0.
`run_check` is run on this kernel's own log before the slot is counted, and the census must read 7 of 7:
`farseat_v1` is what a silent member drop costs, and a dropped member and a harmful member look identical
from the board.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.946 and above** — the gold ruler transfers, and it is the first instrument in this project's history
  to move the board. Re-pin `comp/champion.json`; the masked-table route takes the card ahead of
  everything else, and the next adjudication batch becomes the highest-value work there is.
- **0.9455 to 0.9459** — a real gain, one resolution step clear. Seat the second gold-ranked member
  (`v2_corrected`, Contusion 0.883) next and keep the ruler.
- **0.9425 to 0.9455** — FLAT. Then a member that is better on 469 adjudicated cells is not better on the
  Kaggle test set, and the honest reading is that the gold ruler measures a real property of the LABELS
  that does not carry to the competition's own targets. That would not retract the label finding; it
  would bound where it pays, and the next spend goes to the corpus-scale T1 rescore instead.
- **Below 0.9425** — `lr4e-5_ws` was carrying something the gold ruler cannot see, or `discord_mask`'s
  unreadable sensitivity cost is real and large. Check `run_check`'s census FIRST, then re-run the swap
  as an ADDITION rather than a replacement before concluding anything about the ruler.

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
        # THE ONLY LINE THAT DIFFERS FROM THE CHAMPION IS A SWAP, NOT AN ADDITION. `lr4e-5_ws_cf` stood
        # here and has been VACATED; `discord_mask_s2026` takes its seat. Seat count, order, vote and
        # foreign weight are all unchanged, so this entry differs from `stackseat_v1` in exactly one
        # member and nothing else.
        #
        # WHY THIS SWAP. 469 holdout cells were hand-adjudicated blind (ten chunks, key sealed until the
        # last verdict was on disk). Scored on those cells with `research/scripts/adjudicated_rescore.py`,
        # `lr4e-5_ws` is the WORST of the champion's six on the cut-free column — AUC-vs-verdict 0.5781
        # against a 0.6008 control — and it is the member the v2 ruler ranked joint-best of the entire
        # pool. The member most aligned with v2's error distribution wins on the v2 ruler and loses on
        # gold, which is also why three differently-seated entries all read 0.944: the seat search was
        # optimising v2-alignment, and every v2-aligned set ties.
        #
        # `discord_mask` is the opposite member. It was KILLED on the v2 ruler at -0.0417 for masking the
        # 7,369 cells four report extractors disagree on, and on the gold cells it ranks FIRST of
        # everything on disk: AUC-vs-verdict 0.6982, balanced accuracy 0.649, specificity 0.860. Its
        # accuracy lead (0.832) overstates it -- the split is 93% negative, so constant-0 scores 0.932 --
        # and its sensitivity cost is UNREADABLE at 32 affirmed cells (0.438 [0.28, 0.61] against the
        # control's 0.531 [0.36, 0.69]). The cut-free lead is what this seat is bought on, and it is
        # +0.097. It never trained on the cells the other six overfit, which is the diversity argument.
        #
        # It lives on the SECOND shelf (`dimiturnt/kneemri-members-2`) and needs no encoder the champion
        # does not already mount -- same 12-slice recipe family, fold 0.
        "discord_mask_s2026",
        "through_dense_cf",
        # THE SEVENTH SEAT, AND IT IS AN ADDITION RATHER THAN A SWAP. `labelwise_depth24_s2026` is the
        # ONLY candidate of D367's six with a POSITIVE marginal add on EITHER ruler: +0.0012 on the v2
        # proxy and +0.0011 on the corrected T1 proxy, 12-column macro AUC@holdout909@midpoint. Both are
        # an order of magnitude under the 1.5% bar and neither is a finding on its own -- it is seated
        # because it is the only member measured tonight that the offline pool does not actively price
        # DOWN, and because it is the only one of the six that is not a label-table variant at all.
        #
        # WHAT IT VARIES. 24 slices with labelwise slice-pooling, trained on plain `report_labels_v2`.
        # Its solo is 0.8617 @holdout909 midpoint, epoch 8 of 12, CONVERGED (the peak is not the last
        # epoch), which is the best new solo in tonight's cohort and +0.0052 over `band_span` 0.8565 --
        # DIRECTIONAL, stated as such. Plumbing was asserted from the checkpoint's own `options` before
        # this seat was built: slices=24, slice_pooling=labelwise, calibrated_targets=True,
        # hold_annotations=True, and `slices.gate.3.weight` non-zero at 768 of 768 entries.
        #
        # THE DISAGREEMENT, DISCLOSED. Its residual correlation with the proxy is 0.897 (v2) / 0.888
        # (T1) -- the SECOND-HIGHEST of the six. So on this member gain and decorrelation disagree in
        # the OPPOSITE direction to `v5blend_span`, which leads decorrelation and loses on gain. D367
        # declines to resolve that and so does this entry: `twoshape` ships the decorrelation end of the
        # disagreement and this ships the gain end, deliberately, so the board prices both.
        #
        # SEVEN OF OURS PLUS THE RAPTOR SEAT. The record says the ensemble saturates at six and degrades
        # past it, so an addition is the weaker prior; it is taken here because the saturation was
        # measured WITHIN one family and this member is the only one on the shelf outside it.
        "labelwise_depth24_s2026",
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
