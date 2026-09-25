"""twoshape — the champion seven with TWO seats swapped, chosen on the two rulers that are not v2.

**THE FACT THAT ORDERS THIS ENTRY.** `comp/board.jsonl` holds 129 rows, and EIGHT of them score exactly
0.9440 — seven distinct 7-member seatings and one 6-member, different seats, different families, two
weeks apart. That is a CEILING, not a coincidence, and it is the mechanical answer to why nothing has
gone past 0.944: reseating the existing pool is saturated. Only a member whose ERRORS are shaped
differently can move it, so this entry vacates two seats at once rather than one.

**THE TWO VACANCIES ARE THE TWO WORST SEATS ON THE GOLD RULER.** `lr4e-5_ws_cf` at AUC-vs-verdict 0.5781
and `grade_ladder_cf` at 0.6070, against a 0.6008 control and a constant-0 floor of 0.932 on all-469.
**THE TWO ARRIVALS ARE THE GOLD RULER'S TOP TWO** — `discord_mask_s2026` 0.6982 and `v5blend_span_s2026`
0.6813 — and `v5blend_span` is separately the DECORRELATION leader of D367's six candidates on both the
v2 and the corrected-T1 residual (0.849 / 0.837, lowest of six on each). Two independent non-v2 rulers,
same two names.

**WHAT IS DISCLOSED AGAINST IT.** D367 prices both arrivals NEGATIVE as marginal adds on the offline
proxy — `v5blend` -0.0025 (v2) / -0.0014 (T1), `discord_mask` -0.0029 / -0.0019 — and every label-table
member costs that proxy points on both rulers. It ships anyway because the proxy was selected on the
same v2 ruler its members were fitted under and cannot price a new shape, which is the eight flat
0.9440s seen from the offline side. This is a maximum-VARIANCE entry against a hard ceiling, and it is
said here rather than after the score.

---

Inherited from `goldseat`, the one-seat version of the same argument:

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


def main() -> None:
    boot()

    from kneemri.kaggle.seated_entry import SeatedEntry  # noqa: PLC0415 - the kit is on sys.path only after boot

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        # SECOND VACANCY. `grade_ladder_cf` stood here (gold AUC-vs-verdict 0.6070, fifth of the
        # champion's six) and has been VACATED for `v5blend_span_s2026`. This is the seat bought on
        # DECORRELATION rather than on either ruler's ordering, and the disagreement is disclosed below.
        "v5blend_span_s2026",
        "stack_increments_ff",
        # TWO SEATS DIFFER FROM THE CHAMPION, BOTH SWAPS, NEITHER AN ADDITION. `lr4e-5_ws_cf` and
        # `grade_ladder_cf` are VACATED; `discord_mask_s2026` and `v5blend_span_s2026` take their seats.
        # Seat count, order, vote and foreign weight are unchanged, so this entry differs from
        # `stackseat_v1` in exactly two members and nothing else.
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
        #
        # THE SECOND SEAT, AND THE DISAGREEMENT THAT BUYS IT. `v5blend_span_s2026` is the DECORRELATION
        # leader of six candidates measured offline on 2026-09-11 (D367): residual correlation with a
        # six-member fold-0 proxy ensemble 0.849 on the v2 ruler and 0.837 on the corrected T1 ruler,
        # the lowest of the six on BOTH. On the gold cells it ranks SECOND of everything on disk
        # (AUC-vs-verdict 0.6813 against a 0.6008 control, constant-0 floor 0.932 on all-469). Two
        # rulers that are not v2 -- blind adjudication and residual decorrelation -- name the same two
        # members, and neither of them names `grade_ladder` or `lr4e-5_ws`.
        #
        # WHAT THE OFFLINE POOL SAYS AGAINST IT, STATED BEFORE THE SCORE. D367 prices `v5blend_span`'s
        # marginal add at -0.0025 (v2) / -0.0014 (T1) on 12-column macro AUC@holdout909@midpoint, and
        # `discord_mask`'s at -0.0029 / -0.0019. EVERY label-table member costs that proxy points on
        # both rulers, and the only positive candidate (`labelwise_depth24`, +0.0012) is the
        # second-MOST correlated of the six. The gain ranking and the decorrelation ranking are
        # near-anti-aligned, and D367 declines to resolve them. The reason this entry ships anyway is
        # that the proxy pool was SELECTED on the same v2 ruler its members were fitted under, so a
        # differently-shaped error cannot register in its arithmetic by construction -- and eight
        # separate board entries now read exactly 0.9440 across seven distinct 7-member seatings and
        # one 6-member, which is the same blindness measured from the other side. Against a ceiling
        # that deep, the play is maximum VARIANCE, not maximum expected value.
        "discord_mask_s2026",
        "through_dense_cf",
    )
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
