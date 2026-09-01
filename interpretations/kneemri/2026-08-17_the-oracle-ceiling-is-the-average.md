# The per-label ceiling IS the plain average, and the v3 member does not pay

2026-08-17 · `python -m kneemri.evaluation.member_curve` over seven members, fold 0, split seed 2026,
no TTA, cached readings. Log: the run of 06:16 UTC.

## The measurement

    member                        alone
    res392_ws_s2026               0.8523
    calibrated_targets_s2026      0.8493
    sliceattn_s2026               0.8473
    extended_ws_s2026             0.8462
    intensity392_ws_s2026         0.8459
    dinov3_base_lora_s2026        0.8269   (new: DINOv3-base + LoRA, 12 epochs, 295 s/epoch)

The column is `member_curve`'s own reading of each checkpoint. It is not the trainer's number: the arm
recorded 0.8268 for the same weights, because the trainer scores its holdout as it goes and the curve
re-reads every member through one shared cache and one shared study order. A 0.0001 gap between the
two is the two rulers agreeing; a large one would mean the curve is not reading what the arm fitted.
    lora_r8_v3small_224_s2026     0.8215

    oracle 0.8590   plain average 0.8588   best single 0.8523   headroom +0.0002

## What the oracle settles (knee-4ir)

The oracle takes, for each of the twelve findings, whichever member scores that finding best ON THIS
HOLDOUT — a choice made with the answers, so it is an upper bound by construction and can never be a
submission. It bounds every per-label weighting scheme that could ever be written over these members.

It sits **+0.0002 above the plain rank-average**. Two orders below the 0.0062 fine-tuned floor and
below the 0.0053 board floor. So per-label member weighting is worth nothing here, and it is worth
nothing not because our weighting is bad but because there is nothing to weight: the members do not
disagree per label in a way any selection rule could exploit. `LabelCombine`'s split-half column says
the same thing from the reachable side — its top-4 line (0.8612) equals the plain best-4.

That closes a class of work before it is written, which is what a ceiling is for.

## What the same run says about the DINOv3 member (knee-98j)

`dinov3_base_lora_s2026` is the strongest v3 member we have (0.8268 alone, against 0.8217 for the
v3-small LoRA). It still does not earn a slot:

    best 4  0.8612   res392 + intensity392 + calibrated_targets + sliceattn
    best 6  0.8603   the same four + dinov3_base_lora + extended_ws
    best 7  0.8588   all seven

Every subset that includes a v3 member scores below the best subset that excludes them. The diversity
argument for v3 — different pretraining, so different failures — does not show up as ensemble gain on
this holdout. It is a real 0.8269 model; it is not a member.

## What this does NOT say

- It does not price a member trained differently *in supervision* (ordinal severity, knee-mxj) or in
  *data coverage* (knee-jk3). The oracle bounds combination of THESE seven; a member that fails on
  different studies would move the average itself, not the headroom above it.
- The holdout is 58 annotated studies' worth of ruler with a 0.0062 fine-tuned floor. A +0.0002
  headroom is inside noise, but here that is the finding: the ceiling is not *distinguishable* from
  the average, so no weighting scheme can be argued to clear the floor either.
- Nothing here was submitted. The numbers are holdout only.
