#!/usr/bin/env bash
# bead: knee-5i6d
# base_clean.sh — dinov2-base at the best solo recipe, with the three confounds removed.
#
# WHY THIS ARM. `base_dense` read -0.0118 against `stackfit` and that reading is what the ledger has
# been quoting as "the bigger trunk is worse". It is not a clean reading of the trunk. `run_comparison`
# prints the diff itself, and it is FOUR flags, not one:
#
#     batch 12->6, ema True->False, calibrated_targets True->False, variant small->base
#
# Three of those four are known-positive levers on their own, and all three were LOST in the arm that
# was then read as a verdict on width. `v3base_dense` (-0.0176) carries the identical three-flag
# confound, so the whole "scale hurts here" reading rests on two arms that each gave up EMA, calibrated
# targets and half the effective batch to buy the width. That is the harness being a suspect, and the
# fix is one arm rather than an argument.
#
# WHY THE BATCH CONFOUND IS NOW REMOVABLE and was not when `base_dense` ran. Base at 336 px and 20
# slices derives 56.3 GB at batch 12 against a 32 GB card, which is why that arm halved it. Gradient
# accumulation did not exist then (knee-55k8, since landed) and does now, so `--batch 6 --accumulate 2`
# restores the control's EFFECTIVE batch of 12 at half the activations. The optimiser sees the same
# twelve studies per step it saw on small.
#
# ONE VARIABLE against the control: `--variant base`. Everything else is the `stackfit` line, flag for
# flag, including `--calibrated-targets --ema --hold-annotations` and the 4e-5 backbone rate. The rate
# is deliberately NOT rescaled for the wider trunk, for the same reason `base_dense` gave: rescaling it
# would confound the encoder with its schedule, and no isolated base-appropriate rate exists to borrow.
#
# THE MECHANISM, stated so it can fail. Base is 768-wide against small's 384, twelve blocks at roughly
# four times the arithmetic; the recipe freezes 0-5 and fits 6-11, so what doubles is the width the head
# reads and the capacity of the half that adapts. knee-urvf reads the ensemble as best-single + ~0.0143
# and that bonus has not grown in fourteen days, which makes MEMBER QUALITY the only free term left;
# knee-nj4l wants a member whose errors are somewhere else, and a 768-wide trunk in the family that
# carries every current seat is the cheapest such member that does not also change the corpus.
#
# PRE-REGISTERED, before the run:
#   * >= +0.0062 on the MEAN — the three-flag confound WAS the -0.0118, width is a live lever, and the
#     next arms are the base rate sweep and base under the two levers `band_dense`/`band_wide` carry.
#     It also re-opens `v3base_dense` and `raddino_normfix`, which carry the same confound.
#   * +0.0015 to +0.0061 — directional. Keep, never a finding, no slot. Width is not free and the
#     confound explained only part of the gap.
#   * -0.0061 to +0.0014 — FLAT. This is the branch that would settle it: base at effective batch 12
#     with EMA and calibrated targets ties small, so the -0.0118 was the confounds and WIDTH ITSELF IS
#     WORTH NOTHING at this operating point. That closes scale as a quality lever and sends the search
#     to routing and slice budget (knee-nj4l axes 2 and 3). A tie still goes to `member_axes` +
#     `seat_value` before any verdict: a 768-wide member that ties on the mean is exactly the
#     different-in-KIND candidate the pool census says is missing.
#   * <= -0.0062 — SUSPECT THE HARNESS FIRST. Verify this run recorded `accumulate: 2`, `batch: 6`,
#     `ema: True`, `calibrated_targets: True`, `hold_annotations: True` and `variant: base`, and that
#     the peak memory did not force a silent fallback. Only after those come back clean does base at
#     small's rate read as genuinely mis-fit, and the follow-on is the RATE and the freeze depth
#     (`unfreeze_last 6` was derived against small's width and has never been re-derived for base's),
#     never the abandonment of the encoder.
#
# COST. Roughly four times small's arithmetic per image at the same 336 px and 20 slices; `base_dense`
# is the honest precedent for the epoch. `--recompute-activations` is in $THROUGHPUT and the VRAM cap
# reads the encoder (knee-kwdh). SMOKE IT before queueing.
#
# PRICE: ~4.5 h of card, 0 slots.
# P(clears +0.0062) ~ 0.35 — three known-positive levers were lost in the arm being corrected, and they
# plausibly account for the whole -0.0118; against that, two independent wider trunks both read down.
#
# control  stackfit_s2026 = 0.8631 (best epoch 7), this line on dinov2-small at batch 12.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train base_clean \
  $RECIPE --epochs 12 --batch 6 --accumulate 2 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --ema --hold-annotations \
  --labels "$LABELS" --out "$PROCESSED/base_clean_s2026.pt"
