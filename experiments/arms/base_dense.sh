#!/usr/bin/env bash
# base_dense.sh — the settled recipe on the bigger encoder, which has never been measured.
#
# WHY THIS ARM AND NOT ANOTHER KNOB. Eleven arms re-read on 2026-08-20 came back negative on the mean,
# eleven out of eleven, and the oracle over seven members prices per-label recombination of what we
# already have at +0.0002 (2026-08-17). Both readings say the same thing from different directions: the
# member POPULATION is exhausted, not the tuning. Every one of those members is dinov2-small — 384-wide,
# 22M parameters — and the whole operating point was tuned on it.
#
# WHAT HAS ACTUALLY BEEN MEASURED ON BASE, and it is less than it looks. `compose_base` is the only
# dinov2-base arm in the ledger and it DIED at epoch 7 of 24 and was never rerun. It also predates the
# entire current operating point: no `--whole-stack`, 12 slices rather than 20, no band. Its 0.8283
# control belongs to a recipe nothing here still runs. So base at the settled operating point is not a
# weak result, it is an ABSENT one.
#
# THE MECHANISM, stated so it can fail. Base is 768-wide against small's 384 and carries twelve blocks
# of roughly four times the arithmetic. The recipe freezes blocks 0-5 and fits 6-11, so what doubles is
# the width of the representation the head reads and the capacity of the half that adapts. The findings
# this corpus is weakest on are the focal ones — Lateral Meniscus at 0.7981 is our worst of twelve —
# and a wider representation is the one change that plausibly buys small structure without touching the
# sampling geometry that eleven arms just showed is not the constraint.
#
# ONE VARIABLE against the control: `--variant base`. Everything else is byte-identical to
# `band_dense`, including the backbone rate. The rate is DELIBERATELY not rescaled for the larger model
# even though the usual practice would lower it: rescaling it here would confound the encoder with its
# schedule, and there is no isolated measurement of a base-appropriate rate to borrow. If this arm
# reads flat-to-slightly-down, the rate is the first suspect and the follow-on arm is the rate, not the
# abandonment of the encoder.
#
# THE FALSIFIABLE PART. Three outcomes, separated before the run:
#   * A LIFT OF 1.5% OR MORE ON THE MEAN clears the project bar on its own and makes the backbone the
#     axis the next several arms live on — the rate sweep, then base under the two levers band_dense and
#     band_wide already carry.
#   * A LIFT UNDER THE 0.0062 FLOOR, or a small loss, is NOT a refutation of the encoder. Base at
#     small's learning rate is a known-underspecified pairing, and the seat question is separate anyway:
#     a member that ties on the mean but is 768-wide is exactly the different-in-KIND member the oracle
#     says our current pool lacks, so it goes to `member_curve` before any verdict is written.
#   * A COLLAPSE — anywhere below 0.83, the territory `compose_base`'s dead recipe occupied — says the
#     larger encoder is being mis-fit rather than being worse, and sends this to the rate and the freeze
#     depth. Blocks 6-11 of base hold four times the parameters that `unfreeze_last 6` was chosen
#     against on small, and that constant has never been re-derived for this width.
#
# COST. Roughly four times small's arithmetic per image at the same 336px and 20 slices, so expect an
# epoch well past small's 361 s. `--recompute-activations` is already in $THROUGHPUT and the VRAM cap
# reads the encoder (knee-kwdh), so the batch is not the thing to hand-tune first. SMOKE IT before
# queueing: `python -m kneemri.experiments.arm_check --smoke` writes the truncated cache, then
# `--studies 60 --epochs 1` prices the epoch honestly.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical recipe on dinov2-small.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# --batch 6 IS A SECOND VARIABLE, and it is forced. MEASURED 2026-08-20: base at 336 px and 20 slices
# needs 56.3 GB at the control's batch of 12 against a 32 GB card, so there is no version of this arm
# that holds the control's batch. 6 derives 28.25 and SMOKED at 27.14 GB peak. The proper fix is
# gradient accumulation — the control's effective batch at half the activations — which does not exist
# yet (knee-55k8). Until it does, a flat-to-down read here is confounded with the halved batch and the
# follow-on is accumulation, not the abandonment of the encoder.
run_train base_dense \
  $RECIPE --epochs 12 --batch 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$PROCESSED/base_dense_s2026.pt"
