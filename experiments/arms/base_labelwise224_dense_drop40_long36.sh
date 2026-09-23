#!/usr/bin/env bash
# base_labelwise224_dense_drop40_long36.sh — the schedule step, on the validated base trunk.
#
# bead: knee-mm6v0
# reading: base_labelwise224_dense_drop40_long36_s2026.holdout.npz
#
# WHAT LICENSES THIS ARM. Two measured facts meet here. `base_labelwise224_dense` landed 0.8950 ledger75
# at epoch 8 of 12 on 909 studies, the best the base trunk has read. And on the SMALL trunk the step from
# `labelwise_dense` (0.8964) to `labelwise_dense_drop40_long36` (0.9030) is exactly two flags — `--epochs 36`
# and `--slot-dropout 0.40` — worth +0.0066, above the 0.0062 paired floor. This arm asks whether that step
# transfers to the wider trunk. Composed naively it projects ~0.9016 against the 0.9030 best solo.
#
# WHY THIS AND NOT MORE 336 WORK. Four OOMs sat against the base trunk at 336 when this arm was written.
# CORRECTION, recorded after the fact: this paragraph originally said the earlier death at a 27 GB cap had
# checkpointing OFF. It did not. `experiments/recipe.sh:31` defines THROUGHPUT as exactly
# `--recompute-activations` and every one of these invocations passes `$THROUGHPUT`, so gradient
# checkpointing was ON for ALL of them. `base_labelwise_dense336` (20 slices) died at 21.06 GiB allocated
# + 3.57 requested; `base_labelwise336` (12 slices) died at 23.21 + 2.38, byte-identical to its own earlier
# death. So the footprint neither fell with gradient checkpointing nor scaled with the slice count, and the
# binder is allocator fragmentation (~28.5 GiB reserved against 23.21 in use) — see `base_labelwise336.sh`'s
# header for all seven crashes and the four causes ruled out by measurement. That is a diagnosis to make off
# the card, not four more crashes to pay for. 224 is an instrument corner and this arm cannot be the deliverable on
# its own, but it is the only mechanism-sound step that is KNOWN to fit: the control measured 12.99 GiB
# allocated at this exact corner and neither new flag touches the activation footprint.
#
# ONE CHANGE against `base_labelwise224_dense`: `--epochs 36 --slot-dropout 0.40`, taken as one step because
# that is how it was measured on the small trunk — a longer one_cycle without the regulariser is a different
# claim and was never run separately. Everything else is byte-identical: same base trunk, same 224 px, same
# 12 slices, same 0.2-0.8 band, same batch 12, same `--calibrated-targets --hold-annotations`, same array.
#
# MECHANISM. Slot dropout removes whole series slots from a study during training, so the readout cannot
# lean on one sequence being present; the longer schedule is what makes the harder problem learnable. The
# reason to expect it to pay MORE on base than on small is the same reason capacity paid at all here: a
# wider trunk has more capacity to spend memorising 909 studies, and `base_labelwise224_dense` peaked at
# epoch 8 of 12 — it was still improving into a budget that ended, which is the profile a longer schedule
# with a stronger regulariser is for. The competing outcome is that 0.40 is tuned for the small trunk's
# capacity and over-regularises the wider one, which would show as a peak well before epoch 36.
#
# PRICE, out loud. The control ran 171 s/epoch at this corner, so 36 epochs is ~1.7 h. Zero decode, zero
# submission slots, control free on disk. P(clears 0.9016) ~ 0.30. P(clears the 1.5% bar) ~ 0.08.
#
# BANDS, pre-registered, ledger75, against base_labelwise224_dense 0.8950.
#   * >= 0.9012  the schedule step transfers to the wider trunk at full size or better, and the base trunk
#                is within noise of the best solo the project owns from an instrument corner. The next arm
#                is this recipe at whatever resolution the VRAM diagnosis then permits, and that is the
#                solo deliverable.
#   * 0.8950-0.9012  directional. Keep nothing, compose nothing — and note that the small trunk read
#                +0.0066, so a smaller delta here is the step paying LESS with width, which answers the
#                transfer question and points the remaining card at the VRAM diagnosis instead.
#   * <  0.8950  0.40 is tuned to the small trunk's capacity and over-regularises this one. Read the epoch
#                of the peak: an early peak is that verdict, a peak at 36 is an undertrained budget and a
#                different arm.
#
# READ THE FOCAL LABELS, NOT ONLY THE MEAN. Lateral Meniscus -0.1925, ACL -0.1225, MCL -0.0964 lead the
# model-side error census. Slot dropout's claim is specifically about not leaning on one sequence, and those
# three are the labels a single sequence most often carries. A mean that moves without them is not this
# mechanism. Say so.
#
# HARNESS CHECK. The printed recipe must name "variant": "base" AND "image_size": "224" AND "slices": "12"
# AND "slice_pooling": "labelwise" AND "calibrated_targets": "True" AND "slot_dropout": "0.4" AND
# "epochs": "36". Stamped values are STRINGS; a grep for lowercase true returns empty and reads like a
# missing flag.
#
# control  base_labelwise224_dense_s2026 = 0.8950 ledger75 @ epoch 8, 909 studies. No control run is owed.
#
# deliberate: --epochs 36 --slot-dropout 0.40 (the arm's thesis, taken as one step)
# deliberate: --vram-gb 20 — the cap the control ran under at this exact corner (12.99 GiB allocated),
#             not a new guess. Neither new flag changes the activation footprint.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train base_labelwise224_dense_drop40_long36 \
  $RECIPE --epochs 36 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --slot-dropout 0.40 \
  --image-size 224 --variant base --vram-gb 20 --calibrated-targets --hold-annotations \
  --labels "$LABELS" --out "$MODELS/base_labelwise224_dense_drop40_long36_s2026.pt"
