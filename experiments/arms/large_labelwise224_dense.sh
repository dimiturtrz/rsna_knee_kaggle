#!/usr/bin/env bash
# large_labelwise224_dense.sh — the capacity rung above base, spent at the corner that fits.
#
# bead: knee-jzho6
# reading: large_labelwise224_dense_s2026.holdout.npz
#
# WHAT LICENSES THIS ARM. Capacity has paid TWICE at this exact corner on the current ruler.
# `base_labelwise224` 0.8899 against `labelwise224` 0.8832 is +0.0067 over a 0.0062 paired floor, and
# the schedule step on that trunk read `base_labelwise224_dense_drop40_long36` 0.8996 against
# `base_labelwise224_dense` 0.8950. Base at this INSTRUMENT corner now sits 0.0034 under the best solo
# the project owns (`labelwise_dense_drop40_long36` 0.9030, small at 336/20sl) — inside the paired
# floor. The trunk is not what is missing at 224; the rung above it has never been tried here.
#
# WHY NOT AT THE SHIPPING CORNER. Seven OOMs now sit against a wider trunk at 336, and the causes are
# ruled out by measurement rather than argument (see `base_labelwise336.sh`'s header): gradient
# checkpointing was ON for all seven (`recipe.sh:31` defines THROUGHPUT as `--recompute-activations`),
# 12 slices died at a LARGER allocated figure than 20, `--eager-encoder` reproduced the request to the
# byte, and the 27 GB attempt died at 25.59 GB of demand under its own cap. The binder is allocator
# fragmentation — ~28.5 GiB reserved against 23.21 in use — and `expandable_segments` is a confirmed
# no-op on this Windows machine. So 336 is machine-closed for a wide trunk and 224/12sl is the only
# corner a capacity step can be spent at. This arm cannot be the deliverable on its own; the deliverable
# is this recipe at `--epochs 36 --slot-dropout 0.40` if this clears.
#
# ONE CHANGE against `base_labelwise224_dense`: `--variant large`. Byte-identical otherwise — same 224
# px, same 12 slices, same 0.2-0.8 band, same batch 12 (so the pairwise `--rank-weight` term sees the
# same 66 pairs per step), same `--calibrated-targets --hold-annotations`, same 12 epochs, same array on
# disk. The control is free at 0.8950 ledger75, epoch 8 of 12, 909 studies.
#
# MECHANISM. Large is 304M parameters and 1024-wide against base's 86.6M and 768. The claim is the one
# the 224 corner has already twice sustained: a labelwise readout can SPEND width — it holds a separate
# query per label, so a wider feature gives each of the twelve its own subspace rather than forcing them
# to share. The competing outcome is that 909 studies cannot support the parameters and the run overfits,
# which shows as a peak well before epoch 12; that is distinguishable from a flat result by the epoch of
# the peak, and its re-test is `--slot-dropout 0.40 --epochs 36` rather than a retreat to base.
#
# DINOV2 ONLY, and SHIPPABLE. Large is the DINOv2 family (`backbone.py:208-216`), which is the owner's
# standing constraint; DINOv3 stays refuted at -0.038. The scored kernel runs a T4, which is the reason
# every strong public notebook stops at base — that constraint binds INFERENCE only, and the mount
# `metaresearch/dinov2/PyTorch/large/1` exists at 1.2 GB, so a large member can be packaged. Training is
# on our own card. If this clears, `mount_check --live` is what proves the packaging, not this comment.
#
# PRICE, out loud. The control ran ~108 s/epoch warm at this corner; large is ~3x base's compute, so
# ~330 s/epoch and ~1.1 h for 12 epochs. Zero decode (the 224px/12sl array is on disk), zero submission
# slots, control free. P(clears 0.9030, the best solo) ~ 0.30. P(clears the 1.5% bar) ~ 0.12.
#
# VRAM, and the risk named before it is paid. Base measured 12.99 GiB allocated at this exact corner.
# Only 6 of large's 24 blocks train under the inherited `--unfreeze-last 6` and the frozen prefix runs
# under no_grad, so the dominant term scales with WIDTH rather than depth: ~1.78x on 1024/768, near
# 23 GiB. That is the same neighbourhood the 336 arms fragmented in, so P(OOM) ~ 0.4 — but an OOM here
# costs ~3 minutes of card under a named cap, not the hour, which is what makes this worth launching
# rather than modelling further.
#
# BANDS, pre-registered, ledger75, against base_labelwise224_dense 0.8950.
#   * >= 0.9012  the rung pays. CORRECTION, made while this arm was running: the band originally read
#                "already level with the best solo the project owns", anchored on
#                `labelwise_dense_drop40_long36` 0.9030. That is NOT the best solo. `dml_b_s2026` reads
#                **0.9078** ledger75 @ e18 on the same 909-study population, and a weight SOUP is a single
#                averaged checkpoint — so `basinsoup_dmlb_k4` 0.9092 (README:17763) is a solo too. The
#                anchor was 0.0048 low, most of a paired floor. Clearing 0.9012 is the one-variable
#                capacity verdict and nothing more; the deliverable still has to reach 0.9078+. The next
#                arm is this one at `--epochs 36 --slot-dropout 0.40`.
#   * 0.8950-0.9012  directional. Keep nothing, compose nothing. Read the epoch of the peak: still
#                improving at 12 means the budget bound it and the 36-epoch version is the re-test;
#                an early peak means 909 studies will not support 304M parameters and the capacity axis
#                is DATA-closed rather than machine-closed, which is a different verdict and a bigger one.
#   * <  0.8950  capacity has a ceiling between base and large on this data. Read HOW far below: below
#                0.8899 (the base 224 control's own control) says the width is actively harmful and the
#                axis closes at base, which retires every future large/giant proposal.
#
# READ THE FOCAL LABELS, NOT ONLY THE MEAN. Lateral Meniscus -0.1925, ACL -0.1225, MCL -0.0964 lead the
# model-side error census. Width's claim is per-label subspace, so it should move the labels that are
# currently sharing a representation worst — the three focal ones. A mean that moves without them is not
# this mechanism. Say so.
#
# HARNESS CHECK. The printed recipe must name "variant": "large" AND "image_size": "224" AND
# "slices": "12" AND "slice_pooling": "labelwise" AND "calibrated_targets": "True", and the trunk must
# load `facebook/dinov2-large`. Stamped values are STRINGS; a grep for lowercase true returns empty and
# reads like a missing flag. If "variant" reads "base" this is the control re-run and the card is being
# spent on nothing.
#
# control  base_labelwise224_dense_s2026 = 0.8950 ledger75 @ epoch 8, 909 studies. No control run is owed.
#
# deliberate: --variant large (the arm's thesis)
# deliberate: --vram-gb 26 — the cap that leaves the driver its room on a 31.84 GB card. A cap set
#             higher PERMITS PCIe paging, which reads as 100% utilisation at a quarter of the power
#             budget rather than as a crash. Diagnose by img/s against the control, never by utilisation.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train large_labelwise224_dense \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --variant large --vram-gb 26 --calibrated-targets --hold-annotations \
  --labels "$LABELS" --out "$MODELS/large_labelwise224_dense_s2026.pt"
