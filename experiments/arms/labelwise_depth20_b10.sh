#!/usr/bin/env bash
# The 24-slice corner of the depth axis, under a head that can SEE depth.
#
# `knee-ezz` read -0.0001 for 20 slices over 12 and closed this axis. It ran under
# `--slice-pooling max`, and CLAM (PMC8711640) says max "uses the gradient signal from only a
# single instance in each slide" -- the training signal is one slice deep no matter how deep the
# bag. So that number measured max pooling's indifference, not depth. Downgraded to UNREAD.
#
# The only depth evidence taken under a working head is foreign: `foreign_gate` on Raptor's
# per-finding attention model through our caches, 58 gold studies --
#   336px_12sl 0.8700 -> 16sl 0.8727 (+0.0027) -> 20sl 0.8773 (+0.0046)
# the second increment LARGER, still steepening where we stopped.
#
# PRE-REGISTERED FLAT-TO-SLIGHTLY-NEGATIVE. This is an INSTRUMENT, not a predicted gain. The two
# on-regime published curves disagree: Ilse et al. (arXiv:1802.04712) has attention still gaining
# to bag size 100; Li et al. (MICCAI 2025, doi:10.1007/978-3-032-04927-8_30) over CT slices reads
# 9sl 81.12 -> 18sl 82.54 -> 27sl 81.77, saturating by ~18 and degrading at 27. No published paper
# separates accuracy-vs-instance-count by class-specific vs shared/max aggregation (D323 NULL), so
# the 12/16/20/24 sweep under both heads is the first such curve.
#
# BATCH IS A SECOND VARIABLE AND THAT IS DELIBERATE. DeviceBudget: 1.335*(slices+1)*(batch/12)+0.22
# puts 24sl at batch 12 at 33.59 GB against a 31.8 GB card; batch 10 lands 28.03, the same reserve
# as `labelwise_dense` (20sl b12, 28.25) which runs fine. But RECIPE carries --rank-weight 1.0, and
# `head_trial.py:231` builds RankedObjective whose pairwise population IS the batch, so a batch trim
# moves a second knob. `labelwise_depth20_b10` is the matched control that absorbs it -- and, read
# against labelwise_dense's 0.8613 @holdout909, prices the batch/rank-population axis for free.
#
# Dump the checkpoint and assert `slices.pool.gate.3.weight` is non-zero before any number here is
# written down.
#
# reading: D:/models/rsna_knee_abnormality/labelwise_depth20_b10_s2026*.holdout.npz
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train labelwise_depth20_b10 \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --labels "$LABELS" --out "$MODELS/labelwise_depth20_b10_s2026.pt"
