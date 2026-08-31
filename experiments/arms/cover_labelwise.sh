#!/usr/bin/env bash
# cover_labelwise.sh — the same whole stack, read by an aggregator that a thin finding survives.
#
# bead: knee-tlya
#
# WHAT stack224 ACTUALLY MEASURED. It read all 36 slices of a 30-40 slice volume, the cell no member has
# ever occupied, and finished at 0.8476 against `native224`'s 0.8502. Its own header pre-registered the
# reading: a flat mean with the thin findings UP is the finding. They went down hardest, and everything
# thick went up.
#
#   Lateral Meniscus 0.8028 vs 0.8291  -0.0263      Synovitis 0.8740 vs 0.8638  +0.0101
#   Fracture         0.7929 vs 0.8109  -0.0180      ACL       0.8809 vs 0.8768  +0.0041
#
# A twelve-label mean of -0.0026 hides a two-label collapse and a nine-label rise. That is not scatter and
# it is not the coverage idea failing on its own terms — it is the shape of an AGGREGATOR failure.
#
# THE MECHANISM. `stack224` pools slices with `--slice-pooling max`. The expected maximum of a set of
# noisy per-slice scores grows with the SIZE of the set even when nothing in the volume changed, so a
# maximum over 36 candidates has a strictly higher noise floor than one over 20. A finding present on two
# slices is exactly what a spurious maximum drowns; one present on twenty is not. Coverage and max pooling
# are therefore in direct conflict, and the conflict gets worse the more of the volume is covered. The
# per-label pattern above is what that predicts, label for label — which is why this arm exists rather
# than a closure.
#
# ONE FLAG MOVED against `stack224`: `--slice-pooling labelwise`. `LabelwiseAttention` gives each of the
# twelve findings its own attention map over the slices instead of one shared extremum, which is the
# mechanism-matched read for a finding that lives on two slices out of thirty-six. It is not a new recipe:
# `labelwise_dense_s2026` scores 0.8613 with it at 20 slices. Everything else is `stack224`'s invocation
# untouched, so the result is matched against BOTH `stack224` (aggregator moved, coverage held) and
# `native224` (the standing 224 control, 0.8502).
#
# `$THROUGHPUT` IS KEPT, AND THE FIRST LAUNCH IS WHY. `--recompute-activations` costs a measured 19% (798
# img/s against 946, `kneemri/models/encoder_surgery.py:285`) and was calibrated on a 336-px geometry about
# 4.5x larger than this one, so knee-ujd3 argued it was a tax on a memory problem 224 px no longer has.
# It is not. Launched without it at 08:02 on 2026-08-31 this arm died with
# `torch.OutOfMemoryError: Tried to allocate 488.00 MiB ... 25.50 GiB allowed; 25.09 GiB is allocated`,
# against the 21.9 GB `stack224` peaked at WITH the flag. So the threshold sits between 20 and 36 slices
# at 224 px: the flag is dead weight at 20 and load-bearing at 36, and the 19% is bought, not wasted.
# That is the whole of knee-ujd3's throughput half, answered by one launch and costing three minutes.
#
# READ IT PER LABEL, and against BOTH controls. The mean is the last thing to look at: the whole reason
# this arm exists is that a mean averaged a two-label effect away and nearly closed a live axis.
#
# HARNESS CHECK BEFORE ANY READING. The printed recipe must show `"image_size": "224"`, `"slices": "36"`,
# band [0.05, 0.95], `"whole_stack": "True"`, `"fold": "0"`, and `"slice_pooling": "labelwise"` — that
# last one is the entire arm, and a run that silently kept `max` would repeat `stack224` at full price.
# `"recompute_activations"` must read "True".
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train cover_labelwise \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 36 --band-low 0.05 --band-high 0.95 \
  --image-size 224 \
  --labels "$LABELS" --out "$PROCESSED/cover_labelwise_s2026.pt"
