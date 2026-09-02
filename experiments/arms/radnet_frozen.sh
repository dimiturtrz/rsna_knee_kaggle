#!/usr/bin/env bash
# radnet_frozen — RadImageNet DenseNet-121 with the trunk held STILL.
#
# WHY THIS ARM EXISTS, AND WHY IT IS NOT A REPEAT OF `radnet_family`. That arm fitted the same trunk at
# `--backbone-lr 4e-4` and read 0.6716 against a pre-registered 0.75, which was written down as the cell
# being empty: a 7M supervised medical trunk does not transfer here. On 2026-09-02 the frozen features of
# that same checkpoint read **0.8197** under a ridge probe — above `v3large`'s 0.8046 and above
# `dinov2-large` — once each trunk is read at the penalty it wants rather than a shared 100.0 (knee-P0,
# the shared-penalty defect). Both numbers are about the same weights. A trunk whose FROZEN features
# outrank the incumbent and whose FINE-TUNED score collapses by 0.148 is the textbook shape of feature
# distortion, not of a trunk with nothing in it, and the published fix is to stop moving the features.
#
# So this arm changes exactly one thing about the recipe that failed: the trunk does not move.
#
# HELD AT THE PROBE'S OWN READING, not at `radnet_family`'s. The 0.8197 was measured over the
# `features_radnet121_336px_12sl_r256_cls_mean_focal_max` cache — 336px, 12 slices, read size 256, focal
# max over the slice axis. `radnet_family` ran 20 slices at 352 and a plain max. Aligning to the probe is
# what makes the probe a FLOOR for this arm; aligning to the failed arm would confound the freeze with
# three other changes and leave the result unreadable either way.
#
# PRE-REGISTRATION, against the probe floor 0.8197 and the seat bars 0.8459 (MERIT) / 0.8521 (non-DINOv2).
#   >= 0.8521 — a foreign-family member that clears the size bar on its own. That has never happened here
#     and it would re-open the member axis that seven board flats closed, because every one of those flats
#     reseated DINO-family members among themselves.
#   0.8197-0.8521 — the EXPECTED branch, and it is a keep, not a failure. A frozen radnet121 probe seated
#     beside the five champion ancestors under RankAverage already reads **+0.0041** over the five alone
#     (paired bootstrap, 909 holdout studies, p5=+0.0029, P(>0)=1.000), and with `ortholarge` beside it
#     **+0.0054**. A head above the probe should not read below the probe; anything in this band buys the
#     diversity seat the probe measured, at a member the roster can actually ship.
#   < 0.8197 — the head is COSTING against its own frozen features. That is a statement about this head on
#     these features, not about the trunk: read it against the probe's per-label rows before concluding,
#     since the probe beats `v3large` on Lateral Meniscus by 0.040 and that is where the seat would pay.
#
# `--unfreeze-last 0` is the whole intervention. It is not a LoRA arm and it is not a warm start; the
# trunk is exactly the published RadImageNet checkpoint at every step, which is what makes the probe
# number a floor rather than an analogy.
#
# bead: knee-jyb6 (the family), plus the probe-seat finding filed 2026-09-02.
#
# deliberate: --unfreeze-last --variant
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train radnet_frozen \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack \
  --variant radnet121 --unfreeze-last 0 --slices 12 --band-low 0.1 --band-high 0.9 --train-size 336 \
  --vram-gb 26 \
  --labels "$LABELS" --out "$MODELS/radnet_frozen_s2026.pt"
