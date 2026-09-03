#!/usr/bin/env bash
# sided_pool.sh — `band_dense`'s recipe with `--pool sided`, giving the head a medial/lateral axis it has
# never had. One flag off the best single member on disk, so the read is clean.
#
# bead: knee-2xll
#
# THE MECHANISM, and it is a TYPE SIGNATURE rather than a suspicion. Every token pooling this project has
# ever fitted collapses the patch axis: `TokenPooling.reduce` is typed `"n t d" -> "n f"` in all three,
# and the focal one discards even the indices `topk` hands back. The head therefore cannot express WHERE
# in the field a response sat. Five of the twelve targets are defined by exactly that — both menisci,
# both tibiofemoral compartments, and the MCL.
#
# WHY A COLUMN SPLIT IS AN ANATOMICAL SPLIT: `Laterality` mirrors every study onto one canonical side
# inside the cache pipeline (`core/data/corpus/study.py`), covering 100% of the corpus because the
# geometric rule reaches the half the DICOM tag does not. In that frame medial is always the same side of
# the array. So this arm needs no new input, no cache rebuild and no label — the information is already
# in the pixels and is being pooled away in the last few lines before the head.
#
# WHAT MADE IT THE CANDIDATE. `side_dense` asked the LOSS for the same contrast and regressed -0.0641 with
# every one of the twelve labels falling. Its return was not "the side does not matter": its compartment
# contrast ROSE 0.062 on demand while the metric fell, which says the encoder HOLDS the side and the
# pooled head is where it is lost. That is a statement about the head, and this is the arm that tests it.
# Separately, knee-yrg has now refuted all four data-side explanations of the lateral gap — the laterality
# flip, the slice band, prevalence, and the in-plane crop (which clips MEDIAL harder, pointing the wrong
# way). The head is what is left.
#
# WHY IT SUPERSEDES THE DEFAULT RATHER THAN COMPETING WITH IT. The halves partition the grid, so their
# mean IS the patch mean `cls_mean` already emits; the head can form that sum and can additionally form
# the difference. One extra part, nothing removed. A flat result is therefore informative in a way the
# other pooling arms' were not: it says the head could have ignored the split and did.
#
# PRICE: 72 min of card, one arm, no cache build. P(clears 1.5% on the mean) ~ 0.10.
# The 1.5% reading is the honest one and it is why this is priced as a LONG SHOT on the headline number.
# The board reading is different and is the reason it runs first: MCL (0.8039, n=171) and Lateral OA
# (0.8267, n=233) sit 2.6 and 2.0 standard errors below the pool median on the 909-study holdout, and
# lifting just those two to 0.87 is +0.0091 on the mean — which the measured board transfer of +4.67
# multiplies past the 0.006 gap to the 0.940 goal several times over. Fracture is NOT a target here
# despite reading 0.8216: n=68 gives it an SE near 0.04, so it is not separable from 0.90 and any gain
# there would be noise. Stating the denominator is what removed it from the list.
#
# PRE-REGISTERED, on the 909-study fold-0 holdout against `band_dense` = 0.8578:
#   >= 0.8628   the split carries; the head was the bottleneck; go to a finer grid than two halves.
#   0.8548-0.8627  inside the 0.0062 floor. NOT a refutation of the mechanism — read the PER-LABEL
#                  columns for MCL / Lateral OA / Lateral Meniscus before writing any verdict, since the
#                  mean can absorb a real move on three labels of twelve.
#   <= 0.8547   the extra part costs more than the contrast buys. Suspect the head width first (three
#               parts against two changes the first linear layer's fan-in, not just its meaning), then
#               the mirror — assert `Laterality` fired on this cache before blaming the idea.
#
# WHAT A NULL WOULD NOT RULE OUT: a finer spatial split (quadrants, or the compartment columns rather
# than halves); the same pooling on a head that also sees depth regions; and the small-weight side
# REGULARISER `side_dense`'s row left standing, which was explicitly deferred to "a head that sees
# position" — this arm is what would give it one.
# control  band_dense = 0.8578 (declared 2026-09-03 from this header own prose, knee-sgzgc)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train sided_pool \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --pool sided \
  --labels "$LABELS" --out "$MODELS/sided_pool_s2026.pt"
