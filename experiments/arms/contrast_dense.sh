#!/usr/bin/env bash
# contrast_dense.sh — the first arm that asks the REPRESENTATION a question instead of the decision.
#
# THE OWNER'S IDEA, and the reason it is not another pooling knob. "Torn meniscus is probably physically
# separated as a link." Every locality lever this repo owns is an INTENSITY STATISTIC — where the bright
# voxels are, how they pool across slices, which band of the stack they sit in — and all of them are flat
# to negative: cls_mean_focal +0.0007/+0.0003/-0.0002, depth_regions 0.8549, narrow 0.8554, span 0.8505,
# all under band_dense's 0.8578. Those answer "how bright, and where". None of them can say "is it
# CONNECTED", because no pooling operator at any percentile expresses continuity.
#
# THE CHEAP FORM OF THAT QUESTION, which is what this arm is. A supervised-contrastive term, per finding,
# over the vector each diagnosis reads from the study before the readout collapses it to a logit. Every
# other term in this project is a function of the logits — a per-finding dot product against ONE learned
# direction — so two studies read for completely different reasons can share it, and nothing in a logit
# says whether the head separated them for the right reason. This asks the vector itself to put torn
# beside torn and intact beside intact. It is the weakest phrasing of "the pathological ones look alike"
# that costs no masks, no segmentation and no second dataset.
#
# control  calib_dense_hold_s2026, holdout 0.8634 — the best solo member this project owns. SINGLE
#          VARIABLE against it: --contrast-weight and nothing else. Same recipe, same twelve epochs, same
#          held split, so a difference is the term.
# cost     one 12-epoch fit at ~200 s an epoch, ~40 min. Zero submission slots.
#
# WHY WEIGHT 1.0 AND NOT A SWEEP. Parity is the neutral start, and it is honestly noted rather than
# tuned: at initialisation the contrastive term is roughly 3x the cross-entropy one, so this arm runs the
# term LOUD. That is deliberate for a first read — a term at 0.1 that changes nothing is unreadable
# between "the mechanism is absent" and "the term was too quiet to matter". Khosla et al.'s temperature
# 0.1 is taken, not tuned, for the same reason.
#
# THE WEAKNESS, STATED BEFORE THE RUN. Contrastive learning conventionally uses batches two orders of
# magnitude larger than our twelve studies, and a batch of twelve sees a 7%-prevalence finding roughly
# once. So the term's supervision is biased toward the common findings, which are exactly the ones the
# metric already scores well. `RememberedRanking` is the named fix — a remembered pool of past readouts —
# and reaching it means a mixin rather than a sixth branch in the objective dispatch.
#
# PREDICTION. This is an INSTRUMENT, not a board bet. calibsix's measured 23% board pass-through ceilings
# any member-quality gain near 0.943, and the house bar is 1.5% absolute. What this arm buys is the answer
# to whether shaping the representation moves the number AT ALL — the whole class of change (segmentation,
# healthy-vs-pathological contrast, shape/continuity features) is untried here, and this is its cheapest
# probe. A score at or near 0.8634 says the representation was already separating as well as these labels
# can ask, and sends the topology question to the strong form (SKM-TEA masks, blocked on knee-2sbk)
# rather than to a weight sweep.
#
# WHY THE RANKING TERM STAYS. $RECIPE carries --rank-weight 1.0, and so did the control. The objective
# extends `RankedObjective` and carries the weight through rather than replacing it, so this arm adds one
# term and removes nothing — the first wiring excluded it, which would have made the run a two-variable
# change (contrast added, ranking removed) against the only control worth comparing to.
#
# VERIFY THE FLAG LANDED before believing either outcome: `contrast_weight` is in `landed.py`'s registry
# and `run_setup` raises on every flag it cannot compose with, so a silent no-op is not among the failure
# modes — but read the recipe line in the log rather than assuming it.
#
# bead: knee-ek5s
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB.
# control  calib_dense_hold_s2026 = 0.8634 (declared 2026-09-03 from this header own prose, verified on disk, knee-sgzgc)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train contrast_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --contrast-weight 1.0 \
  --labels "$LABELS" --out "$MODELS/contrast_dense_s2026.pt"
