#!/usr/bin/env bash
# warm_v3base_lora.sh — the CAPACITY half of the warm-start claim, which has never been tested.
#
# WHAT THE FIRST ARM ACTUALLY PRICED, AND WHY IT WAS NOT THIS. `warm_calib_dense` read 0.8562 against its
# control's 0.8634 and was recorded as a −0.0072. That number is real and it is about the wrong claim:
# neither the arm nor its parent passes `--lora-rank`, so the parent's backbone holds no `.down`/`.up`
# tensors at all, and `FittedEncoder.merge_adapters` (fitted_encoder.py:88-97) skips DOWN/UP and rewrites
# only BASE_WEIGHT/BASE_BIAS. Over a rank-0 state it folds NOTHING IN. So that arm asked "does a converged
# trunk start better than a stock one" — an init-TIME question — and answered no. The doubled-adaptation
# argument `knee-wbwq` rests on was never on the card.
#
# THE PARENT IS CHOSEN SO THE MERGE IS NON-EMPTY, and it was checked rather than assumed before this was
# written: `dinov3_base_lora_s2026` carries 72 adapted layers at rank 8, and merging them moves an example
# k_proj weight by 0.0056 at the maximum. The encoder file is 211 tensors, variant v3base, written by
# `python -m kneemri.training.fitted_encoder` to $MODELS/encoders/v3base_lora.enc.pt.
#
# THE MECHANISM UNDER TEST, stated so a null is readable. The parent learned a rank-8 correction on a stock
# trunk. Merging folds that correction into the base weights, and `--lora-rank 8` here builds a SECOND,
# fresh rank-8 adaptation on top of the already-adapted trunk. Effective adaptation capacity doubles with no
# widened layer, no extra parameter at serve time, and no shape change — the member costs exactly what its
# parent costs to serve. If adaptation capacity is what binds these members, this is the cheapest way to buy
# more of it that this project owns.
#
# control  dinov3_base_lora_s2026, holdout 0.8268. SINGLE VARIABLE against it: `--encoder-weights` and
#          nothing else changes. Same variant, same 336/224 sizes, same rank, same 5e-4 backbone rate, same
#          twelve epochs, same held split — the recipe below is dinov3_base_lora.sh's line for line.
# cost     one 12-epoch fit at 224 train size, ~40 min. Zero submission slots.
#
# WHAT IT CAN AND CANNOT BUY. The parent is a WEAK member: 0.8268 against the 0.8634 of the best solo this
# project owns. Even a large relative gain here does not produce a best single, and this arm is not proposed
# as one. It is an INSTRUMENT, and priced as one — it retires or licenses the capacity claim that gates the
# entire warm-start lane, and that lane is what would let the modal recipe skeleton (31 of 98 members, one
# decode path re-run with a different loss or label table) stop paying a full cold fit each time. A positive
# here sends the next arm to a STRONG parent; a null closes `knee-wbwq`'s second claim for good, which is
# worth having because the bead is currently first in the queue when the card returns.
#
# PRE-REGISTERED, against the control's 0.8268 and the 0.0062 arm-vs-arm floor.
#  - >= 0.8330 — the doubled adaptation is real and usable. Warm-start a strong parent next, and the lane
#    reopens as a member-quality lever rather than a time saving.
#  - 0.8207-0.8329 — inside the floor. Capacity buys nothing measurable; the warm start is a time saving
#    only, and `knee-wbwq`'s quality claim is closed with the same evidence that closed its time claim.
#  - <= 0.8206 — a loss, and the schedule is the named suspect BEFORE the mechanism, exactly as it was for
#    arm 1: one_cycle ramps the backbone rate above 5e-4 before annealing and can walk an already-adapted
#    trunk off its optimum. That result sends the next arm to a lower --backbone-lr, not to the graveyard.
#
# VERIFY THE FLAG LANDED BEFORE BELIEVING ANY OF THE THREE. `encoder_weights` is in `landed.py`'s registry
# and `PretrainedEncoder.into` refuses a variant or key mismatch, so a silent no-op is not a failure mode —
# but the log says so explicitly and the log is what to read. Arm 1's lesson is the sharper one: a flag can
# land perfectly and still carry no mechanism, so confirm the MERGE was non-empty too.
#
# THE UNSTATED GEOMETRY IS DELIBERATE, and `arm_check` is right to say so out loud. This arm names no
# --slices, --band-low or --band-high, so it inherits the pre-settlement 12 slices / 0.20-0.80 band
# (knee-aixw). That is not an oversight carried forward: `dinov3_base_lora.sh` inherits exactly the same
# defaults, and the control this arm is read against IS that run. Naming the settled geometry here would
# buy a better feed and cost the only thing the arm is for — a delta attributable to one flag. The feed is
# worth about +0.005 by the foreign-gate curve, which is comfortably enough to swamp the effect under test.
#
# PRICE. ~40 min of card for one 12-epoch fit at 224. As a MEMBER the expected gain is near zero and is not
# claimed: the parent is 0.8268 against a 0.8634 best, so no outcome here produces a competitive member.
# As an INSTRUMENT the expected value is the whole reason to run it — P(clears the 0.0062 floor) is maybe
# 0.3, and either branch retires a claim that currently sits first in the card queue. A null closes
# `knee-wbwq`'s capacity half; a positive redirects the lane at a strong parent, where the same mechanism
# would be worth a real member. Forty minutes to resolve the head of the queue is the best ratio on the
# no-card list, and it is explicitly NOT sold as an improvement.
#
# bead: knee-wbwq
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
require "$MODELS/encoders/v3base_lora.enc.pt"

HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

run_train "warm_v3base_lora" \
  $RECIPE --epochs 12 $PATIENCE \
  --slice-pooling max --whole-stack --lora-rank 8 --backbone-lr 5e-4 \
  --variant v3base --image-size 336 --train-size 224 \
  --encoder-weights "$MODELS/encoders/v3base_lora.enc.pt" \
  $THROUGHPUT \
  --labels "$LABELS" --out "$MODELS/warm_v3base_lora_s2026.pt"
