#!/usr/bin/env bash
# bead: knee-jucq
# vjepa_probe.sh — is a VOLUME-PRETRAINED trunk worth a seat, measured on frozen features before a
# member is fitted.
#
# WHY THIS AXIS AND NOT ANOTHER RECIPE. The 2026-08-21 pair settled that member COUNT is not the seat
# cap and REDUNDANCY is: `sixfresh`'s recipe-distant sixth took 0.931 while `sixstrong`'s near-twin of
# an incumbent — the better checkpoint by 0.0040 solo — cost a tick. Distance inside this recipe family
# is nearly exhausted; every member reads the same pixels through a DINOv2 or DINOv3 trunk and differs
# only in band, rate, resolution, schedule or calibration. A trunk whose pretraining never saw this
# family at all is the largest distance still purchasable, and V-JEPA 2 differs in pretraining domain,
# architecture family and depth handling at once.
#
# IT IS ALSO THE RESIDUE OF THE THROUGH-PLANE CHAPTER. `through_dense` inflated the patch-embedding stem
# to read slices (i-1, i, i+1) and scored -0.0020: at a window of 3, Conv2d IS Conv3d, and the stem emits
# ONE token per patch whose three filters were pretrained to mean colour, so it cannot carry "the slice
# at i" and "how it changes with depth" as separate quantities. That refutation leaves standing exactly
# two things — a window WIDER than 3, and a stem emitting SEVERAL depth positions. A tubelet trunk is
# both, and is the only asset in reach that is.
#
# THE CONFOUND THIS ARM EXISTS TO REMOVE. The obvious probe — V-JEPA2-large against the DINOv2-SMALL
# cache already on disk — compares a 1024-wide 24-block trunk to a 384-wide 12-block one. A win there
# reads "large beats small", which `basedense`'s 0.932 clause already refuted as a lever, and a loss
# reads as nothing at all. So the control is encoded here rather than reused: DINOv2-LARGE, 1024 wide,
# 304M parameters, the same width and depth class as vjepa2-vitl. Two encode passes instead of one, and
# the difference that survives is PRETRAINING DOMAIN rather than capacity.
#
# 252 AGAINST 256, and why that is the right asymmetry to accept. DINOv2 is patch 14 and cannot tile
# 256; V-JEPA 2 is patch 16 and was pretrained at a 256 crop. Forcing either onto the other's grid
# means interpolated position embeddings — a silent wrong answer — so each reads the nearest size its
# own grid tiles. 252 against 256 is 1.6% of linear extent against a 2.7x capacity confound that this
# arm removes; taking the smaller error to avoid the larger one.
#
# WHAT IS DELIBERATELY *NOT* CONTROLLED: the READING. The control reads per-slice and pools after; the
# tubelet trunk consumes the whole 12-slice stack inside the encoder. That bundle is not separable —
# volume pretraining without volume reading spends the trunk's only advantage — and it is the
# decision-relevant bundle anyway, because a member ships as a whole. Named so a reading is not later
# claimed as clean on an axis it never held.
#
# COST, priced before it runs. Two encode passes over 4,407 studies x 6 slots: the control is 317k
# images at ~250 img/s for a ViT-L (~21 min), the tubelet pass is 26k twelve-frame clips at 1,536 tokens
# each (~5 min). Both head fits are minutes on cached features. Call it ~35 min of card for the largest
# distance axis left in the pool — against ~40 min for one fine-tuning arm that answers about a knob.
#
# READ IT ON, against the control fitted in the same command on the same studies and the same split:
#   * ABOVE the control by more than the holdout's 0.0062 floor — volume pretraining carries something
#     2D pretraining does not, and a fitted member is worth the card. That is the point at which
#     knee-2nxh (the VRAM cap's tubelet blind spot) stops being deferred and starts blocking.
#   * WITHIN the floor — the trunk is not BETTER, which does not decide the seat question. A seat is
#     bought by DISAGREEMENT and the holdout is structurally blind to it (see SUBMISSIONS.md, the
#     sixfresh/sixstrong split), so a tie here still licenses fitting one member and pricing it by
#     `seat_value` against the champion five. Do not read a tie as a kill.
#   * BELOW by more than the floor — the frozen features are worse, and the honest next question is
#     whether the FROZEN reading is what fails rather than the trunk: V-JEPA's features are trained for
#     prediction in latent space, not for linear separability, so a weak frozen probe is a known
#     property of that objective and not yet evidence about a fine-tuned member. Record it as blocking
#     the CHEAP path only.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# The card is taken around the ENCODE passes for the same reason run_train takes it around training:
# these spend GPU and do not go through run_train, so without this they are ungated by construction —
# the hole that put a smoke onto an occupied card on 2026-08-19.
hold_card

uv run --no-sync python -m kneemri.embeddings.build_embeddings \
  --variant large --train-size 252 --pool mean --slice-read max --batch 4 --paths paths.yaml

uv run --no-sync python -m kneemri.embeddings.build_embeddings \
  --variant vjepa2l --train-size 256 --pool mean --slice-read channels --batch 4 --paths paths.yaml

release_card

# Both trials read the SAME pool and the SAME split, so the only difference that reaches the number is
# which cache they open.
uv run --no-sync python -m kneemri.training.head_trial \
  --variant large --read-size 252 --pool mean --slice-read max --slice-pooling max \
  --labels "$LABELS" --paths paths.yaml

uv run --no-sync python -m kneemri.training.head_trial \
  --variant vjepa2l --read-size 256 --pool mean --slice-read channels --slice-pooling max \
  --labels "$LABELS" --paths paths.yaml

echo "ARM DONE vjepa_probe"
