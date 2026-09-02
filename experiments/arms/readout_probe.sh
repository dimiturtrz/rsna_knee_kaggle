#!/usr/bin/env bash
# Is "DINOv3 is a worse trunk" a fact about the TRUNK or about the POOLING it was read through?
#
# WHAT IS ALREADY MEASURED, AND WHAT IS WRONG WITH IT. One frozen-probe run, matched flags, n=909:
# dinov2-large 0.7852, v3large 0.7446, ortholarge 0.7387. Those three numbers closed two members and
# carried the whole "the features are absent, not mis-fitted" reading in `comp/members.declines`. Every
# one of them was taken under `--pool mean` — the patch MEAN, the pooling that dilutes a small response
# hardest. `kneemri/models/token_pooling.py` says so in its own words about FOCAL: "a finding occupies a
# small part of the field, so a plain mean over the patches dilutes it".
#
# WHY THAT IS A LIVE SUSPECT AND NOT A QUIBBLE. DINOv3's named contribution is Gram anchoring, which
# exists to stop PATCH-LEVEL features degrading during long training; its classification headline belongs
# to the 7B model, and the ViT-L we run is a distillation of it. So the -0.0406 was measured on the one
# axis that release never claimed, through the one pooling that discards the structure it does claim.
# Our findings are 1-3 mm and `kneemri/models/model.py:124` already records that the head pools patch
# tokens, so the axis is not hypothetical for this task.
#
# WHAT MOVES AND WHAT DOES NOT. Same three trunks, same weights, same studies, same n=909, same read
# size, same slice read. The ONLY free variable is `--pool`: `focal` (class token + patch mean + the top
# 1/8 of each channel's patch responses) instead of `mean`. `focal` is a SUPERSET of what `mean` gave the
# probe, so a trunk cannot lose information by being read this way and any movement is the tail term.
#
# PRE-REGISTERED, before the pass runs. The quantity is the GAP dinov2-large minus v3large, which is
# -0.0406 under `mean`. Read the gap, not the absolute numbers: focal widens every trunk's feature vector
# threefold, so all three will move up and only the SPREAD is comparable across poolings.
#
# With two readouts there are two gaps. The branch below is taken on the SMALLEST of them, because the
# question is whether ANY fair readout closes the ranking, and a lane is closed only when every readout
# tried leaves it closed. Report both gaps whichever branch fires: which AXIS moved it is the finding,
# and the two have different follow-ups (token -> a member at focal; slice -> a learned query head).
#
#   gap closes to <= 0.010     THE RANKING WAS THE READOUT. Three closed trunk readings are facts about
#                              our pooling, and the trunk lane re-opens on an axis costing one probe per
#                              trunk rather than one fit per trunk. Re-probe every declined trunk next.
#   0.010 < gap <= 0.030       PARTIAL. The pooling carries some of it but not the ordering. Worth one
#                              member at focal on the best non-DINOv2 trunk, and nothing wider.
#   gap > 0.030                THE TRUNK IS THE TRUNK. dinov2-large stays ahead under a pooling built
#                              for small findings, which is the strongest case the alternative had. The
#                              trunk lane closes for good and no further card is spent on it.
#
# It also prices FOCAL for OUR OWN trunk at the same time, which is the second reading and free: if
# dinov2-large itself climbs materially under focal, the champion's `CLS_MEAN` members are leaving a
# small-finding term on the floor, and that is a member arm rather than a trunk one.
#
# `knee-fxgm` is the ramp gate this queues behind; this arm owns no bead yet and one is filed on its read.
set -euo pipefail
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
cd "$REPO"

# TWO readouts, not one, because there are two poolings between a patch and a score and the closed
# reading held BOTH at the diluting end. `--pool` is the TOKEN pooling (patches -> one vector per
# slice); `--slice-read` is the SLICE pooling (slices -> what the probe is handed). The three closed
# numbers were `--pool mean --slice-read max`: a patch mean, then a whole-stack read that consumes the
# slices INSIDE the encoder. Kimi's Q-AL (knee-02vn) prices the second axis from AnyMC3D: on a frozen
# 2D foundation trunk over volumes, max pooling reads 0.946 and a learned query pooling 0.962 (+0.016,
# and up to +0.11 in their other setup) -- WITHOUT changing the trunk. So the slice axis is the larger
# published term of the two, and holding it at `max` while sweeping only tokens would answer half the
# question and close the lane on the other half by omission.
#
# `attend` is per_slice=True, so the cache KEEPS a slice axis and `feature_probe` reduces it by mean
# over slots and slices (feature_probe.py:91). That is not AnyMC3D's LEARNED query -- it is the
# cheapest readout that stops the encoder max-pooling the depth away, and it costs one flag rather
# than a head. If the gap moves on this axis and not on tokens, the learned version is the next arm.
SIZE=256
for READ in max attend; do
  for VARIANT in dinov2-large v3large ortholarge; do
    echo "=== ${VARIANT} focal ${READ} ==="
    uv run --no-sync python -m kneemri.embeddings.build_embeddings \
      --variant "${VARIANT}" --train-size "${SIZE}" --pool focal --slice-read "${READ}" --batch 4
  done
done

uv run --no-sync python -m kneemri.evaluation.feature_probe --trunks dinov2-large v3large ortholarge

echo "ARM DONE readout_probe"
