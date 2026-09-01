#!/usr/bin/env bash
# Does knee-specific SSL continuation move the FEATURES? OrthoFoundation-L against the weights it
# continues from, under one frozen linear probe. `knee-440m`.
#
# WHY A PROBE AND NOT A MEMBER. The external-data lane is the only one this project still has open —
# member quality ceilings near 0.943 against a leader at 0.952, the stack lane closed 7/7, seating is
# flat over seven board entries and the trunk lane closed EMPTY on RadImageNet at 0.6716. So the
# question is worth hours, not minutes; but the CHEAPEST published instrument for it is a frozen linear
# probe (Ilse et al. 2025 measure their own scaling law with one), and a probe costs two inference
# passes rather than two training runs.
#
# WHAT MAKES THIS A CONTROLLED READING. `ortholarge` IS `v3large` plus 1.25M knee images of continued
# self-distillation — same 24 blocks, 1024 width, 16 heads, patch 16, 4 register tokens. Every flag
# below is identical between the two passes, so the only free variable is the continuation corpus.
#
# PRE-REGISTERED, before either pass runs, and REVISED once before it ran — see the note under the
# branches, which is the more important half of this header.
#
#   ortho - v3large >= +0.01   THE CONTINUATION MOVES THE FEATURES, and it does so in the weakest place
#                              it could. Mount it; the fine-tuned member arm becomes the priority.
#   +0.01 > delta >= +0.003    DIRECTIONAL. Consistent with the continuation paying under fine-tuning
#                              and only partly surfacing in a linear read. Fit the member.
#   +0.003 > delta > -0.003    FLAT. Says the gain is not linearly separable in pooled frozen features.
#                              It does NOT say the trunk is worthless — fit the member anyway.
#   delta <= -0.003            REVERSED, the shape Ilse et al. report for RAD-DINO on the out-of-domain
#                              VinDR benchmark and attribute to catastrophic forgetting. The only branch
#                              that argues against spending the member arm, and even it argues weakly.
#
# WHY THE THRESHOLDS MOVED, AND WHY THE ARM RUNS ANYWAY. The first version of this header set the top
# branch at +0.02 and had the flat branch CLOSE the external-encoder lane. Kimi's Q-W review found both
# wrong before either pass ran. OrthoFoundation's published numbers — ACL 0.9849, MCL 0.9334, cartilage
# 0.9338 — are FULL FINE-TUNING, and its only DINOv3-L comparisons are prose (`4-7 percentage points` on
# TKR, `3-4` cross-anatomy) on other tasks with no matched row. So there is no published frozen-probe
# delta to calibrate against, and +0.02 was an aspiration wearing a threshold's clothes.
#
# The consequence that matters is the flat branch. A frozen probe is a FLOOR: it pools over slots and
# slices, so a trunk whose gain lives in a positional or non-linear arrangement — or one that needs its
# weights to move at all — reads flat here for a reason about the instrument. A positive probe is strong
# evidence; a null is weak evidence, and cannot close a lane that the published evidence only ever
# measured under fine-tuning. This arm is therefore a CHEAP POSITIVE TEST, not a gate: the fine-tuned
# member arm is queued behind it whatever it reads, because the card is otherwise idle and because this
# is the only lane still open.
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/lib.sh

SIZE=256
for VARIANT in v3large ortholarge; do
  echo "=== ${VARIANT} ==="
  python -m kneemri.embeddings.build_embeddings \
    --variant "${VARIANT}" --train-size "${SIZE}" --pool mean --slice-read max --batch 4
done

python -m kneemri.evaluation.feature_probe --trunks dinov2-large v3large ortholarge

echo "ARM DONE ortho_probe"
