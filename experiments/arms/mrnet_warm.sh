#!/usr/bin/env bash
set -euo pipefail
#
# STAGE TWO, and the deliverable. `base224`'s invocation with ONE flag added: `--init` at the trunk
# that stage one fitted on MRNet ACL. Everything else — variant, rate, slices, band, resolution, epochs,
# label table, scheme — is byte-identical to the control, so the delta is attributable to the foreign
# pretrain and to nothing else.
#
# control  base224 = 0.8509 (best epoch of 12, converged, `experiments/README.md` 2026-08-25)
#
# PRE-REGISTERED BRANCHES, against 0.8509 with a 0.0062 holdout noise floor:
#   * >= +0.015. Sequential cross-corpus pretraining is a real lever and the merged-corpus flat was a
#     fact about CONCATENATION, not about foreign data. The next arms are the same route at larger
#     foreign scale (merged3 = MRNet + Rijeka as stage one) and on a second variant to check it is not
#     an encoder accident.
#   * +0.006 to +0.015. Directional, not a finding. Keep the checkpoint if it seats; do NOT plan the
#     next week around it, and do not spend a confirmation seed on it.
#   * within +/- 0.006. The route is flat at MRNet's scale. That does NOT close foreign pretraining —
#     A-BV's bound says 1248 exams is far under the scale where deep features become reusable, so the
#     surviving question is scale, and the answer costs a stage one on the merged corpus, not a rethink.
#   * <= -0.015. Sequential pretraining SUBTRACTS, which is the `orthofoundation-L` result reproduced at
#     a thousandth of the data. That would be the strongest evidence yet that knee-supervised
#     continuation distorts a DINOv2 trunk, and it closes the family rather than the flag.
#
# WHAT THIS ANSWERS THAT NOTHING ON DISK DOES. Every external-corpus member the project owns trained on
# a merged cache. None started from a trunk fitted on foreign data alone. The 12-label matrix already
# hints the answer is not zero: on the three labels we are weakest at, the best member is an
# external-trained one every time (MCL 0.8253 `merged_blend_s2026`, Fracture 0.8440 `merged_blend_cf`,
# Baker's 0.9279 `merged_dense_cf`).
#
# HARNESS CHECK. The printed recipe must name the init path, and the epoch-1 holdout must NOT equal
# `base224`'s epoch-1 holdout. Equal first epochs means `--init` did not land and the arm is measuring
# the control twice — the plumbing failure mode, not a null.
#
# deliberate: --init mrnet_acl_pretrain_s2026.pt (the whole point of the arm)
# deliberate: --init-scope trunk. MRNet has three series and this corpus six, so head.slot_emb is
#             (3, 256) there against (6, 256) here and head.prior (12, 3) against (12, 6): a whole-model
#             load raises, which is what killed the first launch AFTER stage one had spent its card. The
#             head fitted over a foreign series layout is not a representation of anything here, and the
#             transfer mechanism this arm tests is about the trunk.
# deliberate: every other flag copied from base224.sh verbatim (the control has to stay a control)
#
# bead: knee-xlqgm
#
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
require "$MODELS/mrnet_acl_pretrain_s2026.pt"

run_train mrnet_warm \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --image-size 224 \
  --init "$MODELS/mrnet_acl_pretrain_s2026.pt" --init-scope trunk \
  --labels "$LABELS" --out "$MODELS/mrnet_warm_s2026.pt"
