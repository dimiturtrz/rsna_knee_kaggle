#!/usr/bin/env bash
# bead: knee-vrvus
# reading: mergedk_corpus_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `base224_control`'s exact line with two flags changed: `--scheme mergedk` and a labels table that
#   carries 1,088 more studies. Same recipe, same epochs, same slices, same geometry, same eval labels.
#
# WHY IT IS NOT THE CLOSED SITE-COUNT AXIS
#   `merged3_corpus` measured the site axis at +0.0043 for the SECOND site and -0.0024 for the THIRD,
#   and that closure is what refuses fastMRI as a fourth. This is a second site — 4,407 + 1,088 = 5,495
#   — which is the corner that PAID, not the one that closed.
#
#   And the supervision is a different mechanism from every earlier merge. Rijeka labels ACL alone and
#   MRNet three findings; KMAR-50K ships no findings at all. Its 1,088 rows are supervised by
#   `kneemri.inference.pseudo_labels` — four 224-native teachers' pooled belief over all twelve columns,
#   each cell weighted by how much those teachers agreed (`1 - 2*sd`, mean 0.6548). `experiments/README.md:7411`
#   retires in-domain PRETRAINING on external data and states in its own words that external data as
#   LABELLED augmentation is "a different mechanism and untested here". This is that mechanism.
#
# THE LICENCE IS THE POINT
#   Every other merge on this machine is unshippable — Rijeka CC BY-NC-ND, MRNet's Stanford agreement
#   bars an adaptation, fastMRI bars distributing a derivative. KMAR-50K is CC BY 4.0. A member fitted
#   here can be packaged, mounted and submitted, which makes this the only corpus lever whose answer is
#   readable on the board rather than only offline.
#
# THE HONEST LIMIT, STATED BEFORE THE RUN
#   The teachers include DARK members, which is legal here because a foreign corpus is in no holdout of
#   ours and there is nothing to leak. It does NOT license reading a gain on OUR holdout909 as a
#   generalisation result: the teachers saw those 909 studies, so a gain prices the distillation. What
#   the holdout CAN price is whether the extra rows help at all — and the board, where the teachers'
#   holdout is not the test, is where the generalisation question is actually asked.
#
#   KMAR subjects are NEW PATIENTS from a different hospital, so `foreign_readout`'s decode_cost no
#   longer separates corpus shift from decode shift on them. No cross-site number may be quoted off
#   this arm without saying so.
#
# PRE-REGISTRATION (written before the run, against base224_control)
#   >= +0.015          the labelled-augmentation mechanism is real and separates from site-count.
#                      Package the member, ship it, and scope a larger pseudo-labelled corpus.
#   +0.006 .. +0.015   directional only. Keep the member for the pool — diversity is priced by
#                      `record_mix`, never by this line — and re-open nothing.
#   within +-0.006     the extra rows are already fitted. This does NOT retire pseudo-labelling: mean
#                      teacher agreement is 0.6548, so a flat here is equally consistent with the
#                      teachers being too weak on a foreign site, which is a separate arm.
#   <= -0.015          the distilled rows are actively worse per-sample and the confidence weight is
#                      not discounting them hard enough.
#
# deliberate: --image-size 224 — a 336 px merge needs ~43 GB of new array and /d holds 23 GB (knee-rijuv).
# deliberate: --eval-labels report_labels_v2.csv — pinned, so the holdout is a v2 holdout by assertion
# deliberate: rather than by the accident of which table happened to be loaded.
#
# control  base224_control_s2026, the arm queued immediately before this one on the identical line.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGEDK_LABELS="$PROCESSED/labels/mergedk_labels.csv"
require "$MERGEDK_LABELS"
require "$LABELS"

run_train mergedk_corpus \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --image-size 224 --band-low 0.2 --band-high 0.8 \
  --scheme mergedk \
  --labels "$MERGEDK_LABELS" --eval-labels "$LABELS" --out "$MODELS/mergedk_corpus_s2026.pt"
