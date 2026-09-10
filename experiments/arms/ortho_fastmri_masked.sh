#!/usr/bin/env bash
# `ortho_fastmri` with the ABSENCE PRIOR REMOVED. The second half of the fastMRI reading, and the arm
# that separates "the corpus does not transfer" from "the corpus transfers and its absences eat the
# gain" — two results an overall flat cannot tell apart, which is why `fastmri_corpus` calls the
# per-finding breakdown mandatory in writing.
#
# THE HAZARD, restated from `fastmri_corpus` rather than invented here. fastMRI+ grades three of our
# twelve findings one-to-one — ACL, MCL, Effusion. For the other nine it supplies NEGATIVES ONLY: its
# meniscus and cartilage findings are unsided while ours are per-compartment, so an exam with no tear
# recorded becomes a confident zero on Medial Meniscus, Lateral Meniscus, three OA columns, Synovitis,
# Baker's, Contusion and Fracture. 588 studies of that shape are 12% of the training table, all pulling
# one direction on nine columns, on findings the corpus cannot locate.
#
# WHAT THIS ARM CHANGES, and it is one path. `MergedCache.narrow` blanked those nine families to NaN on
# the fastMRI rows only — 6,692 cells over 588 rows — and `Targets.load` reads `said = ~np.isnan(value)`,
# so a blanked cell arrives at weight 0.0 with no flag, no loss term and no trainer change. The images
# are identical, the cache is identical, the canonical 4,406 rows are byte-identical. fastMRI now
# supervises only what its readers looked at, and contributes its pixels to every shared parameter.
#
# CONTROL: `ortho_fastmri` (the same recipe on `mergedf_labels.csv`), and behind it `ortho_dense` 0.8452
# on our corpus alone. Three points on one axis: no fastMRI, fastMRI whole, fastMRI narrowed.
#
# PRICE. ~490 s/epoch x 12 = ~1.6 h, the same shape as its control. The gain is not a knob: if the
# corpus transfers at all, this is the arm that lets it, and the difference between the two fastMRI arms
# is a DIRECT measurement of what nine columns of manufactured absence cost — a number this project has
# never had for any foreign corpus, and one that prices Rijeka and MRNet's merges too. P(clears
# `ortho_fastmri`) ~ 0.6; P(clears `ortho_dense` + 0.015) ~ 0.2.
#
# PRE-REGISTERED, against `ortho_fastmri` (call it F) and `ortho_dense` 0.8452:
#
#   s >= F + 0.006 AND s >= 0.8452   The absence prior was real and costly, and narrowing is the way to
#               merge ANY partially-graded corpus. Then narrow Rijeka and MRNet the same way and re-run
#               `merged_corpus` (-0.0005) and `merged3_corpus`, both of which were measured WITHOUT it.
#   s ~ F (within 0.006)   The absences were not what limited it. Then one filled slot of six is the
#               binding constraint, and the next move is the 336/20 decode of the fat-suppressed half at
#               `band_dense`'s geometry rather than more label surgery.
#   s <= F - 0.006   Those nine columns of negatives were HELPING — the corpus's absence statements are
#               real information about a real population and our own table is the one with the prior
#               problem. That is a finding about OUR labels, not theirs, and it is the most interesting
#               of the three.
#
# deliberate: --labels mergedf_acl3_labels.csv (the whole arm; every other flag matches ortho_fastmri)
# reading: ortho_fastmri_masked_s2026.holdout.npz
#
# bead: knee-i3ahx
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MASKED_LABELS="$PROCESSED/labels/mergedf_acl3_labels.csv"
require "$MASKED_LABELS"

run_train ortho_fastmri_masked \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant ortholarge --backbone-lr 4e-5 --slices 12 --image-size 224 \
  --band-low 0.2 --band-high 0.8 \
  --scheme mergedf \
  --labels "$MASKED_LABELS" --out "$MODELS/ortho_fastmri_masked_s2026.pt"
