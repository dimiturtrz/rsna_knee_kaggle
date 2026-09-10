#!/usr/bin/env bash
# noisy_student.sh — the third site again, with its SUPERVISION replaced rather than its pixels.
#
# bead: knee-cy3wj
#
# WHAT MOVES, against merged3_corpus. Two flags, and they are one variable: `--distill-weight 0.7
# --teacher-targets teacher_merged3.npz`. Same 6,393-study cache, same twelve slices, same 0.20-0.80
# band, same whole-stack, same 4e-5, same twelve epochs, same 336px read, same 909-study fold-0 holdout,
# same merged3_labels.csv. The only thing that changes is that every study — ours and the 1,987 foreign
# ones alike — now carries a fold-0-clean ensemble's ORDERING beside whatever labels it has.
#
# WHY THIS IS THE ARM AND merged3_corpus WAS NOT. merged3_corpus read 0.8521, BELOW the two-site
# merged_corpus 0.8545, with Lateral Meniscus worst of twelve at -0.0217. Its own consequence line reads
# "a ROTATION, NOT AN ADDITION": the foreign pixels displaced our supervision rather than adding to it.
# The corner that kept paying at a fixed cache was SUPERVISION (merged_blend_cf 0.8597, merged_span_cf
# 0.8618). This arm puts the two together — foreign pixels WITH supervision that speaks all twelve
# columns — which is the only untested cell of that 2x2.
#
# WHY NO LABELS REWRITE. An earlier draft was going to write the teacher's values into the labels array
# for the foreign rows, so those rows would be taught by the teacher alone. It is unnecessary:
# `Targets.pair` already weights a silent column at 0.0, so MRNet's eleven unsupervised columns carry no
# label gradient at all, and its ACL column carries a REAL label rather than a mismatched one. The 0.3
# label term on a foreign row is therefore either a true label or nothing. Leaving it alone keeps this
# arm at ONE variable against merged3_corpus and adds no new code path — this project's most expensive
# recurring failure is a flag that never landed, not a mixture that was slightly wrong.
#
# WHO MAY TEACH. `oof_clean_fold[1-4]_s2026`, the twins rebuilt so that NO teacher ever saw fold 0
# (teacher_clean.sh). Six of the seven champion members were fitted with `--fold -1` and have seen every
# study including the holdout, so they cannot teach an arm scored on fold 0 — `teacher_readings` refuses
# `--fold -1` outright. The teacher is built by INFERENCE over the merged3 cache rather than joined from
# the twins' own holdout files, because the foreign rows have no holdout file to join and because a rank
# is a position within a population: two shards ranked over two different populations would put a
# meaningless separation on every cross-population pair, which is exactly what the distill term reads.
#
# THE SCHEME OVERRIDE THIS NEEDED, AND THE DEFECT IT UNCOVERED. A teacher must read every study the
# student is fitted on — `TeacherFile.aligned` raises on any training study it does not cover, correctly,
# since the alternative is a student taught another study's reading. So the twins have to be pointed at
# the STUDENT's cache, and they could not be: `canonical`, `merged` and `merged3` hold the SAME six slots
# and differ only in which studies they hold, `Member.scheme` resolves by first matching slot list, and
# no checkpoint on disk records a scheme key at all. Every member fitted on a merged cache has therefore
# been resolving to the canonical arrays at serve time. Harmless so far, because our studies lead every
# merged cache in the same order — and silently wrong the moment anything wants a study only a merged
# cache holds, which is this arm. Fixed by `Member.on_scheme`, composed with the existing
# `SharedRead.regrouped` for the depth and band.
#
# THE WITNESS, and the geometry claim it was built on WAS WRONG IN THE SAFE DIRECTION (corrected
# 2026-09-10, off the run). This paragraph used to read "the twins were fitted at 336/20 over the
# 0.1-0.9 band and teach here at 336/12 over 0.2-0.8", and worried that whole-stack plus max pooling
# making the depth flexible was an argument rather than a measurement. `teacher_clean.sh` actually
# fits them at `--slices 12 --band-low 0.2 --band-high 0.8 --image-size 224`. So SLICES AND BAND MATCH
# THE TEACHER PASS EXACTLY and the depth mismatch this witness existed to catch does not exist. The one
# axis that does differ is the source array — the twins were fitted on a native 224 decode and teach off
# the 336 merged3 cache downsized to 224 by `Member.read_size`, which is a resample difference, not a
# geometry mismatch. `teacher_readings` logs `reading N training studies at %d px` from `member.read_size`,
# the member's OWN trained size, NOT from `--cache-size`: the 224 in that line is correct behaviour and
# reading it as a defect costs a kill on a healthy run.
#
# COST. merged3_corpus measured ~328 s/epoch at this geometry; twelve epochs is ~1.1 h. The teacher pass
# is a separate ~10 min over a predecoded array (no pydicom), not part of that.
#
# THE FALSIFIABLE PART, written before the run. Against merged3_corpus_s2026 = 0.8521 at this exact
# geometry — NOT against seat_clean_distill's 0.8634, which is 336/20 on a different corpus; the whole
# 336/12 family sits ~0.852-0.862 and reading this against a 336/20 bar would fail it for the geometry.
#   * >= 0.865 (+0.013). Supervision was the whole blocker; foreign pixels pay once the label mismatch is
#     removed. The six-seat retrain then moves to the merged corpus and the same card-hours buy more.
#   * 0.856-0.865. Real and above the noise floor, below the 1.5% bar. Kept as a member, priced on the
#     board, never planned around.
#   * 0.852-0.856. Flat. Foreign pixels do not transfer even under matched supervision, and the
#     external-data axis closes on its own measurement rather than on a confound.
#   * < 0.852. Worse, and it says the teacher's pseudo-labels are noisier than the mismatched real ones —
#     itself a finding about the teacher rather than about the corpus.
#
# TWO DELTAS, STATED SO THE READING IS NOT OVERSOLD. The distill term is worth ~+0.0056 on OUR rows alone
# (seat_clean_distill 0.8634 vs band_dense 0.8578, at 336/20). So a result inside the +0.004..+0.013 band
# is consistent with "the distill term works and the foreign rows contributed nothing". Only the top band
# separates the two, and the per-column read — Lateral Meniscus, which merged3_corpus lost 0.0217 on — is
# what says whether the foreign block stopped displacing our supervision.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The printed recipe must name merged3 on "scheme" AND
# merged3_labels.csv on "labels" AND report_labels_v2.csv on "eval_labels" AND 336 on "image_size" AND
# 0.7 on "distill_weight" AND teacher_merged3.npz on "teacher_targets", and the log must load 6,393
# studies and hold out 909.
#
# deliberate: --distill-weight 0.7 (the settled point on the weight curve; not re-searched)
# deliberate: --eval-labels report_labels_v2.csv (the pin merged3_corpus carries, same values either way)
# reading: noisy_student_s2026.holdout.npz
# control  merged3_corpus_s2026 = 0.8521 (ledger row 332, same geometry, same corpus, same recipe)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED3_LABELS="$PROCESSED/labels/merged3_labels.csv"
TEACHER="$PROCESSED/teacher_merged3.npz"
require "$MERGED3_LABELS"
require "$TEACHER"
require "$LABELS"

run_train noisy_student \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged3 \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --labels "$MERGED3_LABELS" --eval-labels "$LABELS" --out "$MODELS/noisy_student_s2026.pt"
