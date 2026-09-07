#!/usr/bin/env bash
# seat_oof5_distill — the distillation term at FULL teacher strength, with no holdout for it to leak into.
#
# THE TWO ARMS ARE NOT THE SAME QUESTION. `seat_clean_distill` removes the leak by WEAKENING the
# teachers: the four clean blocks are fitted on ~2,600 studies against the leaky ones' ~3,500, and that
# handicap costs a teacher 0.0067 on its own holdout (measured, four folds). So a miss there is
# ambiguous between "the leak was the gain" and "the teachers got worse". This arm removes the leak the
# OTHER way — by removing the holdout. Five full-strength blocks, each predicting the fold it held out,
# cover all 4,406 studies out-of-fold; the student is fitted at `--fold -1` and therefore has no
# evaluation set for a memorised label to leak into. Its ruler is the BOARD.
#
# WHY IT RUNS EVEN IF THE CLEAN STUDENT MISSES ITS BAR. `seat_clean_distill` prices the term through a
# weakened teacher on a held-out fold; this prices it at full strength on the corpus that actually
# ships. They are two dilutions of one term, and the starved reading cannot refute the full-strength
# one — that is the whole reason the starvation table was worth taking.
#
# ONE VARIABLE against `seat_clean_distill`: the teacher file and `--fold -1`. Same 336 px / 20 slices /
# 12 epochs / 4e-5 / max pooling / whole stack, same labels, same distillation weight.
#
# THE TEACHER IS BUILT, NOT REUSED. `teacher_oof_corpus.npz` covers folds 1-4 only, which is what
# `TeacherTargets.rows` raises on for a `--fold -1` student; `teacher_clean.npz` is the starved set.
# `teacher_oof5.npz` is `oof_fold[0-4]_s2026` — the four on disk plus the fifth twin `oof_fold0.sh`
# trains, all five at the same cheap 224/8 geometry so no block is a different KIND of teacher.
#
# COST one twelve-epoch fit at 336/20, ~361 s an epoch measured on `band_dense` -> ~1.2 h, behind
# `oof_fold0`'s ~12-25 min.
#
# deliberate: --fold -1 (the fit that ships; there is no holdout, by construction)
# deliberate: --distill-weight 0.7 (the settled point on the weight curve; not re-searched)
# deliberate: --teacher-targets teacher_oof5.npz (full-strength five-block OOF, built by the watcher)
# deliberate: every other flag verbatim from seat_clean_distill.sh
# unmeasurable: a --fold -1 fit holds nothing out, so no ledger row can price it. The board is its ruler
# unmeasurable: and the entry that reads it is the one this arm exists to build.
# bead: knee-ygi2k
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

TEACHER="$PROCESSED/teacher_oof5.npz"
require "$TEACHER"

run_train seat_oof5_distill \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT --fold -1 --ema \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --labels "$LABELS" --out "$MODELS/seat_oof5_distill_s2026.pt"
