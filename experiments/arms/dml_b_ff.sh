#!/usr/bin/env bash
# bead: knee-8bmi2
# unmeasurable: dml_b_ff fits --fold -1, so it holds no study back and no ledger row can price it
#               (knee-s8kim). Its holdout TWIN dml_b_s2026 reads 0.9078 ledger75, the highest this
#               project owns. Its only ruler is the board row of the entry it feeds, which costs a slot.
#
# WHAT THIS ARM IS
#   `labelwise_depth24_drop40_long_ff`'s line -- the entry that scored board 0.944 as `ffsolo_v1` -- with
#   ONE term added: the peer-supervision target `dml_b_s2026` was trained against. Flag for flag it is
#   that arm, `--vram-gb 22` included, plus `--distill-weight 0.3 --teacher-targets`.
#
# THE ONE VARIABLE, AND WHY THE ANCHOR IS TIGHT
#   `ffsolo_v1` 0.944 is the whole-corpus refit of a base at 0.9053 ledger75. `dml_b_s2026` is that SAME
#   recipe at 0.9078, +0.0025, the peer term being the only difference. So this arm changes one variable
#   against a board row we own, rather than against a transfer line fitted across families.
#
# PRICE AND BRANCH live in knee-8bmi2, which arm_check reads. Summarised: ~3.5 h of card, predicted
#   0.9465 against a 0.946 champion at the +1.00 in-band transfer constant, P(clears) ~0.35.
#
# WHY THE +0.0025 BASE GAIN IS NOT ITSELF EVIDENCE
#   It sits INSIDE the 0.0053 paired floor, so the offline ruler cannot order `dml_b_s2026` against its
#   base at all. Mutual supervision moved its partner -0.0015 the other way, equally unresolved; the
#   cohort result is null in both directions. This arm is therefore bought as a REFIT of the highest
#   offline member on disk, not as a confirmation of the peer term -- that term is unresolvable on any
#   ruler this project owns, and a board row will not resolve it either.
#
# THE TRANSFER OBJECTION DOES NOT BITE, AND THE NUMBER BEHIND IT WAS WRONG
#   The audit's "-2.7 sd" mixed rulers and is retracted (knee-jw57g). Refitted on offline75 the three
#   highest offline members read -1.9 / +0.9 / -1.0 sd, combined -1.17, not significant. This arm's own
#   base is the +0.9 -- `l75solo`, ABOVE the transfer line -- so the prediction below is conservative.
#
# LEAK
#   `teacher_readings` refuses `--fold -1` as an ARGUMENT, because teaching FROM a full-data member
#   leaks the evaluation labels. `--teach-all` is the sanctioned spelling and is legal exactly when the
#   STUDENT holds nothing out, which is this arm. The teacher `dml_a_s2026` is a fold-0 member; the
#   student holds no study back, so there is no holdout for a teacher reading to reach into.
#
# WHY THE TEACH PASS IS NEEDED AT ALL
#   `dml_read_a_r2.npz` covers the fold-0 TRAINING split only, and `TeacherTargets.rows` raises on a
#   study the file does not cover. A whole-corpus student asks for every study, so the file is rebuilt
#   over all of them -- ~25 min of inference off the fold-aware twin.
#
# deliberate: --fold -1 --ema — the champion's own fit, matched to `labelwise_depth24_drop40_long_ff`.
# deliberate: --distill-weight 0.3 — `dml_b_s2026`'s own weight, and `distill_pairs`' control weight.
# deliberate: --vram-gb 22 — owner VRAM cap 50-70% of the card; the twin's 28 predates DeviceBudget.needed().
#
# CARD. ~25 min teach pass then ~3 h train. Nothing else of ours on the card.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$MODELS/dml_a_s2026.pt"

TEACH="$PROCESSED/dml_read_a_all.npz"
FIT="--fold -1 --ema"

run_gpu uv run --no-sync --project "$REPO" python -u -m kneemri.inference.teacher_readings \
  --checkpoints "$MODELS/dml_a_s2026.pt" --out "$TEACH" --teach-all

run_train dml_b_ff \
  $RECIPE $FIT --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --distill-weight 0.3 --teacher-targets "$TEACH" \
  --labels "$LABELS" --out "$MODELS/dml_b_ff.pt"
