#!/usr/bin/env bash
# bead: knee-z81wd
# reading: dml_a_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   Deep mutual learning (Zhang 2018) at K=2: two members train side by side and each is taught, from
#   round two on, by the OTHER's reading of the training split as that peer stood at the end of the
#   previous round. The deliverable is ONE checkpoint -- `dml_a_s2026`, the `headlr_ship` recipe. B is
#   scaffolding and is discarded like a teacher, so this does not seat a pair.
#
# THE ONE VARIABLE: TEACHER LIVENESS
#   `distill_pairs` is this arm's control and it read +0.0016, under the 0.0062 paired floor. It used
#   the ORDERING term at `--distill-weight 0.3` against a CONVERGED teacher. This arm uses the same term
#   at the same weight against a teacher that is still training. Everything else in the distillation
#   wiring is byte-identical, which is what makes the delta readable against that row. A KL on softened
#   posteriors -- which the bead's first design asked for -- would have moved two variables and could
#   not have been read against it at all; `teacher_targets.py` also argues a KL targets the calibration
#   an AUC metric discards.
#
# WHY THE PEER TERM IS NOT A ROUNDING ERROR
#   Peer disagreement is 28-42% of the supervised residual across five top members (909 studies, 12
#   labels; knee-z81wd notes). Measured at the HARDEST place to find it -- converged members, held-out
#   rows, one family -- so the early-training training-split disagreement this arm feeds on is larger.
#   Disagreement is NECESSARY, not SUFFICIENT: it licenses the run, it does not predict the gain.
#
# ROUNDS, AND WHY NOT `--epochs 8` THREE TIMES
#   `one_cycle` is built over `--epochs` (run_setup.py:571), so asking for 8 COMPRESSES the anneal into
#   a different recipe -- measured head to head, the compressed cycle leads at epoch 0 (0.7510 against
#   0.7393) and trails at the end (0.8429 against 0.8457). `--stop-after-epoch` bounds the LOOP and
#   leaves the schedule built for the full 24, so the epochs either side of an exchange are the epochs
#   this arm was priced for and the result is readable against a 24-epoch control that was never stopped.
#
# LEAK
#   `teacher_readings` writes the ordering of the TRAINING split, the complement of the holdout, and
#   refuses a `--fold -1` member. A and B hold out the same honest fold 0, so a peer reading of training
#   studies carries nothing either model held out.
#
# OVERWRITE, DELIBERATELY
#   `run_train` refuses an `--out` that exists, and rounds 2 and 3 write the one round 1 wrote. Nothing
#   is destroyed: `RunMetrics.begin()` rotates the finished round aside as `_run<n>.pt`, its holdout npz
#   and `.metrics.<n>.jsonl`, and the weights this round resumes from are the `.partial` / `.resume`
#   pair, which nothing rotates. The one real loss is the FULL LOG: `tee` truncates, and every round
#   shares a log name because the log is named for the checkpoint. The per-epoch record survives in the
#   rotated metrics files, which is what the ledger quotes.
#
# CARD. `--vram-gb 28` of a 32 GB card, inherited from both bases. Sequential by construction -- one
#   model on the card at a time -- so nothing else of ours may share it (`card_check`).
#
# PRE-REGISTRATION (delta = dml_a_s2026 minus headlr_ship_s2026 0.9059, ledger75)
#   >= +0.0062   clears the paired floor. Teacher liveness is the thing distillation was missing here,
#                and K, cadence and weight all become knobs worth buying on a mechanism that fired.
#   +0.000..+0.0061
#                directional only. Keep the member for the pool, spend no slot, open nothing.
#   <= 0.0000    mutual supervision is closed for this family at this weight. Record it beside the four
#                null distillation rows; the peer-disagreement measurement stands and is NOT re-opened
#                by re-quoting it.
#   peak         if A peaks on its LAST epoch the budget is still binding and no verdict may be read.
#
# deliberate: --distill-weight 0.3 --teacher-targets — the control's operating point, not a swept knob.
# deliberate: --stop-after-epoch 8/16/24 — the round boundary, on a schedule built for 24 throughout.
# deliberate: B is `labelwise_depth24_drop40_long`'s line — the pair MEASURED to disagree (ratio 0.284).
#
# control  headlr_ship_s2026 0.9059 ledger75 (on disk, 24 epochs, same line as A minus the peer term).
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

A="$MODELS/dml_a_s2026.pt"
B="$MODELS/dml_b_s2026.pt"

fit_a() {
  run_train dml_a \
    $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
    --calibrated-targets --hold-annotations --slot-dropout 0.40 \
    --head-lr 2e-4 \
    --stop-after-epoch "$1" "${@:2}" \
    --labels "$LABELS" --out "$A"
}

fit_b() {
  run_train dml_b \
    $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
    --calibrated-targets --hold-annotations --slot-dropout 0.40 \
    --stop-after-epoch "$1" "${@:2}" \
    --labels "$LABELS" --out "$B"
}

read_peer() {
  uv run --no-sync --project "$REPO" python -u -m kneemri.inference.teacher_readings \
    --checkpoints "$1" --out "$2" --fold 0 --slices 24 --band 0.1 0.9
}

fit_a 8
fit_b 8

read_peer "$A" "$PROCESSED/dml_read_a_r1.npz"
read_peer "$B" "$PROCESSED/dml_read_b_r1.npz"

ALLOW_OVERWRITE=1 fit_a 16 --resume --distill-weight 0.3 --teacher-targets "$PROCESSED/dml_read_b_r1.npz"
ALLOW_OVERWRITE=1 fit_b 16 --resume --distill-weight 0.3 --teacher-targets "$PROCESSED/dml_read_a_r1.npz"

read_peer "$A" "$PROCESSED/dml_read_a_r2.npz"
read_peer "$B" "$PROCESSED/dml_read_b_r2.npz"

ALLOW_OVERWRITE=1 fit_a 24 --resume --distill-weight 0.3 --teacher-targets "$PROCESSED/dml_read_b_r2.npz"
ALLOW_OVERWRITE=1 fit_b 24 --resume --distill-weight 0.3 --teacher-targets "$PROCESSED/dml_read_a_r2.npz"
