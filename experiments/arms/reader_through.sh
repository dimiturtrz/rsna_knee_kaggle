#!/usr/bin/env bash
# reader_through.sh — the NEW corpus reader's table as supervision, one moving token.
#
# THE TABLE. `.agent-runtime/scratch/report_labels_v3.csv` was written by the rebuilt corpus reader on
# 2026-09-06T14:53Z (4407 studies, both census sidecars, `finish_reasons` all `stop`) and then sat four
# days unscored. `reader_end_chain.sh` scored it on 2026-09-10 against the 58 radiologist-annotated
# studies — the only truth in this project not produced by a reader — and it PASSES v2's gate:
#
#   v2   140 net wrong of 696 cells   108 confident false positives + 32 misses
#   new  121 net wrong of 696 cells    47 false positives          + 74 misses
#
# The count improves by 19 cells and the FAILURE MODE INVERTS. v2 invents positives; this reader
# withholds them (45 of its 74 misses are UNK, 29 are an explicit NO), and its silent share rises from
# 27.3% to 32.1% of all cells. Synovitis (-13) and Effusion (-14) carry most of the under-call.
#
# INSTALLED UNDER A NAME THAT CANNOT COLLIDE. It is `report_labels_v5reader.csv`, not v3. There are
# already two files called `report_labels_v3.csv` with different md5s — a level-remap table in the
# labels directory (2026-08-30, the one `recal_through` trains on) and this reader's output in scratch
# — and that ambiguity is knee-ibfrw. A third name spelled `v3` would make it permanent.
#
# THE ONE MOVING TOKEN is `--labels`. Everything else is `recal_through`'s, which is itself
# `through_dense`'s line for line. `--eval-labels` is PINNED to v2 and is NOT a second knob: it is the
# ruler refusing to move with the arm. `recal_holdannot` left it unset, was scored against its own
# re-mapped table, and its +0.0477 was two rulers rather than a result.
#
# THE RULER ASYMMETRY, PRE-REGISTERED BEFORE ANYTHING RUNS, because it points the wrong way here.
# The v2 ruler over-calls (108 confident false positives) and this reader under-calls. An arm taught to
# withhold positives is therefore scored by a ruler that rewards inventing them, and the proxy should
# read FLAT TO NEGATIVE even if the table is better. That is knee-8wa0's asymmetry, and it is why the
# gold reading is first and the holdout second. A holdout regression inside the 0.0062 floor is NOT a
# refutation of this arm; a gold-AUC regression is.
#
# WHY IT IS WORTH THE CARD ANYWAY. D283 measured the board−offline gap at +0.0608: the board reads our
# members HIGHER than our own ruler does, which is what a ruler that over-calls positives produces. The
# board is scored against real readings and our proxy is scored against a reader that invents 108 of
# them in 696 cells. Halving that is the one supervision change with a mechanism pointed at the board
# rather than at the proxy. VisualCheXbert is the same move in the literature (D292): a BETTER labeler,
# not a different mapping of the same one, is what moved weighted AUROC 0.83 -> 0.87.
#
# WHY IT COULD FAIL — the mechanism to write down if it does. 74 misses against v2's 32 is a real cost,
# and under-supervision is not symmetric with over-supervision: a withheld positive teaches the model
# the finding is ABSENT, while an invented one at least puts the model near the right region. If the
# gold AUC falls, the reader traded 61 false positives for 42 misses and lost on the exchange, and the
# next table has to fill silence rather than suppress calls.
#
# PRE-REGISTERED against the control below, own holdout, L75mid, n=909, and on gold AUC (n=58).
#   gold >= +0.005 and holdout >= 0.8805   THE TABLE IS BETTER — v5reader becomes the supervision table
#                                          and every label-side seat is re-fit on it.
#   gold >= +0.005 and holdout <  0.8805   THE ASYMMETRY IS REAL AND MEASURED — the table is better and
#                                          our proxy cannot see it. Ledger it and ship a board entry to
#                                          break the tie, since the board is the third ruler.
#   gold within +/-0.005                   NULL — a better agreement count does not buy a better model;
#                                          reader quality closes as an axis at this magnitude.
#   gold <= -0.005                         COSTS — the over-call/under-call exchange loses; write the
#                                          silence-filling mechanism above and point the next reader at
#                                          Synovitis and Effusion, which carry -27 of the -30 net.
#
# ---------------------------------------------------------------------------------------------
# PRE-REGISTRATION CORRECTED 2026-09-10T11:10Z, MID-RUN, BEFORE ANY NUMBER LANDED. D295 measured the
# gold ruler's own resolution and the branches above are written below it. They are WRONG AS STATED
# and are replaced here rather than edited away, so the record shows what was claimed and when.
#
#   study bootstrap of a macro gold AUC, n=58:            sd 0.0178
#   jackknife SE:                                         0.0183
#   PAIRED difference sd at the measured arm-to-arm corr 0.84:  0.0101
#
# So a +/-0.005 gold branch is HALF a standard deviation of the instrument reading it. The 0.0062
# floor quoted for gold in knee-8wa0 is this project's THREE-SEED PAIRED 909-STUDY floor, borrowed
# onto a 58-study ruler — 15.7x fewer studies. Same denominator error the project rule names.
#
# AND THE GOLD SET IS 78% IN-SAMPLE. `--hold-annotations` does NOT hold the annotated studies out:
# `options.py:314` -> `targets.py:453` skips only the block that OVERWRITES those cells with the
# radiologist value at ANNOTATED_WEIGHT. The rows stay in the training table at their report-derived
# values. 45 of the 58 gold studies sit in the training fold and 13 in the 909 holdout. This arm
# never passed the flag and its supervision table is reader-derived, not fitted on gold, so the
# knee-8wa0 circularity channel does not reach it — but the IMAGES are seen either way, for the arm
# and its control alike.
#
# THE BRANCHES THAT ACTUALLY READ:
#   proxy (909 studies, own holdout, L75mid, floor 0.0062) is the only reading with the resolution to
#   decide anything, and it is the one biased AGAINST this arm. So:
#     holdout >= +0.0062   THE TABLE IS BETTER AND THE ASYMMETRY DID NOT EVEN COST IT. Strongest
#                          available outcome; v5reader becomes the supervision table.
#     holdout within the floor   the table is not worse under a ruler built to punish it. KEEP THE
#                          MEMBER for the ensemble (a genuinely different supervision table is a
#                          diversity axis the pool does not have) and treat the table as OPEN.
#     holdout <= -0.0062   the 74 misses cost real signal; point the next reader at Synovitis and
#                          Effusion and stop paying card for reader tables.
#   gold (n=58) is reported as a LEVEL, never as a selector, and only outside +/-0.020 (2 paired sd)
#   may it be read as a direction.
#   gold-on-the-honest-13 (the gold studies inside the 909 holdout) is the only CLEAN gold read this
#   project can take. It costs no card — both records are on disk. Its sd is ~0.018 on 13 studies, so
#   it is recorded as an observation and is not allowed to decide anything either.
# ---------------------------------------------------------------------------------------------
#
# deliberate: --labels (the single moving token vs through_dense / recal_through)
# deliberate: --eval-labels (the ruler must not move with the arm; recal_holdannot is the incident)
# bead: knee-07gvn
# reading: reader_through_s2026.holdout.npz
# control  through_dense_s2026 = 0.8867 (own holdout, L75mid, n=909)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

READER="$PROCESSED/labels/report_labels_v5reader.csv"
require "$READER"
require "$LABELS"

run_train reader_through \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT  \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --through-plane  \
  --labels "$READER" --eval-labels "$LABELS" --out "$MODELS/reader_through_s2026.pt"
