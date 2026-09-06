#!/usr/bin/env bash
# v3labels_dense.sh — the same recipe, a different RULER. The labeler arm.
#
# WHY THIS ARM EXISTS. Every label this project trains on comes from one lexicon pass over the free-text
# reports (`report_labels_v2.csv`). `2026-09-05_the-holdout-is-the-distorted-ruler.md` measured what that
# pass costs: on the 58 radiologist-annotated studies it reads 108 CONFIDENT FALSE POSITIVES against 32
# misses, and one weight set scores 0.8956 against the radiologist and 0.857 against the reports. The
# reader over-calls, one-directionally, and the whole corpus is supervised by it. This arm swaps in an
# LLM-read table (`report_labels_v3.csv`, Qwen over the 4,406 reports, v2 schema) and changes NOTHING
# else, so the delta is attributable to the labeler and to nothing else.
#
# SINGLE VARIABLE. `band_dense_hold` is the control, byte-for-byte the same command with `--labels`
# pointing at v2. `--hold-annotations` is carried for the same reason it carries there: only a checkpoint
# that HELD the 58 annotation cells is readable by `kneemri.evaluation.annotated_readings`, which
# REFUSES a checkpoint that trained on them. Without the flag the gold ruler cannot be computed at all.
#
# THE TWO RULERS ARE PRE-REGISTERED, AND NEITHER IS THE v2 REPORT HOLDOUT.
#
#   RULER A — GOLD 58, per-label paired against `band_dense_hold_s2026` (0.8956 gold, 0.857 report).
#             The radiologist is the only truth in this project not produced by a reader, so it is the
#             only ruler a change of reader cannot flatter.
#   RULER B — THE 909 HOLDOUT, RE-SCORED AGAINST v3 AS TRUTH. Answers the different question: does the
#             corpus become more learnable when it is read more consistently.
#
#   DISQUALIFIED, RECORDED HERE BEFORE THE RUN: the 909 holdout scored against v2. `band_dense`'s 0.8578
#             and every other ledger number on this corpus is that ruler. A model trained on v3 and
#             scored on v2 is measured by the reader it was built to replace; a DROP there is the
#             expected reading of a successful arm, not a refutation. It will be printed and it is not a
#             branch.
#
# BRANCHES.
#   gold UP and v3-holdout UP        KEEP. The labeler is better supervision, the axis opens, and the
#                                    next step is the roster retrain (every member is supervised by v2).
#   gold UP, v3-holdout FLAT         RULER-LIMITED. Keep the checkpoint, seat it, and ship it as an
#                                    instrument: the gain is real and the offline ruler cannot see it.
#   gold FLAT                        KILL THIS READER, not the axis. A different prompt or a different
#                                    model is a different arm; the corpus defect is unrefuted.
#
# THE GATE THAT COMES FIRST, and it is not this arm. `reader_agreement` on the 58 must read v3 at BETTER
# than v2's 140 net wrong (108 FP + 32 miss). Kimi's endpoint gate read Qwen at 122 net (48 FP / 74
# miss). If the full v3 table lands at 140 or worse, neither this arm nor the ruler swap proceeds — a
# table that is merely DIFFERENT reorders seats and moves scores exactly as happily as one that is
# better. Run the gate, then this.
#
# PRICE. ~40 min of card, band_dense's pixel cache, no decode owed. Expected gain: the mechanism is that
# 108 confident false positives in 684 scoreable gold cells is a 16% one-directional error rate on the
# supervision itself, which is an order of magnitude above the ~2.3% the label-error arms are picking at.
# P(gold ruler moves >= 0.0062) ~ 0.35 — higher than any label-error arm on this corpus, because the
# treatment is the whole table rather than a 3.65% selection of it, and lower than certain because a
# second reader can be differently wrong rather than less wrong.
#
# WHAT WOULD MAKE ME DISBELIEVE IT. `--labels` is a PATH. A path that does not exist, or a partial table
# that covers 600 of 4,406 studies, trains happily on whatever it finds and every log line looks healthy.
# CHECK THE RECIPE PRINTS report_labels_v3.csv AND CHECK ITS ROW COUNT before reading any score. The
# partial Kimi sends for the smoke is 600 rows and must never reach this arm.
#
# deliberate: --labels (the arm's thesis, and its only difference from band_dense_hold)
# deliberate: --hold-annotations (the control carries it; it is what makes the gold ruler computable)
# reading: logs/v3labels_dense_ledger.log
#
# bead: knee-9198
#
# control  band_dense_hold_s2026 (0.8956 gold / 0.857 report). NO --vram-gb: DeviceBudget derives the
#          cap from --slices — 20 slices reserve 28.26 GB.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
V3="$PROCESSED/labels/report_labels_v3.csv"
require "$V3"

run_train v3labels_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --labels "$V3" --out "$MODELS/v3labels_dense_s2026.pt"
