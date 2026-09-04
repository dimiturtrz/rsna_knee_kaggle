#!/usr/bin/env bash
# The first arm this project fits on labels the READER did not get a vote on. `knee-teh4`.
#
# THIS ARM HAS RUN — 2026-09-02, twelve of twelve epochs, best 0.8420 at e6 against `band_dense`'s
# 0.8578. DO NOT RE-RUN IT; a session on 2026-09-04 re-armed and re-queued it from the belief that it
# had not, and the four card-hours were recovered only by chance (`knee-xi4h2`). The reading is the
# arm's own lowest branch, IT COSTS, and it is written up in `experiments/README.md`.
#
# TWO NUMBERS BELOW ARE THE WRONG ROW OF THE GATE, and they are left in place because they are what the
# arm was pre-registered on. `logs/label_errors.log` reports a budget CURVE and, separately, a row for
# cleanlab's own confident-joint mask — and the MASK is the selector this arm ran. On the 108 confident
# false positives the mask reads precision 0.258, lift 1.63x. The 4.05x cited at lines 17 and 43 is the
# plain RANKING read at a 3.7% budget this arm never used. More than half the 5,360 pruned cells were
# cells the radiologist agreed with, which is the measured mechanism for the −0.0158. The strong number
# is still available to a different selector: `knee-jom26`.
#
# THE DEFECT IS MEASURED, NOT HYPOTHESISED. Over the 58 radiologist-annotated studies
# `reader_agreement.errors` counts 108 confident false positives against 32 misses, and
# `supervision_gap` reads that where the report says YES the radiologist agrees 65.3% of the time. Both
# label mechanisms that ship today are functions of what the reader says about ITSELF — `Supervision.applied`
# maps the reader's five levels, `weight_floor` sets the cell's loss weight to `0.25 + 0.75*conf` — and
# `report_reader.CONFIDENCE` is CONSTANT per verdict (std 0.0 over 52,872 cells). So the current weight is a
# deterministic function of the verdict it multiplies, and it up-weights the reader's dominant error by
# construction. No refinement of a self-report can separate a confident-and-wrong cell from a confident-and-right one.
#
# THE GATE IS ALREADY SPENT AND GREEN, which is why this arm is armed and not proposed. Commit f5e1f808 ran the
# pre-registered gate: cleanlab's confident joint over `PublishedRuler.per_seed()` (four seeds x five folds whose
# oof id-sets partition all 4,407 studies, so no study is scored by a model that trained on it), twelve
# independent binary problems because the defect's unit is the (study, finding) CELL. 57 of 58 gold studies
# covered, 684 scoreable cells, and it beats a random flagger of the same size 4.05x on the 108. The detector
# cost zero card and it is not this arm's question.
#
# ONE FLAG AGAINST THE CONTROL. Every flag below is `band_dense`'s — 0.8578, the best member the project owns —
# except `--label-errors`, which points at the table `label_errors --flags` wrote from that same detector run.
# It reads `band_dense`'s own pixel cache, so no decode is owed.
#
# PRUNE, NOT DOWN-WEIGHT, AND THAT IS THE LIBRARY'S CHOICE RATHER THAN OURS. The confident joint estimates how
# many cells to disbelieve, so the SIZE of the flagged set is an estimate. A down-weight would need a magnitude,
# and any magnitude fitted on our own holdout is a hidden seed on a corpus where seeds are banned. An annotated
# cell is never pruned: the radiologist's reading is the truth the detector was gated against.
#
# IT MOVES THE RULER, AND THE READ HAS TO ACCOUNT FOR IT. `Supervision.moves_the_ruler()` returns True here for
# the most direct of its four routes: a pruned cell carries zero weight, and a holdout AUC over a table missing
# those cells is not the table `band_dense` was read on. Re-read the pair on the PLAIN table before quoting any
# delta, exactly as `calib_span` had to.
#
# PRE-REGISTERED, before the arm runs:
#
#   >= 0.8640   THE LABEL AXIS OPENS. +0.0062 or better over the best member the project owns, clearing the
#               holdout floor on the one axis nobody has priced. The next arms are the dose curve (how many
#               cells) and the treatment (prune vs re-label to the model's own held-out probability), and the
#               58 gold stay a bound rather than becoming a target.
#   0.8640 > s >= 0.8578   DIRECTIONAL. Free and mechanism-sound, so keep the checkpoint and price its seat,
#               but it is not a finding and nothing gets planned around it.
#   0.8578 > s >= 0.8516   FLAT. Pruning the cells an independent signal calls wrong does not move the mean,
#               and the 4.05x detection result stands while its TREATMENT does not. That splits the axis: the
#               flags are real and the loss does not care, which points at re-labelling rather than dropping.
#   < 0.8516    IT COSTS. Then the flagged cells were carrying signal the metric wants — the most likely
#               reading is that a report-derived positive the models disbelieve is still enriched for the
#               finding, and pruning it removes a weak-but-real positive from a rare class. Report the
#               PER-LABEL rows before writing anything: the rare findings are where this would show.
#
# WHAT WOULD MAKE ME DISBELIEVE ANY OF IT. That the flag never landed. `--label-errors` is a PATH, and a path
# that does not exist reads as no flags at all while every log line looks healthy — `Supervision.pruned` drops
# a flag naming a study the table has no row for, by design, so a wrong path and a legitimately-subset table
# are the same silent zero. Check the run's own recipe prints `"label_errors"` at the flag table's path, and
# check the flag table's row count against the `corpus flagged` line in `logs/label_flags.log`, BEFORE reading
# the score.
#
# deliberate: --label-errors (the arm's thesis, and its only difference from band_dense)
# deliberate: --slices 20 --band-low 0.1 --band-high 0.9 (band_dense's read exactly — the control)
#
# bead: knee-teh4
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  band_dense_s2026 = 0.8578 (declared 2026-09-03 from this header own prose, verified on disk, knee-sgzgc)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
FLAGS="$PROCESSED/label_flags.csv"
require "$LABELS"
require "$FLAGS"

run_train errors_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --label-errors "$FLAGS" \
  --labels "$LABELS" --out "$MODELS/errors_dense_s2026.pt"
