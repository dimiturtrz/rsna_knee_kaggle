#!/usr/bin/env bash
# blendsix.sh — a roster where EVERY seat saw the blend supervision, not one seat out of six.
#
# bead: knee-4i3m
#
# THIS ARM HAS RUN, AND ITS QUESTION IS ANSWERED. DO NOT RE-QUEUE IT. Four of its five members exist
# under the processed root — blend_ws.pt, blend_calib.pt, blend_ladder.pt, blend_through.pt, all written
# 2026-08-26 and declared in `comp/members.roster` — so re-running it would OVERWRITE members that a
# scored board row depends on being reproducible. The paragraph below says "this arm builds five more of
# them" because it was written before they were built: it is a plan, not a state.
#
# AND THE DOSE IS FLAT. `kneemri-blendthree` served FOUR blend seats out of six and scored 0.939, against
# blendseat_v1's ONE blend seat at the same 0.939. Dose one to dose four is +0.000, which is this arm's
# own pre-registered FLAT band, already read. The label axis closes at the DOSE term. What it leaves open
# is the TABLE term, and — untried, unpriced — blend and calibration composed on the SAME seat, which only
# `blend_calib` and `merged_blend_cf` carry today.
#
# WHAT MOVES, AND IT IS ONE THING. `comp/champion.json` pins blendseat_v1 at 0.939: six members, of
# which exactly ONE (`merged_blend_s2026`) was fitted on the merged corpus and the three-reader blend
# label table. This arm builds five more of them, so the roster goes from a DOSE of one blend seat in
# six to six in six. No board entry has ever moved that variable.
#
# WHY THIS AND NOT v3six. v3six is six DINOv3-base members, ~14 h on the measured 538-789 s/epoch times,
# and its own pre-registration names 0.933-0.939 as "the expected reading" — a fourteen-hour arm whose
# author predicts a null. It is an ENCODER-FAMILY arm, and that is the axis with the most refutations
# already against it here: seven board entries at 0.936-0.937 with count, member identity, fold status,
# regularisation, encoder family and architecture family each moved alone. The competition forum reached
# the same reading independently on 2026-08-25 (topic 735826) — labels first, crop geometry second,
# explicitly NOT backbone capacity, with a counterexample where DINOv2-small plus diagnosis-specific slot
# attention scored 0.674 OOF BCE against a ResNet34 control at 0.664, the BIGGER model losing. v3six is
# not retired, it is out-priced; its file stays, unrun, for a card with nothing better to do.
#
# WHAT THE LABEL AXIS ACTUALLY HAS BEHIND IT, at its real strength and no higher:
#   * blendlabels, one member, fold-0: +0.0034 mean (BELOW the 0.0062 paired floor) and +0.0070 on its
#     own label-bottlenecked six (barely above it), MCL +0.0239.
#   * merged_blend, one member: 0.8587 holdout, seat value +0.0017.
#   * blendseat_v1, one seat on the board: 0.939 against votefit's 0.938 — ONE TICK, noise.
#   * The forum's first public benchmark of report-label tables against the 58 gold studies (topic
#     737454): stevenleehans v4 blend 0.893 macro AUC, v2 0.887, pilkwang report_labels_v2 0.870, with
#     "averaging two independent tables beats either alone on several labels". A 0.023 spread between
#     label TABLES, above the house bar — but read on 58 studies, and our own `severity_boundary` puts
#     59.9% of this corpus outside English, so the ORDERING is the finding and the values are not.
# None of that is a result. All of it points one way, and it is the only axis on this board that does.
#
# GEOMETRY IS HARMONISED, DELIBERATELY, AND THIS IS THE ONE DEVIATION FROM COPY-VERBATIM. Three of the
# champion's seats are DENSE (20 slices, 0.1-0.9) and one is 12/0.1-0.9; the merged scheme has only ever
# been decoded at 12 slices, 0.2-0.8 (`images_336px_12sl_merged.npy`, 41.8 GB). Reproducing the
# champion's geometries here would mean two new merged caches at roughly 70 GB and 42 GB plus an
# unmeasured decode pass over 5,143 studies, to move an axis this project has already measured FLAT
# (`merged_blend.sh`: "the pixel axis measured flat (-0.0023)"). So every member below runs the cached
# 12/0.2-0.8 geometry and the diversity is carried by MECHANISM instead — calibrated targets, the grade
# ladder, through-plane, and a DINOv2-base trunk. Cost of the deviation, stated: the roster loses its
# geometric spread, which is a live suspect if this reads at or below 0.935 (see the last band).
#
# COST. `merged_corpus` measured 297 s/epoch at fold-0 over this cache (4,234 train). At --fold -1 all
# 5,143 studies train, so ~361 s/epoch; twelve epochs is ~1.2 h a member. Four small members plus one
# base member at batch 6 is roughly 7 h. Half of v3six, on an axis with a mechanism. No cache is built
# and no disk is spent: the arrays exist and the label table was written by `build_merged --labels-only`.
#
# NONE OF THESE FIVE CAN EVER BE SCORED OFFLINE. --fold -1 means each has seen every holdout study, so
# no output may be seat-scored, curve-scored or distance-scored, and none may enter a pool that reports
# an offline mean. Like blendseat before it, this ships blind against the board incumbent.
#
# HARNESS CHECK BEFORE READING ANYTHING. The printed recipe must name merged on "scheme" AND
# merged_blend_v1.csv on "labels" AND 336 on "image_size" AND -1 on "fold", and the log must load 5,143
# studies with no holdout. Three runs have entered this ledger as facts about an architecture when they
# were facts about a flag that never landed, and this arm carries three flags on every member.
#
# THE FALSIFIABLE PART, written before the run, read on the BOARD against blendseat_v1 0.939:
#   * >= 0.947. The dose is the lever, the label axis is open, and every remaining hour goes to better
#     label TABLES — the forum's v4 blend first.
#   * 0.943-0.946. Real but under the 1.5% bar as a single step; the axis carries and composes, and the
#     next dose question is whether a SEVENTH blend seat still pays.
#   * 0.940-0.942. Directional, inside the scatter our nine flat 6-7 member entries already span. The
#     DOSE is not the lever; supervision QUALITY might still be, and that is a different table, not more
#     seats of this one.
#   * 0.936-0.939. FLAT. Six blend seats buy what one bought, which is nothing measurable. The label axis
#     closes at the DOSE term and stays open at the TABLE term.
#   * <= 0.935. Two suspects, in this order. First the harmonised geometry above: the champion's spread
#     over 12 and 20 slices may have been carrying the ensemble, and `member_distance` over the five new
#     members answers that without a card. Second, five members fitted on one finer table agreeing with
#     each other more than the champion's five did — the same reading from the other side. Only after
#     both come back clean does this indict the blend supervision itself.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED_BLEND="$PROCESSED/labels/merged_blend_v1.csv"
require "$MERGED_BLEND"
require "$LABELS"

FIT="--fold -1 --ema"
GEOMETRY="--slices 12 --band-low 0.2 --band-high 0.8"
# deliberate: --eval-labels "$LABELS" is inert under --fold -1 (there is no holdout to pin); it is kept
# deliberate: so every member here is argv-identical to merged_blend.sh apart from the fold.
BLEND="--scheme merged --labels $MERGED_BLEND --eval-labels $LABELS"

run_train blend_ws \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOMETRY \
  $BLEND --out "$MODELS/blend_ws.pt"

run_train blend_calib \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOMETRY --calibrated-targets \
  $BLEND --out "$MODELS/blend_calib.pt"

run_train blend_ladder \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOMETRY --grade-ladder \
  $BLEND --out "$MODELS/blend_ladder.pt"

run_train blend_through \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $GEOMETRY --through-plane \
  $BLEND --out "$MODELS/blend_through.pt"

run_train blend_base \
  $RECIPE $FIT --epochs 12 --batch 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --backbone-lr 4e-5 $GEOMETRY \
  $BLEND --out "$MODELS/blend_base.pt"
