#!/usr/bin/env bash
# read_group_queue.sh — the same weights, read a second way, seated BESIDE themselves.
#
# WHY. `traversal_census` (326f2b9b) measured the chunk-outer memo removing 47.5% of the champion's
# slice decodes and 66.7% of its header parses on the shipped path, 96 of 96 bags byte-identical. That
# freed budget is not margin: `decode_bench --union --extra` (795fb35a) prices a FOURTH read group at
# +2.3% to +6.3% of the deduplicated decode, where under group-outer it was a whole extra pass (+33%).
# So a second reading of weights we already own is now affordable, and a second reading is a diversity
# seat that costs no training.
#
# WHAT IS SERVED. The five fold-0 twins of `raptorweight_v1`'s seats — the `twinfive` pool that reads
# 0.8706 LogitAverage over 909 studies. Their native groups, off their own recorded options:
#
#   calib_span_s2026      12 sl  0.1-0.9  336
#   grade_ladder_s2026    20 sl  0.1-0.9  336
#   merged_blend_cf       12 sl  0.2-0.8  336
#   lr4e-5_ws_s2026       12 sl  0.2-0.8  336
#   through_dense_s2026   20 sl  0.1-0.9  336
#
# All five are fold 0 of 5 at split seed 2026, which is the whole reason this queue can be READ at all.
# Five of the champion's six seats are `--fold -1` and hold nothing out, and `comp/champion.json`
# records what comparing across that line cost on 2026-08-25: twins read 0.8687 against peak8's 0.8731
# offline while the board read the opposite by six ticks. So every number this queue produces is
# TWINS-AGAINST-TWINS — regrouped twins beside native twins, never a champion seat.
#
# THE ORDER, and why it is not the marginal-decode order alone. Each candidate costs the SAME card:
# five members over the same 909 studies. So AUC-per-card-hour ranks by expected gain, and the decode
# marginal is the SERVING price that decides whether a winner can be shipped at all. Both are stated.
#
#   1. 392 px, native groups      + 0.0 files/study   resolution is NOT in the read key — `SlotReader`
#                                                     .cropped opens its files before any resize — so a
#                                                     twin served at 392 adds ZERO decode and costs only
#                                                     forward time. It is also the one axis with prior
#                                                     evidence: `res392_ws_s2026` is the best solo member
#                                                     this project owns.
#   2. 16 sl 0.1-0.9              + 2.4 files (+2.3%)  a depth BETWEEN the two served, band already open
#   3. 12 sl 0.15-0.85            + 4.7 files (+4.5%)  the cheapest band SHIFT — new pixels, not a subset
#   4. 24 sl 0.1-0.9              + 6.1 files (+5.8%)  deeper than anything served
#   5. 20 sl 0.15-0.85            + 6.6 files (+6.3%)  shift and depth together
#
# reading: *_r392.holdout.npz
#
# THE BAR, pre-registered. Ten seats (five native + five regrouped) must read >= 0.8751 — 0.8706 plus
# the 0.0045 board resolution — for a candidate to be worth a slot. Below that it is DIRECTIONAL and
# goes to the ledger only. A regrouped seat that reads BELOW its own native twin solo is not
# disqualifying: the question is what the pair adds, not what the copy scores.
#
# WHAT WOULD MAKE THIS A HARNESS FAULT RATHER THAN A NULL. `holdout_backfill` writes the regrouped
# record BESIDE the native one (`SharedRead.beside`, `ReadSize.destination`), so a candidate that
# silently reused the native record would read EXACTLY 0.8706 for its ten seats. An exact tie is the
# signal to check the record's filename before believing the null.

set -euo pipefail
cd "$(dirname "$0")/.."

source experiments/roots.sh
TWINS=(calib_span_s2026 grade_ladder_s2026 merged_blend_cf lr4e-5_ws_s2026 through_dense_s2026)
LOGS=experiments/logs
mkdir -p "$LOGS"

checkpoints() {
    for twin in "${TWINS[@]}"; do printf '%s/%s.pt ' "$MODELS" "$twin"; done
}

# One candidate: back-fill the five regrouped records, then read the ten-seat blend off disk.
# `vote_ceiling` costs no card — every record it reads is already written by the time it runs.
serve() {
    local tag=$1 suffix=$2
    shift 2
    local log="$LOGS/read_group_${tag}.log"
    echo "=== $tag $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
    # shellcheck disable=SC2046 - the checkpoint list is a word list on purpose
    .venv/Scripts/python.exe -m kneemri.inference.holdout_backfill \
        --checkpoints $(checkpoints) "$@" >>"$log" 2>&1
    # FULL PATHS INTO THE MODELS ROOT, never bare names and never `all_ours/`. Two traps, one line.
    # (a) `DataRoot.checkpoints` resolves a bare name through `Path.stem`, and a regrouped record is
    #     spelled `..._n16b0.1-0.9.pt` — whose stem is `..._n16b0.1-0`, because `.9` reads as the
    #     suffix. A path with a parent is taken as written and never goes through stem.
    # (b) A `.holdout.npz` lives BESIDE its checkpoint in the models ROOT (`core/paths.py:50`), and
    #     `all_ours/` is a packaging COPY of the `.pt` with no records in it. Pointing at `all_ours/`
    #     makes every seat read `no holdout record, skipped` and the run dies with "none of the 5
    #     named records carries a holdout array" — which reads exactly like the `--fold -1` refusal
    #     and would have been blamed on the fold line.
    local seats=()
    for twin in "${TWINS[@]}"; do seats+=("$MODELS/$twin.pt" "$MODELS/${twin}${suffix}.pt"); done
    .venv/Scripts/python.exe -m kneemri.evaluation.vote_ceiling \
        --candidates "${seats[@]}" >>"$log" 2>&1
    echo "--- $tag done $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
}

# WHAT THREE FREE READINGS ALREADY SAID (2026-09-06 11:35Z, zero card — the records were on disk
# from earlier work and nobody had scored the ten-seat blend). Native five reproduce 0.8706 EXACTLY,
# so the harness is not the suspect:
#
#   + _r224          0.8693   -0.0013
#   + _r280          0.8707   +0.0001
#   + _n12b0.1-0.9   0.8698   -0.0008
#
# Three points, all at or below baseline, none within 0.0044 of the 0.8751 bar. The mechanism that
# explains it: a twin of yourself under an UNWEIGHTED LogitAverage is the most CORRELATED seat the
# pool can hold, so the diversity it adds is smaller than the dilution it costs. Reweighting would be
# the obvious answer and is a seating operation — closed five times, not reopened here.
#
# So the queue is CUT to its one candidate with a live mechanism. `_r224` and `_r280` are DOWNSCALES
# from the 336 the twins were fitted at; 392 is the other direction, and `res392_ws_s2026` — a member
# TRAINED at 392 — is the best solo this project owns. That is a different sign of the same knob, not
# a third point on the one already measured. `_n12b0.1-0.9` is the depth/band axis's first point and
# it is null, so #2-#5 below stay parked unless r392 moves.
serve r392            _r392            --read-size 392

# PARKED on the r392 reading — each is a depth/band regroup, and that axis already has a null point.
# serve n16b0.1-0.9     _n16b0.1-0.9     --served-slices 16 --served-band 0.1 0.9
# serve n12b0.15-0.85   _n12b0.15-0.85   --served-slices 12 --served-band 0.15 0.85
# serve n24b0.1-0.9     _n24b0.1-0.9     --served-slices 24 --served-band 0.1 0.9
# serve n20b0.15-0.85   _n20b0.15-0.85   --served-slices 20 --served-band 0.15 0.85

echo "QUEUE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
