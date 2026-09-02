#!/usr/bin/env bash
# bead: knee-56w5
# ema_dense.sh — `band_dense`'s recipe with `--ema` as the ONLY variable.
#
# THIS IS AN INSTRUMENT, NOT AN IMPROVEMENT, and it is priced as one (knee-56w5). PRICE: ~72 min of
# card. P(clears 1.5%) = 0.02. It is worth 72 minutes anyway because of what it CREDITS OR DEBITS: the
# only board movement this project has ever produced.
#
# WHAT IS CONFOUNDED. `kneemri-allfit` read 0.937 on 2026-08-24, +0.003 over `finalfit` and +0.004 over
# the 0.933 chain, and that reading was written up as the ALL-DATA axis: six of seven seats refitted at
# `--fold -1` on 4,406 studies instead of 3,497. But every one of those six `_ff` recipes is its
# `_s2026` twin's flags PLUS `--fold -1` PLUS `--ema`, and the twins carry no EMA at all. Two variables
# moved together on the entry that moved the board. The interpretation named this at the time
# (interpretations/kneemri/2026-08-24_the_all_data_dose_response.md: "`--ema` rides along on every `_ff`
# recipe and is not held fixed") and then read the number as a data result regardless, because the
# alternative was untestable on that pool — `--fold -1` members have no offline number of any kind.
#
# WHY IT IS TESTABLE HERE AND NOT THERE. EMA is not a data flag. It can be measured at fold 0, where the
# 909-study holdout still works, against a control already on disk. That is the whole trick: the
# confounded pair cannot be separated where it occurred, and the confounding VARIABLE can be measured
# somewhere the confound does not exist.
#
# SINGLE VARIABLE. This is `band_dense.sh`'s invocation with `--ema` appended and nothing else touched —
# same rate, slices, band, pooling, whole-stack, epochs, schedule, batch, labels and split. `band_dense`
# at 0.8578 (best epoch, 909-study fold-0 holdout) is therefore a matched control that costs no card,
# which is why no second arm is queued. Quick over matched: the comparison exists already.
#
# cost   band_dense ran ~361 s/epoch x 12 = ~72 min. EMA adds one parameter-sized buffer and an
#        elementwise update per step; VRAM should move by well under a gigabyte off 26.92 GB reserved,
#        and step time by under a percent. If either moves visibly, the flag is doing something other
#        than what it says and THAT is the finding.
#
# PRE-REGISTERED, four outcomes, written before the run. Floor 0.0062, and note that the floor is what
# makes two of these bands mean opposite things.
#
#   >= +0.0062 over 0.8578 (i.e. >= 0.8640) — EMA IS A REAL LEVER AT THIS OPERATING POINT, and the
#     allfit reading is DEBITED: a gain of this size at fold 0 is easily enough to account for the whole
#     +0.004 the board paid, so "the all-data axis is real and sub-linear" becomes "an axis and a
#     regulariser moved together and the split is unknown". Consequence: EMA is then a FREE lever — it
#     needs no data, no card beyond the refits, and applies to all seven seats including the fold-0 one
#     — and the next entry is the champion's seven with EMA held ON everywhere, which is a cleaner
#     question than the seventh `_ff` seat this project just declined to buy.
#
#   +0.0015 to +0.0062 — DIRECTIONAL, INSIDE THE FLOOR, and the honest reading is that it is not
#     resolvable here. It neither credits nor debits allfit, because a delta this size at fold 0 could
#     still compound across six seats or could be nothing. Consequence: allfit's data reading STANDS but
#     is recorded as carrying an unresolved co-variable, and no further card is spent chasing it — an
#     instrument that returns "cannot tell" has still retired the cheap version of the question.
#
#   -0.0015 to +0.0015 — FLAT. EMA does nothing at this operating point, so it cannot be what the board
#     paid for, and the all-data reading of allfit is CLEAN. knee-tfj3's closure stands as written, and
#     this arm has done its whole job: it converts a named suspect into a dead one for 72 minutes.
#
#   <= -0.0015 — NEGATIVE, which is the most interesting outcome and the one this ledger has seen
#     before. `calib_span` is positive at one operating point and negative at the one we ship. If EMA is
#     negative at fold 0 and the `_ff` members carry it, then allfit scored 0.937 while PAYING for EMA,
#     the data axis is LARGER than +0.004 rather than smaller, and the seventh-seat decision was made
#     against an understated curve. Consequence: re-open the seventh seat, and refit without EMA.
#
# WHICH LINK TO SUSPECT IF THE ANSWER IS FLAT. Verify the flag LANDED before believing it did nothing:
# `--ema` is a boolean with default False, so a flag that never reaches the trainer produces a run
# byte-identical to the control, and a byte-identical run is MY bug rather than EMA's null. Check the
# checkpoint's recorded config for `ema: true` and check that the saved weights differ from the
# control's — same-seed, same-split, same-data runs that differ only in a no-op flag will otherwise
# agree to float noise, and that agreement is the tell.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train ema_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --ema \
  --labels "$LABELS" --out "$MODELS/ema_dense_s2026.pt"
