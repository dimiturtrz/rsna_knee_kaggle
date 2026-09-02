#!/usr/bin/env bash
# Does the label-noise treatment TRANSFER, or did one member get lucky? `knee-teh4`, second rung.
#
# WHY THIS EXISTS BEFORE ITS PREDECESSOR HAS READ. `errors_dense` is one run of one trunk. A single member
# moving is not an axis opening, and the cheapest thing that separates the two readings is the same treatment
# on a DIFFERENT trunk family. If pruning the same 5,360 cells moves both, the flags are carrying something
# about the LABELS. If it moves only `errors_dense`, it was that member's variance and the axis stays shut.
# This arm is therefore armed only in the two upper branches of `errors_dense` (>= 0.8578) — below that the
# treatment has nothing to transfer and the fork goes to re-labelling instead.
#
# ONE FLAG AGAINST ITS OWN CONTROL, AND THE CONTROL IS NOT band_dense. Every flag below is `errors_dense`'s
# except `--variant v3small`, which changes the trunk family. So this arm's delta is read against the
# v3small member trained on the PLAIN table, never against `band_dense` — a cross-family comparison would
# confound the trunk with the treatment, which is exactly the confound that voided `ortho_dense` tonight.
#
# IT MOVES THE RULER for the same reason `errors_dense` does: pruned cells carry zero weight, so the holdout
# table is not the one the plain member was read on. Re-read the pair on the PLAIN table before quoting.
#
# PRE-REGISTERED, against the v3small plain member's own holdout (call it C):
#
#   >= C + 0.0062   TRANSFERS. Two trunk families, same flag table, both clear the holdout floor. That is the
#                   label axis opening for real, and the dose curve (how many cells) becomes the next arm.
#   C + 0.0062 > s >= C   DIRECTIONAL, SAME SIGN. Keep it, price the seat, plan nothing around it. Two
#                   same-sign directionals are worth more than one but are still not a finding.
#   C > s           DOES NOT TRANSFER. Then `errors_dense`'s move, whatever its size, is about that member
#                   and not about the labels, and the honest reading of the pair is variance. Report both
#                   per-label row sets before writing it up: if the two trunks disagree on WHICH findings
#                   moved, that is a different and more interesting failure than a flat mean.
#
# WHAT WOULD MAKE ME DISBELIEVE IT. The same silent zero as its predecessor: `--label-errors` is a PATH, and
# a wrong one reads as no flags while every log line looks healthy. Check the run's recipe prints
# "label_errors" at the flag table's path AND that "variant" reads v3small, BEFORE reading the score. The
# second check is this arm's own hazard — $RECIPE already names a variant, and this arm relies on the later
# flag winning.
#
# deliberate: --variant v3small (the arm's thesis, and its only difference from errors_dense)
# deliberate: --label-errors, --slices 20, --band-low 0.1, --band-high 0.9 (errors_dense's read exactly)
#
# bead: knee-teh4
#
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
FLAGS="$PROCESSED/label_flags.csv"
require "$LABELS"
require "$FLAGS"

run_train errors_v3small \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --variant v3small \
  --label-errors "$FLAGS" \
  --labels "$LABELS" --out "$PROCESSED/errors_v3small_s2026.pt"
