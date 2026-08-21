#!/usr/bin/env bash
# cnx_dense_448.sh — the retry `cnx_dense` pre-registered for itself, and ONLY that.
#
# DO NOT ARM THIS. ITS GATE OPENED AND ITS MECHANISM CLOSED, in the same result (2026-08-21).
# `cnx_dense` landed at 0.7927 solo, so the "under 0.80" trigger below DID fire — and everything else
# that landed with it says the trigger is pointing at the wrong variable. Three independent readings,
# none of them the mean, and any one of them is enough:
#   1. The structure-scale split reads focal -0.0585 against regional -0.0585, equal to four decimals.
#      A too-coarse grid predicts the FOCAL findings lose while the regional ones hold; they lose
#      identically, which is a uniformly worse model. `run_comparison` prints its own verdict on that
#      line: "nothing gained at either scale, so resolution was not the bottleneck".
#   2. Resolution is already priced on this data and it is worth nothing. `resolution` at 392 px held at
#      the control's own encoder rate scored 0.8459, +0.0002 (knee-a8x); the +0.0066 that row once
#      claimed was the RATE, not the pixels.
#   3. At MATCHED TRAIN LOSS rather than matched epoch, `base_dense` holds 0.8371 at loss 0.8456 and
#      `cnx_dense` holds 0.7927 at loss 0.8332 — a 0.044 generalisation gap at equal fit. The trunk is
#      not fitting less; it generalises worse from the same fit, and a 14x14 grid does not buy that back.
# A pre-registered clause is a commitment against reading the result to taste, not a commitment to run
# an arm whose mechanism the same result refuted. The clause is honoured by RECORDING that it fired and
# why it is not being followed — which is here and in the ledger — rather than by spending forty minutes
# on a hypothesis three readings already exclude. Encoder diversity closed as an axis with this arm;
# the next member comes from the data side.
#
# The original text is kept below unchanged, because it is what the arm was for.
#
# ARM THIS ONLY IF `cnx_dense` READ OUT UNDER 0.80 SOLO. That is its own third clause, written before it
# ran: "the head, the band and the slice pooling were all settled on token sequences from a patch-14 ViT;
# a 121-token grid may simply be too coarse for the slot head's spatial pooling, in which case the reading
# is about RESOLUTION and the retry is 448 px (a 14x14 grid) before the trunk is judged." If the parent
# instead landed in 0.82-0.85 the axis moves to roster diversity (knee-ewi7), and if its seat value came
# out at or below zero the axis moves off the encoder entirely (knee-ec0p). This script answers one
# question and it is not the question those two ask.
#
# THE ONE VARIABLE. 336 -> 448 is the SAME 336 px cache read through `--train-size`, exactly as the parent
# reached 352, so the decode, the band, the slice count and the pixels on disk are byte-identical between
# the two runs. An upsample adds NO INFORMATION — and that is the point rather than a caveat. It moves the
# token grid from 11x11 to 14x14, 121 tokens to 196, while holding the image content fixed, which isolates
# GRID COARSENESS from pixel detail. A gain here is the slot head wanting more spatial slots; a flat
# result says the trunk was judged fairly at 352 and the family, not the resolution, is what lost.
#
# 14x14 is chosen because it is the ViT incumbents' own grid at 336/patch-24-equivalent — the geometry the
# head, the band and the max slice pooling were all settled on. So this is the ConvNeXt evaluated on the
# spatial budget its competitors were tuned for, not merely a bigger one.
#
# MEMORY IS THE RISK AND IT IS PRICED, NOT ASSUMED — AND THIS ARM NAMES NO CAP. Activations in a
# convolutional trunk live in the early high-resolution stages, so they scale with PIXELS and not with the
# final token count: (448/352)^2 = 1.62, against the parent's measured 10.49 GB reserved, projects ~17 GB.
# `DeviceBudget.needed()` independently derives 17.7 GB for this exact shape, and the agreement is the
# whole reason the flag is absent. The parent DID name one, and its header argued the cap could not price
# a convolutional trunk because the formula read only the final token count — that argument was true when
# it was written and is not any more: `units` now comes from `Encoder.activation_units`, summed over every
# block at the size the run reads, so the cap sees the early stages it used to miss. Naming 24 anyway would
# be the overstatement knee-ylp0 closed, and with a co-tenant on the card an overstated ask buys
# `await_headroom` polling and then a crawl.
#
# `--batch 1 --accumulate 6` holds the effective batch at the control's 6 while halving the resident stack,
# a substitution `small_accum` already measured to be free (0.8579 against batch 6's 0.8598 and the
# control's 0.8578, three points inside a 0.0020 spread). SMOKE IT FIRST anyway — `python -m
# kneemri.experiments.arm_check --smoke` prices the real bill before forty minutes are spent discovering
# it, and 448 is a shape neither the fit nor the parent's smoke has seen.
#
# Everything else is `cnx_dense`'s, which is `base_dense`'s: 4e-5, 20 slices, band 0.10-0.90, whole-stack,
# max slice pooling, 12 epochs one-cycle, the standing $RECIPE.
#
# control  cnx_dense itself — same trunk, same data, same recipe, one geometry apart. base_dense = 0.8512
#          is the ViT this family is competing with, band_dense = 0.8578 the recipe's reference point.
# seat     comp/champion.json basedense_v1 = 0.8690 over the 909-study fold-0 holdout, five members.
# cost     ~1.6x the parent per step at half the batch, so expect the epoch to run longer than 663 s.
#
# THE CONFOUND THIS ARM DOES NOT REMOVE, NAMED BEFORE IT RUNS, because after the fact it is an excuse.
# 4e-5 was settled on DINOv2, so a ConvNeXt inheriting it is a recipe suspect in its own right. This arm
# holds the schedule fixed on purpose, so it separates GRID from PIXELS and separates NEITHER from the
# schedule. A flat result here therefore reads as "resolution is exonerated", never as "the trunk is
# finished": the rate-and-epochs question is a different arm (knee-9otf) and it has to be asked before
# ConvNeXtV2 is judged as a family.
#
# HOW STRONG THAT SUSPECT IS CHANGED WHILE THE PARENT RAN, and the correction is recorded here rather
# than quietly dropped. At epoch 6 the parent was climbing +0.0098 an epoch where the control had
# flattened to +0.0026, which read as UNDERFIT and made knee-9otf the likelier explanation than geometry.
# Two epochs later that reading is wrong: the holdout went +0.0050 then +0.0013 to 0.7921 while the train
# loss kept falling hard, 0.9419 -> 0.9088 -> 0.8833. Loss still dropping and holdout already flat is a
# GENERALISATION gap, not a trunk short of steps — more epochs or a larger rate would buy more of the
# same fit. So the schedule suspect weakens, geometry survives as the live one, and THIS arm is the
# honest next question rather than knee-9otf. Both are still confounded with each other; only the
# ordering changed, on data taken before either arm's number existed.
#
# PRE-REGISTERED, three outcomes. Floor 0.0062, board resolution 0.0015.
#   THE GRID WAS THE PROBLEM — solo clears the parent by more than 0.0062. Then the ConvNeXt was never
#     judged at 352: the slot head needs spatial slots, the trunk is viable, and the family axis reopens
#     at the resolution the incumbents were tuned on.
#   THE GRID WAS NOT THE PROBLEM — solo within the floor of the parent. Then 62% more pixels bought
#     nothing and resolution is exonerated as the explanation. It is NOT a verdict on the trunk: the
#     schedule confound above survives this arm untouched, so what follows is knee-9otf (the rate the
#     ConvNeXt inherited from a ViT), and only a flat reading THERE retires ConvNeXtV2 as a member and
#     sends the next one to the DATA side, knee-ec0p.
#   VRAM — if it will not fit at batch 1, the retry is 392 px (a 12x12 grid, 1.24x the parent's pixels)
#     rather than fewer slices or a narrower band. Slices and band are load-bearing — whole-stack reading
#     is the +0.0147 lever — and the batch is measured not to be.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train cnx_dense_448 \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack \
  --variant cnxbase --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 448 \
  --labels "$LABELS" --out "$PROCESSED/cnx_dense_448_s2026.pt"
