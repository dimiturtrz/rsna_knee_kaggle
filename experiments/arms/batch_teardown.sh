#!/usr/bin/env bash
# batch_teardown.sh — price the BATCH alone, and prove accumulation buys it back.
#
# WHY IT EXISTS. `base_dense` carries a second variable it did not choose: base at 336 px and 20 slices
# asks 56.3 GB at the control's batch of 12 against a 32 GB card, so it runs at 6. A flat-to-down read
# there is therefore confounded — encoder, learning rate and batch all moved at once — and the ledger
# would record a verdict on the encoder that the batch may own. This arm removes the batch from that
# list by measuring it on the encoder where 12 IS affordable, so the number transfers as a correction
# rather than as another suspect.
#
# TWO RUNS, ONE QUESTION EACH, both byte-identical to `band_dense` except where named:
#   small_batch6  --batch 6              prices the BATCH alone against the control's 12.
#   small_accum   --batch 6 --accumulate 2   asks whether accumulation buys the 12 back.
#
# THE SECOND RUN IS THE ORACLE FOR knee-55k8 AT SCALE. The unit test proves two micro-batches of three
# step the optimiser exactly where one batch of six does, on a six-study corpus with augmentation
# bypassed. That is an equality, and it is the strongest statement available on CPU — but it says
# nothing about the part accumulation deliberately does NOT restore: the ranking term forms its pairs
# INSIDE a micro-batch, so `--rank-weight 1.0` at batch 6 sees fifteen pairs a step where batch 12 sees
# sixty-six, accumulation or not. Whether that matters is an empirical question about this loss on this
# corpus, and this is the run that answers it.
#
# THE FALSIFIABLE PART, written before the run. Control band_dense = 0.8578 (best epoch 8). Floor
# 0.0062.
#   * If small_batch6 lands within the floor of the control, the batch is NOT a lever here and
#     `base_dense`'s halved batch is not an alibi for whatever it read — the encoder and the rate are
#     the only remaining suspects, and the follow-on is the rate.
#   * If small_batch6 lands materially BELOW the control, the batch is a real cost, `base_dense` is
#     confounded exactly as its header warns, and base must be re-read at effective 12 before any
#     verdict on the wider encoder is written.
#   * If small_accum lands within the floor of the CONTROL while small_batch6 does not, accumulation
#     buys the batch back at half the activations and the base re-read is unblocked at a cost of time
#     rather than memory. If small_accum instead tracks small_batch6, the pairs are what the batch was
#     buying, accumulation cannot restore them, and the honest fix for the wider encoder is a pair pool
#     that outlives the micro-batch (`--pair-memory`, already built) rather than a bigger step.
#
# WHAT THIS ARM CANNOT SAY. Nothing about the SEAT. Both runs are dinov2-small members of a pool that
# is already seven small members deep, so a tie on the mean is a correlated member and not a vote the
# ensemble lacks. Neither goes to `member_curve` unless the mean surprises.
#
# COST. Two runs at the control's 361 s an epoch, 12 epochs each, one cache build shared — about 2.5 h
# for the pair. The card is released between them (run_train takes it per call), so a queued chain can
# interleave.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  band_dense = 0.8578 (declared 2026-09-03 from this header own prose, knee-sgzgc)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train small_batch6 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --batch 6 \
  --labels "$LABELS" --out "$MODELS/small_batch6_s2026.pt"

run_train small_accum \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --batch 6 --accumulate 2 \
  --labels "$LABELS" --out "$MODELS/small_accum_s2026.pt"
