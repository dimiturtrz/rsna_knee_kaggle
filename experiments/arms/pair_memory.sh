#!/usr/bin/env bash
# pair_memory.sh — let a ranking pair span steps, because the metric ranks the whole corpus.
#
# THE MISMATCH. The competition score is one ROC AUC per label over the ENTIRE test set: every positive
# study against every negative one, ~4,400 studies deep. `--rank-weight` optimises a local version of
# exactly that statistic — but with a batch of 12 it only ever forms 66 unordered pairs, all drawn from
# twelve neighbours the sampler happened to put together. A model can order every batch it is shown
# perfectly and still order the corpus badly, because nothing in the objective ever compares two studies
# that were never batched together.
#
# And the batch size is not a statistical choice. Twelve studies x 6 slots x 12 slices = 864 images
# through the encoder already fills a 26 GB budget, so the pair pool is currently set by the card.
#
# THE MECHANISM. Keep the last 8 steps' readouts — detached, three tensors of 144 floats each, against
# activations measured in gigabytes — and form the cross-step pairs too. 96 extra studies on the other
# side of every comparison is roughly 17x the pair mass, and the pairs grow with the MEMORY, which costs
# kilobytes, rather than with the batch, which costs gigabytes. Gradient still flows only through the
# live step; the remembered side is a fixed target, as in a contrastive memory queue (MoCo, He et al.
# 2020 — same trick, applied to a ranking term instead of a contrastive one).
#
# WHY IT MIGHT DO NOTHING, stated first: the encoder moves while a readout sits in the queue, so a
# remembered logit is judged against weights that no longer exist. Eight steps is under 3% of an epoch's
# 291 and the encoder runs at 8e-6, which is the argument that the drift is small — not a proof.
#
# THE FALSIFIABLE PART: if this works it works by ORDERING, so the gain shows in AUC without the loss
# necessarily falling — a lower training loss with a flat AUC means it found an easier objective, not a
# better ranking. And it should be LARGEST on the rare labels (Fracture 314 positives, Synovitis 580),
# whose separating pairs are the ones a batch of twelve most often fails to contain at all.
#
# control  wsfold0_e6 = 0.8457 — same recipe, same 12 epochs, memory off (batch-local pairs only).
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train pair_memory \
  $RECIPE --epochs 12 \
  --slice-pooling max --whole-stack --pair-memory 8 \
  $THROUGHPUT \
  --labels "$LABELS" --out "$MODELS/pair_memory_s2026.pt"
