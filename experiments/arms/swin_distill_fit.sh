#!/usr/bin/env bash
# swin_distill_fit.sh — stage two: the labelled fit, warm from the distilled trunk. THIS is the reading.
#
# ONE VARIABLE against a control already on disk. `swin_supervised` is this invocation with
# `--encoder-weights` removed: 12 epochs, batch 1 / accumulate 6, max pooling, whole-stack, swinbase,
# 384 px, `--backbone-lr 1e-4`, 20 slices, band 0.1-0.9, vram-gb 8, the same $RECIPE and $THROUGHPUT. It
# scored 0.8331 solo. So the only difference between the two runs is which weights the trunk started
# from, which is exactly knee-8009's question and nothing else.
#
# THE EXPORT IS PART OF THE ARM, not a step to remember. `fitted_encoder` converts stage one's member
# checkpoint into the file `--encoder-weights` reads, and it is run HERE rather than by hand because a
# hand-run conversion is how a warm arm silently becomes a cold one: `PretrainedEncoder.into` refuses a
# variant or key mismatch, but nothing refuses an encoder file that is simply the PREVIOUS export sitting
# at the same path. It costs seconds and no card.
#
# PRE-REGISTERED, against swin_supervised's 0.8331. Floor 0.0062, board resolution 0.0015. P(clears the
# floor) ~0.20 — stated low, because two prior distillation arms read +0.0016 and +0.0035 and nothing
# licenses expecting more from a third arrangement of the same teacher.
#   THE CHANNEL IS LIVE — >= +0.0062 solo. A teacher-dominant PRETRAINING stage does what a mixed-in
#     teacher term did not, which licenses knee-8009 (b): feature distillation, label-free, the route
#     external unlabelled data can be used on. Next arm is (b) on this same student.
#   DIRECTIONAL — +0.0015 to +0.0061. Keep the member, report it as directional and never as a finding,
#     and do NOT spend a slot on it. (b) stays unlicensed: a gain this size is what warm starting alone
#     has bought elsewhere in this repo, and warm starting is not the varied term here only because the
#     control is cold.
#   FLAT — inside +/-0.0015. Ordering-over-twelve-targets does not survive a stage boundary. That does
#     NOT refute (b), for the reason stage one's header states in advance: twelve logits are a lossy
#     channel for a REPRESENTATION claim, so a null here is silence about feature distillation rather
#     than evidence against it. Record it and close (a).
#   SUSPECT THE HARNESS FIRST — <= -0.0062. Before writing anything: check stage one recorded
#     `distill_weight: 20.0` and a non-null `teacher_targets`, and that THIS run recorded a non-null
#     `encoder_weights` and did not silently start cold. Only after both come back clean does a negative
#     mean a teacher-shaped trunk fights the labels.
#
# AND THE SEAT IS READ WITH THE SCORE OR NOT AT ALL. `swin_supervised` is the most decorrelated member
# this project owns (nearest rank correlation 0.8346 against a seated band of 0.8772-0.9225). A distilled
# trunk that gains solo score while collapsing toward the pool has spent the exact property that made
# this lane open, and a report quoting only the gain would hide that. `run_train` prints the seat value;
# read `member_axes` beside it and quote both.
#
# WHAT NO BRANCH LICENSES. Nothing here says anything about the distillation WEIGHT (20 is the stage-one
# knob, held, never swept before the axis is shown to transfer), about warm starting in general (the
# control is cold BY DESIGN, so this arm cannot separate "distilled trunk" from "warm start" — a positive
# reading owes a follow-up warm-from-swin_supervised control before it is called a distillation result),
# or about any trunk that is not swinbase.
#
# cost   12 epochs at 384 px = ~90 min, plus seconds for the export. Zero slots.
#
# bead: knee-8009
# bead: knee-7ybf
# control  swin_supervised_s2026 = 0.8331 — this invocation without --encoder-weights, already on disk.
# deliberate: --encoder-weights  (the distilled trunk — the ONE varied term)
# deliberate: --variant
# deliberate: --train-size       (384, pinned by the window tables — see swin_supervised.sh)
# deliberate: --backbone-lr
# deliberate: --batch
# deliberate: --accumulate
# deliberate: --recompute-activations
# deliberate: --vram-gb
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$MODELS/swin_distill_pre_s2026.pt"

uv run --no-sync --project "$REPO" python -m kneemri.training.fitted_encoder \
  "$MODELS/swin_distill_pre_s2026.pt" "$MODELS/encoders/swin_distill_pre.enc.pt" || exit 1

run_train swin_distill_fit \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant swinbase --backbone-lr 1e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 384 \
  --encoder-weights "$MODELS/encoders/swin_distill_pre.enc.pt" \
  --vram-gb 8 \
  --labels "$LABELS" --out "$MODELS/swin_distill_fit_s2026.pt"
