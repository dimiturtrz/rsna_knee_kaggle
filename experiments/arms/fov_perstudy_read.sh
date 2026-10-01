#!/usr/bin/env bash
# fov_perstudy_read.sh — the same reading as `fovjitter160_read.sh`, kept this time.
#
# WHY IT IS RUN TWICE. `foreign_readout` scored the two shifted passes and threw the probabilities away,
# so the eleven-member toll table and the `fovjitter160` reading are means with no subset in them. An AUC
# is a POPULATION statistic — it has no per-study value — so every question of the form "does this
# member's toll concentrate on the studies where X" was, until commit `1e9f02ad`, another readout on the
# card with the encoders reloaded. `--per-study` writes both shifted passes, the study ids and the truth
# once; every later split is then arithmetic.
#
# THE QUESTION IT IS BOUGHT FOR is D269 (c), the padding composition: wide fixed field -> pad-filled
# borders -> pad learned as a site cue -> transfer toll. Each component is published separately (Weng
# arXiv:2403.06748 [SEG], Alsallakh arXiv:2010.02178, Islam arXiv:2101.12322, Gichoya Lancet Digit Health
# 2022) and the composition has never been priced on a classifier transfer endpoint. `crop_coverage` reads
# only ~40% of series wide enough to fill a 160 mm field, so 60% of the wide members' studies are padded.
#
#   EXTENT SURVIVES  — the wide members' advantage is concentrated in the studies that FILL the field.
#                      Then width is the mechanism and the pad is a tax on the rest.
#   EXTENT FAILS     — the advantage is flat across filled and padded studies. Then the reading charged
#                      to extent in the ledger is wrong, whatever the pooled -0.0265 says.
#   PADDING BITES    — the padded studies carry a WORSE toll than the 130 mm member's on the same studies.
#                      That is the composed failure mode, measured, and it is a training-side lever.
#
# THREE MEMBERS, AND IT WAS FOUR UNTIL 2026-09-09. The fourth was `fovjitter160_reflect`, which differs
# from `fovjitter160` in the pad RULE alone, so one file would have answered the pad question twice —
# once by subset within a member, once by rule across two members. That arm was killed at epoch 6 of 12
# and its checkpoint does not exist, so this arm asked `torch.load` for a file that was never written and
# died before it read a single study, leaving `PERSTUDY DONE` in its watcher's log and no `.npz` on disk.
# The BY-RULE half is therefore withdrawn, not answered: it returns if and only if the reflect arm is
# re-run to completion. The BY-SUBSET half — the one D269 (c) actually turns on — needs only the members
# below and is unaffected. The two 2026-09-09 anchors ride along again; if either moves off its recorded
# cost the ruler moved and none of it reads.
#
# THE DEVICE IS THE CARD, never cpu: knee-j99t.
#
# bead: knee-10f10
# reading: fov_perstudy.npz
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

UV_NO_SYNC=1 uv run python -m kneemri.evaluation.foreign_readout \
  --checkpoints \
    "$MODELS/fovjitter160_s2026.pt" \
    "$MODELS/crop160_span_s2026.pt" \
    "$MODELS/band_dense_hold_s2026.pt" \
  --labels "$LABELS" --foreign D:/data/volumetric/mri/processed/raptor/raptor-corpus --device cuda \
  --per-study "$MODELS/fov_perstudy.npz"
