#!/usr/bin/env bash
# sided_tissue.sh — `sided_pool` with the crop taken about the TISSUE instead of the image. One flag,
# and it is the suspect that arm's own ledger row named and did not clear.
#
# WHAT THIS IS NOT. Not a retry of a refuted idea, and not a second draw on the same experiment.
# `sided_pool` read 0.8561 against `band_dense` 0.8578, landed in its MIDDLE pre-registered band, and its
# three side-defined targets moved by a mean of -0.0004 with every interval spanning zero. That row was
# written carefully: it verified the mechanism FIRED (the saved config carries `'pool': 'sided'` and
# `head.proj.1.weight` is (256, 1152) = 384x3 parts against `cls_mean`'s 768 = 384x2, so the head was
# built three-part and did see the split), and it then named ONE alternative to "the idea is wrong":
#
#   "The config reads 'centre': 'image', crop_mm 130. The pooling splits the patch grid at its centre
#    COLUMN, which is the anatomical midline only if the crop is anatomically centred. An image-centred
#    crop leaves the true midline off-column by a study-varying offset, which smears the halves into each
#    other. This is a live alternative to 'the idea is wrong' and it is cheap to test."
#
# This is that test. `--centre tissue` moves the crop box onto the tissue centroid, so the centre column
# of the patch grid IS the anatomical midline, which is the assumption the pooling rests on and the one
# thing `sided_pool` did not supply.
#
# THE CONFOUND IS REAL BUT PARTIAL, AND THE SIZE IS MEASURED — this arm is not sold as a rescue.
# `core/data/corpus/crop_centre.py` measured the tissue centroid against the image centre over 60 studies:
# median 7 mm, p90 18 mm, max 32 mm. Against a 130 mm crop the half-width is 65 mm, so the median study's
# midline is off by 11% of a half and the p90 study by 28%. That blurs the halves for the worst tenth; it
# does not destroy the split for the median. So if the sided contrast were LARGE, a 7 mm median offset
# could not have flattened it to -0.0004 across three labels. The honest prediction is a PARTIAL recovery
# of a small effect, not the appearance of a large one.
#
# WHY IT RUNS ANYWAY, priced out loud. ~75 min of card and ZERO submission slots, on a card that is
# otherwise idle. The pot it aims at is the largest single lever written down in this repo and the only
# one that clears the owner's 1.5% bar: `2026-08-14_the-ruler-was-five-times-too-small.md` measures the
# four weakest labels over the whole 4,407-study corpus — Lateral Meniscus 0.7981, PF OA 0.8165, MCL
# 0.8166, Lateral OA 0.8215 — and states that lifting them to 0.87, a level two other findings already
# reach, is worth +0.0189 on the twelve-label mean. Three of those four are side-defined, which is exactly
# the set this pooling addresses. Nothing else in the inventory aims at that pot without new data.
#
# AND IT IS THE RIGHT SHAPE OF INTERVENTION, which `dark6` is what makes decisive. dark6 read 0.937 on
# the board with six all-data members and NOTHING ELSE, tying `allfit` and `lastseat` which are those six
# plus a seventh: the marginal seventh member is worth zero. Members are saturated, and their per-label
# profiles correlate +0.92 within a fold, so every member is weak on the SAME labels and no vote over them
# can fix it. A deficit shared by every member is not reachable by ensembling; it has to be trained out.
# This arm changes what the head can SEE, which is a change to what is optimised rather than to how
# features are spelled — the distinction the architecture-closure note turns on.
#
# WHAT IS DELIBERATELY NOT TRIED HERE. `sided_pool`'s row also left standing the small-weight side
# REGULARISER that `side_dense` deferred to "a head that sees position". That is loss-engineering aimed at
# a named weakness, which project policy forbids: needs get fixed with balanced data and with what the
# model can see, not by bending the objective toward a label we are behind on.
#
# FOLD 0, not `--fold -1`, and that is on purpose. This is a MECHANISM read and it needs per-label
# columns; an all-data member is unscoreable forever and could answer nothing.
#
# PRE-REGISTERED against `sided_pool` = 0.8561 and `band_dense` = 0.8578 on the 909-study fold-0 holdout.
# Read the THREE SIDE-DEFINED COLUMNS first in every band; the mean over twelve can absorb a real move on
# three, and that instruction is what kept `sided_pool`'s own row honest.
#   >= 0.8640   the midline WAS the bottleneck and the split carries once it is anatomical. Go straight to
#               a finer grid (quadrants, or compartment columns rather than halves) and convert this to a
#               `--fold -1` member.
#   0.8580-0.8639  the mean recovers to `band_dense` or past it. Read the side-defined columns: if they
#               carry the move, the mechanism is real and small, and the finer grid is licensed. If they
#               do not, the gain is elsewhere and belongs to `--centre tissue`, not to the pooling.
#   0.8500-0.8579  flat against `sided_pool`. Anatomical centring does not recover the contrast, and with
#               the crop confound removed the head-sees-position idea is refuted AT HALVES on this data.
#               A finer grid is then a different bet on the same premise and needs a new argument, not a
#               rerun. This is the outcome the 7 mm median makes most likely.
#   <= 0.8499   worse than both parents, which no single-flag crop change should produce. Suspect the
#               tissue centroid on THIS band first: `--slices 20 --band-low 0.1 --band-high 0.9` reaches
#               end-of-stack slices that hold little tissue, and `TissueCentre` takes a median over slices
#               precisely because one such plane is unreliable. Assert the centring fired before blaming
#               anything.
# bead: knee-86vi
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train sided_tissue \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --pool sided --centre tissue \
  --labels "$LABELS" --out "$PROCESSED/sided_tissue_s2026.pt"
