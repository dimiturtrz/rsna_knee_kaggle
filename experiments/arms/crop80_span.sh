#!/usr/bin/env bash
# THE HALF OF knee-nas5 THAT THE BAND LINE DID NOT SERVE. That bead says the all-ours pool has no
# crop/band/centre diversity on the one axis measured to transfer at 2.4x, and six band members later
# the band half is answered while the crop half is exactly where it was: crop80_tissue (80 mm, tissue
# centre) is the only member in the whole pool off 130 mm / image centre, and it is not seated.
#
# THE MECHANISM is that these two decode axes should compose the way the span x density square did,
# and for a reason the square supplies. Span is COVERAGE ALONG the stack and it buys regional findings
# (+0.0079 against focal +0.0011). The crop is coverage ACROSS the slice, and a tighter one spends the
# same 336 px on less field of view, which is resolution on the anatomy rather than on the air - the
# opposite trade, on the perpendicular axis. If they are independent the way slices and span were,
# this reads the two gains together on a pixel group no seated member occupies.
#
# THE CONTROL IS band_span (0.8565), not crop80_tissue, and the arm is not expected to beat it. The
# only 80 mm member the project has scored 0.8345 against a `whole_stack` control at 0.8457 - the
# tighter crop COST 0.0112 on the mean - and `centre_tissue` since established that the loss is the
# resolution change rather than the centring. So the honest prediction is a mean BELOW its control by
# about that much: 0.8420 to 0.8500, i.e. band_span's 0.8565 less the crop's own measured cost, plus
# the floor's width. An arm predicted to lose is worth the slot only for the reason below.
#
# THE PREDICTION THAT MATTERS is therefore about the SPAN, not the mean: the regional group that span
# buys at 130 mm (ACL, Contusion, Baker's, Effusion) reproduces at 80 mm, positive against a 130 mm
# control's own regional deltas, and the seat value against comp/champion.json is POSITIVE despite the
# lower mean - which is what a member on an unoccupied decode group is for.
#
# THE REFUTATION CONDITION is the regional group coming out flat or negative here. Span reproduced at
# both slice counts, so a span that fails to reproduce at a second crop would mean the gain was never
# coverage of anatomy but coverage of the 130 mm FIELD - the outer tenths paying only because the wide
# crop lets them reach something a tight crop already holds - and the two axes are substitutes rather
# than complements, which retires stacking them and with it this arm's whole reason to exist.
#
# It is the cheaper of the two candidates on the queue in card time and the dearer in disk: the cache
# 336px_12sl_80mm_tissue_0.1-0.9band did not exist and is decoded alongside band_full's training.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train crop80_span \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT --vram-gb 26 \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 --crop-mm 80 --centre tissue \
  --labels "$LABELS" --out "$PROCESSED/crop80_span_s2026.pt"
