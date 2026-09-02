#!/usr/bin/env bash
# Read 80 mm about the tissue instead of 130 mm about the image centre — the gating arm for knee-1nn.
#
# THE HYPOTHESIS, and it is falsifiable in a specific shape rather than "does it score higher".
# Measured against radiologist annotations on the 58 annotated studies (knee-dtk), the model BEATS its
# own labels on the diffuse findings and loses badly on the small ones:
#
#   Effusion +0.13, Contusion +0.10, Medial OA, Synovitis, Fracture, Baker's   -- label-bottlenecked
#   Lateral Meniscus -0.19, ACL -0.12, MCL -0.10, Medial Meniscus -0.09        -- model-bottlenecked
#
# That split is by structure SIZE. The geometry says why: the field of view is about 160 mm, we crop
# 130, so we remove 19% of the linear extent and spend two thirds of every image outside the joint at
# 0.387 mm/px — while a meniscal tear is 1-2 mm and DINOv2 patches at 14 px, so a tear is a fraction of
# ONE TOKEN. An 80 mm crop at the same 336 px is 0.238 mm/px: 1.6x finer for identical compute.
#
# SO THE PREDICTION IS DIRECTIONAL AND TWO-SIDED. Small structures up, diffuse findings DOWN — Baker's
# is posterior and effusion spreads, so a tighter window should cost them. A uniform lift is a
# different result and probably a confound; a uniform fall refutes the resolution mechanism outright
# and kills the multi-scale build for one arm's cost, which is the point of running this first.
#
# WHY IT CARRIES THE TISSUE CENTRE TOO (knee-yrf). At 130 mm a half-extent of 65 mm leaves ~15 mm of
# margin and the measured p90 positioning error is 18 mm, so the image centre is survivable. At 80 mm
# the half-extent is 40 mm against a joint some 70-80 mm wide, and an 18 mm error cuts into it — so an
# 80 mm crop about the IMAGE centre would test "a tight crop that sometimes misses the joint" and would
# fail for a reason that has nothing to do with resolution.
#
# THE COUPLING IS STATED RATHER THAN HIDDEN: this arm changes two things, so if it fails both are
# suspects. centre_tissue.sh isolates the centre alone at 130 mm against the existing baseline, and is
# the disambiguation to run if this one disappoints.
#
# control  the 130 mm image-centred members on the same fold and labels. Read PER LABEL, never on the
#          mean alone — the mean averages the two halves of the prediction against each other.
#
# NEEDS ITS OWN CACHE, which no other arm builds:
#   experiments/arms/build_cache.sh --image-size 336 --slices 12 --crop-mm 80 --centre tissue
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train crop80_tissue \
  $RECIPE --epochs 8 \
  --slice-pooling max --whole-stack --recompute-activations \
  --crop-mm 80 --centre tissue \
  --labels "$LABELS" --out "$MODELS/crop80_tissue_s2026.pt"
