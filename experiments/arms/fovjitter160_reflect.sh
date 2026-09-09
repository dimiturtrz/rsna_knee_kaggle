#!/usr/bin/env bash
# fovjitter160_reflect.sh — fovjitter160 with the OUT-OF-FRAME RULE moved. One flag, one axis.
#
# WHY THIS ARM EXISTS. `fovjitter160` widens the extent jitter to +/-20% on a 160 mm cache, and above
# scale 1 the sampling grid leaves that cache. What it reads there was decided by `_warp` and never
# argued: `padding_mode="border"`, edge replication, hardcoded until knee-10f10. Its stated reason was
# ROTATION — a turned knee should run out into tissue-like values rather than a black wedge the encoder
# can use to infer how far the image was turned. That is sound for an 8-degree rotation, which leaves a
# thin corner out of frame. It was never an argument about EXTENT: at scale 1.2 the whole margin is out
# of frame and a zoom-OUT fabricates tissue by smearing one edge row across it.
#
# So `fovjitter160` moves TWO things at once even though it moves one flag: how much extent varies, and
# how much fabricated margin the encoder is fed. This arm holds the first and moves the second. It is
# the disambiguation, not a second attempt at the axis.
#
# WHAT THE LITERATURE SAYS (D258, 2026-09-09). It declines this choice rather than making it, which is
# itself the finding: MAE and DINO crop INSIDE the image (RandomResizedCrop never leaves the frame);
# nnU-Net oversamples its patch by the scale envelope so a zoomed-out sample is real tissue; Chen/Bai
# 2020 states its crop runs "without black borders". torchvision and MONAI pad with zeros, TorchIO with
# an Otsu background estimate. NO medical-imaging paper reports reflection beating crop-inside. This is
# therefore a first read, not a replication, and reflection is the CHEAP APPROXIMATION of the thing the
# field actually does — the real alternative is crop-from-a-larger-field, which needs a wider cache and
# is the follow-up if this reads.
#
# WHY REFLECTION AND NOT ZEROS. Zeros reintroduces exactly the black wedge the rotation argument was
# about, on every zoom-out draw, and 60% of our series are already padded to fill the frame before any
# jitter runs (`crop_coverage`: image@160 over 733 of 1822 series) — a zeros margin would be
# indistinguishable from that existing padding and would teach the encoder that the two are one thing.
# Reflection mirrors real tissue outward: the texture statistics stay in-distribution and the anatomy is
# wrong, which is the same trade every geometric augmentation makes.
#
# control  fovjitter160_s2026 — same cache, same recipe, same seed, same 0.20 jitter, one flag moved.
#          A matched control that costs no card: it is on disk by the time this runs.
# cost     12 epochs at 420 px, ~620 s an epoch = ~2 h, the measured crop160_span/fovjitter160 figure.
# price    ~2 h to separate an axis from its confound. Not sold as an improvement — it is an INSTRUMENT,
#          and it is the one that decides whether the wider jitter D258 points at (nnU-Net's +/-30%) is
#          worth a slot or is fabricating half its draws. Running the wide arm without it prices two
#          things as one for a second time.
#
# PRE-REGISTERED ON THE FOREIGN NUMBER, read with `foreign_readout` against the same eleven-member pool,
# on the card (knee-j99t: cpu is not the ruler). The comparison is against `fovjitter160`'s decode_cost,
# NOT against the pool mean — this arm's subject is the difference between two padding rules at one
# jitter width.
#   PADDING CARRIES  — decode_cost at least 0.0038 better than fovjitter160's (the pool's own sd, the
#                      smallest difference this ruler has ever resolved). The out-of-frame rule is a real
#                      lever, reflection becomes the default for any widened jitter, and the wide-jitter
#                      arm runs under it.
#   NULL             — within +/-0.0038 of fovjitter160. The rule does not matter at this width; border
#                      stays (it is what every member on disk carries, so keeping it costs nothing and
#                      changes no ledger meaning), and the wide arm may run without re-deciding this.
#                      This is the branch the literature's silence predicts.
#   PADDING COSTS    — 0.0038 or more WORSE. Edge replication was doing something reflection is not.
#                      The likely mechanism is that a replicated row is closer to the fill our own padded
#                      60% already carries, so border keeps the two paddings consistent while reflection
#                      introduces a third texture. Border stays, and the wide-jitter arm inherits the
#                      question of whether it is fabricating at all — the answer is then the wider CACHE,
#                      not a wider rule.
# NATIVE IS NOT THE GATE, for the same reason `fovjitter160` gives: finding 3 of the 11-member readout
# says the two numbers are uncorrelated (rho 0.1727, p 0.61).
#
# WHAT THIS ARM CANNOT SETTLE. It prices the rule at ONE width (0.20) on ONE cache (160 mm). A rule that
# is null at 0.20 may not be null at 0.30, where more of every draw is fabricated — the effect is
# monotone in how much margin leaves the frame, and this arm measures the bottom of that range. It also
# cannot separate reflection from crop-from-a-larger-field, which is what the literature actually does;
# reflection is the approximation that needs no new decode.
#
# bead: knee-10f10
# reading: fovjitter160_reflect_s2026.holdout.npz
#
# deliberate: --padding-mode (reflection -- THE MOVED FLAG, against the border every member on disk and
#                             every ledger number carries)
# deliberate: --scale-jitter (0.20 -- fovjitter160's, HELD; moving it would price two things as one,
#                             which is the defect this arm exists to retire)
# deliberate: --crop-mm      (160 -- held with the jitter, same cache as the control)
# deliberate: --image-size   (420 -- carried by --crop-mm to hold mm/px)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train fovjitter160_reflect \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --crop-mm 160 --image-size 420 --scale-jitter 0.20 --padding-mode reflection \
  --labels "$LABELS" --out "$MODELS/fovjitter160_reflect_s2026.pt"
