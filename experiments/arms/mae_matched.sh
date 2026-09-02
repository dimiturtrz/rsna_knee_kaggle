#!/usr/bin/env bash
# Domain-matched MAE pretraining, read as an initialisation for the best recipe on disk.
#
# THE ASSET. pretrain_small_matched_e8 (logs/pretrain_matched_e8.log): DINOv2-small, 8 epochs over 4407
# studies x 20 slices = 402600 slices, masked reconstruction with a pixel term and a spectral term.
# Every encoder on the roster was pretrained on natural images; this one has seen nothing but knee MRI.
#
# WHAT IS ACTUALLY UNDER TEST, NAMED BEFORE THE NUMBER (knee-eoiq). The pixel loss went 0.9859 -> 0.9704
# in two epochs and then sat there: 0.9704 at epoch 2, 0.9708 at 3, 0.9702 at 4. The spectral term fell
# 1.2171 -> 0.4878 over the same span. The objective is being carried ALMOST ENTIRELY by its spectral
# half, so this is not 'reconstruction pretraining' in the plain sense — it is a SPECTRALLY-SHAPED
# INITIALISATION. The obvious wrong verdict here is 'MAE pretraining does not transfer', when what was
# tested is a trunk that never learned to reconstruct pixels at all.
#
# ONE VARIABLE. Byte-identical to band_dense except `--encoder-weights`. Four epochs rather than twelve
# because this arm reads a DIRECTION at the epoch its controls are already recorded at, not a final
# member — both controls below are epoch-3 numbers from full-length runs.
#
# control  adapt_null 0.8240 @ e3 and band_dense 0.8232 @ e3, both already on the ledger, neither re-run.
#          Collapsed arms sit at 0.63-0.73, so the floor that says 'the weights loaded and broke the
#          trunk' is far below the band this lands in — a null here will not be that.
# cost     ~25 min. The pretraining it reads cost ~65 min and is already spent.
#
# PRE-REGISTERED READ, from knee-eoiq, written while epoch 2 of 8 was still running.
#   >= 0.838 (+1.5% over adapt_null): the axis is real, harden it.
#   0.824-0.838: directional at best, and per the house bar not a finding. Do NOT spend a slot on it.
#   < 0.824: the arm failed, and WHICH LINK comes before any verdict — did `--encoder-weights` actually
#            land (assert the trunk's first-block weights differ from the fetched DINOv2, do not assume
#            it), is four epochs enough for a differently-initialised trunk to catch one that starts
#            pretrained, and is the pixel-flatness above the whole explanation.
# WHAT A NULL DOES NOT RULE OUT: a pixel-weighted or longer MAE, and volume-native pretraining, which is
# a different asset entirely (knee-rpkw, armed separately).
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
PRETRAINED="$MODELS/pretrain_small_matched_e8.pt"
require "$LABELS" "$PRETRAINED"

run_train mae_matched \
  $RECIPE --epochs 4 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --encoder-weights "$PRETRAINED" \
  --labels "$LABELS" --out "$MODELS/mae_matched_s2026.pt"
