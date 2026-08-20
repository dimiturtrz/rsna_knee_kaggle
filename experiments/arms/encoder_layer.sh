#!/usr/bin/env bash
# encoder_layer.sh [layer] — read an intermediate block instead of the last one.
#
# NOT A BUG FIX, and worth saying because of how it came up. `last_hidden_state` is what the API
# returns and what linear probes conventionally use; nothing in the code claimed otherwise. This is an
# axis nobody has varied, for either encoder, in the whole project.
#
# WHY IT MIGHT MATTER HERE. The final block of a self-supervised ViT is the most specialised toward its
# PRETRAINING objective - which is why both DINOv2 and DINOv3 ship `get_intermediate_layers` for dense
# work. Our findings are 1-3 mm and our head pools PATCH tokens, so a representation tuned for a global
# image-level objective is not obviously the one they live in.
#
# ON DINOv2, WHICH IS THE POINT. It surfaced while asking why DINOv3 underperforms, and DINOv3's Gram
# anchoring makes it a plausible v3-specific story - but DINOv2 is the encoder we actually ship, and a
# gain there is worth more than a gain on a member we may never submit.
#
# LAYER 9 OF 12 by default: far enough back to escape the last blocks' specialisation, close enough to
# keep the abstraction the head was designed around. hidden_states is 13 long (the embedding output
# plus twelve blocks), so 9 is the ninth block rather than an off-by-one into the middle.
#
# COSTS NOTHING EXTRA: output_hidden_states returns every block from the same forward pass.
#
# control  wsfold0_e6 = 0.8457, identical recipe reading the last block.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
LAYER="${1:-9}"
require "$LABELS"

run_train "encoder_layer_${LAYER}" \
  $RECIPE --epochs 12 $PATIENCE \
  --slice-pooling max --whole-stack --encoder-layer "$LAYER" \
  $THROUGHPUT \
  --labels "$LABELS" --out "$PROCESSED/encoder_layer_${LAYER}_s2026.pt"
