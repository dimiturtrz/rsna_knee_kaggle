#!/usr/bin/env bash
# distill_v3base.sh — the same teacher, a DIFFERENT ENCODER FAMILY, because order is architecture-agnostic.
#
# GATED. Do not queue this until `distill_pairs` has read. Its own pre-registration says which branch
# licenses this arm: >= 0.860, or 0.850-0.860 with a RARITY-ORDERED per-label profile. A flat profile at or
# below its control does NOT license it, and 2.3 h buying a second measurement of a null is the exact trade
# the pricing rule exists to refuse.
#
# WHAT IS HELD AND WHAT MOVES. This is `v3base_dense`'s invocation with the teacher term added and nothing
# else touched — same variant, rate, batch, slices, band, pooling, whole-stack, epochs and schedule. So the
# control is `v3base_dense` at 0.8455 (peak e6, 2026-08-21) and it costs ZERO extra card: the control is
# already in the ledger. The single variable is `--distill-weight`.
#
# WHY THE FAR FAMILY IS WORTH THE 2.3 h. A teacher supplies ORDER, and an ordering is a property of the
# corpus rather than of the architecture that produced it — which is the whole reason distillation is worth
# more here than a knob. The champion is seven members of ONE family, and the only mechanism still live at
# the top of this board is the seventh member's TYPE OF DISTANCE. A student that is both taught the pool's
# own ordering AND built on the far encoder is the only construction this project has that could be strong
# and structurally unlike the pool at the same time.
#
# WHAT IS ALREADY SETTLED, so this does not re-litigate it. The FAMILY alone is worth nothing: v3base_dense
# 0.8455 against base_dense 0.8512 is -0.0057, inside the 0.0062 floor. And the roster's two weak DINOv3
# members (0.8215, 0.8268) were MIS-ATTRIBUTED for five days — that was a ~0.019 LoRA-over-frozen-trunk
# penalty measured on the one encoder where both regimes exist, not a statement about DINOv3. The family is
# a live axis at FULL ADAPTATION and only there, which is the regime this arm runs in.
#
# THE FALSIFIABLE PART, against v3base_dense's 0.8455 — and read the DELTA beside distill_pairs' delta,
# because two students, two families and one teacher is a TRANSFER measurement, not two solo numbers.
#   * >= 0.8598 — beats the best solo member this project has ever fitted, on the far encoder. Seat it; the
#     pool stops being one family, which is the axis the board has been asking for since allfit.
#   * 0.8455-0.8598 — the term transfers across families. If its delta matches distill_pairs' delta, what
#     the teacher carries is ORDER and nothing family-specific; that is the reading that generalises.
#   * <= 0.8455 — the interesting negative: the term is family-specific, so what the teacher hands over is
#     not order but something the DINOv2 student's own family already had. File it as a property of the
#     teacher, not as a verdict on distillation.
#
# COST. ~2.3 h of card (~680 s/epoch x 12). The teacher file is already written and is reused at zero.
# ZERO submission slots.
#
# bead: knee-1x4j
# control  v3base_dense = 0.8455 (peak e6), byte-identical recipe with no teacher.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

# 0.3 is distill_pairs' weight, held rather than re-derived: a weight swept on a second encoder before the
# term is shown to transfer at all confounds the family question with the operating-point question.
run_train distill_v3base \
  $RECIPE --epochs 12 --batch 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3base --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/teacher_fold0.npz" \
  --labels "$LABELS" --out "$PROCESSED/distill_v3base_s2026.pt"
