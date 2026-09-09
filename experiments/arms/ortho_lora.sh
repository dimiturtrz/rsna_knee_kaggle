#!/usr/bin/env bash
# ortho_lora.sh — does PEFT beat full fine-tuning on the LARGEST trunk we own, at 4.4k studies?
#
# WHY IT EXISTS, and it is a published mechanism rather than a hunch. D273 (2026-09-09) answered M275's
# third ask with the one measured crossover on corpus size in the literature: Dutt et al.
# (arXiv:2305.08252, 600+ controlled experiments) find that on HAM10000 full fine-tuning is optimal at
# 100% of the data, but the crossover sits at ~50% ≈ **3.8k images — our scale** — below which LoRA beats
# full FT by up to ~6% F1, and **the gain grows with trunk size** (~4% at ViT-B, ~6% at ViT-H). The rest
# of D273 is corroboration that capacity is not paying at our corpus size; this is the only line in it
# that names an intervention, and the intervention is not the one our large-trunk rows were run under.
#
# WHAT OUR OWN RECORD SAYS, INCLUDING THE PART THAT ARGUES AGAINST THIS ARM. On a v3base trunk we have
# the pair already and it points the OTHER way: `base_dense` (full fine-tune, 4e-5) 0.8512 against
# `dinov3_base_lora` (rank-8 adapters over a frozen trunk, 5e-4) 0.8268. Full FT won by a wide margin, and
# on that reading Dutt does not transfer. But the ledger flags that pair as CONFOUNDED at :3040 — the two
# arms differ on adaptation AND on 224 -> 336 plus the dense recipe, and the whole delta was charged to
# adaptation — and it names the missing cell in as many words: nobody has run LoRA against a matched
# control. So the base pair is not evidence about PEFT; it is evidence about two things at once.
#
# THIS ARM IS THAT MISSING CELL, MOVED TO THE CAPACITY WHERE DUTT PREDICTS THE LARGEST GAIN. It copies
# `ortho_dense` flag for flag — same trunk, 12 slices, 224 px, band 0.2/0.8, whole stack, max pooling,
# 12 epochs — and changes the adaptation and nothing else.
#
# THE RATE IS A CONFOUND THAT CANNOT BE REMOVED, and `lora.sh` already argues it: a frozen-base run and a
# fine-tuned run cannot share a backbone rate, because the parameter being rated is not the same
# parameter. The adapters are zero-initialised, so 5e-4 is a from-scratch peak under OneCycle, not a
# nudge; the control's 4e-5 is a nudge to pretrained weights. Every LoRA arm this project owns ran 5e-4,
# so holding it is what keeps this row comparable to `dinov3_base_lora` and `lora_r8_v3small_224`.
#
# RANK 8 IS HELD FOR COMPARABILITY AND IT IS THE ARM'S WEAKEST POINT, stated before it runs. ViT-L is
# 1024-wide against base's 768, so rank 8 is a SMALLER relative subspace here than in the two rows this
# is meant to sit beside. If the mechanism is real but the rank is too small, the curve sits flat and low
# at every epoch — which reads exactly like a refutation and is not one. M276 has gone to the researcher
# asking for Dutt's own rank, alpha and injection sites; if D276 comes back with a rank tied to corpus
# size or to width, a rank-16 re-run is the follow-up and this row is its control rather than its verdict.
#
# THE PRICE, AND THE ESTIMATE BELOW WAS MEASURED WRONG. `ortho_dense` measured 409 s/epoch, 1.36 h for
# twelve; this arm was budgeted at ~1.6-1.8 h on the reasoning that adapting all 24 blocks keeps more
# activations than fine-tuning six. Epoch 1 measured **713 s**, 1.74x the control, so twelve epochs cost
# **2.38 h** and not 1.8. The missing term is `--accumulate 3`: at batch 4 the arm takes three times as
# many optimiser micro-steps per effective batch as the control does at 12, and the per-step overhead is
# paid on all 24 adapted blocks. A memory fix bought back its VRAM in wall-clock, and the price of that
# trade belongs in the arm that made it. GAIN AS A MEMBER: the control is 0.8452 and the
# seat bar is 0.8521, so a member needs +0.007 over the control — inside what Dutt's effect size would
# predict if it transfers, and nowhere near it if our base pair was right. P(clears the seat bar) ~ 0.25,
# and it would be an OUT-OF-DISTRIBUTION architecture, which is the only kind of seat that has ever moved
# this board. GAIN AS AN INSTRUMENT: it prices the adaptation suspicion hanging over all four large-trunk
# rows (`ortho_dense` 0.8452, `ortho_stack` 0.8417, `v3large_dense` 0.8258, `v3large_stack` 0.8277) the
# way `band_deep_matched` priced the depth suspicion over the foreign-trunk rows. Either answer is worth
# 1.6 h; only one of them is worth more card after it.
#
# READ IT ON, against control `ortho_dense` = 0.8452 (best epoch, 224 px / 12 slices, 4e-5 over six blocks):
#   * ABOVE 0.8514 — clears the 0.0062 paired floor. PEFT pays on a large trunk at our corpus size, Dutt
#     transfers, and the four large-trunk rows above are readings of OUR SCHEDULE rather than of capacity.
#     Then the family is owed a re-run under adapters before any of those numbers is quoted again, and
#     `v3large` gets the same treatment on its own control.
#   * 0.8390 to 0.8514 — inside the floor. PEFT is neutral on a large trunk. Combined with the base pair
#     that is a statement about rank rather than about PEFT, and D276's rank is what decides whether a
#     rank-16 arm follows or the axis closes.
#   * BELOW 0.8390 — PEFT loses on a large trunk too, in the same direction and roughly the same size as
#     the confounded base pair. Then the base pair's confound did NOT carry its delta, Dutt does not
#     transfer to this endpoint, and the ADAPTATION lever closes on every trunk we own. That is a real
#     closure: it says the large-trunk deficit is capacity or corpus, not schedule, and it sends the whole
#     weight of the search back to the data axis.
#
# WHAT NO BRANCH ABOVE SAYS. None of them is a statement about ortholarge's ceiling — the budget is held
# at 12 epochs on purpose so that the adaptation is the only thing that moved, and the control carries
# whatever budget qualifier it carries into both sides of the delta, where it cancels.
#
# THE CARD REFUSED THE FIRST VERSION OF THIS ARM, and the refusal is a fact about LoRA on a large trunk
# rather than a nuisance. `encoder_surgery.backbone_groups` puts an adapter on EVERY block and freezes
# every base weight -- there is no depth flag under LoRA, by design -- so where the control opens six of
# twenty-four blocks this arm opens all twenty-four, and `retained_units` prices the activations of all
# of them. At the control's batch of 12 the cost model derives **45838 MiB against a 32 GB card**, and
# the smoke refused it at 14:38Z with 30991 free. Measured on the ladder: batch 12 -> 45838, 6 -> 25335,
# 4 -> 18500, 2 -> 11666.
#
# SO THE MICRO-BATCH DROPS TO 4 AND `--accumulate 3` HOLDS THE EFFECTIVE BATCH AT THE CONTROL'S 12.
# Dropping the batch alone would have been the cheap fix and it would have confounded the arm: batch is
# in the optimiser's noise scale, and a LoRA row read against a full-FT control at a third of its batch
# is a reading of two changes. Four rather than six because the card is shared by arrangement with the
# other project -- 18.5 GB leaves it room where 25.3 GB does not, and the difference between the two is
# a few minutes of step overhead on an arm already priced at over an hour.
#
# deliberate: --batch 4 (the card cannot hold the control's 12 under adapters on all 24 blocks — 45838 MiB derived)
# deliberate: --accumulate 3 (holds the effective batch at the control's 12, so batch is not a second change)
# deliberate: --lora-rank 8 (held from every LoRA arm this project owns, so the row is comparable)
# deliberate: --backbone-lr 5e-4 (the LoRA peak; an irremovable confound, argued in lora.sh)
# deliberate: --variant ortholarge --slices 12 --image-size 224 --band-low 0.2 --band-high 0.8 (ortho_dense's read exactly — this is the matched control)
#
# bead: knee-rjuc6
# reading: ortho_lora_s2026.metrics.jsonl
#
# control  ortho_dense = 0.8452 (best epoch, 12 epochs at 409 s)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

# The gated DINOv3 weights need a token; ortholarge continues from DINOv3-L. Read into the environment,
# never echoed and never written into a log.
HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

require "$LABELS"

run_train ortho_lora \
  $RECIPE --epochs 12 --batch 4 --accumulate 3 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant ortholarge --lora-rank 8 --backbone-lr 5e-4 --slices 12 --image-size 224 \
  --band-low 0.2 --band-high 0.8 \
  --labels "$LABELS" --out "$MODELS/ortho_lora_s2026.pt"
