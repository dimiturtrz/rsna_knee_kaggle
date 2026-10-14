#!/usr/bin/env bash
# v3lora_dense.sh — the distant-and-strong cell, with ADAPTATION as the only variable.
#
# knee-u0my. `v3base_dense` (0.8512) and `dinov3_base_lora` (0.8268) are the ledger's two DINOv3-base
# members and the +0.0244 between them was charged to adaptation. They differ on FIVE flags:
#
#     flag          v3base_dense        dinov3_base_lora
#     adaptation    full FT @ 4e-5      LoRA r8 @ 5e-4
#     train size    336 (train_size 0)  224 (--train-size 224)
#     slices        20                  12 (the default)
#     band          0.1 - 0.9           0.20 - 0.80 (the default)
#     batch         6                   12 (the default)
#
# Four of those are the dense recipe. Charging their sum to adaptation is the arithmetic error this arm
# exists to undo: it is `v3base_dense`'s invocation with `--lora-rank 8` added and the backbone rate
# moved to the rate LoRA is trained at. Slices, band, batch, image size, pooling, whole-stack, epochs
# and schedule are all held at the values 0.8512 was measured under, which makes `v3base_dense` the
# matched control and adaptation the single variable.
#
# WHY IT IS WORTH A CARD, and it is not the solo score. On 2026-08-24 every non-DINOv2 trunk was priced
# on the distance ruler against `greedy7_v1`'s seven members, and distance ordered exactly by
# architectural distance from DINOv2 across seven points with no exception. The two LoRA members are the
# distant ones:
#
#     lora_r8_v3small_224   nearest 0.8455   solo 0.8217
#     dinov3_base_lora      nearest 0.8467   solo 0.8268
#     v3base_dense          nearest 0.8829   solo 0.8512
#     roster of 7, its own nearest-neighbour spread: 0.8724 .. 0.8893
#
# Both LoRA members sit 0.027 outside the roster's own spread; the fully fine-tuned sibling sits INSIDE
# it. The mechanism is named and it predicts that ordering rather than being fitted to it: LoRA freezes
# the base, so the member keeps DINOv3's representation instead of being fine-tuned toward the solution
# every incumbent converges on. The v3 axis was closed on `v3base_dense` — the one v3 member that is not
# distant — and closed with the quality ruler alone on the very day `MemberDistance` was built to answer
# the other half. See experiments/README.md, "the v3 axis was closed on the wrong member".
#
# So the open cell is distance AND strength in one member, which is what knee-053z asked for and never
# got: `v3base_dense` bought strength by becoming a twin, `dinov3_base_lora` kept distance at a recipe
# that costs 0.0244. Nobody has run LoRA at the dense recipe.
#
# VRAM SHOULD FALL, NOT RISE, AND SMOKE IT ANYWAY. `v3base_dense` peaked at 25.38 GB allocated at batch
# 6 while fully adapting 86M parameters — weights, gradients and Adam moments. LoRA freezes the trunk,
# so the optimiser state and the weight gradients go away and only activations remain, and activations
# are unchanged because the token count, batch and slice count are all held. The backward pass still
# traverses the whole trunk to reach the early adapters, so expect a modest step-time saving rather than
# a large one. Two minutes of smoke against ~2.5 h of arm.
#
# control  v3base_dense = 0.8455 (best epoch 8), byte-identical recipe on the same encoder, fully adapted.
#          CORRECTED 2026-09-08 by control_check: this line read 0.8512, which is `base_dense`'s number —
#          the DINOv2 base, a different encoder family. The arm would have been judged against a control
#          0.0057 too high, close enough to the 0.0062 floor to flip an UNDECIDED into a decline.
# seat     priced against comp/champion.json's greedy7_v1, seven members, roster 0.8712.
# cost     v3base_dense ran ~750-850 s/epoch; LoRA trims the backward modestly, so roughly 2.2-2.6 h.
#
# PRE-REGISTERED, four outcomes. Floor 0.0062, BOARD_RESOLVABLE +0.0015, roster spread bottoming at 0.8724.
#   PREDICTION — solo at or above 0.85 AND nearest still under 0.88. The dense recipe carries the
#     strength while the frozen base keeps the distance, and this is the distant-and-strong member the
#     roster has never held. Ship it; the whole architecture table re-opens behind it.
#   REFUTATION OF THE DIVERSITY MECHANISM — solo at or above 0.85 AND nearest at or above 0.88. The
#     recipe bought quality by making the member a DINOv2 twin, distance and strength are genuinely
#     exclusive on this axis, and the v3 axis closes — this time on the right member and with the right
#     ruler. That is a real result and it retires the axis honestly.
#   DISTANT AND WEAK, AGAIN — solo under 0.84 with nearest near 0.85. Same cell as `vjepa_warm`, whose
#     seat value was +0.0002. Distance alone is confirmed to buy nothing and this project stops paying
#     for it; the remaining diversity budget goes to the DATA side.
#   UNDECIDED — solo between 0.84 and 0.85. Inside a floor of the control either way. That needs a board
#     slot to resolve and must NOT be resolved by another offline reading, which is how the v3 axis got
#     closed wrongly the first time.
#
# The distance ruler still has ONE paired board observation behind it (the 2026-08-21 split). Every
# nearest figure above is quoted as a RANKING, never as a price.
#
# bead: knee-u0my
#
# THE TWO MOVED FLAGS ARE ONE CHANGE. LoRA is not a rate-free adapter: 4e-5 is a full-fine-tune rate for
# an 86M trunk and would drive the r8 adapters at roughly a tenth of the rate every LoRA member in this
# ledger was fitted at, making the arm a test of an UNDER-DRIVEN ADAPTER rather than of adaptation. 5e-4
# is `dinov3_base_lora`'s own rate and the one both mounted LoRA members were measured under. The pair
# moves together or the variable is not adaptation at all. Everything else is `v3base_dense`'s.
# deliberate: --lora-rank
# deliberate: --backbone-lr
#
# THE VRAM PARAGRAPH ABOVE WAS WRONG, MEASURED 2026-08-25, and the arm OOMed on its first launch:
#
#   torch.OutOfMemoryError: Tried to allocate 1.84 GiB ... 28.54 GiB allowed; 25.75 GiB allocated by
#   PyTorch, 2.56 GiB reserved but unallocated.   buf74 = empty_strided_cuda((720, 446, 3072), bf16)
#
# LoRA does not cut activation memory, and activations are what this recipe is made of. It removes the
# trunk's optimiser state and weight gradients — real, roughly a gigabyte — but the backward still
# traverses all 72 layers to reach the early adapters, so every trunk activation is still held, and the
# adapters ADD their own at each of those layers. Net it rose. The prediction was made from the
# parameter count and the memory is bought by the token count.
#
# WHY --eager-encoder AND NOT A SMALLER BATCH. The named allocation is an inductor buffer, so the
# fused-kernel intermediates torch.compile materialises are the part that does not fit. Compilation is
# an execution backend and not arithmetic: `--eager-encoder`'s own help records the throughput pair at
# 0.7604 eager against 0.7606 compiled, a fifth of the noise floor. It costs 21% per step (~2.7-3.1 h
# instead of 2.2-2.6 h) and leaves the recipe's mathematics untouched, which is the whole point of the
# arm.
#
# `--batch 3 --accumulate 2` is the obvious reach and is WRONG HERE, for a reason `Accumulation`'s own
# docstring states: anything pooling WITHIN a step still sees `batch` studies, and this recipe carries
# `--rank-weight 1.0`, so halving the micro-batch halves the ranking term's pair population. That is a
# second variable in an arm whose entire argument is that adaptation is the only one. Accumulation is
# free for a pointwise loss and is not free for this one.
#
# `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` would reclaim the 2.56 GiB of fragmentation and is
# dead on this machine — see experiments/lib.sh, which records three arms carrying it unhelped.
# AND THE GATE'S ESTIMATE IS ITSELF WRONG FOR LoRA, which is why --vram-gb appears on a line that
# otherwise names only recipe flags. `DeviceBudget.needed()` derives the non-activation term from the
# TRAINABLE trunk — "weights, gradients and Adam's two moments, held once per step" — so rank-8
# adapters make that term nearly vanish and the gate reserves 29230 MiB. The activations it derives
# from slices are unchanged, and the adapters' own activations are not in the model at all, so the
# estimate falls exactly where the true cost does not. Eager measured 22.53 GiB allocated + 4.39 GiB
# reserved-unallocated = 26.9, and the failing request wants 1.84 more: ~28.8 against a 28.54 cap.
#
# 29.5 GB is 30208 MiB against 31045 free on an idle card, so it leaves ~800 MiB of DEVICE slack. That
# is the number that matters and not the cap: lib.sh's warning is about a cap set above what the card
# can SPARE, because Windows pages GPU memory over PCIe instead of failing, and a run that does not fit
# then crawls at a fraction of its power budget while reporting 100% utilisation. Here the device keeps
# room, so the cap converts the shortfall into a clean OOM rather than into paging, and
# `kneemri.training.heartbeat` covers the residual risk by reporting a collapse against the run's own peak.
#
# --vram-gb is NOT a second design variable: switch_coverage records it as "a memory cap, set by the
# queue against the card it runs on". It changes what the allocator may hold, never what is computed.
# deliberate: --eager-encoder
# deliberate: --vram-gb
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"


# `v3lora_dense smoke` runs the SAME line at 60 studies for one epoch, to a checkpoint whose name can
# never be mistaken for the arm's. It exists so the queue can chain the smoke and the arm together and
# the card is never idle waiting for a human to run the two-minute check by hand — the gap the queue was
# built to close (four hours on 2026-08-19). It is a NAMED MODE rather than a `"$@"` passthrough on
# purpose: a queue entry that can override any flag is how `resolution` spent a slot re-fitting a member
# we already had and overwrote its own checkpoint (knee-8hu7). Only these three flags can move.
SMOKE=""
[ "${1:-}" = "smoke" ] && SMOKE="--studies 60 --epochs 1 --out $MODELS/v3lora_smoke.pt"

run_train v3lora_dense \
  $RECIPE --epochs 12 --batch 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3base --lora-rank 8 --backbone-lr 5e-4 --slices 20 --band-low 0.1 --band-high 0.9 \
  --eager-encoder --vram-gb 29.5 \
  --labels "$LABELS" --out "$MODELS/v3lora_dense_s2026.pt" $SMOKE
