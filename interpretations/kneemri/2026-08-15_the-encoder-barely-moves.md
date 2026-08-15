# The encoder barely moves, and half of it never moves at all

*2026-08-15, morning. Every number here is CPU arithmetic over two state dicts. No GPU, no forward
pass, no data.*

knee-c5n proposed adapting the encoder to knee MRI without labels, and gated itself on a cheap
measurement first: how far did supervision actually move the encoder away from the weights it started
from? If twelve epochs of report labels already rewrote the representation, pretraining it differently
beforehand would simply be overwritten, and the axis is dead for an afternoon's CPU.

The gate passes, and the profile says something more specific than the bead asked for.

## The measurement

Both sides were already on disk. Member checkpoints hold the fine-tuned encoder under a `backbone.`
prefix; the stock weights come from the same `AutoModel.from_pretrained` the trainer used. Per block,
`‖W_finetuned − W_pretrained‖ / ‖W_pretrained‖`.

| member | LB / holdout | blocks 0–5 | block 6 | block 11 |
|---|---|---|---|---|
| `wholestack_s2026` | 0.8457 | **0.0000** | 0.0054 | 0.0023 |
| `res392_ws_s2026` (best single member) | 0.8523 | **0.0000** | **0.0214** | 0.0079 |
| `calibrated_targets_s2026` | 0.8495 | **0.0000** | 0.0054 | 0.0023 |
| `dinov3_v3small_224_s2026` | 0.8057 | **0.0000** | 0.0048 | 0.0037 |
| `large_deep_rank24` (24 blocks, 12 unfrozen) | 0.8450 | 0.0000 ×12 | 0.0110 | 0.0027 |
| `lora_r8_v3small_224_s2026` | 0.8217 | **0.0000 throughout** | 0.0000 | 0.0000 |

**The frozen blocks reading exactly 0.0000 is the oracle, not a boring row.** It is what proves the two
state dicts are aligned and the same architecture is being compared to itself. A misalignment would
have produced noise there, and every other number would have been meaningless.

## Two findings, and the second is the one that matters

**One: the movement is tiny.** The most-adapted block of our best member changed by 2.1% of its own
norm; every other member sits between 0.2% and 1.1%. Whatever our recipe is doing, it is not rebuilding
a representation.

**Two — and this is the mechanism argument the bead did not make: half the encoder is untouched BY
CONSTRUCTION.** `--unfreeze-depth 6` freezes blocks 0–5, so six of twelve blocks in every DINOv2 member
we have shipped are bit-identical to weights fitted on natural images. Not "barely adapted" —
*identical*. On-domain pretraining would write into exactly the blocks the fine-tune is incapable of
overwriting. The usual objection to domain-adaptive pretraining, that the downstream fine-tune undoes
it, does not apply to the frozen half at all.

## The supporting correlation, and its size

`res392_ws_s2026` carries 5× the encoder learning rate of the control and shows ~4× the drift
(0.0214 vs 0.0054) — the plumbing did what the config said. It is also our best member, +0.0066 above
the control. That is two members, not a study: it is consistent with "more encoder adaptation, better
score" and it does not establish it. The ledger already warns that this member's +0.0066 confounds
resolution with rate, and this measurement does not disentangle them either.

## What it says about DINOv3, which was a live confusion

The ledger reads the LoRA arm as *"the recipe was damaging v3"* — frozen-base adapters beat fine-tuned
v3 by +0.0160 while the same frozen base **lost** on v2. The drift profiles now say the two encoders
were being adapted essentially identically: v3 moves 0.0037–0.0048 per unfrozen block, v2 moves
0.0023–0.0054. So whatever separates them, it is not that our recipe drags v3's weights further from
their pretraining than it drags v2's. The LoRA arm's gain has to come from what adapters do differently
(a low-rank update available at *every* block, against a full-rank update available at only the top
six), not from the magnitude of the damage.

## The honest limit, stated because it is load-bearing

**A relative weight norm is a crude proxy for representational change.** Attention is not linear in its
weights; a 0.5% change in the right direction can move a function far more than 0.5%. So "the encoder
moved 0.5%" must NOT be read as "the representation is 99.5% unchanged". What the number does support
is the narrower claim the gate was built for: supervision here re-weights a pretrained representation
rather than replacing one, and it is arithmetically incapable of touching blocks 0–5.

The measurement that would settle the stronger version is representational rather than parametric —
CKA or a linear probe between stock and fine-tuned features on the same slices — and it needs forward
passes, which needs a GPU. That is worth doing before an SSL run, not after.

## Verdict

**Proceed** on knee-c5n, against the thresholds the bead wrote in advance: drift is concentrated (top-4
share 0.56–0.65 on the small members) *and* small in magnitude, which was the "the room is real" branch.
The refuting branch — drift large and spread across all blocks — is not what the data show for any
member measured.

The gate itself also caught two ways of measuring nothing and calling it zero: a compiled checkpoint
whose every key carries `_orig_mod.` matched no stock tensor and reported a drift of 0.000, and a LoRA
member legitimately has no base drift to concentrate. Both now say so out loud rather than printing a
reassuring number. That is the third time in a week a tool here has reported a clean pass over an empty
set — `arm_check` did it twice — and it is worth naming as a class of defect rather than as three bugs.
