# The champion is an architecture monoculture

*2026-08-28 · beads knee-nj4l, knee-7l4e · measured from the six `calibsix` member configs*

## What the roster actually is

`comp/champion.json` pins `calibsix_v1` at **0.941**, six members. Dumping all six configs:

| axis | value | varies across the six? |
| --- | --- | --- |
| backbone | `facebook/dinov2-small` — DIM 384, 22M params | **no** |
| read size | 336 px | **no** |
| slots | `SAG_FLUID_FS`, `COR_FLUID_FS`, `AX_FLUID_FS`, `SAG_FLUID_NOFS`, `COR_T1`, `SAG_T1` | **no** |
| `whole_stack` | `True` | **no** |
| token pool | `cls_mean` | **no** |
| slice pooling | `max` | **no** |
| slot pooling | `attend` (`SlotHead`, per-diagnosis query, anatomy prior, absent slots masked) | **no** |
| `encoder_layer` | `-1` | **no** |
| `lora_rank` | `0` | **no** |
| `crop_mm` | `130.0` | **no** |
| learning rates | head `1e-3`, backbone `4e-5` | **no** |
| `rank_weight` | `1.0` | **no** |
| `slices` | 12 or 20 | yes |
| `band` | `[0.1, 0.9]` or `[0.2, 0.8]` | yes |
| `through_plane` | `True` on `through_dense_cf` alone | yes |

Three varying axes, all of them recipe. One encoder, one resolution, one head, one slot scheme.

The roster's only non-DINOv2 member ever fitted is `lora_r8_v3small_224_s2026.pt` at 0.8217 — too weak
to have earned a seat.

## Why that is the explanation and not just a description

Seven board entries have now varied seat count (6/7/8/11/12), member identity, fold status, vote rule,
serving views, corpus, supervision and pixel group. They span **0.937–0.941** — a 0.004 range against a
0.0053 Hanley–McNeil floor. Seven ways of reseating six copies of one lens is what a 0.004 range looks
like.

The ledger reaches the same wall from the offline side and calls it the twelfth flat-or-negative reading
on ensemble construction: seat count flat past four, membership flat across five different sevenths,
per-label selection (`label_combine`, 2026-08-26) optimal at the flat rank-average, the vote inside its
pre-registered flat band, overlap TTA flat, external corpus flat on score *and* on ordering. Its own
conclusion — *"the next real number costs a card, not a slot"* — is this finding stated as a budget rule.

## The single-model ceiling is not DINOv2's fault

On 2026-08-27/28, two submissions:

- `calibsolo` — one of our own members, served alone → **0.923**
- `raptor_v4_swa` — `coatnet_rmlp_2_rw_384` at 384 px, a foreign training corpus, a foreign decode, a
  foreign head, served on our pipeline → **0.923**

Two unrelated stacks, the same day, one number. That coincidence is the load-bearing fact here. It says
the single-model ceiling is not set by the block layout, because changing the block layout entirely did
not move it. What the two trunks *do* share is that both were pretrained on **natural images**.

It also settles knee-9g28 in knee-3uje's favour: the claimed +0.030 foreign member-quality gap does not
exist.

## What that costs us against the goal

The ensemble-bonus law (knee-urvf) fits `board(ensemble) ≈ board(best single) + bonus` over three clean
pairings: +0.011, +0.014, +0.018, mean **+0.0143**. `calibsolo → calibsix` is +0.018, the top of that
range.

So **0.950 needs a single near 0.9357**. Our best is 0.923 and the best foreign member we can serve is
also 0.923. No reseating of this pool closes 0.012 of single-model quality, and the twelve flat readings
say the bonus term is already maxed.

## The axis this pool does not vary at all

The ledger's own honest-limits paragraph names the exit: *"a member from an axis this pool does not vary
at all (a genuinely different encoder, image-derived supervision) is not tested by any of it: saturation
of a pool is a statement about that pool."*

The most pixel-distant trunk the repository owns is **RAD-DINO** — a DINOv2-base ViT finetuned on 882,775
chest radiographs, MIT-licensed, already wired into `backbone.py` as `variant=raddino`. Its ledger entry
reads 0.8137, −0.0375 against its control, and it has been on the ban list as refuted since 2026-08-23.

**That kill does not stand.** The run is dated 2026-08-23; the normalization fix landed 2026-08-25 in
`3b74685`. For its entire life the arm fed RAD-DINO the ImageNet constants while its own
`preprocessor_config.json` declares 0.5307 / 0.2583 — inputs 0.20 std off its pretraining distribution
with a 12.8% scale error. The *control* was correct, because DINOv2-base does declare the ImageNet
constants, so the one preprocessing difference the arm header called shared was in fact **differential and
pointed against the arm being tested**. On top of that the process was truncated at epoch 7 of 12 under
`one_cycle`, which makes 0.8137 a floor rather than a landing, and its recipe was tuned across ~189
DINOv2 runs.

Three defects, one number. No claim about radiology pretraining survives any of them. `raddino_dense.sh`
was rewritten on 2026-08-25 to say exactly this and then sat unarmed for three days — which is the
cheaper half of the lesson: the audit had already been done and written down, and nobody read it back.

The cold re-run is armed (knee-7l4e), queued behind the fleet teacher pass. It is not the banned warm
start — a warm start is impossible, because the 2026-08-23 checkpoint carries the wrong constants baked
into its own persistent buffers.

## What this does not say

It does not say DINOv2-small was the wrong choice; it is the trunk that got us to 0.941 and every
correctly-normalised control on this page is one. It does not say a different architecture will pay —
`raptor_v4_swa` is a different architecture and scored exactly level. The claim is narrower and testable:
**the pretraining distribution is the axis this pool has never varied, and it is the only one left that
the bonus law can still be paid on.**
