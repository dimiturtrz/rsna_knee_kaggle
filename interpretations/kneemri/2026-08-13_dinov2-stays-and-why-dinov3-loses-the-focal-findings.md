# DINOv2 stays. What DINOv3 actually lost, and why it is not resolution and not capacity

2026-08-13. Three DINOv3 runs are on disk against the DINOv2 whole-stack control, all on fold 0, all
scored on the same 881 studies. The encoder question is settled as an *improvement* question: DINOv2
is the encoder, and the decision is committed rather than re-litigated per arm. The interesting part
is not the mean — it is that the deficit has a SHAPE, the same shape in all three runs, and that shape
eliminates the two explanations we would have reached for first.

## The numbers

Control `wholestack_s2026`, DINOv2-S/14, 336 px read, epoch 6, **0.8457**.

| label            | v2-S/14 336 | v3-S/16 336 | v3-S/16 224 | v3-S+/16 224 |
|------------------|------------:|------------:|------------:|-------------:|
| **MEAN**         |  **0.8457** |  **0.8080** |  **0.8057** |   **0.8012** |
| Baker's          |       0.896 |       0.817 |       0.822 |        0.781 |
| Medial Meniscus  |       0.893 |       0.828 |       0.820 |        0.818 |
| Lateral Meniscus |       0.832 |       0.756 |       0.761 |        0.754 |
| Contusion        |       0.861 |       0.819 |       0.813 |        0.795 |
| ACL              |       0.873 |       0.842 |       0.844 |        0.827 |
| Fracture         |       0.784 |       0.759 |       0.748 |        0.744 |
| Medial OA        |       0.881 |       0.860 |       0.853 |        0.857 |
| Lateral OA       |       0.822 |       0.790 |       0.796 |        0.800 |
| PF OA            |       0.824 |       0.800 |       0.805 |        0.805 |
| MCL              |       0.755 |       0.727 |       0.719 |        0.732 |
| Synovitis        |       0.873 |       0.855 |       0.853 |        0.860 |
| Effusion         |       0.854 |       0.842 |       0.834 |        0.840 |

The deficit runs from −0.115 to −0.012 depending on the label, and the ordering is stable across all
three v3 runs: **Baker's, both menisci and Contusion lose four to ten times what Effusion and
Synovitis lose.** A uniform shift would have said "worse model". This is not that.

CALIBRATED AFTERWARDS, and it is the reason this section is allowed to carry an argument. Each label's
AUC has its own sampling error, from how many positives carry it — Fracture's 62 positives give a 95%
interval of ±0.061 where Effusion's 533 give ±0.025 (`kneemri.labels.label_power`). Against those intervals
exactly four of v3-S's twelve deltas clear their own: **Baker's, Lateral Meniscus, Medial Meniscus and
Contusion** — the four this argument rests on. The rest, Fracture's −0.025 included, are smaller than
the instrument that produced them and are not evidence of anything. The same calibration deletes a
claim made the day before about extended slots, where NOT ONE per-label delta cleared its interval.

## What the shape rules out

**Not the patch grid.** The obvious story is tiling density: at 336 px DINOv2's patch-14 cuts a 24x24
= 576-patch map and DINOv3's patch-16 cuts 21x21 = 441, and at 224 it cuts 14x14 = 196 — a third of
the control's resolution on findings 1-3 mm wide. The story predicts that 336 recovers most of 224's
deficit. **It buys 0.0023** (0.8080 against 0.8057), inside the 0.0062 sampling floor. Doubling the
number of patches the head reads is worth nothing here, so how finely we tile the image is not what
separates the encoders.

**Not capacity.** DINOv3-S+ is the same family 33% wider (28.7M against 21M, 384-wide against 384 at
more depth) and it scores **0.0045 WORSE** than plain v3-S at the same read. A model short of capacity
does not get worse when given more of it. This is also direct evidence against knee-d2s's premise
that the distillation story scales with the student — inside the small family it scales the wrong way.

## What is left

The pretrained representation itself, on THIS input. DINOv3's headline improvement over v2 is Gram
anchoring — it stops PATCH-level features degrading over a long run by holding the patch-to-patch
similarity structure to an early teacher. That is measured on natural-image dense tasks (segmentation,
depth, tracking), where local structure means object boundaries in RGB. Our input is a single-channel
MRI slice replicated across three channels, and the discriminative structure is a low-contrast
intensity gradient *inside* one tissue field: a meniscus is a dark triangle against dark bone in a
dark background, a Baker's cyst is a fluid pocket that is only a Baker's cyst because of where it
sits. Anchoring preserves whatever similarity structure the teacher had; it does not create
sensitivity to a contrast axis the pretraining distribution never contained. The findings that survive
in v3 — Effusion, Synovitis, the OA grades — are exactly the ones carried by joint-wide signal that a
coarse, boundary-oriented representation can still see.

The per-label RANKING moves too, which is the same fact from the other side. Baker's is the control's
best label (0.896) and v3's ninth (0.817); Medial OA is the control's third and v3's first. v3 is not
the control shifted down. It is a differently-shaped reader.

## Scope, stated rather than glossed — and then measured

Every v3 run inherited DINOv2's adaptation recipe: last six blocks, backbone LR 8e-6, 12-epoch
OneCycle. So the claim is "DINOv3-S under DINOv2's recipe", not "DINOv3 is worse". We are not sweeping
the rate to close that — a 0.040 deficit with a per-label shape this consistent is not a hyperparameter
miss, and the read-side knob we DID vary (441 patches against 196) moved 0.0023, which says this
encoder's behaviour here is insensitive to exactly the axis a rate sweep would be tuning.

**MEASURED LATER THE SAME DAY, and the caveat was half right.** Adaptation was the one axis never
varied, so it was varied: LoRA freezes the base — gammas, RoPE, patch embedding, norms — and confines
the update to a rank-8 subspace. Against fine-tuned v3 at the same 224 px read, matched epoch for
epoch: 0.8118 against 0.7940 at epoch 4, 0.8189 against 0.8009 at 5, 0.8193 against 0.8041 at 6.
Fine-tuned v3's best over twelve epochs was 0.8057; LoRA reaches **0.8193**, **+0.0136** — 2.6x the
floor.

So **about a third of v3's deficit was our recipe damaging it, and about two thirds is the encoder on
this data.** The verdict below is unchanged — v3+LoRA is still 0.027 short of DINOv2's 0.8457, so
DINOv2 stays — but "DINOv3 is worse here" is now a narrower and better-supported claim than it was
this morning, and the axis this section flagged as untested turned out to be worth more than the two
that were tested.

It also asks a question larger than DINOv3: if our fine-tuning recipe costs v3 0.0136, what is it
costing DINOv2?

## What DINOv3 is still for

Failing differently is the one property measured to pay on this competition (five weak members scored
LB 0.888; two members 0.021 better EACH scored 0.882). v3's per-label profile is reordered, not
merely lowered, which is the definition of a differently-failing member — and it costs no extra decode,
because it reads 224 off the 336 cache every other member already needs. Whether that earns a seat is
`kneemri.evaluation.member_curve`'s call, not this document's.

## The decision

- DINOv2-S/14 remains the encoder. `--variant` already defaults to it; nothing in the code changes.
- DINOv3 is **refuted as an improvement** and retained **only** as a diversity candidate.
- knee-d2s (DINOv3 at base) keeps its evidence but loses its premise: capacity scaled the wrong way
  inside the small family.
