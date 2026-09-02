# The trunk axis has four measured points and the incumbent wins all four

**2026-09-02.** For two days the plan carried "the trunk family is the only open lever that could
plausibly deliver 1.5%". That sentence was written from a MECHANISM (the pool is an architecture
monoculture — six members, all `facebook/dinov2-small` at 336 px, seven differently-seated board
entries spanning 0.937-0.944 against a 0.0053 Hanley-McNeil floor) and it was never checked against the
runs this repo already holds. It is not supported by them.

| trunk | pretraining corpus | supervision | holdout AUC |
| --- | --- | --- | --- |
| dinov2-small @336 (the whole pool) | natural | SSL | **0.8502 - 0.8592** |
| dinov3-large (`v3large_dense`) | natural | SSL | 0.8258 |
| ConvNeXt (`cnx_dense`) | natural | supervised / FCMAE | 0.7927 |
| RadImageNet DenseNet-121 (`radnet_family`) | **medical** (1.35M, CT/MRI/US) | supervised | 0.6716 |

Four families, one ordering, and every alternative is below the incumbent — the nearest by 0.024, the
furthest by 0.19. Two of the four were declined as seats in writing before today; what was missing was
reading them TOGETHER as a statement about the axis rather than one at a time as statements about
members.

## What this does and does not close

It does NOT say a different trunk cannot help. `knee-y59c` measured distance-from-DINOv2
ANTI-correlating with solo strength across seven trunks, so a weak solo is not by itself a refutation of
a diverse seat, and `raptor_v4_swa` — a foreign trunk at 0.923 solo on the board — bought +0.002 as a
seventh seat. The four rows above are about SOLO quality, and the seat question was asked separately and
answered flat (`vjepaseat` -0.001, `distinct14` 0.937).

It DOES say the axis is not a cheap 1.5%. Each of these arms costs ~50 minutes of card, the four spent
so far returned nothing above the pool, and the two families still unfitted (supervised ImageNet
VGG/ResNet, further medical SSL) sit on the same side of the ordering as the two natural-image and one
medical point already measured. Ranking the open levers by plausible size, this one has now been
demoted by its own evidence rather than by a preference.

## What replaced it, and why the arithmetic is better

The 2x2 that closed the same morning (see `experiments/README.md`) says the curriculum SCHEDULE carries
+0.007 solo on two unrelated difficulty signals, while the same signals applied flat cost -0.0036. That
is a member-side change, and member-side changes were priced at a **23% seat pass-through** on
2026-08-26 (`calib_dense_hold`: +0.0056 solo, +0.0013 seated) — which would make +0.007 worth +0.0016,
below what the board resolves.

The pass-through is the wrong number for this change. It prices ONE better member joining an otherwise
unchanged pool, where the ensemble's ambiguity term absorbs most of the gain. `calibsix` measured what a
UNIFORM lift does on this exact pool instead: +0.0038 pooled solo went to **+0.0056 and +0.0084**
ensembled on two nested pools — more than the solo gain, not a fraction of it, because every member
moves the same way and the ambiguity term is preserved rather than spent.

The ramp's dose is ~1.8x calibsix's. `rampsix` (P1) refits five of the champion's six seats with the
ramp and nothing else, pre-registered at +0.004 or better on the offline ensemble, gated on the first
two members each clearing +0.002 solo. Cost ~4.2 h of card. The board gap to the leader is 0.008.

## The confound this reading must carry, and it is a large one

`2026-09-01_the-trunks-are-tied-frozen-and-not-fitted.md` (knee-gd62) measured ridge probes over the
FROZEN caches and read DINOv2-large 0.7996, V-JEPA2-large 0.7985, DINOv2-small 0.7896 — the two large
trunks 0.0011 apart frozen against about 0.019 apart fitted. Two independent frozen probes now say the
same thing: what separates a foreign trunk from the incumbent in this repo is **not what its features
contain**. The gap opens during FITTING.

Every one of the four rows in the table above was fitted with a recipe tuned on DINOv2-small — the same
`--backbone-lr 4e-5`, the same 12-epoch one_cycle, the same EMA decay, the same slot geometry. So each
row measures TRUNK-PLUS-RECIPE and attributes the whole difference to the trunk. That is precisely the
"faulty TRAINING" branch of this project's own failure-chain rule, and `radnet_family`'s 0.6716 is the
row it bites hardest: a 7M-parameter densely-connected CNN handed a ViT's rate is not a fair reading of
RadImageNet.

So the honest statement is narrower than the one I opened with. **The trunk axis is not refuted; it is
demoted on cost-per-signal.** Each point costs ~50 minutes of card and answers a confounded question,
and answering it unconfounded means a recipe search PER FAMILY — several hours before the axis's first
clean number. `rampsix` costs about the same total and asks a question whose control already exists.
That is the whole argument for the reordering, and it is an argument about arithmetic rather than about
architecture.

The re-opening condition is stated so it cannot be quietly dropped: a foreign trunk gets its own recipe
(rate, schedule, and feed geometry fitted for it) before its solo number is read as a fact about the
family. Until someone spends that, none of the three rows below the pool may be cited as evidence that
a different architecture cannot help — only as evidence that the DINO recipe does not transfer.
