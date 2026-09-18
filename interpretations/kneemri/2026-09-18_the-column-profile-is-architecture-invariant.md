# The twelve-column difficulty profile is the same under every architecture we own

**What was measured.** Per-label ROC AUC for sixteen members spanning five encoder families, all on the
same fold-0 909-study holdout, truth cut at 0.75, scored with `LabelScores.per_label`. No card, no
submission slot — every number came off holdout `.npz` files already on disk. 2026-09-18.

The question it answers is the one asked most often in this project and never answered with a
measurement: *what is the structurally different member that covers what DINOv2 cannot see?*

## The answer is that there isn't one, and the gaps are not DINOv2's

| family | macro | profile r vs DINOv2 | columns where it beats DINOv2 |
| --- | ---: | ---: | --- |
| DINOv2 labelwise (5 members) | 0.9122 | — | — |
| ResNet50d — a CNN, not a ViT | 0.8736 | **+0.926** | **none of twelve** |
| V-JEPA — video/volumetric pretraining | 0.8755 | **+0.920** | **none of twelve** |
| MedSAM — medical segmentation encoder | 0.8458 | **+0.812** | **none of twelve** |
| DINOv3 | 0.8583 | **+0.779** | **none of twelve** |

The profile is each family's twelve AUCs minus its own macro — the SHAPE of which findings are hard,
with the family's overall quality divided out. Across a 0.078 span of macro quality, and across
self-supervised ViT, supervised CNN, segmentation-pretrained and video-pretrained encoders, that shape
correlates **+0.78 to +0.93** with DINOv2's, and **+0.92 to +0.99 between the foreign families
themselves**. A CNN and a ViT disagree about almost everything except which findings are hard.

The shared shape, as DINOv2 mean AUC:

| Baker's | MedMen | ACL | Effusion | Synovitis | MedOA | Contusion | PF OA | LatMen | MCL | LatOA | **Fracture** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9598 | 0.9444 | 0.9433 | 0.9401 | 0.9186 | 0.9215 | 0.9068 | 0.8995 | 0.8949 | 0.8950 | 0.8755 | **0.8488** |

**Within the DINOv2 family the columns are fixed to a hundredth.** Five members — three recipes, a
mutual-distillation member, a head-lr variant — have SD across members of 0.0011 to 0.0111 per column,
while the columns themselves span 0.11. Member-level work moves a column by ~0.003. The column
structure does not move.

## Scaling explains most of it, and the residual is consistent too

A weaker model degrading multiplicatively toward 0.5 would preserve the ordering trivially, so the
scaling null has to be removed. It is real but partial: `(auc − 0.5) / (dinov2 − 0.5)` has mean 0.83–0.88
per family with SD 0.065–0.099, and after fitting each family its own best scale factor the residual SD
is 0.026–0.040 — structure scaling cannot absorb.

That residual is itself consistent across families. All three of DINOv3, ResNet50d and MedSAM are worse
than their own scaling predicts on **Lateral Meniscus (−0.05 to −0.08) and MCL**, and better than
predicted on **Synovitis and Effusion (+0.03 to +0.06)**; V-JEPA's worst residual is Lateral Meniscus
too. So DINOv2 holds a genuine relative advantage on the two ligament/meniscus columns and a genuine
relative disadvantage on the two fluid ones.

**It does not convert into a complementary member.** The relative disadvantage is not an absolute one:
DINOv2 reads Synovitis 0.9186 against 0.9093 for the best foreign family and Effusion 0.9401 against
0.9203. The gap narrows and never crosses. Zero columns change hands in sixteen members.

## What this explains that was already on the record

- **Why the seven-member champion buys +0.001 over the one-member solo.** Architectural diversity does
  not produce error-location diversity. Members built on different pretraining objectives are wrong in
  the same places, so there is nothing for an average to cancel. That is the mechanism behind the
  efficiency re-ranking recorded in `2026-09-18_state_of_the_levers.md` and `comp/README`.
- **Why serve-time column mixing is a null** (`2026-09-04_serve-time-column-mixing-is-a-null.md`,
  ceiling −0.0019 under both L2 and sparse L1). There is no leftover cross-column information because
  every member already read the same pixels the same way.
- **Why the label lane retired at −0.0040** (`experiments/README.md:6738`). `blend_hybrid_fx` wins
  Fracture by +0.12 on the 57 gold studies and loses 0.0040 under training. A label table cannot move a
  column whose difficulty five unrelated encoders agree on.
- **Why `side_dense` regressed −0.0641 with its mechanism confirmed firing** and `sided_pool`'s explicit
  medial/lateral axis went unused. The lateral deficit is not a representational blind spot a loss term
  or a pooling axis can address; it is present in every representation tried.

## What it does NOT close

The truth here is **our report-mined supervision binarised at 0.75, not Kaggle truth** — the caveat
`column_headroom.py` states about itself applies unchanged. This gives the ORDERING of column difficulty
under our labels, not the board's per-column areas.

It therefore cannot yet separate the two live explanations of the invariant shape:

1. **An image ceiling** — the finding is genuinely less visible in these pixels, for anything.
2. **A label ceiling** — the report-mined column for that finding is noisier, so every model is fitted
   and scored against a noisier target and no architecture can escape it.

These predict different things and the distinction is worth an instrument, because (2) would mean the
weak columns are capped rather than unsolved, and effort belongs in the columns that are not. The
discriminating measurement is per-column reader accuracy on the gold annotated studies against this
profile: under (2) they correlate, under (1) they need not. `report_labels_v2` measures 79.2% overall
against the annotations and the per-column breakdown is what is missing.

Fracture is the case to be careful with either way: 42 positives in 909 studies, so its absolute AUC
carries a wide interval against the population even though the sixteen-member comparison is paired on
the same studies and precise.

Also untouched: an encoder nobody here has tried, a genuinely different INPUT (a locator, a segmentation
prior, a different plane or reconstruction) rather than a different encoder over the same input, and
supervision that is not report-derived at all.

## The consequence for choosing work

At `d/12` per column — settled, `kneemri-colnull` shipped constant 0.5 and read exactly 0.500 — the four
weak columns carry 0.19 of deficit between them and lifting them to 0.92 would be +0.0145 macro, a full
1.5% lever from one mechanism. This measurement says that lever cannot be reached by **swapping the
encoder**, and the record already says it cannot be reached by **relabelling**, by **mixing the output
columns**, or by a **loss term naming the axis**.

So a candidate member is now cheaply gated before it costs a card: score it per column on this holdout
and require it to WIN a column outright. Sixteen members across five families have won zero. A candidate
that only raises the macro is another member of the same shape, and the shape is what binds.

Related: `kneemri/evaluation/column_headroom.py` (the per-column lift pricer and the `d/12` arithmetic),
`interpretations/kneemri/2026-09-18_state_of_the_levers.md` (the audit this sharpens),
`experiments/README.md:6738` (the label lane), `experiments/README.md:2754` (`side_dense`).
