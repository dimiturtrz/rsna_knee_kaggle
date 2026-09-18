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

Two explanations of the invariant shape were live when this was written:

1. **An image ceiling** — the finding is genuinely less visible in these pixels, for anything.
2. **A label ceiling** — the report-mined column for that finding is noisier, so every model is fitted
   and scored against a noisier target and no architecture can escape it.

**(2) is refuted, and the measurement that does it was already on disk.**
`2026-08-26_label-ambiguity-and-our-deficit-do-not-overlap.md` ran `reader_disagreement` over four
independent readings of the same reports on 4,406 studies with all three foreign readers aligned
4406/4406. Correlating its per-finding reader spread against this profile's per-column AUC:

**r = +0.081, Spearman +0.147.** Zero, and the faint sign runs the WRONG way — the columns readers
argue about most score slightly HIGHER. One of the five weakest columns (Fracture) is among the five
most-disputed; the other four are not. The disputed columns are Synovitis (spread .300), Contusion
(.240), Baker's (.218) and Effusion (.208), and three of those four are in our strong half, Baker's
being our best column of twelve.

That interpretation reached the same place on a second ruler sharing no denominator — UNK mass on the
58 radiologist-annotated studies is diffuse (Synovitis 41 of 58, Baker's 29, Fracture 26) while the
focal findings are decided (ACL 4, Medial Meniscus 2, MCL 6) — and stated the consequence plainly:
*"Text ambiguity is a severity problem, not a detection problem."*

So the invariant profile is an **image ceiling**, not a label ceiling. The weak columns are capped by
what five encoder families can extract from these pixels, not by the noise in the target — which means
they are unsolved rather than closed, and worth attacking, just not with another encoder or another
label table.

Fracture is the case to be careful with either way: 42 positives in 909 studies, so its absolute AUC
carries a wide interval against the population even though the sixteen-member comparison is paired on
the same studies and precise.

Also untouched: an encoder nobody here has tried, and a genuinely different INPUT (a locator, a
segmentation prior, a different plane or reconstruction) rather than a different encoder over the same
input.

**The lane the 2026-08-26 chain pointed at has since been run, and it closed — on the same shape.** That
interpretation named image-derived supervision as what survives: *"MRNet's ACL and meniscal tears, made
by radiologists FROM THE IMAGES."* `merged_corpus` added Rijeka's 736 image-read studies (+0.0020, inside
the floor, but the deficit set moved as a set: Lateral Meniscus +0.0196\*), and `merged3_corpus` then
added MRNet's 1,370 (`experiments/README.md:346`). It reads **−0.0024**, and **Lateral Meniscus ranks
LAST of twelve at −0.0217** — the exact column the MRNet argument had been built on, the one Rijeka had
moved most. Second site +0.0043, third site −0.0024: the site-count corner is closed negative.

**Its verdict is this finding arriving from the other side, and that is why it belongs here rather than
only in the ledger.** The ledger's words are *"THE SHAPE IS A ROTATION, NOT AN ADDITION … Two foreign
sites trade the same small set of columns instead of stacking"* — Synovitis was Rijeka's one starred cost
(−0.0206\*) and MRNet hands it straight back (+0.0234) while the columns Rijeka bought go down. So the
column profile is invariant under a change of **corpus** as well as a change of **encoder**. Adding a
site does not add to the profile; it rotates which columns are paid for out of the same total. Two
independent axes, the same conserved shape.

That makes the shape the object to attack, and it narrows what could move it to things that change
neither the encoder nor the site count: the corpus corner the ledger says still pays — **supervision at
a fixed two-site cache** (`merged_blend_cf` 0.8597, `merged_span_cf` 0.8618) — and **Q-POOL**, the
combinations `merged3_corpus` explicitly did not try (per-site conditioning, harmonisation,
pool-then-specialise). Representation remains open on its own evidence (`knee-jk3`, a 0.044 gap at
matched train loss).

## What predicts the shape, and why the obvious correction is already closed

Neither label noise (r = +0.081) nor architecture explains the ordering, so the question is what does.
The strongest variable found is **positive count**: `r(AUC, log n_pos) = +0.536`, `r(AUC, prevalence) =
+0.438`. Both matched anatomical pairs track it — Medial Meniscus 1,779 positives at 0.9444 against
Lateral Meniscus 706 at 0.8949 (+0.0495), Medial OA 1,102 at 0.9215 against Lateral OA 644 at 0.8755
(+0.0460). It is partial (r² ≈ 0.29): Baker's is the best column of twelve on 566 positives and PF OA is
weak on 1,448.

**The correction that follows from it has been run, at both ends of the pipeline, and it is negative.**
At the objective, `per_label_loss` weighted by inverse prevalence read **0.8414 against 0.8457, −0.0043**
(`experiments/README.md:570-602`), and its pre-registered clause is what settles it: the gain was
required to concentrate in the rare labels, and it concentrated there with the wrong sign — eleven labels
inside −0.007, and **Fracture, the rarest at 314 positives and the most up-weighted, at −0.022**.
Up-weighting a 314-positive column multiplies its gradient variance without adding information to it,
and AUC being rank-based offers no positive-rate gain to set against that. At the ensemble, `member_curve`'s
oracle — which picks per finding the best member *with the answers*, and therefore upper-bounds every
per-label weighting scheme writable over these members — reads **+0.0002 over the plain average**
(`experiments/README.md:602-633`).

So prevalence predicts the profile without being a lever on it. The ledger's reading is the right one:
prevalence here *is* the budget that buys the common labels their scores, not a mismatch to correct. This
is the sixth route closed on the conserved shape. It does NOT close the loss-weight axis over **studies**
(`severity_weight`, `experiments/README.md:2102`, knee-t1ho), which is a different mechanism and has its
own evidence.

## The consequence for choosing work

At `d/12` per column — settled, `kneemri-colnull` shipped constant 0.5 and read exactly 0.500 — the four
weak columns carry 0.19 of deficit between them and lifting them to 0.92 would be +0.0145 macro, a full
1.5% lever from one mechanism.

The profile is now measured invariant under **five encoder families** (this document) and under **three
corpus sites** (`experiments/README.md:345-346`), and it does not track **label noise** (r = +0.081).
So that lever is not reachable by swapping the encoder, adding a site, relabelling, mixing the output
columns, a loss term naming the axis, or reweighting the columns by prevalence — six routes, each closed
on its own measurement. What is left is not a longer list of the same kind of move.

So a candidate member is now cheaply gated before it costs a card: score it per column on this holdout
and require it to WIN a column outright. Sixteen members across five families have won zero. A candidate
that only raises the macro is another member of the same shape, and the shape is what binds.

Related: `kneemri/evaluation/column_headroom.py` (the per-column lift pricer and the `d/12` arithmetic),
`interpretations/kneemri/2026-09-18_state_of_the_levers.md` (the audit this sharpens),
`experiments/README.md:6738` (the label lane), `experiments/README.md:2754` (`side_dense`).
