# The published frontier on twelve-label knee MRI, and we are not behind it — 2026-08-24

Gathered on a night with the card busy and no submission slots, against the standing rule that a
from-scratch method loses to a replicated frontier plus one real idea. The question was narrow: is there a
published mechanism, sized for our 0.015 board gap, that this repo has not read?

## The closest prior art is the same task, and its number is below ours

**CoPAS** — *Learning co-plane attention across MRI sequences for diagnosing twelve types of knee
abnormalities*, Nature Communications 15:7637 (2024), HKUST Smart Lab with the Third Affiliated Hospital of
Southern Medical University. Code at `github.com/zqiuak/CoPAS`.

- 1748 subjects, **twelve** abnormality types — the same label count our metric averages over.
- Overall **AUC 0.812**, against MRNet 0.781, MPFuseNet 0.754, ELNet 0.760.
- Beats junior radiologists (0.65 accuracy), competitive with senior (0.80 vs 0.78).

Absolute numbers do not transfer — different corpus, different truth, different denominator — so 0.812
against our holdout 0.873 says nothing about who is better. What transfers is **mechanism and its size**:
plane-aware fusion buys **+0.031** over MRNet on their own ruler, which is twice our board gap.

## The mechanism is already ours, at finer granularity

CoPAS runs three plane-specific ResNet3D-18 branches (sagittal, coronal, axial) with weight sharing, then
cross-plane attention where Q comes from the main plane and K, V from the others resampled into its
representation, then pools to `R^(D x C)` to keep depth-wise slice order.

`core/data/corpus/slots.py` routes each study's series into **six** slots — plane crossed with weighting
axis and fat-suppression state — chosen so each of the twelve findings has at least one sequence that
shows it well, and the slot head attends over them. The docstring carries the same anatomical argument
(cruciates oblique and best sagittal, collaterals and meniscal body coronal, patellar cartilage and
retinacula axial) plus a correction CoPAS has no equivalent of: measured on 150 random series, the shipped
`Fluid_Sensitive` column matches recovered fluid sensitivity only **78.7%** of the time, so routing on
vendor flags alone would put fluid-weighted series in the T1 slots for a fifth of the corpus.

Three planes fused, versus six plane-and-contrast slots attended over. **The headline mechanism is not a
lever here because it is already spent.**

## The ablation table, read 2026-08-25 — the components do not add, and CPA has no isolated number

The ablation is NOT in the article; it is Table 10 of the supplementary PDF
(`static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-51888-4/MediaObjects/41467_2024_51888_MOESM1_ESM.pdf`,
page 10). Centre A, mean AUC-ROC over the twelve labels:

| row | configuration | mean AUC |
|-----|---------------|---------:|
| (i) | MRNet baseline, **ROI-cropped** | 0.781 |
| (ii) | baseline **w/o ROI** | 0.725 |
| (iii) | baseline, multitask 12-class | 0.744 |
| (iv) | + CPA (cross-**plane** attention) | 0.774 |
| (v) | + CSA (cross-**sequence** attention) | 0.799 |
| (vi) | + CRM (correlation mining) = **CoPAS** | 0.812 |
| (vii) | CoPAS, single-task ×12 | 0.808 |

**There is no clean CPA-only step anywhere in this ladder.** The text builds (iv) "on the basis of model
(ii)", i.e. WITHOUT the ROI crop, while (iii) is compared in the prose against (i), i.e. WITH it. So every
subtraction that reaches CPA also crosses the crop or the single-to-multitask change. Any quoted
"cross-plane attention is worth +0.0xx" is a mis-subtraction, including one this project made: +0.049 is
(iv) − (ii), which silently skips (iii).

(Their prose and their own table disagree, which is why only the table is quoted here: the text says the
multitask adaptation drops "0.781 to 0.705" and row (iii) reads 0.744.)

**The clean steps are the two we have least use for.** CSA is (iv)→(v) = **+0.025** and it buys T1W and T2W
as separate branches — a multi-contrast axis, not a plane one. CRM is (v)→(vi) = **+0.013**, under our own
1.5% bar on its own ruler before any corpus transfer.

**The arithmetic that settles it.** Those two clean steps sum to +0.038 on the no-crop ladder, while the
whole stack over the ROI-cropped baseline (i)→(vi) is **+0.031**. Different ladders, so this is not a strict
subtraction — but the stack cannot be worth less than its own two later components unless the crop and the
attention modules overlap heavily. Their own sentence says exactly that: *"cropping along ROI can align the
spatial position across sequences."* **Alignment is what the crop buys, and it is most of what CPA was
buying on top of an unaligned baseline.** We already crop to 130mm and our slots are orthogonal within
87.3°, so we sit at (i) and the residual for cross-plane feature attention is somewhere near zero by their
numbers — not by an argument about our geometry.

That correction runs the other way too: "our slots are already orthogonal, so cross-plane attention cannot
help" is not a valid argument on its own. Orthogonality is the PRECONDITION their module operates on, not a
substitute for it. What kills the lever is the table, not the geometry.

## What is genuinely left of their delta

A **3D encoder with depth-preserving features and geometric cross-plane resampling**, against our 2D slice
encoder plus pooling. Their claim for it is enhancing spatial information along slices without
interpolation artefacts. That is the **3D-stem instrument already named in `knee-ombx`'s refutation
clause**, so it is on the list rather than new — and the same clause pre-registers what a flat reading
from it would mean.

## What the sweep is worth

It hands us no mechanism, and it does two smaller things that are still worth the twenty minutes:

1. **The plane axis is spent, not unexplored.** Our finer routing was built from anatomy and then measured;
   the published state of the art fuses coarser and gains +0.031 doing it.
2. **It removes an explanation for the gap.** "The leaders are running a published architecture we have
   not read" does not survive. Whatever sits at 0.952 on this board is not in this literature, so the gap
   is not a reading failure and there is no paper to go replicate.

Sources: Nature Communications s41467-024-51888-4 · PMC11368947 · github.com/zqiuak/CoPAS · the RSNA 2026
Knee Abnormality Detection AI Challenge page.
