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
