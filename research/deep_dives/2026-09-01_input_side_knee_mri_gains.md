# Input-side gains in knee/volumetric MRI — Q-O

Gathered 2026-09-01 for Claude's Q-O: which INPUT-side choice carried the largest
documented gain in knee-MRI or comparable volumetric-MRI competitions and
literature, with a number, and whether the delta was measured over an ensemble or
a single model.

The five axes are localization/cropping, in-plane resolution, slice sampling,
multi-plane fusion, and sequence selection/weighting.

## Short answer

The largest documented input-side gain is **localization / ROI cropping**: the
CoPAS paper reports **+0.056 mean AUC** (0.725 → 0.781) from cropping to a
meniscus-derived ROI on a single-model MRNet baseline, twelve labels. Every other
axis has either a smaller clean number, a confounded ablation, or no published
curve at all. Several of the levers are already spent in this repo (crop,
plane/contrast slots), and the one cell that remains genuinely unread is **slice
coverage** — reading many more slices than the current 20.

## 1. Localization / cropping — the biggest documented number

**CoPAS** (Qiu et al., *Nature Communications* 15:7637, 2024; PMCID
PMC11368947) is the cleanest source. Its supplementary Table 10 ablates the MRNet
baseline on Centre A (mean AUC over twelve abnormalities):

| configuration | mean AUC |
|---|---|
| MRNet baseline, **ROI-cropped** | **0.781** |
| baseline **w/o ROI crop** | 0.725 |
| Δ crop | **+0.056** |

This is a **single-model** delta, on the same task and data, and it is larger
than any other clean input-side number in the literature reviewed here. The
localizer is a learned but coarse U-Net meniscus segmentation used only to find
the ROI centre; the crop removes irrelevant areas so the same pixel budget is
spent on the joint.

**Whether it is still a lever here.** This repo already crops to a 130 mm
physical FOV (`slot.crop_mm`), and a corpus audit found only 0.43% of series are
narrower than that crop (`research/deep_dives/2026-08-31_public-frontier-sweep.md`). The
project is therefore already operating at the ROI-cropped end of CoPAS's
ladder. A tighter per-label ROI (ACL box, meniscus box, cartilage box) is not
documented with a number in the accessible literature; it is an open experiment,
not a literature-backed move.

## 2. In-plane resolution — no clean curve, evidence for saturation

**No published resolution-only ablation on knee MRI was found.** The available
numbers are either confounded or point estimates:

- **MRNet** (Bien et al., *PLOS Medicine* 2018; PMC6258509) uses **256 × 256**
  inputs and reports ACL AUC ~0.97 with an AlexNet backbone. That is *lower*
  resolution than this repo's 336 px, which is evidence against the hypothesis
  that resolution is the binding constraint.
- **CoPAS** uses **224 × 224 × 24** cubic volumes as input (its own ResNet3D-18
  encoder).
- A competitor forum post (topic 737597, reported in
  `research/deep_dives/2026-08-26_the-ablation-table-a-competitor-published.md`) claimed
  *"slice geometry mattered more than resolution"*, but the comparison was
  **192 px, no crop, 24 equal-spaced** (0.7695) vs **288 px, 140 mm crop,
  12 adjacent** (0.7777). Resolution and crop are not separated, so the number
  is uninterpretable as a resolution effect.
- The public frontier includes self-reported single models at **224 px 0.942**
  and **336 px 0.926** (`research/deep_dives/2026-08-25_competitor-single-model-scores.md`),
  but these are different recipes, not a controlled ablation.

**What this means.** A resolution curve for knee MRI has not been published in
a form that would let us price 224 vs 336 vs 448. The anecdotal evidence
(MRNet at 256, CoPAS at 224) suggests the task saturates at or below 256 px
when the crop and slice axes are handled well.

## 3. Slice sampling — adjacent vs equal-spaced, and the unread coverage cell

**Canonical literature.** MRNet pools per-slice CNN features with an
**element-wise max over all slices**; this is the field's default aggregation
(`research/deep_dives/2026-08-11_slice-aggregation-literature.md`). CoPAS keeps 24 slices
per volume and uses 3D convolutions plus cross-plane attention.

**Competitor ablation.** The same topic-737597 post reports a single-variable
slice-geometry comparison on fold 0:

| geometry | their OOF |
|---|---|
| 288 px, 140 mm crop, **9 adjacent** slices | **0.7815** |
| 288 px, 140 mm crop, **9 equal-spaced across 24** | 0.7635 |
| Δ adjacent vs equal-spaced | **+0.018** |

Their setup: EfficientNet-B0, 4 sequence slots, weighted BCE. This is a
**single-model** delta, but it compares adjacent vs equal-spaced at fixed slice
count and crop, not the project's recipe.

**Project's own measurement.** The repo already has a three-point span sweep at
12 slices (`research/deep_dives/2026-08-26_the-ablation-table-a-competitor-published.md`):

| span | ours |
|---|---|
| 0.20–0.80 | 0.8525 |
| **0.10–0.90** | **0.8565** |
| 0.00–1.00 | 0.8427 |

The optimum is interior: narrowing from 0.10–0.90 to 0.20–0.80 loses 0.0040,
and widening to the full stack loses 0.0138. Their "adjacent" winner sits at a
span (~0.375) narrower than the repo's narrowest tested point, in a direction
where the repo's curve is already falling. The repo's shipped recipe (20 slices
at 0.10–0.90) buys density the same way their result does, by adding samples
inside the good span rather than shrinking the span.

**The unread cell: slice coverage.** The public frontier sweep found that
top notebooks use **64 slices** per study, while every arm in this repo uses
12–20 slices (`research/deep_dives/2026-08-31_public-frontier-sweep.md`). The repo has
never tested reading the full stack; VRAM is the binding reason (20 slices at
336 px already reserve ~27 GB of 32). A `stack224` arm was written to probe 36
slices at 224 px because that is the only resolution where full coverage fits.
This is the only input-side direction the literature points to that the repo has
not priced.

## 4. Multi-plane fusion — spent here, and the ablation is confounded

**CoPAS headline.** CoPAS reports overall AUC **0.812** vs MRNet **0.781**, a
**+0.031** delta. But this is the full stack (3D ResNet3D-18 branches,
cross-plane attention, cross-sequence attention, correlation mining) against a
2D single-plane MRNet baseline. It is not a clean cross-plane-only number.

**The ablation table is confounded.** Supplementary Table 10 shows:

| row | configuration | mean AUC |
|---|---|---|
| (i) | MRNet baseline, ROI-cropped | 0.781 |
| (ii) | baseline w/o ROI | 0.725 |
| (iii) | baseline, multitask 12-class | 0.744 |
| (iv) | + CPA (cross-plane attention) | 0.774 |
| (v) | + CSA (cross-sequence attention) | 0.799 |
| (vi) | + CRM (correlation mining) = CoPAS | 0.812 |

There is **no clean CPA-only step**: (iv) is built on (ii) (no crop), while
(iii) is compared against (i) (with crop). The only clean subtractions are
CSA (iv)→(v) = **+0.025** and CRM (v)→(vi) = **+0.013**. CSA is a
**cross-sequence** (contrast) module, not a plane module, and CRM is a late
correlation layer. The repo's own analysis concludes that the cross-plane
attention residual is near zero once the crop is already in place
(`research/deep_dives/2026-08-24_published_knee_mri_multilabel_sota.md`).

**Counter-signal.** A 2025 multi-plane study reported single-plane sagittal
ACL AUC **0.976** beating three-plane fusion **0.949**
(`research/deep_dives/2026-08-11_slice-aggregation-literature.md`). More planes is not
automatically more signal.

**Whether it is a lever here.** The repo already routes each study into six
plane/contrast slots and attends over them with `SlotHead`. The plane axis is
therefore already spent: the published mechanism is coarser than what is
already shipped.

## 5. Sequence selection / weighting — fluid-sensitive is not enough

**CoPAS external-validation sequence drop.** The strongest published evidence
is not an ablation table but a natural experiment: CoPAS is trained on the
internal dataset with PDW + T1W + T2W, then tested on external datasets where
some sequences are missing.

| setting | mean AUC |
|---|---|
| Internal (PDW + T1W + T2W) | 0.812 |
| External D+E (PDW + T1W/T2W mixed) | 0.721 / 0.726 |
| Δ from dropping T1W/T2W consistency | **−0.086 / −0.091** |

Bone contusion specifically drops **0.821 → 0.698** when T1W/T2W are
unavailable. This is **single-model** evidence that T1/T2 carry disproportionate
signal for at least some labels, even though fluid-sensitive sequences are
usually treated as the primary diagnostic sequence.

**Clinical mapping.** CoPAS also reports plane-abnormality correlations that
match radiology: menisci on sagittal, cruciates on sagittal T2, collaterals on
coronal. The repo's slot routing was built from the same anatomical mapping
(`core/data/corpus/slots.py`).

**Project relevance.** The repo's slot fill rates on 60 studies
(`research/deep_dives/2026-08-31_public-frontier-sweep.md`):

| slot | fill rate |
|---|---|
| AX_FLUID_FS | 100% |
| SAG_FLUID_FS | 98% |
| COR_FLUID_FS | 93% |
| SAG_FLUID_NOFS | 55% |
| COR_T1 | 60% |
| SAG_T1 | 57% |

Fluid-sensitive slots are nearly complete; T1 slots are missing on ~40% of
studies. The CoPAS external result suggests the missing T1 signal may hurt
labels like bone contusion. A learned sequence-gating / imputation module that
handles missing T1 without requiring it to be present is the direction the
literature points to; simply waiting for T1 to be present would throw away 40%
of the corpus.

## Summary table

| axis | largest documented gain | over what | single / ensemble | lever still open here? |
|---|---|---|---|---|
| Localization / cropping | **+0.056 AUC** (CoPAS, w/ vs w/o ROI) | MRNet baseline | single | **No** — 130 mm crop already in place |
| In-plane resolution | No clean curve; saturation anecdotal at ≤256 px | — | — | **Maybe** — no project ablation, but MRNet/CoPAS point low |
| Slice sampling | **+0.018 OOF** adjacent vs equal-spaced (competitor) | fold-0 single model | single | **Partially** — coverage (more slices) is unread |
| Multi-plane fusion | **+0.031** full CoPAS vs MRNet, but confounded | baseline | single | **No** — six slots + SlotHead already cover it |
| Sequence selection | **−0.09** drop when T1W/T2W missing (CoPAS external) | full sequence set | single | **Yes** — T1 missing 40%, and imputation/gating is untested |

## Sources

- Qiu et al., "Learning co-plane attention across MRI sequences for diagnosing
twelve types of knee abnormalities," *Nature Communications* 15:7637 (2024).
[PMC11368947](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/) ·
[Supplementary PDF](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-024-51888-4/MediaObjects/41467_2024_51888_MOESM1_ESM.pdf)
- Bien et al., "Deep-learning-assisted diagnosis for knee magnetic resonance
imaging," *PLOS Medicine* (2018).
[PMC6258509](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- Repo prior art:
`research/deep_dives/2026-08-11_slice-aggregation-literature.md`,
`research/deep_dives/2026-08-24_published_knee_mri_multilabel_sota.md`,
`research/deep_dives/2026-08-26_the-ablation-table-a-competitor-published.md`,
`research/deep_dives/2026-08-25_competitor-single-model-scores.md`,
`research/deep_dives/2026-08-31_public-frontier-sweep.md`.
