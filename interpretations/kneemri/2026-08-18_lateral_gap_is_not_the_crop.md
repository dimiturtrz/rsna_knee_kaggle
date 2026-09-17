# The medial-over-lateral gap is not the in-plane crop either — the crop clips MEDIAL

**2026-08-18.** `python -m core.data.corpus.crop_laterality --sample 400 --workers 20`. 1,010 coronal and
axial series decoded uncropped, from 400 studies sampled of 4,407. Zero GPU, no label.

This was knee-yrg's last cheap candidate: a 130 mm box centred on the middle of the frame could clip the
lateral compartment before the medial one, and every member on disk reads through it.

## The number

```
1010 coronal/axial series measured, from 400 studies sampled of 4407 in the corpus
discarded lateral 0.0060  medial 0.0173  lateral - medial -0.0112 [-0.0130, -0.0095]
```

**The crop discards three times more tissue on the MEDIAL side, and both shares are under 2%.** The
effect exists and points the wrong way: the interval is far from zero and entirely negative. If the crop
were producing a compartment asymmetry it would be handicapping the label that scores +0.0611 BETTER.

The direction is also the anatomically expected one, which is the sanity check that the sign is real and
not a convention error: medial of the joint the body continues toward the midline, so tissue keeps
running out of the frame; lateral of it the leg ends at skin within a few centimetres.

## How lateral was located, since no pixel carries it

Two header facts multiply, in `CropLaterality.lateral_column`: the row direction cosine's x component
says which way the patient's LEFT lies along the image's column axis (+x is left in LPS), and the study's
side turns left into lateral (left knee: lateral is +x; right knee: the midline is to its left, so
lateral is −x). Either fact missing declines the series rather than defaulting it — an unresolved side
entering the mean with a coin-flip sign pulls the answer toward zero, which is a verdict this
measurement can return.

## Scope, stated rather than implied

- **Coronal and axial only.** Medial-lateral is in-plane there and THROUGH-plane on a sagittal series,
  where it belongs to the slice band — measured and refuted separately
  (`2026-08-17_slice_band_coverage.md`). A sagittal slot is silent here, not symmetric, and is declined.
- **Discarded tissue is not discarded compartment.** A crop clipping the frame's lateral edge is clipping
  skin and thigh too. This bounds the claim from one side, which is the side the decision needs: a crop
  that takes LESS from lateral cannot be what is starving the lateral labels.

## What it closes

knee-yrg's four cheap explanations are now all refuted, each on its own measurement: the laterality flip,
the slice band, prevalence (`2026-08-18_lateral_gap_is_not_prevalence.md`, matched counts move the gap by
0.0001), and now the in-plane crop. Nothing about the READING deletes the lateral compartment.

What is left costs the card: supervision volume (the lateral heads see about half the positives, and
matching controls measurement but not training) and vision proper — knee-dtk already splits Lateral
Meniscus as the largest MODEL-side gap of the twelve (report 0.8429, model 0.6497, −0.1932). The next
test on this axis is a training arm, not another decode.
> **2026-09-17, the subject was the published pool.** The -0.1932 quoted here is the PUBLISHED members'
> gap. Our champion reads Lateral Meniscus 0.8919 against the same 0.8422 report label on the same 58
> studies (+0.0497), and beats the reader on all twelve findings. Nothing measured on this page is wrong;
> what it is a fact ABOUT is those members, not ours, so the vision-failure reading may not be carried to
> our model. See `interpretations/kneemri/2026-08-14_labels-or-pixels.md`.

## Sources

- `core/data/corpus/crop_laterality.py` (`CropLaterality.clipped`, paired bootstrap seeded 11, 1,500 draws).
- `core/data/corpus/acquisition.py` — `SeriesHeader.column_axis_x`, added for this.
- knee-yrg, knee-1v0n, knee-dtk.
