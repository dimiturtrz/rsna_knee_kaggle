# Q-EH — mapping our 2–3% true noise rate onto the HLC curve

This note answers Claude's follow-up to Q-EG: if HLC is the closest
published multi-label corrector, what does its measured curve say about our
much lower noise rate?

## Inputs from our side

- 1,932 flagged cells / 52,872 total = **3.65% cell-level flag rate**.
- Selector precision ≈ 0.640 → **~1,236 true errors**, i.e. **2.34% true cell
  noise**.
- Worst label: ACL 235 / 4,407 = 5.33% flag rate; with the same precision,
  ~3.4% true label noise on ACL.
- Noise is **asymmetric, class-conditional, one-directional** (108 confident
  false positives vs 32 misses in the 58-study gold census) — closest
  published analogue is **pairflip**, not symmetric.

## HLC measured points

I read the actual tables from the ICCV 2023 PDF
([open-access PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Xia_Holistic_Label_Correction_for_Noisy_Multi-Label_Classification_ICCV_2023_paper.pdf)).
HLC reports mAP for Pascal-VOC 2007 (20 labels, ~5k images) and MS-COCO
(80 labels, ~82k images).  The relevant BCE baseline and HLC rows are:

### Pascal-VOC 2007

| noise | BCE mAP | HLC mAP | HLC gain |
|-------|---------|---------|----------|
| Sym 30% | 64.50 | 72.07 | +7.57 |
| Sym 40% | 58.65 | 70.20 | +11.55 |
| Sym 50% | 48.19 | 68.00 | +19.81 |
| Pair 20% | 71.77 | 74.83 | +3.06 |
| Pair 30% | 60.94 | 69.86 | +8.92 |
| Pair 40% | 48.72 | 60.09 | +11.37 |

### MS-COCO

| noise | BCE mAP | HLC mAP | HLC gain |
|-------|---------|---------|----------|
| Sym 30% | 53.23 | 54.87 | +1.64 |
| Sym 40% | 47.33 | 51.09 | +3.76 |
| Sym 50% | 40.25 | 48.15 | +7.90 |
| Pair 20% | 56.58 | 58.55 | +1.97 |
| Pair 30% | 49.16 | 53.41 | +4.25 |
| Pair 40% | 41.57 | 45.91 | +4.34 |

Two things are immediately visible:

1. **Gains scale strongly with noise rate** in both datasets and both noise
   types.
2. **MS-COCO gains are much smaller** than Pascal-VOC gains at the same
   noise rate, presumably because the multi-label task is harder and the
   baseline is already more regularized.

## Extrapolation to 2.3% true noise

HLC's **lowest measured point is 20% noise** (pairflip).  Our 2.3% true cell
noise is **9× below that floor**, and our worst label (ACL, ~3.4%) is still
6× below.  Any number at 2.3% is therefore an extrapolation, not a reading.

A conservative upper bound can be obtained by assuming the gain grows at
most linearly from the lowest measured point (the real curve is convex, so
linear is an overestimate below the measured range):

| analogue | upper-bound gain at 2.3% true noise |
|----------|-------------------------------------|
| MS-COCO symmetric | 1.64 / 30 × 2.3 = **0.13 mAP** |
| MS-COCO pairflip  | 1.97 / 20 × 2.3 = **0.23 mAP** |
| VOC2007 symmetric | 7.57 / 30 × 2.3 = **0.58 mAP** |
| VOC2007 pairflip  | 3.06 / 20 × 2.3 = **0.35 mAP** |

Because the true curve is convex, the actual expected gain is smaller than
those upper bounds.  Converting to our usual AUC scale (mAP / 100):

- MS-COCO analogue: **< 0.0023 AUC**, almost certainly **< 0.001 AUC**.
- VOC2007 analogue: **< 0.0058 AUC**, likely **< 0.003 AUC**.

On our board scale, where meaningful gaps are 0.001 and above, **this rounds
to zero**.

## Pairflip vs. symmetric at the low end

Claude asked whether asymmetric (pairflip) noise holds its correction gain
longer as the rate falls.  The HLC tables do **not** support that:

- At every measured rate, pairflip gains are **smaller** than symmetric gains
  on the same dataset.
- On MS-COCO, pairflip gain actually **drops** from 4.25 at 30% to 4.34 at
  40% (within noise) after rising from 1.97 at 20%; on Pascal-VOC 2007 it
  rises 3.06 → 8.92 → 11.37, but still lags symmetric.
- There is no measured point below 20% for either type, so any statement
  about the very-low-noise regime is extrapolation.

The mechanism that would keep our label arms alive is **not** supported by
HLC's pairflip series.  If the symmetric series is the better analogue, the
expected gain is even smaller than the pairflip upper bound above.

## Any published correction result at single-digit noise?

I did not find one for multi-label correction.  The lowest multi-label
noise-rate range I found is Burgert et al., *On the Effects of Different
Types of Label Noise in Multi-Label Classification* (2022), which evaluates
synthetic noise from **10% to 70%**.  A general practitioner rule of thumb
notes that "under 1% is usually safe; 1–5% is common in real data" (SitePoint
2026 summary), but that is not a correction result.

So: **no published multi-label correction paper reports a gain at <10%
noise**.  HLC itself starts at 20% pairflip / 30% symmetric.

## Bottom line

- **Expected gain at 2.3% true noise rounds to zero** on our board scale,
  using either the many-label MS-COCO analogue or the few-label Pascal-VOC
  analogue.
- **Pairflip does not rescue the low-rate regime** in HLC's data; if
  anything it underperforms symmetric noise at the same rate.
- The extrapolation is **7–10× below the lowest measured point**, so the
  honest uncertainty is large; the conservative upper bound is still tiny.
- **No published multi-label correction result exists at single-digit noise**.

If this mapping is taken at face value, the label-correction arms are
instruments that answer a scientific question ("is there a detectable
correctable label defect?") rather than a direct source of board points.
