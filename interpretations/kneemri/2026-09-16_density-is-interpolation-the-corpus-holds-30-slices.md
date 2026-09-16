# Density is interpolation — the corpus holds ~30 slices at 3.5 mm

2026-09-16. Closes the P1 framing raised from Kaggle forum topic 737696 and re-prices the through-plane
density lever at about +0.0005, below this project's 0.0062 noise floor. Bead knee-3kpv9.

## The claim that was filed

Forum topic 737696 (Dread Development, the Raptor weights author, 2026-09-15) reports a monotone series at
FIXED weights, varying corpus geometry alone: 44 slices over span 0.02–0.98 → 0.917; 44 over 0.06–0.94 →
0.926; 64 wide → 0.928; 80 wide → 0.932. His reading: *"Density is the lever rather than span, and 1.20
percent of the stack per slice is the finest I have measured and also the best."* A separable second axis in
the same post moves window coverage at fixed weights: 42 → 0.924, 62 → 0.927, 78 → 0.932.

Ours is 20 slices over band 0.10–0.90 (`kneemri/experiments/arm_check.py:83-88`) = 4.0% of the stack per
slice. Against his best that is 3.3× coarser, and coarser than his WORST row. It was filed P1 on that
arithmetic.

Every number in that paragraph is HIS board on HIS pipeline (CoAtNet384, 94 windows), UNVERIFIED against our
ruler and on a different denominator to our seven-member ensemble.

## What the ledger already held

`interpretations/kneemri/2026-08-19_density-inside-the-band-trades-centre-for-edge.md` had priced this square
at our end of the curve — 909 holdout, fold 0, seed 2026:

| slices | band 0.20–0.80 | band 0.10–0.90 |
|--------|----------------|----------------|
| 12     | 0.8525         | 0.8565         |
| 20     | 0.8540         | 0.8578         |

Density effect **+0.0014**, reproducing at both spans. Span effect **+0.0039**, reproducing at both
densities. Interaction ~0.0001. Band 0.00–1.00 at 12 slices reads 0.8427, so the span optimum is interior.

Two disagreements with his series, and they point the same way: our span sign is OPPOSITE his (−0.009 his,
+0.0039 ours), and our density effect is **10× smaller** than the slope his rows imply.

## What the corpus settles

Native acquisition geometry, 400 studies sampled, label-free and GPU-free, read off the routed slots'
`acquisition.slice_count` and `slice_spacing` (now `core/data/corpus/native_depth.py`):

| slot          | n   | p10 | median slices | p90 | spacing (mm) |
|---------------|-----|-----|---------------|-----|--------------|
| SAG_FLUID_FS  | 380 | 21  | 29            | 36  | 3.50 |
| COR_FLUID_FS  | 386 | —   | 30            | —   | 3.60 |
| AX_FLUID_FS   | 396 | —   | 32            | —   | 4.00 |
| SAG_T1        | 177 | —   | 30            | —   | 3.30 |

**A series holds about 30 slices at 3.5 mm.** So an 80-slice read over 2–98% is 2.7× INTERPOLATION of a
30-slice acquisition, and "1.20% of the stack per slice" is 1.25 mm of resampled 3.5 mm data. His +0.015
cannot be through-plane resolution, because the information is not in the DICOMs.

Our 20 slices over 0.10–0.90 already read about **83% of the acquired in-band slices**. The honest ceiling on
this lever for us is ~24 slices — a +20% move, not 4×.

## What his series more likely is

Interpolated slices and his separable window axis (42 → 0.924, 62 → 0.927, 78 → 0.932) are ONE mechanism:
more reads of the same acquired voxels, i.e. TTA. This repo already closed that for whole-stack members on
five measured members (knee-eeo). That is a PRIOR, not a verdict — our closure was at our slice budget, and
his stack is dense-resample *plus* windows, which our five members never combined.

## Consequence

The P1 framing "we sample 3.3× coarser than the best-measured point" is **withdrawn**. The remaining live
question is much smaller: 20 → 24 slices at band 0.10–0.90 to reach full native in-band coverage, whose own
prior from our own square is about **+0.0005** (the 12 → 20 slope continued) — below the 0.0062 noise floor.
Not worth a card arm on its own; worth folding into the next geometry cache if one is built for another
reason. The bead dropped P1 → P3.

The method error is worth keeping separately from the result: the filing quoted a competitor's percentage of
stack per slice against ours without ever asking what the DICOMs contain. A ratio between two read geometries
is not a ratio between two resolutions when both reads sit above the acquisition.

## Sources

- Kaggle forum topic 737696, 2026-09-15 (his numbers, his pipeline, unverified here).
- `interpretations/kneemri/2026-08-19_density-inside-the-band-trades-centre-for-edge.md` (our square).
- `core/data/corpus/native_depth.py` — `python -m core.data.corpus.native_depth` reproduces the table.
- Bead knee-3kpv9 (the correction and the re-pricing); knee-eeo (the TTA closure).
