# The band is already close to a distance: the span spread is 1.3x, not 2x

**2026-08-30 · instrument, not an improvement · card-free**

## The axis this was built to price

In-plane geometry is already millimetre-normalised: `slot_reader` narrows every stack to
`crop_mm = 130.0` through the acquisition's own `pixel_spacing`. Through-plane is not. The band stays a
FRACTION of whatever the protocol acquired — `--band-low 0.1 --band-high 0.9` keeps the middle 80% of the
series, however many millimetres of knee that happens to be.

The argument for normalising the band to millimetres rests entirely on a claim about spread: if two studies
read at the same fraction see materially different extents of knee, the model is being handed an
inconsistent field of view along the one axis nothing corrects. `band_coverage` reported a MEAN span per
slot, and a mean cannot answer that question — it is the exact statistic that hides it.

## What was measured

`core/data/corpus/band_coverage.py` now carries the per-study spans it previously reduced to a mean, and
reports their p10, p90 and ratio. Over 200 studies / 923 series at band 0.2-0.8:

| slot | span mm (mean) | p10 | p90 | **p90/p10** | n |
|---|---|---|---|---|---|
| AX_FLUID_FS | 126.3 | 99.0 | 150.8 | **1.52** | 197 |
| COR_FLUID_FS | 108.3 | 93.7 | 126.0 | **1.34** | 193 |
| COR_T1 | 108.1 | 94.5 | 126.0 | **1.33** | 133 |
| SAG_FLUID_FS | 103.0 | 87.3 | 117.6 | **1.35** | 190 |
| SAG_FLUID_NOFS | 104.9 | 93.5 | 118.5 | **1.27** | 123 |
| SAG_T1 | 100.8 | 84.9 | 115.5 | **1.36** | 87 |

## The reading

**The instrument argues against the axis it was built to price.** The mm-normalisation case needed the
same fraction to read as materially different distances across studies. It does not: the sagittal slots —
where every band arm this project has run was decided — span 1.27-1.36 between the tenth and ninetieth
percentile. Acquisition protocol is far more standardised through-plane than the axis assumed. Converting
the band from a fraction to a fixed millimetre extent would move the decode's field of view by ~15% on the
studies at each tail and by nothing at all in the middle, which is well inside the region where the band's
own fraction sweep was FLAT.

This is consistent with the axis's other closed result: the band optimum is INTERIOR (0.20-0.80 = 0.8525,
0.10-0.90 = 0.8565, 0.00-1.00 = 0.8427), so the band behaves as a filter that discards uninformative end
slices, not as a budget that has to buy a fixed anatomical extent. A filter does not care about
millimetres.

AX_FLUID_FS at 1.52 is the one slot with real spread, and it is also the slot with the largest mean span —
axial coverage is where protocols disagree. If mm-normalisation is ever re-opened, it should be re-opened
for the axial slot alone and priced against that slot's own contribution, not as a global decode change.

## Status

Axis NOT armed. Cost was one card-free report edit; the expected gain never cleared the bar because the
premise it needed is measurably false. Recorded so the argument is re-opened by answering this
measurement, not by re-stating the premise.
