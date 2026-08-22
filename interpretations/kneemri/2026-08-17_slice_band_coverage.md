# What the 0.20–0.80 slice band actually discards

**Measured** 2026-08-17, `python -m core.data.corpus.band_coverage --sample 400`, 400 studies sampled of
4407 in the corpus, 1848 series decoded, 0 decode failures. Label-free, GPU-free. Denominator is the
FULL series read at depth 24, which is the only denominator that can price what a narrower band removes.

## The constant

`slot_reader.SLICE_BAND = (0.20, 0.80)` carried one line of justification — "the outermost slices of a
knee series are mostly soft tissue outside the joint" — and no number. Every member on disk was fitted
under it (knee-nas5: `band_low`/`band_high` appear in none of the six all-ours checkpoints' recorded
options), so it is the one reading rule the pool has never varied, on the decode axis that transfers to
the board at 2.4× the usual compression.

## The numbers

| slot | discarded | 95% CI | p90 | low end | high end | span mm | cut/end |
|---|---|---|---|---|---|---|---|
| AX_FLUID_FS | 0.4162 | [0.4102, 0.4221] | 0.4796 | 0.1842 | 0.2320 | 125.8 | 25.2 |
| COR_FLUID_FS | 0.3924 | [0.3840, 0.4007] | 0.4950 | 0.0865 | 0.3060 | 107.6 | 21.5 |
| COR_T1 | 0.2917 | [0.2867, 0.2968] | 0.3486 | 0.0903 | 0.2014 | 107.0 | 21.4 |
| SAG_FLUID_FS | 0.3545 | [0.3474, 0.3609] | 0.4286 | 0.1726 | 0.1820 | 103.8 | 20.8 |
| SAG_FLUID_NOFS | 0.3645 | [0.3581, 0.3704] | 0.4246 | 0.1796 | 0.1849 | 104.6 | 20.9 |
| SAG_T1 | 0.3725 | [0.3630, 0.3816] | 0.4474 | 0.1843 | 0.1882 | 101.8 | 20.4 |

**The baseline is 0.4000, not 0.** A band 40% wide discards 40% of a series whose tissue is uniform
along the axis. The constant's claim predicts a share *below* that; a share at or above it means the
discarded ends hold more tissue per slice than the retained middle.

## The verdict is split, and neither half is a result

**Tissue share leans the constant's way.** SAG_FLUID_FS discards 0.3545 against the 0.4000 uniform
baseline — the ends are genuinely sparser than the middle. That is the constant, directionally
confirmed. It is not the 0.02 that would have killed the hypothesis outright (a third of the imaged
tissue is still thrown away), but the cheap kill direction did not go the hypothesis's way.

**Distance leans the other.** 20.8 mm is cut from *each* end of a 103.8 mm sagittal span. The
acceptance test written into knee-1v0n before the measurement was 8–15 mm = mechanism live, 2–3 mm =
dead. 20.8 mm clears the live bar by 40%.

They disagree because they measure different things and **the measurement cannot reconcile them:
discarded tissue is not discarded joint.** The outermost sagittal slices hold thigh and calf — which is
exactly why the share comes in sub-uniform — *and* they hold whatever compartment sits in the outer
20 mm. Nothing label-free separates those two.

**No medial–lateral asymmetry.** SAG_FLUID_FS loses 0.1726 at one end and 0.1820 at the other. This is
the part that matters most, because it is what the hypothesis was built on: knee-yrg's Lateral Meniscus
(0.7981, weakest of twelve, medial beating lateral by +0.0622) suggested something cutting the lateral
side harder. A symmetric cut is not an obvious source of an asymmetric label deficit. **That route is
weakened, and knee-1v0n dropped P1 → P2 on it.**

## The method error, recorded because it moved the headline by 0.10

The first run thresholded tissue with **per-slice** Otsu — `TissueCentre`'s own rule, reused on the
assumption that reusing it was the conservative choice. It is not, for this question. Per-slice Otsu is
correct for a *within-slice* question (where is this slice's centroid) and wrong for a *between-slice*
one (is there more tissue here than there): a background-only end slice gets its own histogram split, so
its noise is thresholded into "tissue" and the tail inflates — the very tail being measured.

Per-slice read: SAG_FLUID_FS 0.4598, *above* the uniform baseline, i.e. "the ends are denser than the
middle, the constant is backwards." One threshold per series: 0.3545, below baseline. **The artefact was
large enough to reverse the sign of the finding**, and it was in the direction that flattered the
hypothesis under test.

## What would settle it

A band-diverse member (0.10–0.90) trained and scored against the current pool. That costs a fresh 99 GB
cache and a GPU slot, and it is now gated behind the asymmetry argument having failed — what remains is
"a diversity axis nothing has ever varied", which is knee-nas5's general claim rather than a
lateral-meniscus mechanism. Not a reason to spend the slot ahead of the P0s.

## What this bought regardless

`Acquisition.through_plane_mm` (`SpacingBetweenSlices` × slice count, thickness only as fallback). The
repo had in-plane `pixel_spacing` only, so before this **no through-plane reading rule in this codebase
could be stated as a distance** — SLICE_BAND was a share of an unknown extent. Sagittal spans measure
101.8–104.6 mm and axial 125.8 mm, which is the scale any future slice-sampling argument gets to use.
