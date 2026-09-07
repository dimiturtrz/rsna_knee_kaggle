# Cell C — pre-registration, written before it starts

Cell C is armed inside a LIVE watcher (`experiments/after_cellc.sh`, pid waiting on A'), and a live watcher
is never edited in place — bash reads a script incrementally, so an edit to a running one is a rewrite of
instructions it has not read yet. So C's header lives here instead, committed before the run begins.

## What actually moved, measured — and it is not one variable

| | his published corpus (A') | our maxspan corpus (C) |
|---|---|---|
| `all_vols.npy` | `(3200, 44, 336, 336)` uint8 | `(3200, 64, 336, 336)` uint8 |
| `extra_vols.npy` | `(1207, 44, 336, 336)` | `(1207, 64, 336, 336)` |
| slot layout | `Sag1 12 · Sag0 10 · Cor1 8 · Cor0 6 · Ax 8` = 44 | `Sag1 18 · Sag0 14 · Cor1 12 · Cor0 8 · Ax 12` = 64 |
| span | `[0.15, 0.85]` | `[0.02, 0.98]` |
| crop | 140.0 mm | 140.0 mm |

**RECOVERED FROM THE PIXELS, 11:08Z (researcher), and it revises this table's first column.** The 44
corpus records none of its own geometry, so it was read back by matching its slices to maxspan's: 50
studies, each 44-slot plane downsampled 4x and cross-correlated against all 64 maxspan slots of the same
study (mean best-match NCC 0.950, 41.9 of 44 slots valid, 99.6% of adjacent matches monotone). Slot layout
confirmed independently of the NCC by per-slot occupancy over all 3200 studies — the missing-plane runs
break at the same fractions in both corpora (Sag0 at 0.97, Cor0 at 0.74). Crop confirmed by tissue
bounding box on matched pairs: ratio median 1.000, IQR [1.000, 1.000] over 2096 pairs, so the two share
the crop code path.

So **C moves TWO variables, not three**: slot count 44 → 64 and span 0.15–0.85 → 0.02–0.98. The crop is
constant. Its ledger row still says UNPRICED-span, because slot count moves with it.

**AND HIS KERNEL'S SPAN MATCHES NEITHER.** `[0.06, 0.94]` is not the corpus he published (0.15–0.85) and
not maxspan (0.02–0.98); a first slot at 0.06 never lands inside any run's IQR. So his serve geometry is
a third setting, and the comment at `:315-317` describing 0.15–0.85 as the span that "threw away" the
collaterals is a claim about the corpus he published — which is the one A' is fitted on.

**WHAT OUR OWN LADDER PREDICTS, now that the numbers are known.** Our three trained span points are
0.20–0.80 = 0.8525, 0.10–0.90 = 0.8565, 0.00–1.00 = 0.8427 — an interior optimum with the outer tenths
measured as active distraction. A' at 0.15–0.85 sits beside our best-but-one point; C at 0.02–0.98 sits
essentially at the point that lost 0.0138. If span behaved on his stack as it does on ours, **C reads
BELOW A'**, and the pre-registered direction is therefore negative rather than neutral. What could
overturn that and is the reason to run it anyway: his k=12 window sampler redraws its window every step,
so the extra depth accrues across epochs instead of thinning a fixed budget — our ladder moved span at a
FIXED slice count and cannot see that case. C is the only cell that separates the two. His loader reads the depth dynamically (`self._D = self.vols.shape[1]`, T:217), so 64
runs without an edit; nothing in his training script hard-codes 44 outside its docstring.

## The reason that reframes the whole 2x2

His own inference kernel states the geometry his RELEASED WEIGHTS were trained on, and it is not the corpus
he published for training:

- `knee-mri-twelve-findings-from-a-single-model.py:93-97` — *"64 slices per study instead of 44, same
  proportions. Must match the corpus the weights were trained on (knee_corpus_v4.py)"*, with
  `SLOTS` summing to 64 and `CROP_MM = 140.0`.
- `:315-317` — *"wide span: the collateral ligaments and lateral meniscus live in the peripheral slices the
  old 0.15-0.85 crop threw away. Must match the corpus the weights were trained on"*, and the span it then
  builds is `[0.06, 0.94]`.
- `knee-mri-training-the-twelve-finding-model.py:13` — the corpus he PUBLISHED is *"a fixed 44 x 336 x 336
  uint8 stack"*.

So **A' is not the replication cell; C is closer to one.** A' fits his recipe on an older corpus than the one
his v4 weights were fitted on, and the A'-minus-v4 gap therefore carries a geometry term nobody had priced.
Our maxspan span is `[0.02, 0.98]` against his stated `[0.06, 0.94]` — two slices at each end of a typical
series, one residual difference, named here rather than discovered later.

## Bands, on OOF-896 under the LEDGER ruler, against A' (never against his published 480/board numbers)

- **≥ A' + 0.005** — the geometry his weights were actually trained on is a lever. Carry it: the effv2l cell
  gets the 64-slot corpus too, and the A'-minus-v4 gap is re-read as geometry rather than as labels or SSL.
- **within ±0.005** — the 44-slot corpus loses nothing his kernel's comment claims; geometry closed at his
  published default, and the remaining A'-v4 gap stays with labels/SSL init.
- **≤ A' − 0.005** — the wider span hurts under a 12-window sample: 64 slots at k=12 spreads the same windows
  over more anatomy, so each window is sparser. Name the columns — the pre-registered loser is Synovitis
  (D14: a pipeline column, our dense geometry wins it) and the pre-registered winner is LatOA/Fracture,
  which is what the peripheral slices are for.

Noise floor for a paired OOF-896 read: 0.0062. A |delta| under that is not a reading in either direction.
