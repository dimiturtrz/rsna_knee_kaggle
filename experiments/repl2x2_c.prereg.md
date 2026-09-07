# Cell C — pre-registration, written before it starts

Cell C is armed inside a LIVE watcher (`experiments/after_cellc.sh`, pid waiting on A'), and a live watcher
is never edited in place — bash reads a script incrementally, so an edit to a running one is a rewrite of
instructions it has not read yet. So C's header lives here instead, committed before the run begins.

## What actually moved, measured — and it is not one variable

| | his published corpus (A') | our maxspan corpus (C) |
|---|---|---|
| `all_vols.npy` | `(3200, 44, 336, 336)` uint8 | `(3200, 64, 336, 336)` uint8 |
| `extra_vols.npy` | `(1207, 44, 336, 336)` | `(1207, 64, 336, 336)` |
| slot layout | 44, undocumented split | `Sag1 18 · Sag0 14 · Cor1 12 · Cor0 8 · Ax 12` = 64 |
| span | not recorded with the corpus | `[0.02, 0.98]` |
| crop | not recorded with the corpus | 140.0 mm |

Slot count 44 → 64 AND span AND crop all move together, so **C does not price span alone**. Its ledger row
says UNPRICED-span. His loader reads the depth dynamically (`self._D = self.vols.shape[1]`, T:217), so 64
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
