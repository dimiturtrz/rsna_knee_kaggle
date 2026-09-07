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

## AMENDED 11:40Z, after A' read and BEFORE any C number exists

A' finished at 11:20Z. C's watcher took the card at 11:26Z and is at ep0-ep1; `logs/repl2x2_c.full.log`
holds no epoch score, so nothing about C has been read and this amendment is still a pre-registration.
The A' numbers below were taken on CPU from the OOF file A' wrote itself
(`raptor_oof_repl2x2_a_fold0.npz`, 896 studies), re-scored under the LEDGER truth table by
`kneemri.evaluation.foreign_holdout` — no card, seconds. The card exception offered at 11:33Z was not
needed and was not taken.

### A' on OOF-896 under the LEDGER ruler

| | macro |
|---|---|
| A' best epoch (ep12) | **0.8307** |
| A' SWA (ep 12/9/14) | 0.8328 |
| `raptor_coat384` — HIS RELEASED WEIGHTS, same architecture | **0.8508** |
| `raptor_cnv2b336` | 0.8550 |
| `raptor_effv2l480` | 0.8703 |

corr(A', coat384) = 0.837 rank-mean over 12 columns, the highest of any pair here (cnv2b336 0.789,
effv2l480 0.758) — A' is in his family, it is simply worse. Overlap 896 of his 909.

**A' − v4(coat384) = −0.0201**, three times the 0.0062 paired floor. Under the bands as issued that is
the **< 0.840 harness-signature** cell. It is not a harness signature, and the per-column table is why.

### The gap is not flat, and its shape is HIS claim

| | A' | v4 coat384 | delta | pos |
|---|---|---|---|---|
| LatMen | 0.8129 | 0.8681 | **−0.0551** | 221 |
| PCL | 0.7448 | 0.7894 | **−0.0446** | 169 |
| Fracture | 0.8088 | 0.8465 | **−0.0378** | 65 |
| LCL | 0.8110 | 0.8444 | **−0.0335** | 232 |
| ACL | 0.8137 | 0.8465 | **−0.0328** | 243 |
| Cartilage | 0.8342 | 0.8550 | −0.0208 | 403 |
| MCL | 0.8640 | 0.8796 | −0.0156 | 327 |
| MedMen | 0.8849 | 0.8956 | −0.0107 | 475 |
| PFOA | 0.8689 | 0.8787 | −0.0098 | 265 |
| Baker's | 0.9087 | 0.9056 | **+0.0030** | 214 |
| MedOA | 0.8153 | 0.8074 | **+0.0079** | 548 |
| LatOA | 0.8016 | 0.7920 | **+0.0096** | 116 |

The five worst columns are the collateral and cruciate ligaments, the lateral meniscus and fracture.
The three columns where A' is AHEAD are the OA columns and Baker's. That is, line for line, the claim at
`knee-mri-twelve-findings-from-a-single-model.py:315-317`: the peripheral slices the 0.15-0.85 span
throws away are where the collaterals and the lateral meniscus live. A' is fitted at 0.15-0.85; his
released weights are fitted at 0.02-0.98.

**Why this is not just a training-budget deficit.** A' is one fold at 16 epochs against his full program,
so a uniform shortfall was expected and is the null. A budget shortfall depresses every column; it does
not leave MedOA, LatOA and Baker's AHEAD by +0.003 to +0.010 while five ligament columns fall 0.033 to
0.055. Budget is still in the gap — it is what the flat columns' size bounds — but it cannot be the
STRUCTURE of it.

### What this does to C's predicted direction: it FLIPS

The direction registered above was NEGATIVE, from our own three-point span ladder (0.10-0.90 = 0.8565
beats 0.00-1.00 = 0.8427 by 0.0138). That ladder moved span at a FIXED slice count, so every widening was
paid for out of central density. C does not pay that: 44 slots become 64, so the periphery is added
rather than traded, and his k=12 sampler redraws its window every step so the extra depth accrues across
epochs. **The A' per-column signature is evidence for his side of exactly that distinction**, and it was
not available when the negative direction was written.

Both readings stand, and C separates them. So the bands are unchanged and a SECOND, sharper reading is
registered on top of them:

- **Macro, unchanged**: ≥ A'+0.005 → geometry is a lever · ±0.005 → closed · ≤ A'−0.005 → wider span hurts.
- **Per column, registered now**: if C's gain over A' lands in **LatMen, PCL, LCL, ACL, Fracture** and
  leaves **MedOA, LatOA, Baker's** flat to slightly down, the mechanism is the periphery and it is his,
  not a budget or a seed. If C gains uniformly across all twelve, it is capacity from 44→64 slots and the
  span claim is unproven. If C gains on the OA columns and not the ligaments, both stories are wrong and
  the cell is a variance reading.
- **The number to beat is A' = 0.8307, never coat384's 0.8508** — C is one fold at 16 epochs like A' is,
  and the budget term cancels only between them.
