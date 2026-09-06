# Reading the same weights a second way does not buy a diversity seat

2026-09-06. Zero card. Three points, measured off records that were already on disk.

## What was asked

After the chunk-outer `FileMemo` (census `326f2b9b`: 201.0 → 105.6 slice decodes per study, 14.0 → 4.7
header parses, 96/96 bags byte-identical), `decode_bench --union --extra` (`795fb35a`) priced a FOURTH
read group at **+2.3% to +6.3%** of the deduplicated decode, where under the old group-outer path it
was a whole extra pass (+33%). That made a proposition affordable that had never been affordable
before: serve weights we already own under a SECOND read group and seat the two side by side. A
diversity seat that costs no training.

`experiments/read_group_queue.sh` was written to test it on the `twinfive` pool — the five fold-0
twins of `raptorweight_v1`'s seats (`calib_span_s2026`, `grade_ladder_s2026`, `merged_blend_cf`,
`lr4e-5_ws_s2026`, `through_dense_s2026`), LogitAverage **0.8706** over 909 studies. Pre-registered
bar: ten seats (five native + five regrouped) must read **≥ 0.8751** — baseline plus the 0.0045 board
resolution.

## What was already on disk

Three regroupings of all five twins had been back-filled by earlier work and **never scored as a
blend**: `_r224`, `_r280`, `_n12b0.1-0.9`. Reading them costs nothing — `vote_ceiling` only reads
`.holdout.npz` files.

| blend (10 seats) | LogitAverage | vs baseline |
|---|---|---|
| native five only | **0.8706** | — |
| + `_r224` | 0.8693 | −0.0013 |
| + `_r280` | 0.8707 | +0.0001 |
| + `_n12b0.1-0.9` | 0.8698 | −0.0008 |

Not one is within 0.0044 of the bar. The best is +0.0001, which is not a direction.

## The harness is not the suspect

Three checks, all clean:

1. The native five reproduce **exactly 0.8706**, the number the pool was recorded at. The record path,
   the fold line and the vote rule are all doing what they did before.
2. The named harness suspect for this experiment was record REUSE — `holdout_backfill` writes a
   regrouped record beside the native one, so a candidate that silently reused the native record would
   read exactly 0.8706 for its ten seats. None of the three did; each moved, and each moved by a
   different amount.
3. Every seat is twins-against-twins, fold 0 of 5 at split seed 2026. No champion seat crosses the
   `--fold -1` line, which is the comparison that gave a fourteen-tick confident wrong answer on
   2026-08-25.

## The mechanism

A twin of yourself is the most CORRELATED seat the pool can hold. It shares the weights, the training
data, the fold and the split — everything except which pixels reached the encoder. Under an
**unweighted** LogitAverage, adding it moves the mean toward a prediction the pool already made, so
the diversity it contributes is smaller than the dilution it costs. The freed decode budget is real;
what it buys is not a seat.

The obvious next move — weight the regrouped seats below the native ones — is a seating operation, and
that axis is closed (seven board flats, five separate closings). It is not reopened by this result.

## What this does NOT rule out

- **Upscaling.** `_r224` and `_r280` are DOWNSCALES from the 336 px the twins were fitted at. 392 is the
  other direction, and `res392_ws_s2026` — a member actually TRAINED at 392 — is the best solo member
  this project owns. That is a different sign of the same knob, not a third point on the one measured
  here. `read_group_queue.sh` is cut to that single candidate; the four depth/band candidates are
  parked behind its reading, since the depth/band axis's one point (`_n12b0.1-0.9`) is null.
- **A regroup of a DIFFERENT member.** Every point here regroups a twin against itself. A regroup that
  changes the weights too is a different proposition and is untouched by this.
- **The decode saving itself.** The memo's 47.5% is measured and stands. This says only what the freed
  budget cannot be spent on.

## Two defects found on the way

- `read_group_queue.sh` pointed `MODELS` at `.../all_ours/`, which holds a packaging COPY of each `.pt`
  and **no `.holdout.npz`** — records live beside their checkpoint in the models ROOT
  (`core/paths.py:50`). Every seat would have read `no holdout record, skipped` and the run would have
  died with *"none of the 5 named records carries a holdout array"* — which is the `--fold -1` refusal
  message, and would have been blamed on the fold line. Fixed.
- `DataRoot.checkpoints` resolves a bare name through `Path.stem`, so `..._n16b0.1-0.9` truncates to
  `..._n16b0.1-0`. The script passes full paths, which never go through stem.
