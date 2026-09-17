# Every knob audit here read the arm scripts, and a default is written in none of them

2026-09-17. Bead knee-e9f83, knee-dry3h. No card spent: this is a read of the run records already on
disk, plus one instrument built to read them.

## The blind spot

`tri.py`, the shipping-corner flag audit, and the weight-decay join all answer the same question the same
way — they parse `experiments/arms/*.sh` and look for two arms that differ in one flag. That works for a
flag an arm WRITES. It cannot see a leaf left at its default, because a default is written in no arm
script at all. This is why the corner's audit reported 40 of its 93 leaves as "sits at the default,
unpriceable", and why `scratchpad/tri.py` reported that a certain three-point curve existed nowhere in
407 arms when in fact its middle point was a default.

The resolved side is on disk and always has been. **`<stem>.metrics.jsonl` line 1 is
`{"recipe": {...}}` with all 93 leaves and every default filled in.** 403 runs carry one, read through
`RecordedRun`, which falls back to the checkpoint's stamped `options` where the metrics file predates
the stamp. The jsonl is text and needs no torch; the checkpoint is the fallback, not the first read.

## The instrument

`kneemri/experiments/recipe_pairs.py` reads every resolved recipe, joins them to the `ledger75` table,
and reports pairs of scored runs differing in EXACTLY ONE leaf:
`python -m kneemri.experiments.recipe_pairs --holdouts <dir> [--key <leaf> …]`. Against an arms-based read that found 16 isolating pairs
for `backbone_lr` and declared several keys unpriceable, it finds **103 pairs across 31 keys**.

Four guards had to be built into it, and each was added only after its absence produced a wrong number
in this session. They are the whole reason the count is 103 rather than 1015 or 162.

**Population.** `ledger75` prints one block per holdout population under the legend *"ranks below do NOT
cross this line"*. The first run ignored those headers and returned 162 pairs across 38 keys, whose two
largest key-groups were `fold` (42 pairs) and `exclude_fold` (4). Those cross populations by
construction — a different fold holds out different studies — so they compare numbers read on different
patients. Confining pairs to population `66c6f231` (909 studies) makes both keys vanish entirely, which
is the check that the fix worked. Reported as findings, they would have shipped 46 nonsense pairs and
`fold` deltas up to ±0.028 read as knob effects.

**Truncation.** A TRUNCATED run holds a `ledger75` row like any other, and under `one_cycle` its score is
a FLOOR rather than a reading. A key's pairs are sorted by |delta|, so the dead processes land at the top
of every table — precisely the rows a reader quotes. **21 scored runs on the main population are
truncated.** Filtering to runs that reached their own declared epoch budget drops 8 pairs and FOUR ENTIRE
KEYS.

**Schema drift.** A leaf absent from an older run's recipe resolves to `None`, which is not a value the
knob ever took. Tolerating that asymmetry — comparing only the keys both sides declare — inflated the
table to **1015 pairs across 55 keys**, admitting pairs whose recipes were eighteen and thirty-six leaves
apart as "isolating". Drift has to BREAK a pair: two runs whose recipes are not the same shape are not
isolating anything.

**Void.** A member fitted on its own holdout reads high and cannot be compared against one that was
not. This axis only appeared when the instrument was promoted into the package and reused
`Ledger75Row.void` rather than re-deriving the completeness test: the run reports `427 of 446 records;
19 excluded as VOID`. The same promotion corrected the completeness test itself — a hand-rolled
`max(epoch)+1 >= asked` drops a `--patience` STOPPED run as though it were a truncation, where
`ConvergenceCheck.settled()` separates the two. That correction is why 108 pairs became 103.

## The retraction this instrument owes

Before the truncation guard, the largest delta in the entire table was:

    layer_decay  0.75  0.9007  ->  1.0  0.8054   -0.0953
    labelwise_dense_stem_llrd_s2026 / labelwise_dense_stem_s2026

The shipping corner runs `layer_decay 1.0`, so this read as the corner sitting on the losing side of the
biggest knob effect the project had ever measured. It is not a knob effect.
**`labelwise_dense_stem_s2026` holds 4 rows, last epoch 3, against 24 asked.** It is a dead process, and
0.8054 is a floor under an anneal that never ran.

What the two runs that DID finish actually say is the opposite, and it is small. The LLRD arm shares the
shipping corner's recipe on every leaf except two — `layer_decay` 0.75 against 1.0, and `unfreeze_stem`
True against absent — and it scores **0.9007 against the corner's 0.9053**. LLRD plus an unfrozen stem is
**0.0046 WORSE**, inside the paired floor. The corner is fine where it is, and after the guard
`layer_decay` has **zero** isolating pairs: it is unpriced, not flat.

`slice_regions` and `unfreeze_last` remain zero-pair too. The default-join does not rescue them. The
suspicion that it would was this session's, and its own instrument refutes it.

## What survives at or near the shipping corner

All `drift0`, all on population `66c6f231`, all between runs that finished:

| key | low | high | delta | where |
|---|---|---|---|---|
| `whole_stack` | False 0.8523 | **True 0.8914** | **+0.0391** | confirms the corner |
| `head_lr` | 5e-3 0.8670 | 2e-4 0.8777 | +0.0107 | endpoints 25x apart, corner between |
| `weight_decay` | 0.1 0.8820 | 0.005 0.8714 | -0.0106 | same shape, corner between |
| `stochastic_depth` | 0.1 0.8907 | 0.05 0.8974 | +0.0067 | NOT at the corner |
| `scheme` | roi 0.8984 | **roinative 0.9018** | **+0.0034** | corner's own drop40_long family |
| `slices` | 20 0.8970 | 24 0.8998 | +0.0028 | confirms the corner |
| `stochastic_depth` | **0.0 0.9048** | **0.05 0.9051** | **+0.0003** | AT the corner, flat |

Nothing here clears the 0.0062 paired floor at the corner. The two 0.010s are the pairs the rate
interpretation already set aside for a reason that still holds: the corner's own value sits BETWEEN two
deliberately extreme endpoints, so the pair prices a gradient across a 25x span and says nothing about
the step from where the corner actually is.

The one thing in the table that is not exhausted is `scheme`. The corner runs **`canonical`**; the only
pair compares `roi` against `roinative`, a third value the corner has never taken, and
`roinative_labelwise_drop40_long_s2026` at **0.9018** is the third-best solo the project owns.

### `scheme` closed by DECOMPOSITION rather than by a pair

The corner and the roinative run are both complete at 24 of 24 and carry zero schema drift, so they are
directly comparable — but they differ on six leaves, not one, which is why no pair exists:

| leaf | corner | roinative | this table's own delta |
|---|---|---|---|
| `slices` | 24 | 20 | 20 -> 24 **+0.0028** |
| `stochastic_depth` | 0.0 | 0.05 | 0.0 -> 0.05 **+0.0003** |
| `scheme` | canonical | roinative | the residual |
| `accumulate` / `batch` | 1 x 10 | 3 x 4 | effective 10 against 12 |
| `vram_gb` | 28 | 0 | bookkeeping, no effect on the recipe |

Observed: 0.9053 against 0.9018, **-0.0035** for roinative. Subtracting the two leaves this table
prices independently leaves about **-0.0010** for `scheme` plus the batch-shape difference — flat, deep
inside the 0.0062 floor.

**This is a subtraction, not a measurement**, and it assumes the `slices` and `stochastic_depth` deltas
transfer from the pairs they were read on to this corner. That is the same additivity assumption that a
single-key pair does not need, which is exactly why a pair is worth more. It is quoted here as a bound:
for `scheme` to be worth a card at the corner, the two subtracted terms would have to be wrong in the
same direction by four times their measured size.

So the knob axis at the shipping corner is **closed**, not merely unfinished. Of the 31 keys with
isolating pairs, the only ones clearing the floor price gradients across 25x spans with the corner
sitting between their endpoints, and the one key left over closes by decomposition. The next 1.5% is not in this recipe's
flags, and further single-flag A/B arms at this corner are priced at roughly zero expected gain — they
are instruments for retiring a suspicion, never improvements.

## The general shape

This is the third instance in two days of the same defect, and the instances are getting cheaper to find
because the shape is now named. The ledger compared numbers read on different study sets and said nothing
about it. The corner carried a rate fitted under a different schedule and said nothing about that. Here
an audit read the side of the record that is written down and was structurally blind to the side that is
not — and then the instrument built to fix that was itself blind to which of its rows were dead
processes. In every case nothing was broken and every check was green. What was missing each time was a
record of what the number is a statement ABOUT.

## What the instrument found that this audit did not look for

`labels` is the leaf it prices most — 30 pairs, 17 clearing the floor, against every other leaf's 0-6 —
and reading that as a lever is a trap. Every clearing pair sits on the older `*_span_s2026` family,
which tops out 0.027 below the corner. See
`2026-09-17_labels_is_the_most_priced_leaf_and_the_corner_has_tried_three_schemes.md`.
