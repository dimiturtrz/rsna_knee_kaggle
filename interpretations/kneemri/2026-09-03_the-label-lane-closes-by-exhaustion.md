# The label lane closes by exhaustion

2026-09-03. `unread_dense` was the strongest shot the label lane had, and it read null.

## What the arm did that no prior label arm did

14,445 of 52,872 (study, finding) pairs — **27.3% of all supervision** — carry the UNK verdict: the
report never named the structure, so `report_reader` emits one constant for every study in the corpus.
Those cells carry no ORDER, and order is the whole of what the metric reads.

`unread_dense` replaced that constant with an out-of-fold score from the PUBLISHED pool (20
checkpoints, 5 folds x 4 seeds, partitioning all 4,407 studies), so every cell was scored by a model
that did not train on it, corpus-wide, for no card hour. Score only, never weight. Mean-preserving
inside each finding's UNK block, band [0.08, 0.48] derived as the nearer gap in `MEASURED_AGREEMENT`
rather than chosen.

Every prior label arm — `--calibrated-targets`, `--finding-calibration`, `--label-errors`,
`--weight-floor`, `--certainty-ramp`, `--severity-weight` — moved a value or a weight on cells the
reader had **already read**. This one is the only arm that touched the quarter of the signal that was
never read at all.

## The reading

Converged 12/12, peak e6 = 0.8577, `convergence_check` clean — not truncated, not a patience stop.

Primary ruler is gold-58 via `annotated_readings`, both members under `--hold-annotations` so neither
trained on those cells:

| member | gold-58 mean AUC |
|---|---|
| `band_dense_hold_s2026` (control) | 0.8956 |
| `unread_dense_s2026` | 0.8976 |
| combined | 0.8994 |

**Paired delta +0.0020 [-0.0155, +0.0200]. INSIDE the interval — the middle pre-registered branch, a
null.** The report proxy agreed (0.8577 vs the control's 0.8573, +0.0004) and carries nothing either
way: the header registered it as uninterpretable BEFORE the run, because the report holdout scores
against report-derived labels whose UNK cells ARE the constants this treatment replaces.

## Why this closes the lane rather than adding one more null

The arm's own header wrote the closure in advance: *"A mechanism touching a quarter of the training
signal is the only label-side thing left that COULD be 1.5%, and if it lands at +0.01 like the rest the
lane is closed by exhaustion rather than by one more null."* It landed at +0.002.

Every measured label-lane delta this project owns sits between +0.006 and +0.017. The arm designed to
break out of that band by moving twenty times more supervision than any of them landed at the bottom
of it.

## Both named fallbacks are killed by arithmetic, not run

The header declined to rule out two variants. Neither survives a division:

* **The weight-side variant** (`--weight-floor` rather than the score) reaches the same UNK cells
  through a weaker channel — how hard the cell pulls rather than what it pulls toward.
* **Restriction to the five findings with positive `supervision_gap`** (Synovitis +0.1075, Fracture
  +0.0715, Lateral Meniscus +0.0522, Baker's +0.0462) reaches a SUBSET of the cells.

Both reach fewer cells than the full application that just scored +0.0020, so both have a lower
ceiling than the number they would have to beat. Two 80-minute arms not run, on arithmetic.

## What is NOT closed by this

The label lane is about the SUPERVISION. It says nothing about the trunk, the input geometry, or the
pool's structural monoculture. `annotated_readings` also shows the two members COMBINE to 0.8994 —
above either alone — which is the ordinary ensembling gain and not a label result.

The one thing this does retire: *"get better labels"* is no longer an answer to the 0.008 board gap.
It was the cheapest-sounding lever left and it has now been measured at its own maximum.

## Withdrawn heading

`experiments/README.md:6690` reads *"The label lane is priced, and it does not pay"*. That heading
rests on `stack_increments_fx`, which was TRUNCATED at 5 of 12 epochs under `one_cycle` — so its
-0.0040 is a FLOOR, not a price. The conclusion happens to survive; the evidence under it does not.
This file is the honest version, and the heading should point here.
