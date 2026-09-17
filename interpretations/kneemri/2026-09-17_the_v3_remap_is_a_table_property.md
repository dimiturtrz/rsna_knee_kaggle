# The v3 re-map is a property of the TABLE, and only the gold ruler can read it

*2026-09-17, 12:00Z. No card spent on this reading — both members were on disk, one since 2026-08-30 and
its gold number since 2026-09-03. `python -m kneemri.evaluation.ledger75 --only recal_dense_s2026
band_dense_hold_s2026` plus `logs/gold_recal_paired.log`.*

## The result was complete for fourteen days and its branch was never selected

`report_labels_v3.csv` is `report_labels_v2.csv`'s verdicts with each of the reader's five confidence
rungs replaced by that rung's MEASURED positive rate on the 58 annotated studies
(`kneemri.evaluation.level_calibration`, knee-7oep): 0.08/0.28/0.68/0.82/0.94 becomes
0.0408/0.1356/0.3776/0.6879/0.9583. It is a recalibration, not a second reader — the map is strictly
increasing, so every per-label rank AUC of the table is unchanged and only what the model is ASKED TO
FIT moves.

`recal_holdannot` landed that re-map's first result at the 224 corner on 2026-08-30 and it read LANDS.
`recal_dense.sh` was then written to ask the one question that decides whether to buy it — *is the gain a
property of the RE-MAP, or of the 224 corner it was measured at?* — with three pre-registered branches.
It ran the same day. Its gold reading was taken on 2026-09-03. Nobody selected a branch.

```
                      proxy ledger75    gold (58 annotated)
band_dense_hold_s2026       0.8877               0.8956      control
recal_dense_s2026           0.8923               0.9128      one moving token: --labels

paired against band_dense_hold_s2026, resampled over studies:
  recal_dense_s2026    +0.0172 [+0.0004, +0.0343]  clears
```

That is REPLICATES, the arm's own first branch, verbatim: *"gold above `band_dense_hold`'s by the floor
or more, proxy not down by the floor. The re-map is a property of the TABLE and not of a corner. Price a
roster rebuild on v3 and spend a slot."* The proxy is not merely not-down, it is **+0.0046 UP**.

## Why this pair is readable at all, when the first one was not

`--eval-labels` is PINNED to v2 on both arms. That is not a calibration detail. The re-map sends the 0.68
rung to 0.3776, which CROSSES the midpoint the holdout truth is binarised at, so an unpinned eval table
re-binarises the truth that travels with the predictions — 2017 positives against the canonical 3336 —
and the ruler moves with the arm. `recal_holdannot_s2026` left it unset and still prints **0.9004
`ledger75`, the highest number on the shelf**, off a truth array no other member shares. Its header
carries the incident; `comp/members.declines:84` is the only thing keeping it out of a seat.

So `recal_holdannot`'s +0.0477 was two rulers, and its corrected re-read was +0.0009 — FLAT. **The dense
corner is the only honest measurement of this axis that exists**, which is exactly why leaving its branch
unselected cost more than it looked like.

## The gold ruler moves 3.7x what the proxy moves, and that is the expected shape

+0.0172 gold against +0.0046 proxy is not a discrepancy to reconcile. The proxy's truth IS the report
table — the same document the supervision was extracted from — so it is biased AGAINST label-side
movement by construction. The gold ruler is 58 studies of radiologist annotation, and the field believes
the Kaggle test labels were made from images rather than reports (knee-7oep). A correction to the
supervision can only show up at full size on the ruler that did not come from the thing being corrected.

Consequence for every future label-side arm: **read gold first, and treat the proxy as a cost check
rather than the reading.** An arm on this axis that reports only its ledger row has measured about a
quarter of its own effect.

The caveat stays load-bearing: n=58 is a LEVEL and not a selector (knee-kddo). The interval
[+0.0004, +0.0343] clears the 0.0062 floor and clears it at the very edge. What makes this more than one
n=58 read is that two corners — 224 px / 12 slices and 336 px / 20 slices, different pooling, different
band — now agree in SIGN on the same ruler.

## What it licenses, and what is on the card because of it

The roster's members are all fitted on v2. The REPLICATES branch buys a rebuild, and
`experiments/arms/recal_labelwise.sh` is that arm already written and never run: `labelwise_dense`'s
every flag, one moving token (`--labels` to v3), `--eval-labels` pinned, `--hold-annotations` so the gold
ruler will accept the checkpoint, control `labelwise_dense_s2026` = 0.8964, three pre-registered bands at
0.9009 / 0.8966 / 0.8964. Armed 12:01Z.

Its own written failure mechanism is worth keeping in view: v3's fitted levels and this recipe's
`--calibrated-targets` may be two answers to the SAME question — what positive rate does this reader
value mean — in which case the second has nothing left to correct and the two over-correct together.

## How the reading was nearly missed a second time, by the same mechanism

The follow-up was filed this morning as "the dense-corner replication has NEVER RUN — no
`v3labels_dense*` checkpoint exists on disk", and an arm was written and gated to spend a granted hour of
card on it. That premise was a **name check**. The experiment existed under a different name, flag for
flag, and `arm_check`'s TWIN rule (knee-7uoq) is what refused it:

```
TWIN v3labels_dense: is recal_dense flag for flag, up to --out — so it spends a card slot to produce a
second name for one recipe, and no comparison between them means anything
```

This is the same error `knee-ibfrw` was filed against an hour earlier, in the opposite direction:
there, two DIFFERENT tables shared one basename; here, one recipe under two names. Both are cases of a
name standing in for content, and in both the only thing that caught it was a check that reads the
content. `v3labels_dense.sh` is deleted rather than kept — a second name for a recipe that has run is a
future duplicate, not an arm.

## The denominators

Proxy: 909 held-out studies, fold 0 of split_seed 2026, `ledger75` mixed cut — ten findings at 0.75,
Medial Meniscus and Baker's at the midpoint. Gold: 58 annotated studies, 696 cells, 34.5% positive,
per-finding positives 9 to 35; both members fitted `--hold-annotations`, so `annotated_readings` confirms
neither trained on those cells. Floor 0.0062 on both rulers, paired.
