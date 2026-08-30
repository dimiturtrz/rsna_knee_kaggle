# The board priced an offline constant to +0.0001, and then priced the pool

2026-08-30. What `greedyseat_v1` (ref 55877491, board 0.936) bought was not a rank. It was a ruler.

## The line

Two earlier fold-0 anchors map an offline holdout score to a board score:

| offline | board |
| ------- | ----- |
| 0.8712  | 0.933 |
| 0.8751  | 0.937 |

That is **1.026 board points per offline point**. A slope fitted on two points is a line drawn through
two points; the whole question is whether a third, taken later and by a different method, lands on it.

## The third point was a prediction, not a fit

`greedyseat_v1`'s seven seats were chosen by a greedy forward search over the holdout, so its in-sample
score is optimistically biased. That bias was measured **entirely offline, before the submission
scored** — 111 records, seven seats, **+0.0028**.

| entered as              | offline | predicts | actual | error       |
| ----------------------- | ------- | -------- | ------ | ----------- |
| raw greedy in-sample    | 0.8768  | 0.9387   | 0.936  | **-0.0027** |
| bias-corrected          | 0.8740  | 0.9359   | 0.936  | **+0.0001** |

The board, which was never shown the correction, prices it at 0.0027 against an offline measurement of
0.0028. Two rulers that share no machinery agree on a constant to a tenth of a thousandth.

Two things follow immediately.

**The transfer line is usable.** It now carries a third point taken by a different selection procedure,
and it predicted rather than fitted.

**An apparent ruler inversion dissolves.** Offline had risen 0.0017 while the board fell 0.001 — the
kind of disagreement that, taken at face value, retires the offline ruler entirely. It was the greedy
bias. Corrected, the ordering is restored. This is the second time an apparent inversion turned out to
be an accounting error rather than a fact about the metric (see
`2026-08-29_the-ruler-never-inverted.md`).

**Consequence for the record:** any search-selected entry goes into `comp/board.jsonl` **bias-corrected,
never raw**. A raw entry does not merely overstate itself — it bends the line every later prediction is
read off.

## What the line then said about the pool, for free

With a validated line, the ceiling of the members we already own is arithmetic rather than an
experiment:

```
fold-0 pool ceiling, honest, 28 seats: offline 0.8753 -> board 0.9372
+ all-data dose (+0.0045, fold -1):                     0.9417
champion raptorweight_v1:                               0.9440
top-10:                                                 0.9460

offline needed to MATCH the champion: 0.8775  -- short 0.0022
offline needed for a TOP-10 spot:     0.8795  -- short 0.0042
```

**Every arrangement of the members we own is already behind the champion.** The seating axis closes for
a fourth time, and this time from the price side rather than from a search that failed to find anything.

**Top-10 costs +0.0042 offline and no mapped axis pays it.** Label-table swap is +0.0034 and is a
*training* effect, not a scoring one (see below). Distillation is +0.0016 to +0.0035. The weighting
oracle is +0.0002. The combiner is flat. The specialist row's +0.0043 is an **oracle** — what closing
MCL and Fracture *completely* would buy — so a real specialist buys some fraction of it.

## The caveat that decides what to build next

**+0.0042 is under the 0.0062 paired holdout floor.** Even the right member cannot be *confirmed*
offline at that size. So the correct response is not to go looking for a +0.0042 knob — it is to reach
for a mechanism whose plausible effect is large enough to clear both the gap and the floor. A different
corpus or a different decode is the only named open axis of that size (`knee-ombx`).

## Stated limits

- The slope rests on three points, all fold-0, all ours.
- The all-data dose (+0.0045) is an estimate from a fold boundary, not a matched control: no hand-picked
  fold-0 seven exists, so procedural-vs-hand-picked (0.936 vs 0.944) crosses that boundary. The dose
  covers roughly half the 0.008 shortfall, leaving ~0.003–0.004 for the selection term.
- The public test set is hidden. Everything here is a statement about *our* holdout mapping onto *that*
  board, and it holds only while the split does.

## The scoring/training distinction that came out of the same window

Three label tables — `confident_v2`, `merged_labels`, `report_labels_v2` — are different files with
different md5s and different row counts (4407 / 5143 / 4407), and they score the same prediction set
**identically to four decimals**. The metric is a per-label rank AUC over a *binarised* truth; the
tables differ only in confidence values that binarise the same way.

So `knee-m69u`'s measured **+0.0034 from a label-table swap cannot be a scoring effect. It is a training
effect** — the table changes what the model learns, not how it is graded. That reframes it from a free
ruler swap into a data-supervision lever, which is a different and more interesting thing.
