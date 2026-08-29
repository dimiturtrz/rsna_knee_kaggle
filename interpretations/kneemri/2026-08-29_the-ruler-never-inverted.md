# The ruler never inverted a pair it resolved

2026-08-29. Re-fit of `kneemri.evaluation.board_transfer` on the now-complete `comp/board.jsonl`
(74 of 74 live submissions; it held 34 of 58 on 2026-08-25).

## What was believed

Two things, both quoted repeatedly in kernel headers, `experiments/README.md` and this project's
own agent instructions:

1. The local-to-board compression **saturates**: +1.72 over the whole record, falling to under +1.00
   — and on the latest read, to **−0.41** — among the strongest entries. A holdout point near the
   champion buys nothing, so no queued arm could reach the top ten.
2. The holdout ruler **inverts** the board: 15 of 149 resolved pairs ordered the other way.

Both were used as planning premises. (1) is what makes a target unreachable; (2) is what makes a
submission decision a coin-flip.

## What is actually there

`raptor_solo` (ref 55829913) carried `offline: 0.8938` against `board: 0.923`. That offline was read
on our fold-0 holdout, and `kneemri.evaluation.foreign_overlap` had already established — the day
before — that 896 of those 909 studies sit inside raptor's published training row set. The number
was 98.6% in-sample. The module whose docstring says **THE DEFAULT IS GUILT** existed, was correct,
and was never called by the fitter.

Voiding that single cell (the board number is honest and stays):

| | whole record | top regime | resolved pairs inverted |
|---|---|---|---|
| with `raptor_solo`'s offline | +1.30 | −0.41 [−0.44, +0.70] | 15/149 |
| without it | **+1.55** | **+0.99 [+0.77, +1.18]** | **0/127** |

Eleven of the fifteen inversions were that one entry: an in-sample offline "beating" fourteen honest
ones, and an honest board declining to agree. The remaining four are all under the 0.0053
Hanley–McNeil floor — pairs the holdout called equal, which the board may order either way without
refuting anything.

## The two readings that change

**The ruler has never once mis-ordered a pair it resolved.** Zero of 127. That is a much stronger
statement about our 58-study proxy than anything previously on record, and it is what licenses
spending a slot on an offline preference at all.

**Compression at the top is ≈ 1.0, not ≈ 0 and not negative.** A holdout point near the champion is
worth about a board point. The whole-record +1.55 is still the wrong number to price with — the fall
from +1.55 to +0.99 is real and is probably the attenuation you get from dividing a board gap by an
offline gap barely wider than the offline noise — but the difference between "≈1.0" and "≈0" is the
difference between a reachable target and an unreachable one, and the unreachable version was an
artefact.

## What this does not say

Nothing about raptor as a member: `raptor_v4_swa` is board-measured at 0.923 solo and its seat in
`raptorseat_v1` stands. Nothing about the level of the two scores — the estimator is pairwise by
construction and refuses the level question.

And, correcting the paragraph above rather than qualifying it: **the top regime does not reach the
champion.** Its six entries span board 0.930–0.937 (`basedense_v1`, `longseat_v1`, `gradeswap_v1`,
`greedy7_v1`, `peak8_v1`, `honest12`), and it forms three resolved pairs. Everything above 0.937 —
`votefit_v1` 0.938, `calibsix_v1` 0.941, `raptorseat_v1` 0.943 — carries no offline number and
`kneemri.kaggle.priceable_entries` says why it never can: those entries are built from `_ff` members
fitted at `--fold -1`, which have seen every holdout study. Thirty-five of sixty ensemble rows are
in that class. So "+0.99 near the top" is measured in the 0.930–0.937 band and the compression at
0.943 is **unmeasured, not measured at 1.0**. What the refit establishes is that the collapse to
zero was an artefact; it does not establish that no collapse happens above 0.937.

Filling the gap would need an entry near the champion built entirely from fold-0 members — which
is a slot and an arm, not a free re-read. `knee-awxb` holds the cheap half: the rows that CAN be
honestly filled.

## The generalisable lesson

A guard only guards what calls it. `foreign_overlap` was written, tested, documented and correct,
and the contaminated number went on being divided by for a day because the fitter never asked. The
cheap structural fix is not another detector; it is that a row's `offline` must be a number read on
studies the entry did not train on, asserted where the row is written.

## The cheap half, filled — 2026-08-29 21:38Z

`kneemri.kaggle.entry_offline` surveys every board row whose `offline` is null and either fills it or
records WHY it stays null. Of 34 such rows, **10 could be filled honestly** and 24 could not: 20 hold an
`_ff`/`final_ws_e12` member fitted at `--fold -1`, 2 hold the foreign `raptor_v4_swa` whose published
training rows cover our holdout, and 2 hold `_cf` merges that wrote no holdout record. Each of those 24
now carries an `offline_scope` field naming its reason, so the null is a recorded decision rather than an
absence — the same move `board_check` made with `scope: excluded`.

The ten are `ratepair_v1`, `solotop_v1`, `rateviews_v1`, `topviews_v1`, `sixstrong_v1`, `sixfresh_v1`,
`sevengrade_v1`, `sixfresh_v2`, `sevenplane_v1`, `accumseat_v1` — board 0.921 to 0.933, scored under
`RankAverage` because every one of them shipped before `kneemri-vote` served `LogitAverage`.

Re-fit of `board_transfer` on the filled record:

| | before | after |
|---|---|---|
| whole record | +1.55 | +1.25 |
| at or above 0.8668 | +0.99 [+0.77, +1.18], **3 pairs** | **+0.97 [+0.94, +1.11], 15 pairs** |
| at or above 0.8691 | — | +0.89 [+0.70, +1.19], 5 pairs |
| resolved pairs inverted | 0/127 | **0/233** |

**The reading that changes is the CONFIDENCE, not the constant.** The top-regime slope moved 0.02 — well
inside its own interval — while the number of pairs carrying it went from three to fifteen and the
interval narrowed from 0.41 wide to 0.17. The premise every slot decision is priced against ("a holdout
point near the top is worth about a board point") was, until this pass, a three-point fit; it is now a
fifteen-point one saying the same thing. That is the whole value: nothing was learned about the board,
and a great deal was learned about how much to trust what we already believed.

The unmeasurable half is unchanged and now says so in the file itself: compression above board 0.937
stays unmeasured, because every entry up there is built from members that have seen the holdout. Reaching
it still needs a near-champion entry built entirely from fold-0 members — a slot and an arm.
