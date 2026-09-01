# What the record says once the stale numbers are out of it

**2026-09-01, after a documentary sweep of `experiments/README.md`, `interpretations/`, `research/`,
`comp/` and all 324 open beads. No card time, no slot. Every number below was recomputed from
`comp/board.jsonl` (87 rows, `scope: excluded` dropped, joined on the `members` field) or read live at
2026-09-01T07:33:16Z.**

This file exists because the corrections landed in five places at once — three beads, a research doc, an
interpretation banner — and none of them states the resulting picture. That picture is the thing that
sets direction, so it gets written down once, here, rather than reassembled from corrections.

## What was wrong, and what it was hiding

Four defects, all of the same kind: a number that was correct when written, quoted afterwards as if it
were current.

1. **The ensemble bonus was said to be flat for fourteen days.** It is not. Best single has been frozen
   at **0.925** since 2026-08-26; best ensemble went 0.939 → **0.944** over the same window. The law is
   `board(ensemble) ≈ board(best single) + 0.019`, and the second term GREW while the first stood still.
2. **The member-count curve was fitted on 34 of 58 submissions** and read as a fact about the record.
   Recomputed over all of it, the span from one seat to our best entry is **+0.019**, not +0.024, and
   nearly all of it is bought by the second and third seat.
3. **Rival singles beat our ensemble** — true on 2026-08-26, not now. Their claimed 0.942 sits INSIDE
   our range: our best six reads 0.943 and our best seven 0.944.
4. **Stale standings in the tracker.** knee-17am carried `rank 84 of 1758, gap 0.032`; the live read is
   **rank 30 of 2814, gap 0.008**. knee-u521 and knee-mwej each named a champion two pins out of date.

None of these were arguments that turned out to be wrong. They were arguments quoted past their date,
which is why the fix is a standing check (`board_claims`) rather than five edits.

## What the corrected record actually says

**The seating axis is closed.** Nine 6/7-member entries span 0.002, and every single-seat operation
since — `holdswap_v1`, `vjepaseat_v1`, `spanseat_v1` — read flat against its control.

**The serving axis is closed too, as of yesterday.** Vote rule, view count, slice budget and now
precision: `halfcham_v1` served fp16 and landed on 0.944, dead level with its fp32 control. That closes
the knob on its own pre-registered terms and makes fp16 the default on runtime grounds alone.

**The count ladder is already shipped and needs no instrument.** `soloseat_v1` 0.925 (1) → `onepair_v1`
0.936 (2) → `trioseat_v1` 0.938 (3) → `fourseat_v1` 0.937 (4), four consecutive days on one pool. Counts
1–4 are as close to a matched sweep as this project owns. Six through eight stay confounded — the
high-reading small entries are late ones built from better members — and no slot should go to
un-confounding them.

**So there is exactly one free term left.** With the bonus law holding and the seating and serving axes
closed, the board score this project can reach is `best single + 0.019`. Our single has not moved in six
days. **Everything that changes our position changes the member.**

## What that licenses, and it is all already filed

The sweep's second half read the backlog rather than the board, and every idea that survived pricing was
already written down by somebody. Nothing here is new; what is new is that the arithmetic above says
these are the only lane.

- **More supervision, not more architecture.** `knee-d35s` (Rijeka, 736 labelled volumes already on
  disk), `knee-3oy8` (fastMRI+), `knee-2jg6` (the merged cache). Backbone size is a closed axis on our
  own measurement and on a competitor's, so capacity is not the shortfall.
- **Better labels.** `knee-teh4` (P0, confident learning over the OOF arrays) and the severity-boundary
  defect, which is reader disagreement written into the gold set rather than a threshold.
- **Unlabelled adaptation.** `knee-c5n` — the encoder sees 2.5× more slices than the supervision uses.
- **Compose the positives that already landed.** `knee-nu1m` and `knee-7d53`: calibration, EMA and mixup
  each moved the member and have never been stacked into one. This is the arithmetic route and the
  cheapest of the five.
- **Published objectives.** `knee-gzhj` (Rep-GLS) and `knee-bdhb` (AUC-margin), the second stalled
  in_progress.

## What this does NOT say

It does not say ensembling was wasted — +0.019 is the largest single effect on our board and it is
banked. It says the axis is spent, not mistaken.

It does not price the member levers against each other. That ordering needs the wall-clock-over-expected-
gain number for each, and this file deliberately stops at "the member is the only free term" rather than
guessing which member lever is biggest.

Related: `2026-08-25_the-seat-axis-is-saturated-and-the-member-is-0.028-behind.md` (three corrections in
its banner) · `2026-08-30_one-board-tick-is-fourteen-places.md` · beads `knee-urvf`, `knee-macc`,
`knee-17am`, `knee-mwej`, `knee-u521`.
