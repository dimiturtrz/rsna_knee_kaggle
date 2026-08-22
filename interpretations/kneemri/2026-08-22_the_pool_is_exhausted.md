# The pool is FLAT, not empty: 63 candidates, best eighth seat +0.0005

*(Filed the same morning as "the pool is exhausted"; the title was corrected within the hour by a
board result — see the CORRECTION section. The filename is left alone so links keep working.)*

2026-08-22 · `python -m kneemri.evaluation.record_mix` over every same-split holdout record on disk,
909 shared studies, card-free from recorded per-study probabilities. Logs: `logs/greedy_all.log`,
`logs/roster_seats.log`. Read the numbers as SIGNS, not scores — a record is the trainer's own reading
of its holdout, comparable across members and not identical to `member_curve`'s GPU numbers.

## The measurement

70 same-split records (79 on disk, 9 refused by the tool as smokes or other folds — it raises rather
than averaging across a split boundary, which is the check working).

**Every candidate priced against the shipped seven:**

    roster of 7: 0.8695
      +small_batch6_s2026    +0.0005     <- the best eighth seat available anywhere
      +crop80_span_s2026     +0.0004
      +aug_long_s2026        +0.0004
      +aprior_dense_s2026    +0.0003
      +attend_dense_s2026    +0.0003
      ... 58 more, descending to -0.0025

**63 candidates. Maximum +0.0005. Zero clear the +0.0015 under which this project has never seen the
board tick.**

## CORRECTION, same day, 09:40 — the title of this document overclaims

`gradeswap` scored **0.933** at 09:34, a new best past `sevenplane`'s 0.932. Its offline seat value on
the very scan above is `+grade_ladder_s2026: +0.0001` — a fifteenth of the +0.0015 threshold this
document just used to declare 63 candidates worthless. **A +0.0001 offline seat produced a board tick.**

So the arithmetic in the paragraph above is wrong in the way this project's own instructions warn about
most: it compared offline deltas against a threshold and read the result as a statement about the board,
without applying the conversion that sits in `knee-7d1` — **median +4.67 board points per holdout
point**, measured over the five pairs the holdout resolves. Run properly, the best available eighth seat
is +0.0005 offline ≈ **+0.0023 on the board**, which is above the board's own resolution, not below it.
`gradeswap` is the measured instance: +0.0001 offline, +0.001 board, a ratio in the same family.

**What survives.** The seat spread across the whole pool really is 0.0005 to −0.0025, so no candidate is
a *large* addition and the count plateau (below) is unaffected — that reading came from the greedy curve,
not from the threshold. **What does not survive is the word "exhausted".** The pool is flat, not empty,
and on a compression of ~4.7x a flat pool can still hold board ticks. The honest statement is that the
BINARY SEAT RULER cannot tell which of the 63 carries one, since its whole live range is a third of its
own stated resolution.

That is also the third strike against reading a seat number as a board prediction, and the ruler that
DID call `gradeswap` correctly was the ordinal one — see below.

## The count axis is a PLATEAU, not a peak, and both earlier readings were on a smaller pool

Greedy forward selection over all 70:

    greedy  1   0.8598   small_batch6
    greedy  4   0.8703
    greedy  5   0.8710
    greedy  8   0.8717   <- peak
    greedy 10   0.8717
    greedy 14   0.8715
    greedy 22   0.8711
    greedy 70   0.8606

The peak is 8-10, and from 5 to 22 the curve spans **0.0007** — a seventeen-member-wide flat plateau.
This supersedes two recorded claims, both taken on smaller pools: `record_mix`'s own docstring says
"count saturates at four", and `2026-08-19_subset_search_is_spent.md` measured the peak at 6-7 over 22
members. Four is actually **0.0014 BELOW** the plateau, so count does pay a little — up to about eight,
and then nothing at all for the next fourteen members.

**This reconciles the board with the holdout instead of leaving them in conflict.** Best board entry at
each count reads 4 -> 0.930, 5 -> 0.931, 6 -> 0.931, 7 -> **0.932**: monotone, +0.002 total. Offline 4 -> 8
is +0.0014, and at the 2.4x board transfer this project has measured for diversity that predicts +0.0034
against a measured +0.002. Same sign, same order of magnitude, two independent rulers.

**Consequence: "twenty members like pilkwang" is dead as a hypothesis.** The curve is flat from 8 to 22
and falling by 70. Whatever buys the frontier its 0.94+, member count is not it. Note this was reachable
card-free the whole time — 135 checkpoints already sit on disk, so this was never a training question.

## It also hardens the distance verdict from n=1 to n=63

`vjepa_warm` is the most distant member this project has produced (nearest seated neighbour 0.8210
against a seated spread topping out at 0.8801) and it prices **+0.0001** here — below `small_batch6`
(+0.0005), `crop80_span` (+0.0004) and `aug_long` (+0.0004), none of which is remarkable on any
diversity axis. The ledger entry for that arm concluded "distance is necessary and not sufficient" from
a single member. Across 63 candidates on a second ruler, the ordering says the same thing: residual
distance does not predict seat value here. The distance ruler is not refuted — it measures what it
claims — but the INFERENCE from distance to seat has now failed twice, independently.

## The honest reading of the greedy peak, which is the one live thread

Greedy-8 reads 0.8717 against the shipped roster's 0.8695: **+0.0022**, and five of its eight members are
not seated. That is the largest offline delta available anywhere in the pool, and at 2.4x it would be
+0.005 on the board.

It is not a slot, yet, and the reason is recorded rather than invented: **greedy is an argmax over the
same 909-study holdout it is scored on**, and `2026-08-19_subset_search_is_spent.md` hit exactly this —
a greedy peak +0.0017 over the shipped entry — and correctly declined to ship it. The two readings here
disagree in a way that says which one to trust: the seat scan says no single addition beats +0.0005,
while greedy claims +0.0022 for a wholesale eight-member replacement. A rearrangement worth four times
the best addition, chosen by argmax on the ruler that scores it, is what overfitting looks like.

What would make it a slot is a forward test of the argmax itself, which is `knee-7d1`'s question and is
currently in flight as `gradeswap` (submitted 08:07 UTC, kernel COMPLETE, score pending). Do not fire a
greedy-8 entry before that reads.

## What this does NOT rule out

- **A member that does not exist yet.** Every number here is over 70 checkpoints from one design family.
  The pool being exhausted is a statement about the pool.
- **The ruler.** 909 studies, 0.0062 floor, and eleven axes in a row have now landed inside it. A ruler
  that returns "inside the floor" eleven times is either describing a flat world or is the flattest
  instrument in the chain (`knee-7d1`, P0).
- **Per-label selection over a distant family** — `knee-clpy`. The 2026-08-17 oracle closed per-label
  weighting, but it was measured before any distant member existed, and its own text names this re-test.
- **Data outside the corpus** (`knee-ombx`), the only remaining item that adds information rather than
  rearranging it, and blocked on a licence decision only the owner can make.
