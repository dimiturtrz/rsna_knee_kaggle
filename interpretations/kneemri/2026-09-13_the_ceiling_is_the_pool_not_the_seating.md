# The 0.944 ceiling is a property of the POOL, not of the seating

*2026-09-13. Read off `comp/board.jsonl` at 129 scored submissions. No card time spent; this is a
re-reading of submissions already made.*

The owner has asked the same question for a week — **"why are we not going beyond 0.944"** — and the
answer has been given four times as a story about the last entry. This is the answer read off the
whole record at once, and it is not a story about any entry.

## The number

```
count    0: n=  1  max=0.500   count    9: n=  1  max=0.936
count    1: n= 14  max=0.933   count   10: n=  1  max=0.928
count    2: n=  5  max=0.936   count   11: n=  1  max=0.941
count    3: n=  6  max=0.939   count   12: n=  1  max=0.937
count    4: n= 14  max=0.937   count   14: n=  1  max=0.937
count    5: n= 11  max=0.941   count   25: n=  1  max=0.894
count    6: n= 27  max=0.944
count    7: n= 28  max=0.944   best overall 0.944
count    8: n= 10  max=0.944
```

**0.944 is the maximum at counts 6, 7 AND 8 — three different ensemble sizes, 65 entries between them,
and not one of them has ever exceeded it.** Counts 9 through 25 are all worse. Count 1 tops out at
0.933, so the ensemble is worth about +0.011 over the best single member and then stops.

## Sixteen different member sets reach it

Sixteen entries sit at exactly 0.944. Twenty distinct members appear across them. Four are near-ubiquitous:

```
16x final_ws_cf   16x calib_span_ff   16x through_dense_cf   15x grade_ladder_cf
10x lr4e-5_ws_cf   8x raptor_v4_swa   7x merged_blend_cf   7x stack_increments_ff
 2x labelwise_dense_s2026   2x warm_calib_lowlr_s2026   2x discord_mask_s2026   2x labelwise_depth24_s2026
```

**The obvious reading — "those four ARE the score" — was checked and is false, in both directions.**

*Not sufficient.* Twenty-eight entries contain all four. Sixteen score 0.944; the other twelve score
0.929 to 0.943:

```
0.929  foreignseat_v1    0.941  calibmix_v1    0.943  raptorseat_v1
0.940  pixgroup_v1       0.941  bend_v1        0.943  vjepaseat_v1
0.941  calibsix_v1       0.942  fleettwo_v1    0.943  champviews_v1
0.942  onegroup_v1       0.943  aucvote_v1     0.943  stackvote_v1  0.943  oofseat_v1
```

*Not necessary.* `tablepool_v1` reaches 0.944 without the full four, on eight members including
`discord_mask_s2026` and `labelwise_depth24_s2026`.

So there is no magic core. **Any reasonable six-to-eight draws from this pool land at 0.944, and which
six barely matters.** A bad draw (a dark foreign seat, a merged member, a low-quality solo) costs a
thousandth or two; a good draw buys nothing back above the ceiling.

## What that means, stated as a mechanism

The board reports at 0.001, so sixteen entries tied at 0.944 with substantially different membership
**bounds their true spread below one thousandth**. The quantity that varies across those sixteen —
which members, and how many — does not move the score at the resolution the competition reports.

The pool is **rank-deficient**. Every member in it is fitted on the same 4,407 studies, against the
same label table, with recipes that differ by a flag or two. Six draws already span whatever subspace
that construction can produce, and the seventh and eighth are inert by construction rather than by bad
luck. That is the measured form of `knee-uco3`'s inert-seventh-seat question.

## What it explains that was previously unexplained

- **Four designed seat changes in a row read exactly 0.944** — `labelseat_v1`, `slotseat_v1`,
  `hidose_v1`, `warmchair_v1`. Each was argued from a different mechanism and each was pre-determined
  flat, because each was a reseating inside the span.
- **`board_transfer` saturates.** Its +1.135 slope [+1.02,+1.25], R² 0.962, is fitted where offline
  quality still buys board; at the top the board stops responding because the pool stops producing new
  directions, not because the transfer breaks.
- **`farswap_v1` 0.934 and the three "quality dominates distance at seat seven" readings.** Distance
  measured inside a rank-deficient pool is distance along directions that are already spanned.
- **Offline 0.85–0.90 sitting under a board 0.944.** These are not in tension; the ensemble recovers
  what the pool jointly contains, and the pool's ceiling is what is on display.

## The consequence, which is the point of writing this down

**A submission that reseats members of this pool is pre-determined flat and may not be sold as an
attempt on the score.** That is a rule, not an observation: sixteen entries and 65 submissions at
counts 6–8 are the evidence, and the owner named the failure mode before it was measured — *"you will
be spamming instruments that never succeed and never teach us anything."* Such a slot may still be
spent, but only as a declared INSTRUMENT, and the submission message has to say so.

**Only two things can move the board from here:**

1. **A member from outside the pool's span.** Not a new recipe on the same corpus and labels — those
   are the span. The three named routes and their status:
   - *corpus* — CLOSED, negative, on a harness that cannot be blamed (`knee-q1pp0`, `knee-0ggxn`). The
     second site makes a distinct member on every distance axis we own and it is worth −0.0012.
   - *labels* — CLOSED three times over (`2026-09-03`, `2026-09-08`, `2026-09-11`). The corpus's labels
     are in better shape than any of the three attacks assumed; the rung prices are right.
   - *objective* — **OPEN, and running.** `severity_drop40_long` against `band_dense_drop40_long`
     (`knee-evdfd`). This is the owner's third named lever and the only one of the three still live.
   - *architecture* — OPEN. A CNN family the board has never held is the one structural axis that does
     not go through the corpus or the label table.
2. **A different decision rule.** Every entry above serves the same way: `LogitAverage`, or that plus a
   ranked vote for a foreign seat. The pool being rank-deficient says nothing about the DECISION being
   optimal, and that has never been varied independently of the seating.

## What this does NOT say

It does not say the pool is bad — 0.944 is rank 71 of 3612, and the gap to the leader is 0.011. It
does not say members are worthless: count 1 at 0.933 versus count 6 at 0.944 is the ensemble earning
+0.011, which is most of the remaining gap all over again. And it does not license reading anything
into differences *between* the sixteen tied entries — their spread is bounded below the board's own
resolution, which is precisely why no search among them can report a winner.
