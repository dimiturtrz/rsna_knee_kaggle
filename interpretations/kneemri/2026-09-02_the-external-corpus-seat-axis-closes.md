# The external-corpus SEAT axis closes, and the dose bends DOWN

2026-09-02. Four board entries, one moving variable: how many of six seats trained on the merged
external corpus. Everything else held at champion — count six plus the raptor foreign seat,
`LogitAverage`, `overlap=False`, each member's own precision, `FOREIGN_WEIGHT=1/3`, same kit.

## The curve

| external seats | entry | board |
|---|---|---|
| 1 | `raptorweight_v1` (champion) | 0.944 |
| 2 | `spanseat_v1` | 0.944 |
| 4 | `corpusdose_v1` | 0.943 |
| 6 | `corpusfull_v1` | 0.940 |

Flat, flat, flat, then **down four ticks**.

**CORRECTED 2026-09-03, and the correction removes a claim this file made.** As written, this paragraph
said `BOARD_RESOLVABLE` is 0.0015, so 1→4 sat inside the resolution and 4→6 was "the only resolvable
move on the axis". The constant was raised to **0.0040** on 2026-09-02 — the same day this was written —
because the old value had been fitted on six paired entries and the record now holds 561 ordered pairs
with 6.2% inverting (`member_curve.py:56`). At 0.0040, **4→6 is 0.003 and is NOT resolvable either.** No
single step on this axis is, and only the full 1→6 span of 0.004 reaches the threshold at all.

What survives and what does not:

* **The closure survives, and is if anything stronger.** Nothing on the axis moves the board by an
  amount the board can read. That is precisely the condition for spending no further slot on it.
* **The attribution below does NOT survive as a reading.** "Down four ticks", the step that evicts the
  pixel- and objective-distant members, and the paragraph that assigns those ticks to the STRUCTURE that
  left rather than to the corpus that arrived — all of that rests on a 0.003 step being resolvable, and
  it is not. It is directional and consistent with the champion pin's earlier structural reading, which
  is why it is worth keeping on the page; it is not evidence, and no arm may be planned on it.

## What each entry pre-registered, and which branch fired

`corpusdose_v1` was pre-registered ≥0.9455 lane / 0.9455-0.9435 sub-linear / 0.9435-0.9425 flat /
≤0.9425 swap cost exceeded corpus gain. It read **0.943 — the FLAT branch**.

`corpusfull_v1` was pre-registered ≥0.9455 the dose compounds late (which would have made
fastMRI/MRNet the plan rather than a side errand) / 0.9455-0.9425 flat, closing the axis / ≤0.9425 the
two distant seats were carrying something. It read **0.940 — the LOW branch**.

Both fired below their own optimistic bands. The reading that was pre-registered as the thing that
would license external data as a *plan* did not happen, and the opposite one did.

## The confound `corpusfull` declared in advance, and why it now carries the reading

`corpusfull_v1`'s own submission message stated the confound before the number came back: reaching six
external seats cannot hold structure the way `corpusdose` could, because no `merged_*` member exists at
the pixel-distant or objective-distant corners. So `grade_ladder_cf` (the severity/ordinal objective)
and `through_dense_cf` (the only member whose input tensor differs) BOTH leave at exactly the step
where the dose rises to six. Two quantities move together and this entry alone cannot separate them.

It does not have to separate them alone, because `corpusdose_v1` is the control that keeps them.
At four external seats the two distant members are still seated and the board reads 0.943 — flat. The
drop appears only in the step that evicts them. So the four ticks are attributable to the STRUCTURE
that left, not to the corpus that arrived.

That is the same quantity the champion pin already bought once, in the opposite direction: *"a near-twin
of a seated member scored 0.930, a recipe-distant member 0.931, an OBJECTIVE-distant member 0.931, and
a PIXEL-distant member 0.932."* Pixel distance ticked up then; evicting pixel- and objective-distance
together ticks down now. Two readings, consistent sign, on the axis the pool has least of.

## What this closes and what it does NOT

**CLOSED: the external-corpus SEAT axis.** One, two and four seats read the same to the board's
resolution, and six reads worse for a reason that is about seating rather than corpus. The corpus a
member trained on is not what its seat is worth. No further slot goes to raising the count.

**NOT closed: external DATA.** A trunk *continued* on external images is a different mechanism from a
head fitted on external labels, and nothing here touches it — that is what `ortho_dense` asks and what
the `ortholarge`/`v3large` probe pair was built to read. This result removes the seat-count route to
external data; it does not remove external data.

**NOT closed, and now sharper: structural distance is the pool's live axis.** Six differently-corpus'd
members buy nothing; losing the two structurally distant ones costs four ticks. Every member-side
result this project owns points the same way — the pool is an architecture monoculture (82% of what
separates 105 members is one scalar) and what it lacks is distance, not accuracy.

## Honesty on the size

Four ticks is 0.004, under the 0.0062 three-seed holdout floor and far under the house 1.5% bar. It is
a single board read of a single seating with a declared confound, and it is only *resolvable* at all
because the board quantises to 0.001 and the drop spans four of them. Read it as directional and as an
axis CLOSURE — which is what it is good for, since closing a lane costs nothing to be wrong about in
the upward direction and saves the slots that would have gone to re-asking it.

Neither of these entries is offline-scoreable: the pools contain `--fold -1` members, so every holdout
study is in-sample for part of the pool and `board.jsonl` records their offline as null by construction.
The board is the only ruler that saw this, which is exactly why it took four slots to read a curve.
