# The feed was the whole shortfall — a foreign checkpoint is now our best member

2026-08-27, 23:12Z. Costs: 24 minutes of card, no submission slot.

## The number

`raptor_ft_coatnet_v4_full.pt` — a public Kaggle checkpoint published by another competitor — served
through **their** decode over **our** fold-0 holdout:

| member | holdout mean, 909 studies |
|---|---|
| `raptor_v4` (foreign, CoAtNet-384 + per-finding attention) | **0.8915** |
| `calib_dense_hold_s2026` (our best member) | 0.8634 |

+0.0281, against this pair's own 0.0062 paired floor. 0 of 909 studies fell back (0.0%), so the number
is over the whole holdout and not over the subset the adapter happened to parse.

## The denominator, first, because three numbers in this story live on two different sets

Their published **0.9167** and our earlier cache read of **0.8819** are both over the **58 gold-annotated
studies**. The 0.8915 here is over the **909-study fold-0 holdout**. These are not points on one line, and
0.8915 is not "between" the other two — it is a third measurement on a different set.

`knee-9g28` pre-registered a test in exactly those terms ("near 0.9167 → the seam is right; near 0.8819 →
my plumbing"), and that test **cannot be read off this run**. What the run gives instead is the comparison
that was actually wanted: every member this project owns has a number on the 909, and on that ruler the
foreign one is first.

## What it settles anyway, by a different route

`2026-08-27_the-foreign-feed-shortfall.md` measured the byte-identical checkpoint at −0.0348 through our
best cache and read that as a fact about foreign members: through our feed, they are weak. The same
weights now read **above every member we have** through their decode.

Same file, same holdout studies, two feeds, opposite verdicts. The shortfall was never about their model
and never about ours — it is upstream, in what a single slice's feature vector contains, exactly where the
earlier interpretation placed it. `knee-xuj5`'s correction ("through our feed it is a weak member") was
right about the feed and wrong to generalise from it to the member.

## The seat is worth buying for the disagreement, not the mean

| direction | findings |
|---|---|
| foreign better | Fracture +0.0779 · Lateral OA +0.0591 · Lateral Meniscus +0.0572 · PF OA +0.0486 · Medial OA +0.0464 · Contusion +0.0391 · Baker's +0.0381 · MCL +0.0350 · Medial Meniscus +0.0256 |
| flat | ACL −0.0037 |
| **ours better** | **Effusion −0.0324 · Synovitis −0.0540** |

By structure scale: focal +0.0383, regional +0.0137. This is not a better copy of our member. It is better
at the focal findings and **worse at the two large fluid collections**, which is the trade a seat exists to
buy. `2026-08-27_the-pool-is-82-percent-one-number.md` measured 81.7% of what separates our 105 members as a
single quality scalar — a pool with almost nothing to trade. This member is off that scalar.

## What this does not say

- **The per-label deltas are not facts.** `run_comparison` marks the pairing INDICATIVE with half-widths up
  to ±0.0456, because `raptor_v4`'s scores are RECONSTRUCTED from kept predictions rather than recorded by a
  run. The mean delta sits outside that half-width and nine labels share a sign; a single label's delta does
  not.
- **It is not a board number.** It is a holdout number on our ruler, and board-vs-holdout compression is its
  own open question (`board_transfer`).
- **It is not the strong half of what exists.** The author states a 0.936–0.938 un-blended model is trained
  and unreleased (topic 737696). This is the 0.924 set.

## The consequence

Seating `raptor_v4` is the next slot. It was not spent tonight on purpose: the last slot was pinned to
`kneemri-calibsolo`, whose kernel is pushed, run and witness-clean, and `raptor_v4` is in no kit — packaging,
mounting and witnessing it could not have finished before the 00:00Z reset. A rushed seat would have spent
the slot on the packaging rather than on the member.

`knee-tlba`, the five remaining checkpoints, is no longer speculative: the adapter is proven end to end at
0.0% fallback, and the five are running.
