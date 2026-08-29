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

## All six, and the adapter validating itself

The other five ran the same way, every one at 0.0% fallback:

| member | our 909 | vs our best | their published gold (58) |
|---|---|---|---|
| `raptor_v4_swa` | **0.8938** | +0.0304 | 0.9150 |
| `raptor_v4` | 0.8915 | +0.0281 | 0.9167 |
| `raptor_v9` | 0.8769 | +0.0135 | 0.9031 |
| `raptor_effv2l480` | 0.8703 | +0.0069 | 0.8716 |
| `raptor_cnv2b336` | 0.8550 | −0.0084 | 0.8901 |
| `raptor_coat384` | 0.8508 | −0.0126 | 0.9025 |

Four of six beat our best member. Three clear the paired floor by a wide margin; the fourth only just.

**The two rulers agree on the top three, in order** — v4 and its SWA twin first and second, v9 third — on
sets fifteen times apart in size. A plumbing bug does not preserve a rival's own ranking of six checkpoints
it knows nothing about. That is the self-check no single checkpoint could provide.

Where they disagree, `coat384` and `cnv2b336` invert. At n=58 with per-label SE 0.031–0.072 those three are
one indistinguishable clump on their ruler, so the inversion is what small-sample noise looks like — and it
is confined to the checkpoints not worth seating.

**The family has one signature.** All six are worse than our member at Synovitis (−0.052 to −0.094), five of
six worse at Effusion, all six better at Fracture. Independently, on every checkpoint. The large fluid
collections are ours and the focal findings are theirs — a structural trade, not one member's scatter.

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

Seating is `knee-p0gk`, and the member is `raptor_v4_swa`.

The remaining question is not the member. It is whether their **decode** can be served inside a Kaggle
kernel without vendoring third-party code, which `knee-p0gk` states as the thing to decide before building.

## Every seating of it is worse than serving it alone

Priced offline with `record_mix` over the same 909 studies, no slot and no card:

| seating | RankAverage |
|---|---|
| `raptor_v4_swa` alone | **0.8928** |
| champion six + `raptor_v4_swa` | 0.8910 |
| `raptor_v4_swa` + `raptor_v4` + `raptor_v9` | 0.8908 |
| `raptor_v4_swa` + our best two calib members | 0.8876 |
| champion six | 0.8597 |

Nothing we own improves it. The whole champion six costs 0.0018, its own siblings 0.0020, our two best
calib members 0.0052. `QualityWeightedRank` on the seven reaches 0.8918 and still does not recover the solo.

This is not a puzzle. The member is 0.030 clear of the pool, a flat vote pays that gap in full, and the
Synovitis/Effusion complementarity above — real, and the reason a seat looked worth buying — is worth about
0.003 against it. A weighted vote is the only shape that could hold both, and the one weighting we have,
fitted split-half, does not.

**So the earlier claim in this file is wrong and is retracted here**: the +0.030 is not what a better seat 1
gets *added to*. On this ruler it is what ensembling *spends*.

## …and the board says the opposite, which is the finding

`calibsolo` scored **0.923** the same evening — the champion's own first seat, alone, on the board. The
six-seat roster scores **0.941**. Ensembling is worth **+0.018 there** while costing 0.002–0.005 here.

The same six that the board pays +0.018 for read 0.8597 offline, *below four of the six foreign checkpoints
individually*. Our holdout cannot see whatever the board is rewarding about a mixture. That inversion was
already visible as the seating axis running opposite to offline gain across seven entries; this is its
sharpest form, and it means the table above must **not** be read as "do not ensemble on the board".

Two slots, and they bracket it rather than one guessing:

- **solo `raptor_v4_swa`** — what the holdout says, and the clean read of a foreign member. Matched against
  `calibsolo` 0.923, taken the same night on the same board.
- **champion six + `raptor_v4_swa`** — what the board's own +0.018 says, on the roster whose board number
  we know exactly.

If solo wins, the board's ensemble bonus is a property of weak members and the offline ruler was right. If
the seven win, the bonus survives a strong seat and the offline ruler is blind to it. Neither number can be
inferred from the other, which is the whole reason to spend both.

## A ruler that disagrees with itself

`knee-3uje` read the same checkpoint against `calib_span_hold_s2026` on the 58 gold studies and found a
genuine null: 0.9167 against 0.9152. On the 909, that same member is **0.8565** — behind by 0.0350.

Going from the 58 to the 909, our member falls 0.0587 and theirs falls 0.0252. The small set flatters ours
by more than twice as much.

The first suspect is **our own selection**, not their model. The 58 are held out of training on both sides,
which is what that bead checked. They are not held out of member *choice*: `calib_span_hold` is the best of
105 arms as read on that set, and a maximum over 105 draws at a per-label SE of 0.031–0.072 buys several
hundredths for free. Raptor's number is one self-reported figure with no such search behind it. `knee-3uje`
made exactly this argument against its own per-label oracle; it applies to the macro too, with 105 draws
rather than two.

So `knee-rh4t`'s member gap is not retired after all. Removing the two-ruler artefact does not leave a null —
it leaves a gap measured on our own ruler, which is a stronger form of the same claim than the forum numbers
ever were.

## CORRECTED 2026-08-29: the 909 is the contaminated ruler, and the gap IS retired

The section above reads the two-ruler split the wrong way round. It is right that the rulers disagree and
right that our 105-arm selection inflates our number on the 58. It is wrong about which side the artefact
sits on, and its closing sentence — *"`knee-rh4t`'s member gap is not retired after all"* — does not survive.

`labels_llm_soft.csv`, their own published row set, contains **896 of our 909 fold-0 holdout studies
(98.6%)**; the 13 it lacks are exactly the gold studies falling in fold 0. Their table is the corpus minus
the 58 and nothing else. So the 909 is the ruler THEY trained on and we did not, and the +0.030 there is
in-sample against held-out.

That makes the board the tiebreaker, and it had already spoken: `raptor_v4_swa` solo scored **0.923, level
with our own best single**. Two rulers where both models are honest (gold 58, board) read null; the one
ruler where only one model is honest reads +0.030. The gap is retired.

`python -m kneemri.evaluation.foreign_overlap --foreign <their table>` now runs this check before a foreign
number is read. Full reading: `interpretations/kneemri/2026-08-29_the-foreign-member-trained-on-our-holdout.md`.
