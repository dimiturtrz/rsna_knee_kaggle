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
   - *architecture* — **MEASURED THE SAME NIGHT, AND IT IS NOT A DIRECTION.** See below.
2. **A different decision rule.** Every entry above serves the same way: `LogitAverage`, or that plus a
   ranked vote for a foreign seat. The pool being rank-deficient says nothing about the DECISION being
   optimal, and that has never been varied independently of the seating.

## The architecture route, priced without spending a slot

The finding above predicts something specific and testable: **seat seven is inert regardless of what is
in it.** `cnnseat_v1` is the sharpest available test of that, and it was already on the board —
stackseat's six plus `resnet_224_short_s2026`, the most out-of-family member this project owns, in the
seventh chair. It read **0.944**, byte-equal to the champion six. Its own pre-registration drew the
right conclusion at the time: it "closes the single-CNN-seat question but NOT the CNN family."

So the open version of the architecture question was never addition — it was **replacement**: a CNN
MAJORITY rather than a CNN passenger. The Einstein gate clears that cleanly. Across 129 scored
submissions the board has held **at most ONE CNN in any entry** (`cnnsolo_v1` 0.915 at count 1,
`resnetlong_v1` 0.907 at count 1, `cleanseat_v1` 0.942 with 1 of 6, `cnnseat_v1` 0.944 with 1 of 7).
A CNN majority is genuinely untried.

It is also refuted, offline, for free — four CNN holdouts are on disk, so the ensemble is a numpy
`LogitAverage` over 909 shared studies and costs no card and no slot. Read under `ledger75` (ten
columns at 0.75, Medial Meniscus and Baker's at the midpoint):

```
CNN-4 alone                  0.8867
CNN-4 + top-3 transformer    0.9012
transformer-only 6           0.9050
1 CNN + 6 transformer        0.9056
```

**The CNN majority is −0.0038 against the transformer-only six**, far under the inherited +0.0040
offline FLOOR, and adding one CNN to six transformers is +0.0006 — noise, which is the same answer
`cnnseat_v1` gave on the board.

The mechanism is in the first line, not the third. **Four CNNs together beat their own best solo
(0.8842) by +0.0025.** A family whose four members are nearly collinear *with each other* cannot be
supplying a direction the pool lacks. "Different architecture" was being treated as a proxy for
"different direction", and on this task it is not one: the CNNs are not an unexplored axis, they are a
worse-conditioned copy of the axis already spanned.

This is written down as a SAVED SLOT. A CNN-majority entry was a well-formed, Einstein-clean,
mechanism-backed proposal, and it was killed by a computation that cost seconds instead of by a
submission that would have cost a fifth of a day's board access. That is the shape the owner asked for.

**What it leaves is the DECISION RULE**, which is now the only untested structural route that does not
need new members. Every entry on this board serves the same way — `LogitAverage`, or that plus a ranked
vote for a foreign seat — and the pool being rank-deficient is an argument about the MEMBERS, not about
the function that combines them. That has never been varied independently of the seating.

**It was varied, the same night, and it is closed too.** Ten aggregators over the all-ours six span
0.0010 at `ledger75`, with the incumbent `LogitAverage` effectively optimal (TrimmedMean-logit +0.0007,
ProbMean +0.0006, both medians +0.0003, RankMean −0.0003, RankMax −0.0034, RankMin −0.0083). The null
was then attacked on a deliberately heterogeneous 8 (+2 CNNs) and a wide 10 (+aux objectives); the
spread stayed 0.0010–0.0011 on all three. The unifying quantity is **mean pairwise rank-correlation
ρ ≈ 0.86–0.88 no matter how the set is diversified** — adding CNNs moves it only 0.884 → 0.860. At that
ρ every aggregator must coincide, which is the 0.0010. Rank deficiency measured directly *between
members*, not inferred from the board, and it explains the seating flat, the architecture flat and the
decision-rule flat with one number.

## The same statement, measured offline out-of-fold rather than read off the board

Everything above is inferred from 129 board rows. The claim it makes — *seat selection inside this pool
cannot buy the score* — is directly testable offline, and was, the same night. Pool: 140 honest members
(declines, every `raptor_*`/`foreign_*`/DINOv3 row, and the licence-tainted `merged_*` family all
removed), each with solo `ledger75` ≥ 0.855 on the 909 shared studies.

**Greedy forward selection reaches 0.9097 at k=8 against the champion six's 0.9050 — +0.0047, which
clears the +0.0040 floor.** That number is selected and scored on the same 909 studies, so it is an
upper bound on availability, not a result. Split the holdout in half, select on one half, score on the
other:

```
fit A -> eval B:  core six 0.9078   greedy-8 0.9097   +0.0019
fit B -> eval A:  core six 0.9016   greedy-8 0.9029   +0.0014
```

**Mean honest gain +0.0017 — under half the floor.** About 65% of the in-sample +0.0047 was selection
overfit. And the two fitted sets share **2 of 8 members**: the selection is not converging on a better
seating, it is fitting the half it was shown.

This is the strongest form of the finding available. It is not "we tried several seatings and they were
flat" — it is *optimal* seating, computed, and it is flat out-of-fold. `knee-gcgbk`'s retirement of
reseating and the INSTRUMENT rule above now rest on a measurement rather than on a board inference.

## The orthogonal-member search, and why its answer was contamination

The obvious escape is a member outside the pool's span. Scanning all 357 holdouts for mean per-label
rank-correlation ρ against the six-member consensus finds an apparently perfect candidate tier —
ρ ≈ 0.75 with solo *above* every core member (`raptor_effv2l480` ρ=0.782/0.9139, `dinov3f0_solo`
ρ=0.753/0.9021, a dozen `foreign_dinov3_*` at 0.750–0.775) against the core's ρ 0.935–0.967.

**Every one of them is already declined in `comp/members.declines` as in-sample void.** That is not a
coincidence to note and move past — it is the mechanism. A model that has memorised part of the holdout
tracks the TRUTH on those rows, and the truth is exactly what the honest consensus fails to track. So
contamination *presents as orthogonality*: low ρ with high solo is the contamination signature, and it
is the same reading `foreign_overlap` exists to refuse, arriving by a different door.

Restricted to the 140 honest members, the range is:

```
honest minimum          rho=0.852  solo=0.8673  vjepa_warm_s2026
honest low-rho tier     rho 0.852-0.888, solo 0.860-0.884
the core six            rho 0.935-0.967, solo 0.883-0.899
```

**No honest member is both off-consensus and strong** — inside the honest pool, low ρ is bought with
low solo. The 0.10 gap between the honest minimum (0.852) and the contaminated tier (0.750) is wide
enough to be usable as a screen in its own right.

## What this does NOT say

It does not say the pool is bad — 0.944 is rank 71 of 3612, and the gap to the leader is 0.011. It
does not say members are worthless: count 1 at 0.933 versus count 6 at 0.944 is the ensemble earning
+0.011, which is most of the remaining gap all over again. And it does not license reading anything
into differences *between* the sixteen tied entries — their spread is bounded below the board's own
resolution, which is precisely why no search among them can report a winner.
