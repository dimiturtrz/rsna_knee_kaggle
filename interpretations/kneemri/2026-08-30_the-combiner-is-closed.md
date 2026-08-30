# The combiner is closed: two selection rules that share no structure agree to 0.0003

*2026-08-30, morning. Every number here is CPU arithmetic over holdout records already on disk. No GPU,
no submission slot, no training.*

For most of this project the ensemble has been the place where score came from. The champion is an
ensemble, the board's best entries are ensembles, and the last several design questions were about how
to combine members rather than about what a member is. This is the measurement that says that door is
shut, and it is worth writing down because it was shut by two experiments that were each aimed at
opening it.

## The pool, stated once, because it is the denominator for everything below

`record_mix --largest-pool` over the processed root: **113 of 202 records became members, over 909
shared studies, voting `LogitAverage`.** The other 89 are checkpoints with no holdout record, twelve
smokes and one member scored under an older label table, all named in the log. Both measurements below
were taken on that same pool on the same day, which is what makes them comparable at all — a routing
number from one pool against a seat number from another would have been two facts about two
denominators.

`logit` and not `rank` because the shipped champion serves logit. Every number in the ledger was taken
under `rank`, so this is also the first reading of both axes in the vote rule that actually ships.

## First door: does the ensemble want more members?

The roster sweep of 2026-08-29 found 76 scoreable members that were in neither the roster nor anywhere
else and added thirty of them. If pool SIZE were a lever, the honest seat curve should have moved when
the pool grew by a third.

    1 seat   0.8633
    6 seats  0.8738
    12 seats 0.8752
    19 seats 0.8763   <- peak
    24 seats 0.8762

**0.0025 across the whole range — 0.4 of the 0.0062 floor.** Against the pre-sweep pool the peak moved
from 0.8754 to 0.8763, +0.0009. So the count axis is not merely plateaued, it is INSENSITIVE TO THE
POOL: growing the pool by thirty members cannot reopen it.

The one thing more seats do buy is a steadier estimate. The greedy selection bias falls from +0.0020 at
six seats to +0.0010 at twenty-four. More members make the number you read more trustworthy; they do not
make it better.

## Second door: is any member a per-label specialist?

The competition score is a mean over labels, and the champion serves ONE flat mix for all of them. If a
member were strong on the worst label and mediocre elsewhere, a flat mix would average that strength
away by construction, and per-label routing would recover it. That is a mechanism that could plausibly
be worth real points, and it costs no card and no slot to ask.

    flat rank-average over all 113                       0.8666
    per-label routing, peak at top-10 members per label   0.8760
    global honest greedy seats, peak at 19 seats          0.8763

**Selection is worth +0.0094 over the flat average** — comfortably above the floor, and the one real
effect in this document. **Routing supplies no part of it.** Its peak sits 0.0003 below the global
greedy's, which is a twentieth of a floor, and the two rules share no structure: one picks a single
subset for every label, the other is free to pick a different subset per label and had every opportunity
to exploit a specialist.

The routing curve refutes the specialist hypothesis a second time, by its own shape. If any member were
a specialist, `top-1 per label` is precisely where it would show — that column is free to hand each
label its single best member. It reads **0.8657**, the WORST point on the entire curve, below even the
flat average over all 113. The curve then rises to its peak at ten members per label and decays
monotonically for a hundred more, back to the flat average at 113. Members here differ in general
competence, not in what they are good at; per-label freedom buys only the noise of choosing on fewer
studies.

## Why the two together are stronger than either alone

Each experiment was designed to find a lever and found a ceiling, and they are not the same experiment.
One varied HOW MANY members vote, the other varied WHICH members vote on WHICH label. They agree that
this pool tops out at ~0.876 under the honest estimator however you pick among it.

That agreement relocates the ceiling. It is not a fact about the combining rule — two rules with nothing
in common found the same wall — so it is a fact about the members. And `comp/champion.json` already
names what the members are: an architecture monoculture, overwhelmingly `facebook/dinov2-small` at
336 px with one head. Every axis this project has closed was closed INSIDE that monoculture.

## What it licenses

The only remaining lever on the offline number is a member that is not from the monoculture. That is
exactly what `swin_supervised` (supervised IN-22k pretraining, shifted-window attention) and
`vit_in21k` (the same objective swap at ZERO architectural distance from `v3base`) were built to test,
and it is why card time is going to trunks rather than to ensembling.

It also says what NOT to do. Do not spend a submission on either combiner axis: the count curve is flat
and the shipped champion already votes logit over a globally-chosen seat set, which is the better of the
two selection mechanisms measured here. There is nothing to ship from this document — its value is the
two experiments it means nobody has to run again.

## The honest caveat

Both numbers are split-half honest on 909 studies, and the floor those studies support is 0.0062. A
0.0003 gap and a 0.0009 gap are far below it; they are reported as "indistinguishable", never as "equal".
What the measurement supports is that neither axis delivers anything CLOSE to the ~1.5% a result needs —
not that the true effects are zero.

Beads: `knee-m69u` (count), `knee-ymek` (routing).
