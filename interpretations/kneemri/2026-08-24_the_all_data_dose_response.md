# The all-data axis is a dose-response, and every band has a move — written before the read

2026-08-24 08:30Z, with `kneemri-allfit` (ref 55736360) still PENDING at 55 minutes. Written now, on
purpose: the four bands were pre-registered in the kernel docstring but what to DO in each was not, and a
consequence chosen after seeing the number is a story rather than a decision.

## Why this is a dose-response and not a one-off

Two board entries differ on this axis and on almost nothing else:

| entry | all-data seats | fold-0 seats | count | vote | board |
|---|---|---|---|---|---|
| `finalfit` | 1 (`final_ws_e12`) | 6 | 7 | RankAverage | **0.934** |
| `allfit` | 6 | 1 (`band_dense_long_s2026`) | 7 | RankAverage | pending |

Count and vote are held fixed, so the pair reads as a DOSE on the one axis the offline holdout is
structurally blind to: a `--fold -1` member trains on all 4,406 studies and scores `nan` forever. This is
the only lever still open — every offline family is closed (member count saturated four steps running,
resolution's candidate count zero, backbone capacity inside the floor from small through large, per-label
weighting oracle-bounded dead, label fidelity not the lever).

It is a dose-response with two points, which is worth saying plainly: it can establish a SIGN and a rough
size, and it cannot establish a shape. One inversion would not refute it and one steep step would not
prove linearity.

## The reading is not confounded by plumbing

Six of allfit's seven seats carry an EMPTY serving fingerprint — the `knee-52mu` hole, where `--fold -1`
took its witness from a holdout it did not have. That would have made a low read ambiguous between the
axis and a reading defect. It is closed by twin transfer rather than left open: each of the six has a
witnessed sibling whose reading config is identical key by key, and all five distinct siblings pass with
the residual spread across 4–8 of 12 labels, which is the arithmetic shape rather than the concentrated
shape a reading defect makes. So the number is about the axis.

## The four bands, and the move each one buys

**≥0.940 — the axis compounds across seats.** Six seats bought ≥0.006 over one seat. The last fold-0 seat
becomes worth converting, which is `band_dense_long_ff` at 7.3 h of card. That is the only band that
justifies the train, and it needs asking biohub for the card properly rather than taking it.

**0.936–0.939 — sub-linear, or a shared ceiling.** The axis is real but the seats are not independent:
six all-data members largely re-learn the same extra studies. Converting a seventh seat buys a seventh of
a diminishing quantity, so the 7.3 h train is NOT justified — the honest move is to stop here and record
the axis as real-but-spent.

**0.933–0.935 — one-member effect, flat.** `finalfit`'s +0.001 over `gradeswap`/`accumseat` (both 0.933)
was already inside the floor, and this says the whole axis is. All-data training is neutral: it neither
helps nor hurts, and the extra 909 studies buy nothing the other six members were not already carrying.
Retires the axis. Nothing left open, which is a real and unwelcome finding rather than a null to work
around.

**≤0.932 — the informative loss, and the one band with a FREE move.** The reverse direction costs no card
at all: the fold-0 members already exist and are already packaged, so the move is to seat MORE of them and
the next slot tests it.

**The mechanism I first wrote for this band was WRONG, and it is corrected here before the read rather
than after.** I wrote that a low number would mean the ensemble had been living on FOLD diversity —
"seven members trained on seven different views of the data". That is false about both entries. Every
`_s2026` member in the roster is SAME-SPLIT: they share one fold-0 holdout of 909 studies, which is
precisely why they can all be seat-scored against each other. `finalfit`'s six fold-0 seats are six
members trained on the *same* 3,497 studies, differing in recipe, band, rate, resolution and encoder —
never in which studies they saw. Neither entry has any fold diversity, so no reading of this pair can
show fold diversity being lost.

What a low read would actually mean is narrower and duller: training on the extra 909 studies makes a
member WORSE as a seat. Candidate mechanisms, none of them tested — `--ema` rides along on every `_ff`
recipe and is not held fixed; a 26% larger corpus at an unchanged epoch count and schedule is a different
operating point on a lever this project has already seen invert (`calib_span` is negative at the recipe
we ship and positive at the one we do not); or the all-data members are simply individually weaker and
the pool is paying for it. Only the first is separable cheaply, and none is separable without the card.

This matters beyond the wording, because the wrong mechanism named a move that cannot be built. Real
fold diversity would need members trained on other folds, and folds are banned here by explicit standing
instruction — seeds and folds being the same thing tried slightly differently. Had the number landed low
and this text stood, the "free move" would have read as a licence for the one family that is closed by
decree. Seating more fold-0 members is still the right response to a low read; it is just a response to
"all-data seats are weak", not to "we lost our diversity".

Note the asymmetry: the two middle bands both end the axis and cost nothing further, the top band costs
7.3 h, and the bottom band hands back a shippable change for free. Only one of four outcomes wants the
card, which is the ratio that should have been stated before the entry was submitted and is stated now.

## What this does not decide

None of the four bands reopens an offline family. If the axis is spent, the candidate count for the next
member is zero and the gap to the leader (0.952 against a champion 0.934) has no named mechanism behind
it. That is the honest position, and inventing an arm to avoid stating it would spend wall-clock on a
family the ledger has already closed.
