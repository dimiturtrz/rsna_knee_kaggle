# The corner is a PLATEAU of twelve members, and the midpoint ruler cannot resolve which one is on top

2026-09-17. Bead knee-wbwq. No card spent: one `ledger75` pass over every holdout record on disk
(`--holdouts D:/data/volumetric/mri/models/rsna_knee_abnormality`, 428 of 447 records, 19 VOID), parsed for the 363 rows on
population `66c6f231`, 909 studies.

**The answer, stated first: twelve distinct members sit between 0.8998 and 0.9053 on ledger75 — a 0.0055
spread, INSIDE the 0.0062 paired floor. Which of them is "the best solo we own" is not a measurable fact.
And the midpoint-to-ledger75 move is not a constant offset: it ranges from −0.0154 to +0.0504, with a
p10–p90 spread of 0.0134. So a midpoint delta under about 0.013 is unreliable in its SIGN, which is more
than twice the floor that this repo has been treating as the resolution limit.**

## The move is a distribution, not an offset

| statistic | midpoint → ledger75 move |
|---|---|
| min | −0.0154 |
| p10 | +0.0218 |
| median | +0.0289 |
| p90 | +0.0352 |
| max | +0.0504 |

A shared offset would cancel in any A/B. It does not cancel, because it varies by member. The rank
churn it causes is large: `chainfour_v5blend_s2026` moves +0.0504 and climbs 97 places (#281 → #184),
`ctl_nofloor_s2026` +0.0377 and 88 places, `hisfluid_hold_s2026` +0.0376 and 76 places. These are not
members anyone would have called mis-priced.

The one member that barely moves is the one whose name explains it: `recal_holdannot_s2026`, +0.0040,
the best member on the midpoint (#0) and eleventh on ledger75. It is recalibrated, so its score
distribution already sits where the 0.75 cut wants it. That is the mechanism of the whole effect stated
in one row — **the move measures how far a member's outputs are from the cut, and it is a property of
the member, not of the ruler.**

## The plateau

Ledger75, population `66c6f231`, 909 studies, top of the table:

| member | midpoint | ledger75 |
|---|---|---|
| `labelwise_depth24_drop40_long_s2026` | 0.8679 | **0.9053** |
| `labelwise_drop_drop40_long_s2026` | 0.8703 | **0.9051** |
| `labelwise_dense_drop40_long_s2026` | 0.8672 | **0.9048** |
| `labelwise_ext_drop40_long_s2026` | 0.8672 | **0.9045** |
| `labelwise_depth24_negpull_s2026` | 0.8666 | **0.9044** |
| `lr8e5_ship_s2026` | 0.8673 | **0.9031** |
| `labelwise_dense_drop40_long36_s2026` | 0.8678 | **0.9030** |
| `roinative_labelwise_drop40_long_s2026` | 0.8641 | **0.9018** |
| `labelwise_dense_stem_llrd_s2026` | 0.8667 | **0.9007** |
| `recal_holdannot_s2026` | 0.8964 | **0.9004** |
| `warm_calib_lowlr_drop40_long_s2026_run1` | 0.8632 | **0.9000** |
| `labelwise_depth24_s2026` | 0.8617 | **0.8998** |

Every one of these is within the 0.0062 paired floor of the top row. The knobs separating them —
`depth24` against `dense` against `drop`, `negpull`, `stem_llrd`, `36` epochs against 24, ROI-native
against not, warm against cold — are twelve different answers to twelve different questions, and the
ruler says all twelve landed in the same place.

**And the midpoint ORDERS THEM DIFFERENTLY.** On the midpoint the top row is `labelwise_drop_drop40_long`
(0.8703), not the corner (0.8679); `recal_holdannot` is first of all (0.8964) and eleventh here. Two
rulers, two orderings, one set of predictions.

## What this closes

**Closed: knob search inside the labelwise / drop40 / long family.** A thirteenth arm that changes one
more knob in this family has twelve priors saying the outcome lands inside the floor, and neither ruler
could tell you if it had won. This is not "the knobs do not work" — the family as a whole is +0.04 over
the pool's median — it is that the family has been walked to its own plateau and the remaining variation
in it is scatter.

**Closed: quoting a solo delta under ~0.013 at the midpoint, for any purpose.** The repo's stated
resolution limit is the 0.0062 paired floor, and that floor is measured on ONE ruler. The
cross-ruler disagreement is twice as wide. A verdict that rests on a midpoint delta between 0.006 and
0.013 has never been supported by its evidence.

## What this does NOT license

- It does not say ledger75 is right and the midpoint wrong in some absolute sense. It says they are
  different statistics, that they disagree by more than the floor, and that **a delta must name its cut
  or it is not a number** — the `nn_corr@pool251` rule, now with a measured size for the violation.
- It does not re-open any closed axis by itself. The move is mostly POSITIVE and mostly uniform-ish in
  the middle, so a decline that read "inside the floor and pointing down" at the midpoint almost always
  still reads inside the floor on ledger75. What the spread licenses is re-pricing a decline whose
  delta sat between the floor and ~0.013 — and the honest expectation there is that a loss becomes a
  flat, not that a flat becomes a gain.
- **It does not identify a better solo.** It identifies that we do not have one, and that the next arm
  must change the trunk, the objective, or the data — because the knobs are spent.

## A false alarm, kept here because it is the third instance today

While reading this table I matched each of the top thirty stems against the raw text of
`comp/members.roster` and `comp/members.declines` and got **seven hits in neither**, including
`labelwise_drop_drop40_long_s2026` at 0.9051. That would have been a shipping defect worth a P0.

It is not one. `roster_check` reports **1 undocumented member, and it is `vjepa_cold_s2026`** — the arm
running on the card as this was written, which gets its line when its checkpoint closes. The seven are
all documented under spellings a raw substring test does not match.

The rule this is the third instance of today: **a question that a check in this repo already answers
must be asked of the check, not of `grep`.** The first instance asserted an arm had never run from a
grep of `experiments/arms/*.sh`; the second compared a README midpoint against a ledger75 number; this
one invented seven unshippable members out of a string comparison. All three were caught, and all three
cost more to catch than the one command would have cost to run.
