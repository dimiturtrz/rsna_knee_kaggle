# The worst tenth is under-supervised, not mislabelled — 2026-09-02

`python -m kneemri.evaluation.study_attribution --run <name>`, read on two members.

## What was asked

"WHY are the wrong studies wrong?" — the second half of the level-2 question. Level 1 closed at
`knee-q1pf`: we lead twenty published members on ten of twelve labels, so neither the loss nor the class
balance survives as an account of the error. `study_errors` then named WHICH studies carry it. This is
the attribution.

## The reading, in admissibility order

Measured on `calib_dense_hold_s2026` (909 studies, tail 90, body 819) and confirmed on
`stackfit_s2026`. The unit is a discordant pair, which is exact rather than a proxy: per label the AUC
is one minus the discordant fraction.

1. **The tail is SHARED, not this member's noise.** 0.656 of the tail is shared between the two members
   against a 0.100 chance rate — 6.6× — and only 4 of 90 tail studies are private to one member.
   Confirmed at 0.667 the other way round. The difficulty is a property of the STUDIES. This is the
   reading that makes every report- and acquisition-side question below admissible at all; without it
   the tail is a fact about one checkpoint.
2. **Acquisition reads nothing.** series_count +0.244, planes 3.000 both, fluid 0.593 vs 0.594, fat
   0.593 vs 0.594. Flat null. Retired as an explanation. (Corpus note found on the way: `Fluid_Sensitive`
   and `Fat_Suppression` are DUPLICATE columns — identical counts 14010/10361. Two of the four
   acquisition rows were one row.)
3. **The report text mostly does not separate.** Of the seven categories, only `missed_negation` clears
   2σ and it goes the WRONG way (tail 0.156 vs body 0.245). `genuine` disagreement is flat
   (−0.011 ± 0.028). So the tail is NOT enriched in "the report plainly says it and we still got it
   wrong".
4. **SILENCE separates** — tail 0.767 vs body 0.653, **+0.113 ± 0.048**, confirmed +0.097 ± 0.048 on the
   second member. Explicitly post hoc, printed as such.

## The correction that had to be made, and why it is the finding

The first reading of (4) was that the tail rests on wrong labels. **That does not survive
`2026-08-26_silence-is-not-the-defect.md`**, which measured the radiologist calling **13.6%** of silent
cells positive against **41.6%** where the reader spoke. Silence tracks ABSENCE. A silent zero is mostly
a correct zero, and a perfect silence-filler fixes at most 24 cells while over-calling costs 108.

So a fourth reading was built to separate a WRONG zero from a NEVER-READ one. The truth array cannot do
it — `NO` and `UNK` both binarise to a negative label and the threshold destroys the difference. Only
the reader's verdict column survives, where `NO` carries value 0.08 and `UNK` carries 0.28, the corpus
base rate the reader fell back to with nothing to say.

**Negative cells the reader never read, tail against body:**

| member | tail | body | difference |
|---|---|---|---|
| `calib_dense_hold_s2026` | 54/90 (0.600) | 374/819 (0.457) | **+0.143 ± 0.054** SEPARATES |
| `stackfit_s2026` | 59/90 (0.656) | 387/819 (0.473) | **+0.183 ± 0.053** SEPARATES |

Larger than the silence effect on both members. **The tail is UNDER-SUPERVISED.** Its negative cells were
trained against a soft 0.28 prior and are scored against a hard 0 — simultaneously the weakest
supervision the model got and the weakest part of the ruler it is judged by.

## What this does NOT license, and it is the whole point

**It is not a reason to remove, fill, or re-label the UNK cells. That was already run and it inverted.**
`confident224` cleared every UNK cell rather than filling it — 14,445 of 52,872 pairs, 27.3% of the
corpus, trained at weight 0.0 — and scored **0.8431 against `native224`'s 0.8502, −0.0071, 1.15 floors
DOWNWARD**. Removing a quarter of the supervision COST score, so the UNK cells are weak evidence the
model is already using at their 0.2875 effective weight. Both other halves of that project are also
already spent: the VALUE is corrected (`MEASURED_AGREEMENT` maps 0.28 → 0.136 plus per-finding logit
shifts) and the WEIGHT is floored (`--weight-floor`, in `$RECIPE` on all six champion seats).

The under-supervision result and `knee-teh4`'s confidently-WRONG cells are DISJOINT populations. This
tail is where the labels say little, not where they say the wrong thing.

## What survives as a lever

The ledger's own surviving clause from `confident224` is the narrow one: the three labels whose UNK share
is extreme — **Synovitis 84.2%, Fracture 56.4%, Baker's 45.9%**. That intersects `knee-q1pf`'s weakest
columns: **Fracture is 0.7827, our second-weakest**, and it is 56.4% unread. A label that is both weak
and majority-unread is the one place where "the ruler is thin" and "the model is weak" are the same
sentence, and it is the only version of the reader project the measurements have not already refuted.

That check is card-free and per-label. It is the next step, not a retrain.

## Method note

Three of the four readings are refusals — acquisition flat, report text flat, silence caveated into a
statement about supervision rather than correctness. The one that separates only became readable after
the repo's own prior was consulted, which is the second time in a week a result was repriced by reading
`interpretations/` before filing. The caveat is now hard-coded into the printed line, so the number
cannot be quoted as label error by a later reader who skips this file.
