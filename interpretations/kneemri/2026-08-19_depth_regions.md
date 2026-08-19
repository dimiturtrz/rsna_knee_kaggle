# Region pooling: the mean says nothing, the labels say two things

`depth_regions` — region pooling (`--slice-regions 4`) on the wide dense band — scored **0.8549** at
epoch 8 against `band_dense`'s **0.8578**. That is −0.0029 on a 0.0062 noise floor, which is a
non-result if the mean is the only thing read. It is not.

## What the arm predicted, and what happened

The arm was written against a specific claim: the wide band and the dense band are two halves of one
pooled budget, `band_wide` bought the collaterals at the centre's expense, `band_dense` bought the
centre back at the collaterals'. Region pooling was supposed to end the trade — one maximum per depth
region rather than one over the whole band — so the prediction was **the centre holds AND the
periphery recovers**.

Per label against `band_dense`, with `band_wide` alongside as the other end of the trade:

| label | `depth_regions` | `band_wide` |
|---|---|---|
| PF OA | **−0.0193\*** | **−0.0194\*** |
| Lateral Meniscus | **+0.0165\*** | +0.0007 |
| MCL | +0.0031 | +0.0111 |
| Medial OA | −0.0086 | −0.0122\* |
| MEAN | −0.0029 | −0.0025 |

(`*` = clears that label's own paired 95% interval; half-widths up to ±0.028.)

**Both clauses of the prediction are wrong.** The centre did not hold: PF OA falls by −0.0193, which
is `band_wide`'s −0.0194 to the fourth decimal. The periphery did not recover either: MCL gains
+0.0031 where `band_wide` gained +0.0111, so region pooling recovered *less* of the collateral than
the plain wide band did.

## What is there instead

An unpredicted signature the mean averaged away. Of thirteen labels exactly two clear their interval,
and they point in opposite directions:

- **PF OA −0.0193, identical to `band_wide`.** The patellofemoral loss is the WIDE BAND, and the
  pooling does not touch it. Whatever `band_dense` bought back at the centre, it bought with the band,
  not with the aggregation.
- **Lateral Meniscus +0.0165, where `band_wide` got +0.0007.** This is the only place the two wide-band
  arms disagree, so it is the only place attributable to region pooling at all — and the mechanism is
  the one the arm was built on, just landing somewhere it was not aimed: a whole-band `amax` lets the
  loudest depth win the whole stack, and a per-region maximum gives a mid-band focal structure its own
  voice.

The pooled-budget story survives as a statement about the BAND. Region pooling is a separate, smaller
effect on one focal off-centre structure.

## What this does not settle

Thirteen labels and two stars is a multiple-comparison risk; a signature that stars once is not a
finding. `depth_regions_narrow` is the single-variable follow-up and it is now worth its slot for two
reasons rather than the one it was written with: on a 0.20–0.80 band there is no patellofemoral loss
to pay, so a surviving lateral-meniscus gain lands net-positive AND reproduces the signature on a band
that shares nothing with this one but the pooling.

Per the project rule, this is a QUESTION about where region pooling belongs, not a verdict against it.
Nothing here refutes the aggregation; it refutes the arm's own written prediction, and it moves the
mechanism from "fixes the band trade" to "protects one focal structure from the whole-band maximum".

## Member value, separately from the mechanism

`depth_regions` is the only member in the pool with +0.0165 on Lateral Meniscus. Decode diversity is
the one axis measured to transfer to the board (0.0059 offline → 0.014 there), so a member whose error
is *differently shaped* is worth more than its solo mean suggests.

IT COSTS NO DECODE GROUP ALONGSIDE `band_dense`, which an earlier draft of this section got wrong.
Read off the checkpoints against `Member.pixel_group`'s key (band, crop mm + centre, tie, group, img,
slices, slots — `members.py:432`), `depth_regions` is band 0.1–0.9 / 20 slices / 336 px / centre=image
— every field identical to `band_dense`. Region pooling happens on features that are already computed,
so it changes the model and not the pixels. The two share one decode, and the greedy four
(`band_dense`, `band_wide`, `depth_regions`, `calibrated_targets`) is THREE groups, not four.

## The lateral-edge explanation, proposed and refuted the same hour

A sagittal slice index IS the medial–lateral axis, so a whole-band maximum lets the medial compartment
— larger, and roughly twice as many positives — win a channel the lateral compartment is competing
for, and lateral structures additionally sit nearest the band edge. That story explains +0.0165 on
Lateral Meniscus, and it makes a second prediction on the same data: **Lateral OA must move too.**

It does not. Lateral OA is −0.0019 [−0.0144, +0.0123] — flat, on 233 positives, a label with enough
power to have shown a move of that size. The explanation is refuted by its own second clause.

STATE THE MULTIPLICITY, too: twelve labels tested at 95% give roughly 0.6 stars by chance, and this
comparison has two. PF OA's is not one of them — it reproduces `band_wide`'s −0.0194 to a thousandth
on a different arm, which is corroboration no chance star gets. Lateral Meniscus has no such second
witness. So the honest state of it is ONE unexplained label, not a mechanism, and `depth_regions_narrow`
is what turns it into either a reproduction or a coin flip.
