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
is *differently shaped* is worth more than its solo mean suggests. It costs a decode group, though —
its `pixel_group` is the wide band's, not the pool's — and the board cannot price member count. That
makes it an ensemble candidate to weigh against a group budget, not a drop-in.
