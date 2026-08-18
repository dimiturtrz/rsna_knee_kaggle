# The outer band holds the collaterals, not the menisci

`band_wide`, 2026-08-19. The sagittal slice band widened 0.20–0.80 → 0.10–0.90 at constant
mm-per-slice (12 → 16 slices over a 1.33× wider span), everything else the control's recipe: 336 px,
whole-stack, max pooling, backbone rate 4e-5, 12 epochs, fold 0, seed 2026.

**0.8553** against the control `rate_wholestack` **0.8525**, +0.0028, peak at epoch 6 of 12.

## What it settles

This was the last live version of "we discard data" in knee-jk3. Every other item on that bead's list
is refuted by our own measurements — cache density (−0.0001), slot count (+0.0002), resolution
(+0.0002) — and each of those is *breadth* or *acuity*. This one is coverage of the series along the
axis the cache never varied, and it is the only one that moved.

It is +0.0028 on a 0.0062 noise floor, so **the mean is directional, not a result**. What is worth
more than the mean is where it landed, because the prediction was written down before the run and the
prediction named the wrong labels.

## The prediction, and how it failed

Written into `experiments/arms/band_wide.sh` before the arm started:

> the band carries joint anatomy the model can use → above 0.854, and the gain lands in the labels
> whose compartments sit off-centre (Lateral/Medial Meniscus, Lateral OA), not uniformly.

Score: hit (0.8553). Uniformity: hit, the move is concentrated. Labels: **missed**, and not narrowly.

| label | Δ vs control | | label | Δ vs control |
|---|---|---|---|---|
| MCL | **+0.0133** | | Effusion | +0.0014 |
| Synovitis | +0.0083 | | Lateral Meniscus | +0.0012 |
| Fracture | +0.0074 | | Medial Meniscus | +0.0008 |
| Contusion | +0.0072 | | PF OA | −0.0002 |
| ACL | +0.0071 | | Lateral OA | −0.0013 |
| | | | Medial OA | −0.0088 |

909 shared holdout studies. Every interval straddles zero, so no single row resolves on its own; the
PATTERN is what is being read, and the pattern is the opposite of what was predicted. The three
menisci/lateral-OA labels that were named are the three flattest rows in the table. The OA labels are
negative.

## The mechanism the table actually points at

The band runs medial ↔ lateral. What sits in the outer tenth at each end is not the menisci — those
are mid-compartment structures that the 0.20–0.80 band already contained, which is exactly why they
did not move. What sits there is the **collateral ligaments**: MCL far medial, LCL far lateral, both
running along the outside of the joint and both routinely cut off by the old band. MCL is the largest
single move in the table by a factor of nearly two, and it is the label whose anatomy the widening
literally added.

ACL is the same story one step in — it is central in the coronal sense but its femoral and tibial
attachments spread toward the notch walls. Contusion and Fracture are peripheral bone: marrow signal
in the metaphyseal flare, off the mid-slices. Synovitis is a distributed finding whose recesses reach
the gutters.

So the reading is: **the outer band was cutting off structures that live at the edges of the joint,
and the model was scoring them from context.** `band_coverage` (2026-08-17) could measure how many
millimetres were being discarded — 20.8 mm from each end of a 103.8 mm span — but could not separate
discarded tissue from discarded joint. This arm asks the model instead of the pixels, and the answer
is that a fifth of what was discarded was joint.

Why OA goes the other way is the honest loose end. Medial OA −0.0088 is the largest negative and OA
is the one finding class that is genuinely central-compartment; a plausible reading is dilution — 16
slices of which four are new peripheral content, max-pooled, gives the central evidence a smaller
share of the pooled representation. That is a hypothesis with no measurement behind it, and it is
testable (band 0.10–0.90 at 20 slices holds the central density AND adds the periphery), but it is
not tested here.

## What it is worth as a member

`band_wide` is the strongest single member this project has produced — 0.8553 alone, past
`rate_wholestack`'s 0.8525 — and it is not merely stronger, it is **different**. Greedy forward
selection over the accumulated pool picks it FIRST, and the five-member mix that included it reads
0.8643 against 0.8632 for the same five with the control in its place. Paired with the control alone
it makes 0.8607, +0.0054 over the better of the two.

Read all of those as signs rather than scores — a training-time record is the trainer's reading of the
holdout, and the holdout prices diversity 2.4× too cheap.

## What it does not say

- **Not "the band should be 0.10–0.90 everywhere."** It is one arm, one fold, one seed, and the mean
  is inside the floor. It says the outer band is not empty.
- **Not a licence for the other knee-jk3 items.** Cache density, slot count and resolution stay
  refuted; that bead is explicit that it must not be read as authority for them, and this result
  changes nothing about them.
- **Not a meniscus result.** The prediction that named the menisci is refuted by its own arm, and that
  is recorded here rather than quietly dropped, because the same reasoning — "off-centre compartments
  are what the band cuts" — would have been reused.
