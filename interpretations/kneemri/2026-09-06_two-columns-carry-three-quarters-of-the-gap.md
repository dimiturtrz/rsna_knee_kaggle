# Two findings carry three quarters of the gap to first place — and a third cannot be read

*2026-09-06 04:26Z, CPU-only while the card ran a dose arm. Zero card, zero slots.*

`pool_ceiling` now reports per finding, not only the mean. The mean said the pool is 0.0097 short of
first place. The columns say WHERE those points are.

```
per finding, weakest first — stack vs the best single member:
  Fracture             stack 0.8232  alone 0.8418    68+  band 0.1068
  MCL                  stack 0.8283  alone 0.8253   171+  band 0.0702
  Lateral OA           stack 0.8369  alone 0.8386   233+  band 0.0583
  Effusion             stack 0.8662  alone 0.8640   558+  band 0.0470
  Lateral Meniscus     stack 0.8783  alone 0.8700   226+  band 0.0534
  PF OA                stack 0.8783  alone 0.8733   409+  band 0.0441
  Medial OA            stack 0.8876  alone 0.8968   333+  band 0.0461
  Synovitis            stack 0.8882  alone 0.8852   124+  band 0.0608
  Contusion            stack 0.8934  alone 0.8867   267+  band 0.0433
  ACL                  stack 0.9066  alone 0.8976   248+  band 0.0429
  Medial Meniscus      stack 0.9086  alone 0.9135   482+  band 0.0401
  Baker's              stack 0.9220  alone 0.9237   217+  band 0.0433
```

`band` is the width of the column's own 95% bootstrap interval over studies, and `68+` is the count of
positives the column is scored on.

**Fracture 0.8232, MCL 0.8283, Lateral OA 0.8369 — against a pool mean of 0.8765 and a top column of
0.9220.** Lift all three to the pool's own mean and the competition score moves by +0.0118, which is more
than the +0.0097 that separates this pool from first place. But the bands say only two of the three can
be read:

- **Fracture is not readable.** 68 positives, band 0.1068 — its deficit below the pool mean is 0.0533,
  half its own band. The column is the noisiest of the twelve by a wide margin and it is where the stack
  most conspicuously fails to beat its best single member (−0.0186). Both of those are what a small
  positive count looks like. **The bead that gated this asked exactly this question and the answer is the
  one it warned about: this column is about the ruler.**
- **MCL is a real deficit.** 171 positives, band 0.0702 (±0.035) against a 0.0482 shortfall.
- **Lateral OA is a real deficit.** 233 positives, band 0.0583 (±0.029) against a 0.0396 shortfall.

So the honest target is two columns, not three, and it is worth

    (0.8765-0.8283 + 0.8765-0.8369) / 12 = +0.0073

against a +0.0097 gap. **Three quarters of the distance to first place sits in MCL and Lateral OA.** Not
all of it — which is itself the useful part, because it means a targeted member is necessary and not
sufficient, and the remaining quarter still has to come from somewhere.

## What this changes

The seat-requirement and weighting readings both closed the COMBINING routes and pointed at "a member
outside the distance/quality frontier". That was true but shapeless — it named no axis. This names one:
**a member that is better on MCL and Lateral OA**, and it does not have to be better anywhere else. A
member that lifts those two to the pool mean and is merely average on the other ten is worth +0.0073;
every member this project owns is instead an attempt to be good on the twelve-column mean, which is
precisely the objective that lets a column sit 0.05 low as long as ten others are fine.

Both surviving columns are also mechanically suggestive in the same direction. MCL is a peripheral
structure at the medial edge of the joint, and Lateral OA is a compartment-specific reading — both are
LOCALISED findings, and every one of our 188 members reads a centre-weighted crop and a fixed slice
budget chosen on the mean. That is an INPUT axis (field of view, slice selection, laterality), not
another recipe over the same corpus, which is the kind of thing the frontier reading said was needed.

## What it does not say

- **`alone` is selection-biased upward and `stack` is not.** The best-single-member column is a maximum
  over 188 members chosen on the very studies it is scored on; the stack column is out of fold. So a
  small negative gain (Lateral OA −0.0017, Baker's −0.0017) is not evidence the stack hurts. Only the
  large ones (Fracture −0.0186, Medial OA −0.0092) are worth a second look, and even those are an upper
  bound on the harm.
- **It does not say MCL and Lateral OA are LIFTABLE.** They are real deficits, not noise, but a finding
  can be genuinely harder to see. The band rules out "this is the ruler"; it does not promise a member
  can close it.
- **The band is the column's ABSOLUTE interval, and it overstates what a targeted arm could not
  measure.** A new member would be compared against this pool on the SAME studies, and a paired interval
  on the difference is far tighter than either arm's own band (that is why `paired_interval` exists).
  What the wide band on Fracture rules out is reading its RANK among the twelve as a fact — not the
  measurement of a change to it.
- The arithmetic is a decomposition, not a prediction. Raising two columns without touching the other
  ten assumes no trade, and a member trained toward MCL may lose elsewhere.

## What to do next

The gating question is answered, so the next one is whether the deficit is about FIELD OF VIEW. Both
surviving columns are localised and off-centre; if a crop or slice-selection variant already in the
ledger scores better on MCL and Lateral OA than on the mean, the axis is confirmed before any card is
spent. That is a ledger read, not a run.
