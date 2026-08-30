# The reader is right about order and wrong about value

2026-08-30. A forum thread said the ceiling here is label making, not capacity. That is a claim other
participants make about their own runs, mid-competition, so it is a hypothesis and not a fact. This is
what it looks like measured on our data.

## The denominator

**58 studies × 12 findings = 696 cells.** `train.csv` annotates a study on all twelve findings or on
none, so the radiologist rows are all-or-nothing and there are 58 of them. Every rate below is over
those 696 cells and over nothing else.

The table compared is `labels/report_labels_v2.csv` — the one `experiments/lib.sh` hands every arm —
read **raw**. Raw is load-bearing: `Targets.load` overwrites derived labels with the annotation
wherever a study has one, which would score the annotations against themselves.

## The reader orders well

| finding          | n+ | rank AUC | best thr | agreement @ best | @ 0.5 |
| ---------------- | -- | -------- | -------- | ---------------- | ----- |
| ACL              | 24 | 0.987    | 0.94     | 0.966            | 0.845 |
| MCL              |  9 | 0.976    | 0.82     | 0.948            | 0.828 |
| Medial Meniscus  | 26 | 0.944    | 0.82     | 0.914            | 0.845 |
| Baker's          | 12 | 0.932    | 0.68     | 0.897            | 0.879 |
| Medial OA        | 15 | 0.909    | 0.82     | 0.862            | 0.828 |
| PF OA            | 21 | 0.886    | 0.82     | 0.862            | 0.793 |
| Fracture         | 18 | 0.870    | 0.82     | 0.845            | 0.828 |
| Lateral Meniscus | 23 | 0.842    | 0.82     | 0.810            | 0.759 |
| Effusion         | 35 | 0.815    | 0.50     | 0.724            | 0.724 |
| Lateral OA       | 11 | 0.789    | 0.68     | 0.845            | 0.828 |
| Contusion        | 19 | 0.753    | 0.94     | 0.724            | 0.621 |
| Synovitis        | 27 | 0.687    | 0.68     | 0.707            | 0.690 |
| **mean**         |    | **0.866**|          | **0.842**        |**0.789**|

Mean rank AUC 0.866, ACL 0.987 — the reader knows which studies are worse. Binarised at the 0.5 this
project uses everywhere, agreement is 0.789. **Ten of twelve findings want a threshold above 0.5**,
eight of them 0.82. The reader over-calls positives on every finding but one.

Synovitis is the exception and it reverses: 14 misses against 4 false positives, at AUC 0.687. The
reader can barely see it either way. Contusion is the worst over-call — 20 false positives against 39
negatives.

## And it is wrong about value

Pooling the levels over all twelve findings:

| reader level | cells | P(gold positive) |
| ------------ | ----- | ---------------- |
| 0.08         | 196   | 0.041            |
| 0.28         | 177   | 0.136            |
| 0.50         |  12   | 0.417            |
| 0.68         |  98   | 0.378            |
| 0.82         | 141   | 0.688            |
| 0.94         |  72   | 0.958            |

**The two ends are nearly right.** 0.94 → 0.958 and 0.08 → 0.041. That is why nothing downstream has
noticed: the reader is confident exactly where it is correct.

**The middle is not.** The 0.68 bucket is 98 cells — 14% of everything — that are **62% negative**, and
they are trained today as 0.68 positive. The target is wrong by about 0.30 in the direction that
invents positives. 0.08 and 0.28 each over-state by roughly a factor of two. (The 0.50 bucket holds
twelve cells and is not separately estimable; the apparent non-monotonicity at 0.50/0.68 is that cell,
not a fact.)

## The mechanism, which is the forum's

The reader's number is confidence in applying a heuristic to **words**. The gold is a radiologist
reading an **image**. Per a host remark in the same thread, the people who made the *test* labels did
not start from the reports either — they read images. So our supervision and the competition target are
two different generating processes that happen to correlate.

Two processes that share an ordering and not a scale is exactly the shape above. They agree at the
extremes, where the words are unambiguous, and come apart in the middle, where "small effusion" is a
coin the report already flipped. Boundary cases in the thread are outright contradictory: two reports
with the same English template and the same sentence carry opposite Effusion labels, and at least one
"moderate effusion" is labelled 0.

**So the fix is not a better reader.** A better reader improves the ordering, and the ordering is
already 0.866. The fix is a monotone re-map of the six levels onto their empirical rate — six
parameters fitted on 696 cells, no model, no card.

## Why this is the axis worth the next arm

`knee-m69u` measured **+0.0034 from swapping the label table** between three files that binarise
identically and score any prediction set to the same four decimals. That gain cannot be a scoring
effect; it is a training effect — the table changes what the model learns. This changes the target
*values* on 14% of cells rather than swapping which file supplies them, so it is the same mechanism
and strictly larger.

The arithmetic that makes it the best ratio on the board: the seating axis is closed from the price
side (every arrangement of members we own is already behind the champion), top-10 costs **+0.0042**
offline, and no mapped knob pays it. This costs a CPU relabel plus one arm at the control recipe.

## The ruler asymmetry, stated before anything is armed

**Our offline holdout truth IS the report-derived table.** A supervision change that moves training
toward the image-read process can therefore lower the offline number while raising the board — the
offline test does not have this axis.

The only rulers pointed at the target process are the 58 gold rows (`foreign_gate --gold-auc`) and a
submission. So an arm here is read on **gold AUC first** and on holdout second, and a holdout
regression inside the 0.0062 floor is **not** a refutation. A gold-AUC regression is.

## Stated limits

- n = 58 studies. Enough to see a 0.30 miscalibration on a 98-cell bucket; nowhere near enough to fit
  twelve per-finding thresholds, which is why the proposed re-map is pooled and has six parameters.
- The gold rows are *our* corpus's annotations, not the competition's test labels. The argument that
  they are closer to the test process is the host's remark plus the mechanism, not a measurement.
- The best-threshold column is fitted in-sample on 58 studies and is a diagnostic, not a proposal. The
  0.842 it reaches is an upper bound on what thresholding alone could buy, not an expectation.

Beads: `knee-7oep` (the measurement), and the arm it licenses.
