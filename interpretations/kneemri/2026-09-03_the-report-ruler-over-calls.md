# The report ruler over-calls, and our own members already beat it

2026-09-03, card-free, from arrays already on disk. Answers `knee-ds2e` in the LABEL-NOISE direction.

## The question

`knee-qrtx` put 87.3% of the ensemble's remaining uncertainty in the ALEATORIC term — what no member
knows. But aleatoric is only defined GIVEN THE LABEL, and our labels are `report_labels_v2.csv`: LLM-read
findings from radiology report text. So that mass is imaging ambiguity PLUS annotation error, and nothing
separated them. The two readings imply opposite work: near-ceiling imaging ambiguity closes the member
search, material annotation error opens a lever no architecture can reach.

## The instrument

`train.csv` carries RADIOLOGIST annotations for 58 studies. Eight members on disk were fitted with
`--hold-annotations`, so those 58 rows were dropped from their training tables and none of them has seen
a single gold cell: `calib_span_hold`, `blendhold`, `holdannot`, `calib_dense_hold`, `band_dense_hold`,
`recal_holdannot`, `finding_calib_dense`, `band_span_hold` — the flag was read off each one's own recipe
row, not off its name (`finding_calib_dense_s2026` carries no "hold" in its name and does hold). Their
predictions over all 58 are already cached under `processed/…/annotated_predictions/`, so this costs no
card and no slot.

## What the report ruler gets wrong

At the project's 0.5 midpoint, over 58 x 12 = 696 cells:

| | count | rate |
|---|---|---|
| cells where report and radiologist disagree | 147 | **0.2112** |
| report says POSITIVE, radiologist NEGATIVE | 115 | |
| report says NEGATIVE, radiologist POSITIVE | 32 | |
| gold positives | 240 | 0.345 |
| report positives | 323 | 0.464 |

**The error is one-directional. The report reader OVER-CALLS**, by 83 net positives on 240 — about a
third again as many findings as the radiologist saw. Per label, the over-call is everywhere except one:

```
label              gold+  rep+     FN     FP  disag
ACL                   24    33      0      9      9
MCL                    9    19      0     10     10
Medial Meniscus       26    33      1      8      9
Lateral Meniscus      23    31      3     11     14
Medial OA             15    23      1      9     10
Lateral OA            11    17      2      8     10
PF OA                 21    25      4      8     12
Effusion              35    51      0     16     16
Synovitis             27    17     14      4     18
Baker's               12    17      1      6      7
Contusion             19    37      2     20     22
Fracture              18    20      4      6     10
```

Synovitis is the single inversion and it inverts hard — 14 under-calls against 4 over-calls. That is the
same finding the tail reading already produced from the other side: Synovitis is 84.2% UNK, so the report
is SILENT about it far more often than it is wrong about it, and silence scores as a hard zero.

## What the members do with it

Scored against the radiologist on the same 58, mean AUC over the twelve labels:

| | mean AUC |
|---|---|
| `report_labels_v2` used as a predictor | **0.8658** |
| the eight-member logit-average pool | **0.9174** |

Paired delta **+0.0516**, 95% CI **[+0.0296, +0.0729]** over 4,000 study-level bootstrap draws,
P(delta > 0) = 0.9998. Eleven of twelve labels move the same way; MCL is the only one that does not
(-0.0057), and it is the label with the fewest gold positives (9 of 58).

```
label                REPORT     POOL pool-rep
Contusion          0.7530  0.9123  +0.1592
Effusion           0.8149  0.9416  +0.1267
Synovitis          0.6870  0.7551  +0.0681
Lateral Meniscus   0.8422  0.9081  +0.0658
Fracture           0.8701  0.9306  +0.0604
Baker's            0.9321  0.9692  +0.0371
Medial OA          0.9085  0.9442  +0.0357
Medial Meniscus    0.9435  0.9772  +0.0337
PF OA              0.8861  0.9138  +0.0277
ACL                0.9865  0.9877  +0.0012
Lateral OA         0.7892  0.7988  +0.0097
MCL                0.9762  0.9705  -0.0057
MEAN               0.8658  0.9174  +0.0516
```

Every solo member also beats the report ruler on its own: 0.8964 to 0.9189 against the report's 0.8658.
The gain is not an ensemble effect. **Models trained on a noisy target already average away a large part
of that target's per-study noise**, and the labels they are then SCORED against still carry it.

The ordering of the per-label gains is the same ordering as the over-call: Contusion (20 FP) and
Effusion (16 FP) are the two labels the report over-calls hardest and the two where the members beat it
hardest. That agreement between an error COUNT and an AUC DELTA, computed from different quantities, is
what makes this a mechanism rather than a coincidence of two small samples.

## What this establishes, and what it does not

**Establishes.** The aleatoric mass is not purely imaging ambiguity. A ruler with a 21% cell error and a
systematic one-directional bias sits between every arm this project has run and the thing it is trying to
measure. `knee-ds2e` is answered in the LABEL-NOISE direction, with a sign and a size.

**Does not establish the RATE over the corpus.** State the denominator: **58 of 4,407 studies, 1.3%**.
Nothing here shows the annotated 58 were sampled at random, and if they were chosen as interesting cases
the 21% is an upper bound rather than an estimate.

**Does not establish which ruler the hidden test uses.** The argument that it is the radiologist form is
that `train.csv` — the competition's own label file — is where these 58 gold rows come from, so gold is
the host's own labelling convention and the report table is ours. That is an argument from provenance,
not a measurement, and it is the load-bearing assumption of everything below.

**Is not a board number.** Nothing here has been shipped, and the compression from an offline gold delta
to a board delta is unmeasured on this axis — `board_transfer`'s slope was fitted on report-holdout
points and does not transfer to a different ruler.

## What it redirects

1. **The 909-study holdout is a report ruler.** Every offline delta this project has quoted — every seat
   value, every member curve, every `BOARD_RESOLVABLE` comparison — is measured against a target that
   over-calls positives by a third. That does not invalidate the ORDERINGS, since the bias is shared by
   both arms of every comparison, but it does mean an arm that reduces over-calling is charged for it.
2. **Label correction is an axis, and it is card-free on the target side.** `knee-kabn` already found one
   instance of it from the other direction (Fracture + 0.25*Contusion, +0.063 on the radiologist ruler).
   The over-call table above says where the rest of it is: Contusion, Effusion, Lateral Meniscus.
3. **The correction must come from a MECHANISM, not from fitting the 58.** Twelve per-label thresholds
   fitted on 58 studies is a guaranteed overfit and the gold set is then spent — it is the only honest
   ruler this project owns and there is no second one. Derive the shift from the reader's behaviour (the
   UNK cell at 0.28, the over-call direction, the Synovitis silence), then CHECK it on the 58 once.

## Reproduce

Eight cached prediction arrays under `processed/…/annotated_predictions/`, `GoldRows().annotated()` on
`raw/train.csv`, `GoldRows().reported()` on `report_labels_v2.csv`. No GPU, no slot, no training.
