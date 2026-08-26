# Our model is past its teacher; the public baseline is not

**2026-08-26 22:17Z.** Answers `knee-fki5`, filed twenty minutes earlier off forum topic 735826, which
proposed the diagnostic: *score your label extractor against the gold labels, then score your image model
against the same gold labels. If the image model does not beat its teacher there is modelling headroom;
if it beats the teacher by a large margin, the labels are the ceiling.*

The instrument already existed. `supervision_gap` runs it, and the only thing standing between us and the
reading was passing `--records`.

## The reading

```
python -m kneemri.evaluation.supervision_gap --labels <merged_labels.csv> --device cpu \
    --records <processed>/holdannot_s2026.pt
```

**58 annotated studies, 696 cells, radiologist calls 34.5% positive.**

| scored against the radiologist | mean AUC | vs teacher |
|---|---|---|
| report-derived labels — THE TEACHER | 0.8658 | — |
| **`holdannot_s2026` — OURS** | **0.8932** | **+0.0274** |
| the published public-notebook model | 0.8392 | −0.0266 |

**The two models sit on opposite sides of the same teacher.** The public baseline is 0.027 BELOW the
labels it learned from; ours is 0.027 ABOVE them. By the forum's own heuristic those are opposite verdicts:
the public pipeline has modelling headroom, and **we have a label ceiling.**

Per finding, ours beats its teacher on **9 of 12**:

| finding | labels | ours | ours − labels |
|---|---|---|---|
| Contusion | 0.7530 | 0.8961 | +0.1430 |
| Synovitis | 0.6870 | 0.7682 | +0.0812 |
| Effusion | 0.8149 | 0.8807 | +0.0658 |
| Baker's | 0.9321 | 0.9674 | +0.0353 |
| Medial OA | 0.9085 | 0.9380 | +0.0295 |
| Fracture | 0.8701 | 0.8917 | +0.0215 |
| Medial Meniscus | 0.9435 | 0.9579 | +0.0144 |
| Lateral OA | 0.7892 | 0.7930 | +0.0039 |
| ACL | 0.9865 | 0.9890 | +0.0025 |
| PF OA | 0.8861 | 0.8790 | −0.0071 |
| Lateral Meniscus | 0.8422 | 0.8273 | −0.0149 |
| MCL | 0.9762 | 0.9297 | −0.0465 |

## The denominator, said out loud

**58 studies. 696 cells. A per-finding row is 58 cells.** Nothing in the per-finding column resolves on its
own — a 0.0025 row and a 0.0039 row are the same row. The aggregate is over 696 cells and 12 correlated
findings, so it is stronger than any row but it is not a 909-study number either, and it has no interval
here. Quote the SIGN and the ORDERING; do not quote +0.0274 as a magnitude.

What survives that discount is the part that does not depend on precision: **the sign is opposite between
our member and the public one, against the same teacher on the same 58 studies.** That is a comparison of
two models against a shared control, not a point estimate.

## The member is a conservative choice, which makes the finding stronger

`holdannot_s2026` reads **0.8506** solo on the 909-study holdout — mid-pack, well under
`calib_dense_hold_s2026` (0.8634) and `small_batch6_s2026` (0.8598). It was used because
`annotated_readings` REFUSES any checkpoint not fitted with `--hold-annotations`: every other member
weights the annotated studies at 3.0, so scoring one here would be scoring its own supervision reflected
back. So the number above is what a MIDDLING member of ours does against the teacher. Our better members
are not measurable on this ruler at all, and there is no reason to think they sit lower.

Read the calibrated `*_hold` members with the caution `annotated_readings`' own docstring states: their
targets carry `Supervision.MEASURED_AGREEMENT`, a constant fitted on these same 696 cells, so their
annotated read is scored against a ruler sharing its origin. `holdannot_s2026` is the clean one, which is
the other reason it is the one quoted.

## What it changes

It moves the label lane from "one lever among several" to **the lever**, on our own measurement rather
than on the field's opinion:

- **`knee-pju4` (P0)** — 27.3% of supervision cells (14,445 of 52,872) are a constant rather than a read.
  We are already above the teacher; the 27% we never read is the largest single thing the teacher does not
  say.
- **`knee-b8g9`** — the near-boundary defect is not noise to average out. LLM self-reported confidence
  measures certainty in applying a heuristic to text; the noise lives in whether the radiologist wrote
  "small" or "moderate", and every calibrated-target arm inherits that pointing the wrong way.
- **Architecture is closed twice over.** A competitor swapped ResNet34 → DINOv2-small with data and labels
  frozen and got WORSE OOF BCE (0.664 → 0.674); our own scaling arm bought +0.0011. Both agree with this
  reading: if the model is already past its supervision, a bigger model reads the same imperfect labels
  more expensively.

**It does not change tonight's authorised slot.** `kneemri-resseat` is already argued as an explicit bet
on decode-path diversity and no label work can be trained, packaged and shipped before 00:00Z. It changes
what the NEXT block of card time is worth spending on, which is the decision this reading was run to
inform.

## The three findings where the teacher still wins

MCL (−0.0465), Lateral Meniscus (−0.0149) and PF OA (−0.0071) are the only rows where the report beats the
image. On 58 cells each that is not a finding; it is a place to look. MCL is the largest and the report's
own MCL AUC is 0.9762, so if any finding is genuinely image-hard rather than label-hard, it is that one.
Worth a look before assuming the label lane covers all twelve.
