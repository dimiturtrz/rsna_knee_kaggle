# The 58-study ruler does not obey the threshold the competition states

2026-08-25. `python -m core.data.labels.severity_boundary`, on `raw/train.csv`.

The competition's introduction grades an effusion positive at **moderate or above**. That is a rule
about words, so it is checkable: bucket each radiologist-annotated report by the size qualifier in its
own effusion sentence, and read the annotation beside it.

## What it reads

```
28 English annotated studies of 58 annotated (Effusion read from the report)
  small / minimal / trace           12 studies   annotated positive   6
  both, e.g. small to moderate       1 studies   annotated positive   1
  moderate or larger                 3 studies   annotated positive   2
  mentioned, no size word            7 studies   annotated positive   6
  effusion denied                    5 studies   annotated positive   0
buckets holding BOTH classes: small (6 of 12), moderate or larger (2 of 3), unqualified (6 of 7)
```

**Seven of the 28 studies contradict the stated rule** — six small effusions annotated 1, one moderate
annotated 0. The identifiers are printed by the tool; they are the evidence, not the rate.

**The denominator.** 28 of 58, because 59.9% of the corpus is not English and this tool refuses to invent
a size vocabulary for seven more languages. So 7/28 = 25% *of the readable English annotations*, and
7/58 = 12% is the floor over all 58. Neither is a rate over the corpus, and nothing here has an interval.

## What it means

**The size qualifier does not separate the gold classes, and it is not even monotone in them.** Small
reads 50% positive, moderate-or-larger 67%, unqualified-mention 86%. On these counts the *presence* of an
effusion sentence carries the signal and the *grade word* carries almost none — the one bucket that is
clean is the denial (0 of 5), and that is also the only one our own reader gets confidently right.

This is consistent with the hosts' stated procedure (topic 735826): the gold and test labels were made by
radiologists **from the images**, not from the reports. The report's adjective and the annotator's
judgement are two different measurements of the same knee, and at the threshold they disagree about half
the time. Two competitors report the same thing from the other direction — the same English template and
the same effusion sentence annotated both ways.

## What our own reader does with that bucket

| bucket | n | gold positive rate | `report_labels_v2` |
|---|---|---|---|
| effusion denied | 5 | 0.00 | 0.08 |
| small / minimal / trace | 12 | 0.50 | 0.68 |
| mentioned, no size word | 7 | 0.86 | 0.79 |
| moderate or larger | 3 | 0.67 | 0.90 |

The denial is well placed. The small bucket sits at 0.68 against a gold rate of 0.50 — but with n = 12,
`P(≤6 positives | p = 0.68) ≈ 0.13`, so **that difference is not separated from zero** and this table is
not a miscalibration finding. Read it as the shape it is: our reader is confident in the direction the
words point, and the words are a coin flip there.

## What this retires, and what it licenses

**It retires "read the reports better for Effusion".** A better extractor recovers wording, and the
wording is what disagrees with the truth. No reader improvement crosses this — the ceiling is in the
map from adjective to annotation, not in the parse. That kills a lane cheaply, which is the value here.

**It licenses one narrow label edit, at one bucket.** A qualifier-boundary case should carry a target
near 0.5 by construction and a confident 0/1 away from it, which is a change to gradient emphasis rather
than to label ordering: all smalls move together, so no ordering inside the table changes and
`supervision_gap`'s AUC cannot move. The claim would have to be tested on a board or a holdout, not on
the gold 58, and it is Effusion-shaped — one of twelve findings — so nothing here is priced near the
1.5% bar. It is a free rider on a label arm, never an arm of its own.

**It does not touch the modelling-headroom reading.** Our model is 0.027 below its own teacher
(`2026-08-25_forum-bottleneck-vs-our-own-diagnostic.md`); this says the teacher itself is noisy at one
finding's boundary. Both can be true, and the first still dominates the queue.
