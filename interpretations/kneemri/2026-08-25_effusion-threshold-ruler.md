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
the gold 58.

**The corpus denominator is what settles its size, and it settles it downward.** The bucket is not 12
studies; over the whole corpus the same rule reads 4407 studies, 1769 of them English (40.1%), and the
small bucket holds **509 — 11.5% of the corpus**. So the edit would touch 509 training rows, on ONE of
twelve findings, with a target moving by at most 0.18. Even a perfect repair of Effusion's boundary
supervision reaches the macro through a twelfth of it. Nothing here is priced near the 1.5% bar: it is a
free rider on a label arm, never an arm of its own.

**And the axis cannot be widened to the other findings on our evidence.** Grade words exist for OA
(mild / moderate / severe), for meniscal tears (partial / complete) and elsewhere, so the same question
is askable there — but it is answerable only against image-derived truth, and that truth is 58 studies
of which 28 parse. Per-finding those counts are single digits. The honest position is that this is
measured for Effusion and unmeasurable for the rest, not that it generalises.

**It does not touch the modelling-headroom reading.** Our model is 0.027 below its own teacher
(`2026-08-25_forum-bottleneck-vs-our-own-diagnostic.md`); this says the teacher itself is noisy at one
finding's boundary. Both can be true, and the first still dominates the queue.

---

## Postscript 2026-08-26 — the per-finding spread this leaned on is chance at the level it leaned on

This document reads Effusion's own boundary rate against the pooled constant, and the `calib_span` ledger
row was extended with the same argument at corpus scale: at the 0.68 level the constant is 0.378 while the
twelve findings' own rates run 0.000 to 0.778. Both readings share one unexamined step — **a wide range
across findings was taken as evidence that the findings differ.**

They hold 5–27 annotated cells each. Ten such findings, every one of them genuinely drawn from the SAME
pooled constant, produce a wide range and cross 0.5 by sampling alone. `LevelAgreement.heterogeneity`
draws that null: 2,000 corpora at each finding's own cell count, scoring the chi-square dispersion about
the pooled rate and the observed range against the range chance produces.

| level | findings | chi2 / df | p | range vs chance | |
|---|---|---|---|---|---|
| 0.08 | 11 | 8.1 / 10 | 0.615 | 0.105 vs 0.211 | consistent with pooling |
| 0.28 | 12 | 26.1 / 11 | **0.021** | 0.341 vs 0.429 | beyond pooling |
| 0.68 | 12 | 18.0 / 11 | 0.100 | 0.778 vs 0.833 | consistent with pooling |
| 0.82 | 12 | 22.9 / 11 | **0.024** | 0.800 vs 0.750 | beyond pooling |
| 0.94 | 11 | 20.3 / 10 | 0.135 | 0.400 vs 0.286 | consistent with pooling |

**0.68 is the level this document is about, and it does not survive.** It carries the 6,461 mild cells, it
holds Effusion's 0.481 against 0.378, and its headline range is *smaller* than what one constant produces.
Heterogeneity survives at 0.28 and at 0.82 — and 0.82 is separately the one level whose pooled constant
fails split-half at 2.0σ, and where four of the five sign inversions sit.

**What still stands.** Everything measured here about the READER is untouched: a small-qualifier effusion
is annotated positive 6 of 12, an unqualified mention 6 of 7, denials 0 of 5, moderate-or-larger 2 of 3.
That is a statement about the map from adjective to annotation and it needed no pooling comparison. The
document's own honest-size paragraph — 509 rows, one of twelve findings, a target moving by at most 0.18,
nothing near the 1.5% bar — was right, and this postscript only removes the corpus-scale corroboration it
was later given.

**What does not.** Do not quote the per-finding spread at 0.68 as evidence that pooling is wrong. The
narrow claim that survives is one level, 0.82, with Contusion (24 cells, 0.458 against 0.688) as the only
thick row — and two of five levels at p ≈ 0.02 fail Bonferroni at α 0.01 individually, so even that rests
on one uncorrected test. The joint count does survive (expected 0.25 tests below 0.05, observed 2,
p ≈ 0.023), which is why the statement is "structure exists somewhere" rather than "the constants are
fine".

---

## Postscript 2026-08-27 — this explains `reader_agreement`'s largest single signal, and thereby spends it

`core.data.labels.reader_agreement` run 2026-08-27 06:42Z reports, over the same 684 cells:

```
 graded   266 positive calls   agrees 63.2% against 77.8% plain: -0.146 [-0.277, +0.005]
 hedged    34 positive calls   agrees 70.6% against 65.3% plain: +0.052 [-0.163, +0.202]
```

**−0.146 is the largest effect anywhere in that census** — larger than the English/non-English gap
(+0.025), larger than any per-finding deviation — and its interval only just touches zero. Read alone it
reads as a lever with a name attached: the reader over-calls on graded reports, so teach it that a grade
word is not a full YES. That reading is wrong, and this document is why.

**The graded penalty is not the reader being wrong; it is the RULER disagreeing with itself on exactly
those cells.** A graded report is, by construction, one carrying a size or severity word — which is the
same population this document buckets. Six of twelve small-qualifier effusions are annotated POSITIVE and
one of three moderate-or-larger is annotated NEGATIVE, because the gold was made from the IMAGES and not
from the reports (hosts, topic 735826). So on graded cells the annotation is near a coin flip, and any
reader — ours, a Qwen-class replacement, a radiologist reading only the text — scores ~50% there no
matter how well it parses. The 14.6-point penalty is the price of being measured against a ruler that
splits, not the price of a parse.

**The consequence is a subtraction from `knee-pju4`'s remaining case.** That bead's sizing argument was
already killed by the silence census (a perfect silence-filler fixes at most 24 cells while over-calling
costs 108), and what survived was "a local multilingual reader is still the instrument that could fix
over-calling". This says a meaningful share of that 108 is not fixable by any reader, because the target
on those cells is genuinely ~0.5. **The two directions the census names are therefore not symmetric in
what they license**: the misses are a coverage problem a better reader could address, while the
over-calls are part reader error and part irreducible boundary ambiguity, in an unmeasured proportion.
Neither this document nor the census separates them, and 684 cells over 57 studies cannot.

**What is NOT retracted.** The graded flag remains the right thing to record — knowing which cells sit on
an ambiguous boundary is exactly what a soft target would need. This document's own honest-size paragraph
still binds: 509 rows, one of twelve findings, a target moving by at most 0.18, nothing near the 1.5%
bar. It stays a free rider on a label arm, never an arm of its own. The addition here is only that its
strongest-looking corroboration from the reader side is the same fact seen twice, not a second one.
