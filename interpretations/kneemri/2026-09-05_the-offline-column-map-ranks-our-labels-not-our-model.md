# Our per-column offline AUC ranks our LABELS, not our model — and the board headroom is in the ten columns we have never read

**Date:** 2026-09-05 · **Card spent:** none · **Slots spent:** none (reads two that were already bought) · **Board read:** 2026-09-05T11:08:57Z

`colnull` fixed the competition metric as a plain unweighted mean of twelve per-column ROC AUCs, which
makes a damaged-column entry a measuring instrument: overwrite one column with a constant 0.5, and the
board drop gives that column's board AUC as `(champion − damaged) × 12 + 0.5`. Two such entries have now
scored against the same 0.944 champion control.

| column | our offline AUC | board AUC | lift |
|---|---:|---:|---:|
| MCL (`colmcl_v1`, ref 56020054, board 0.904) | 0.8104 — our **worst** of twelve | **0.980** | **+0.17** |
| Baker's (`colbaker_v1`, ref 56023244, board 0.903) | 0.9208 — our **best** of twelve | **0.992** | **+0.071** |

**The rounding, stated before the reading.** Board scores carry three decimals, so each is ±0.0005 and a
champion-minus-damaged difference is ±0.001, which the ×12 multiplies to **±0.012 on a column AUC**. The
two column values are therefore 0.980 ± 0.012 and 0.992 ± 0.012: they overlap, and **the ordering between
these two columns is NOT resolved.** Everything below rests on the part that survives that interval —
both columns sit at or above ~0.97 on the board, and both sit far above their offline values.

## What the two points say

**1. The level does not transfer, and this is now measured twice rather than argued once.** Our
champion pool averages 0.8715 across twelve columns offline; the board averages 0.944. Both externally
read columns land higher on the board than offline, by +0.17 and +0.071.

**2. The lift is not uniform, and that is the whole finding.** A constant offset would be a boring ruler
calibration. Instead the column where we look *worst* lifts by 0.17 and the column where we look *best*
lifts by 0.071. MCL is our weakest column offline and is at 0.980 on the board — the model separates MCL
nearly perfectly against the competition's labels while scoring 0.8104 against ours.

**So the offline column ranking is substantially a ranking of OUR SUPERVISION.** MCL has 50% radiologist
agreement, the second-worst-labelled column this project owns, and `overcall_audit` measures our lexicon
at 108 false positives to 32 misses. A model evaluated against noisy labels is attenuated toward 0.5, and
attenuated most where the noise is worst. That is exactly the shape of these two points.

> **CORRECTION, same day, 13:15Z — the paragraph above is WRONG for MCL, and I am leaving it standing so
> the reasoning that failed is visible.** `python -m kneemri.evaluation.supervision_gap --labels
> report_labels_v2.csv` scores the report label ITSELF as a predictor of the radiologist on the 58
> annotated studies. **MCL's report label reads 0.9762 there — the SECOND BEST of twelve**, behind ACL's
> 0.9865 and ahead of Medial Meniscus 0.9435 and Baker's 0.9321. Our MCL labels are not noisy; they are
> among the cleanest we own. The "50% agreement" I reached for is a THRESHOLD statistic and the wrong one
> for an AUC argument: on the 58, gold calls MCL positive 9 times and v2 calls it 18, so the lexicon
> over-calls MCL two-to-one — but AUC reads the RANKING, and the reader's graded output
> (0.08/0.28/0.68/0.82/0.94) ranks MCL almost perfectly. Over-calling at a cut and ranking badly are
> different defects, and `overcall_audit` measures the first while every number in this document is the
> second. Denominator, checked: MCL's corpus positive rate is 17.4% (767 of 4406) and the gold set's is
> 15.5% (9 of 58), so the gold set is not enriched on this column and the 909-study read is not
> thin-positive noise either.
>
> **What that leaves.** Our model reads MCL at 0.8104 against our own labels on 909 studies, at 0.8798
> against radiologists on the 58, and at 0.980 on the board. The labels are clean, so label noise cannot
> be the attenuator. `supervision_gap`'s own split puts MCL on the MODEL-BOUND side at **−0.0964** (the
> model is worse than the label at predicting the radiologist), one of four per-finding bands that exclude
> zero. So MCL is a column where the signal is IN our supervision and our model fails to pull it out of
> the pixels — and the board nonetheless scores it at 0.980. The only story left that fits all three
> numbers is that the board's MCL column is a DIFFERENT AND EASIER TASK: different case mix, or a
> definitional difference between our lexicon's MCL and the competition's. `Q-RULER5` listed that as the
> alternative it could not separate from label noise. With MCL's labels now measured clean, it is not the
> alternative any more — it is the leading explanation, and label noise is the one that has been refuted
> on this column.
>
> **What survives unchanged.** The headroom arithmetic below (two columns at ~0.98 and ~0.99 leave the
> other ten averaging ~0.936) rests only on the two board reads and is untouched. So is the conclusion
> that the offline map is dead as a planning tool — it is dead for a DIFFERENT reason than this paragraph
> gives: not because it ranks our labels, but because our worst offline column is our second-best-labelled
> one and is already at 0.980 externally, so the map does not point at board headroom by any mechanism.
> Baker's is untouched too: its label reads 0.9321 and its board 0.992, which is the same shape without
> the contradiction.

**3. Therefore the per-label lane is dead as a planning tool, and for a better reason than before.** The
+0.0196 bottom-three headroom in the per-column map is priced on the assumption that a column we score
badly is a column with board headroom. On the one column where that assumption has an external check, it
is wrong by 0.17: MCL is nearly solved on the board. "Fix our weak columns by modelling" aims at label
noise wearing a model's clothes.

## Where the headroom actually is

The board mean is 0.944, so the twelve columns sum to 11.328. Two of them are ~0.980 and ~0.992, which
leaves **the other ten averaging ~0.936** — below both measured columns, and we have no external read on
any of them. That is where the 0.010 gap to first place lives, and nothing offline can point at it,
because the offline map has now failed its only two external checks in LEVEL and cannot be trusted in
ordering either.

**This is a case for spending slots on instruments rather than entries.** A damaged-column entry costs
one slot and returns one column. A damaged-GROUP entry costs one slot and returns the SUM over the group,
which localises faster: knock out six, and a low sum says the weak columns are inside that six. Three
slots spent adaptively would narrow twelve unknown columns to a short list, and every seating and
supervision decision after that would be aimed instead of guessed.

## What must NOT be concluded

- **Not that the board labels are clean.** They may be noisy too; nothing here measures that.
- **Not that case mix is ruled out — and as of the 13:15Z correction it is the LEADING explanation, not
  the alternative.** `Q-RULER5` established that case mix and a definitional mismatch between our
  lexicon's MCL and the competition's make *identical* offline predictions and cannot be separated
  without board labels or board cases. Label noise WAS the explanation with measured support; on MCL it
  has now been refuted by `supervision_gap` (report label 0.9762 against radiologists), so what remains
  is the pair Q-RULER5 could not split. A hidden test with different prevalence would also lift AUC.

## The split this correction hands forward: six columns are model-bound

`supervision_gap` prices, per finding, the report label as a predictor of the radiologist against what a
published member fitted on those labels actually reaches, both on the same 58 studies. The aggregate gap
is **+0.0266 [−0.0082, +0.0610]** and SPANS ZERO — 58 studies cannot say which half of the pipeline is
weaker overall. Four per-finding bands do exclude zero, so the shape is a shape and not twelve findings:

| the READER is the weak part (model beats its own labels) | the MODEL is the weak part (label beats the model) |
|---|---|
| Effusion **+0.1304** · Contusion +0.0891 · Synovitis +0.0508 · Medial OA +0.0419 · Baker's +0.0226 · Fracture +0.0160 | Lateral Meniscus **−0.1925** · ACL **−0.1225** · MCL **−0.0964** · Medial Meniscus −0.0938 · PF OA −0.0856 · Lateral OA −0.0793 |

Bands excluding zero: ACL, MCL, Lateral Meniscus, Effusion.

The consequence for planning is sharper than the column map ever was, because it names a MECHANISM per
column rather than a rank. A better label reader (knee-wko44) can only pay on the left column, and it is
led by Effusion, whose label predicts the radiologist at 0.8149 while the model reaches 0.9453 — the
model is already correcting the reader there. Showing the model more of the image can only pay on the
right, and it is led by Lateral Meniscus at −0.1925, the largest single gap this project has measured:
the label carries the finding at 0.8422 and the model extracts 0.6497 of it. Averaged over twelve, either
change is diluted by the six it cannot touch, which is one reason every whole-pool operation has flattened.

**The caution that has to travel with this table.** It is 58 studies, and the per-finding positive counts
are 9 to 35. Only four bands exclude zero. The table is a direction to aim a mechanism at, never a number
to price an arm with, and the aggregate's own interval spanning zero is the honest headline.
- **Not that paired member comparisons are invalid.** A shared bias plausibly cancels when two members
  are compared on the same holdout, which is the only reason the eight declines and seven board flats
  still stand. That instinct remains **unverified** (knee-l4xx6), and it is weakest exactly where members
  differ in SUPERVISION — the `errors_*` family.
- **Not that the two columns are ordered.** ±0.012 covers the 0.012 between them.

Beads: knee-zsyp4, knee-l4xx6, knee-aily5, knee-wko44.
