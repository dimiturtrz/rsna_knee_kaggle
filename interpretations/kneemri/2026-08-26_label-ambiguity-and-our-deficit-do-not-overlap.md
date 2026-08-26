# Label ambiguity and our deficit do not overlap

*2026-08-26, on CPU, while `blendsix` held the card. Two instruments, no GPU, one closed lane.*

## The question, and where it came from

The owner put it plainly: *"text is text but the true label might not be in the text. we extract it from
there but ... and its probably hiding in the less sure diagnostics."* Then, sharper: *"still the test is
annotated, so we should be able to tell where we lose — is it unk?"*

That is a good question because it is **answerable**, and because the alternative to answering it was
spending a card arm on a label-confidence recipe.

## The structural fact that frames it

`train.csv` carries `StudyInstanceUID, Report, ACL, ... Fracture`. `test.csv` carries
`StudyInstanceUID`. One column.

**The hidden test has no text.** Report text is TRAIN-TIME-ONLY supervision — privileged information, in
the standard sense: a channel present at training and absent at inference. So a label cell is not a fact
we must reproduce. It is a teaching signal whose STRENGTH we choose. That reframing is what makes both
measurements below decision-relevant rather than merely interesting.

## Instrument 1 — where do independent readers disagree?

`core/data/labels/reader_disagreement.py` aligns four independent readings of the same reports
(`report_labels_v2`, `steven_v4blend`, `lixin_gpt56`, `hybrid_v4`) through `LabelBlend` and reports each
finding's reader spread. 4,406 base studies, all three foreign readers aligned 4406/4406, so no coverage
artifact. The figures are comparable to each other and not to any external threshold — the readers do
not share a scale, and the docstring says so.

**Prediction, pre-registered in `knee-1ww2`:** ambiguity concentrates on the small focal findings the
model already loses (`knee-lxi1`). If it is flat, the story is wrong and no arm is licensed.

It is not flat. It is **backwards**.

| half | findings | mean reader spread |
|---|---|---|
| diffuse — model already beats its labels | Synovitis .300, Contusion .240, Baker's .218, Fracture .216, Effusion .208, Medial OA .176 | **0.226** |
| focal — the whole member deficit | Medial Men .208, Lateral OA .191, Lateral Men .180, PF OA .161, ACL .150, MCL .135 | **0.171** |

Readers argue about *how much* effusion or synovitis — a grade on a continuum. They do not argue about
*whether* the ACL is torn. **Text ambiguity is a severity problem, not a detection problem.**

## Instrument 2 — is our loss hiding in the UNK cells?

`kneemri/evaluation/unsure_cost.py` answers the owner's question directly, on the only rows where a
radiologist and our reader both spoke: the 58 annotated studies, 696 cells. A cell the reader marked
`UNK` is written at **0.28**, the mild-negative level of `Supervision.MEASURED_AGREEMENT`.

```
58 annotated studies, 696 cells, 177 marked UNK, taught at 0.28
Synovitis          unsure  41  pos 0.341   decided 17  pos 0.765   taught error +0.061
Baker's            unsure  29  pos 0.034   decided 29  pos 0.379   taught error -0.246
Fracture           unsure  26  pos 0.154   decided 32  pos 0.438   taught error -0.126
Lateral OA         unsure  22  pos 0.000   decided 36  pos 0.306   taught error -0.280
Medial OA          unsure  14  pos 0.000   decided 44  pos 0.341   taught error -0.280
PF OA              unsure  14  pos 0.214   decided 44  pos 0.409   taught error -0.066
Contusion          unsure  10  pos 0.100   decided 48  pos 0.375   taught error -0.180
Lateral Meniscus   unsure   7  pos 0.143   decided 51  pos 0.431   taught error -0.137
MCL                unsure   6  pos 0.000   decided 52  pos 0.173   taught error -0.280
ACL                unsure   4  pos 0.000   decided 54  pos 0.444   taught error -0.280
Medial Meniscus    unsure   2  pos 0.000   decided 56  pos 0.464   taught error -0.280
Effusion           unsure   2  pos 0.000   decided 56  pos 0.625   taught error -0.280
pooled unsure - decided: -0.2806  95% [-0.3539, -0.2042]  SEPARATED
```

An UNK cell is **13.6%** positive. A decided cell is **41.7%**. Separated by a wide margin, bootstrapped
over STUDIES rather than cells because the twelve cells in one study share a report and a radiologist.

**So "the reader could not decide" almost always means "the report did not mention it, and there was
nothing there."** The 0.28 default is not a bias against the image; if anything it slightly OVER-states
those cells. The framing I filed hours earlier — that the default teaches a soft NO where the image may
carry a YES — is refuted by this table, and the refutation is the useful part.

## What the two instruments agree on

Read the UNK counts on the findings we actually lose: **ACL 4 of 58, Medial Meniscus 2, MCL 6, Lateral
Meniscus 7.** The reader is confident on those cells, and `supervision_gap` already reads ACL's report
label at 0.9841 against the radiologist. The label is good.

| lane | where the ambiguity is | where our deficit is |
|---|---|---|
| reader disagreement | diffuse | focal |
| UNK mass | diffuse (Synovitis 41, Baker's 29, Fracture 26) | focal (ACL 4, MedMen 2) |

They do not overlap. Two rulers sharing no denominator — 4,406 studies against 58, cross-reader spread
against radiologist positive rate — and they land in the same place.

**The label axis cannot buy the focal deficit.** No confidence weighting, no reader blend, no relabelling
of ambiguous cells reaches the six findings that carry the whole gap, because on those six the text is
not ambiguous. The model is failing to extract from the pixels a thing the label states clearly.

## What survives

**Synovitis.** 41 of 58 studies unsure, 34.1% of those actually positive, taught at 0.28 — the only
finding with a positive taught error, and the top-disagreement finding on the other instrument too. Real
reader blindness, and both instruments see it. But it sits on our strong half (+0.0508 model over
labels), so it is an honest small patch, never a lever.

**And the lane this leaves open is the one `knee-lxi1` already named.** Every geometric route to the
focal findings is closed (392px at constant field, `crop80_tissue` at 0.238 mm/px, joint selection on
cached probabilities). The label route is now closed too. What remains is **supervision from
image-derived labels** — MRNet's ACL and meniscal tears, made by radiologists FROM THE IMAGES, which is
both the failing findings and the measurement KIND the hidden test uses — and **representation**
(`knee-jk3`, the 0.044 gap at matched train loss).

## The habit this is about

The cost was ~30 minutes of CPU inside a gap where the card was busy anyway, and it retired a P1 that
would have cost a 40-minute arm and possibly a slot. The prediction was written down BEFORE the number
came back, which is the only reason a refutation was available — a gate stated after the fact would have
been read as confirmation of whichever direction appeared. And the second instrument was built because
the owner asked the question in the form that made it measurable: not *"are our labels bad"* but *"the
test is annotated, so where do we lose?"*

Beads: `knee-1ww2` (closed on this), `knee-uahl` (closed on this), `knee-lxi1`, `knee-ombx` (re-priced up).
