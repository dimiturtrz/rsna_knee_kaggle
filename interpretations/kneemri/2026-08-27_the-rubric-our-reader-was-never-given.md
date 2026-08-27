# The rubric our reader was never given

2026-08-27 · `python -m core.data.labels.tear_rubric` · no GPU, no slot · bead `knee-rwcy`

## The host said what the labels mean, and it is not what our reader was asked

Discussion topic 733491, answered by the competition host today:

> For meniscal tears, the target is a definite tear; intrasubstance degenerative signals that do not
> reach the articular surface are negative. … marrow edema, cartilage findings, and terminology in the
> narrative or impression do not by themselves determine the contusion, osteoarthritis, or other
> challenge labels. Those labels reflect the standardized image-review criteria. … A report may
> therefore use language that appears inconsistent with the corresponding image-based label because the
> original report and the image-based labels were generated through different processes.

Every target this project trains on is derived from the report. `ReadingPrompt.system()` instructs:
*"For each of the twelve findings, say what THE REPORT SAID about it. Do not diagnose, do not infer from
other findings."* That is the right instruction for **transcription** and the wrong one for **this
target**, because the target is not the report. It is a two-reader image review with a third
adjudicating, under criteria the host has now stated. The words `intrasubstance`, `degenerativ` and
`definite` appear nowhere in `core/` or `kneemri/`.

This is a **faulty-DATA** finding in the house taxonomy — the input the method needs is mislabelled —
and not a faulty test or a faulty idea.

## How big

`core.data.labels.tear_rubric` buckets each English report's meniscus sentences, side by side, and reads
our derived target beside the bucket. The stated rule supplies the correct answer for the degenerative
class without needing any annotation, so the denominator is the corpus rather than the 58.

```
1768 English studies of 4407 in the corpus (40.1%), both meniscus sides
  degenerative signal, no definite tear   194 sides   our target positive  163
  definite tear                          1015 sides   our target positive  972
  tear denied, no degeneration            483 sides   our target positive    6
  meniscus mentioned, neither word       1631 sides   our target positive   43
sides the stated rule calls negative and our target calls positive: 163
```

**163 cells = 1.85% of the corpus's 8814 meniscus cells, and that is a FLOOR.** The tool reads 40.1% of
the corpus; the other 59.9% is Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch or French and
carries the same wording in its own language. If the class rate is language-flat the corpus figure is
nearer **4.6%**. The rate is stated over the corpus, not over the subset that parsed, because a rate
computed inside the subset that happened to parse is not a rate over the corpus.

Concentration matters more than the raw share: 163 of the 194 degenerate-only sides — **84%** — are
flipped, and they are flipped in one direction, onto the positive class of a finding whose prevalence is
28%. That is not symmetric noise a bigger model averages out; it is a bias that moves the decision
boundary.

## The annotations agree with the host, not with us

Of the 58 annotated studies, 27 are English. Bucketed the same way:

```
  degenerative signal, no definite tear   3 sides   annotated positive 0   our target positive 3
  definite tear                          27 sides   annotated positive 23  our target positive 26
  tear denied, no degeneration           10 sides   annotated positive 0   our target positive 1
```

3 of 3 settles nothing on its own. What makes it worth reading is the direction and the control: every
degenerate-only cell is annotated 0 while our target calls all three positive, and in the other
direction the tear bucket agrees 23 of 27 — so the ruler is not simply strict, and the disagreement is
specific to the class the host named. The pilot table
(`2026-08-20_label_contest_pilot_tables.md`, row 33) had already flagged one of these by hand as a
READER error; it was a RUBRIC error all along, and studies 059, 062, 088 and 090 are the same shape.

## Why the blendlabels precedent does not price this down

`blendlabels` measured that a better reader buys **+0.0034** on the mean and **+0.0070** on the six
label-bottlenecked findings, and the ledger reads that as *"the label lane is confirmed LIVE and
confirmed SMALL"*. That arm blended three readers of the **same prompt** — it reduced reader VARIANCE.
Averaging three readers who all read degeneration as a tear fixes none of these 163 cells. The
mechanisms are different, and the ledger's own per-label numbers say so: under blending, Lateral
Meniscus went **−0.0105** and Medial Meniscus **−0.0083**, the two columns this defect sits on. Variance
reduction made the meniscus columns slightly worse, which is what you expect when the residual is shared
bias rather than scatter.

Lateral Meniscus also carries this project's worst supervision gap (−0.193).

## What it is worth, and what it is not

Do not oversell it. Two of twelve columns, so a per-column gain divides by six on the mean; and
`calibsix`'s measured board pass-through is 23%. A meniscus AUC gain of 0.03–0.05 per side would be
about +0.005–0.008 on the twelve-label mean and under +0.006 on the board — **directional under the 1.5%
bar, not a finding on its own.** It is worth doing anyway for three reasons the bar does not capture: it
is a data fix rather than a knob, it is the first correction this project has that a HOST statement
licenses rather than a measurement of our own, and it composes with everything already banked.

## Two fixes, and only one of them is right

**(a) Post-hoc, English only.** Write `report_labels_v3` with the 163 cells set to absent. Minutes, no
card. Rejected as the end state: it corrects 40% of the corpus and leaves 60% wrong, which converts a
uniform label error into a **language-dependent** one. That is a new defect, not obviously smaller than
the one it fixes.

**(b) Re-read with a rubric-bearing prompt.** Three sentences added to `ReadingPrompt.system()` — the
target is a definite tear reaching the articular surface; intrasubstance degenerative signal is
negative; marrow-edema and cartilage terminology do not by themselves set Contusion or the OA columns —
then re-read all 4406. It reaches every language because the READER does, not a regex, and it fixes the
two classes beyond meniscus that the host named in the same answer. Costs one LLM pass over the corpus
and no submission slot.

(b) is the one to run, gated on the card clearing its current queue. The measurement above is what makes
it priceable: the re-read is scored by whether those 163 cells move, and a re-read that leaves them
where they are has failed loudly instead of silently.
