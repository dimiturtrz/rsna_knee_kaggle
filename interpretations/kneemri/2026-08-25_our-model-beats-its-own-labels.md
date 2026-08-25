# Our model beats its own labels, and the deficit page was about someone else's model

**2026-08-25.** `holdannot` — the control's recipe with `--hold-annotations`, so the 58 radiologist-read
studies lose their cells and their 3.0 override — was fitted at 19:40Z and then sat unread for an hour,
because nothing in this repo could predict a member over a study list that is not a fold holdout. Writing
that step (`kneemri.evaluation.annotated_readings`, plus a `--records` path through `supervision_gap`)
took forty minutes and no card. It fired the third of the arm's three pre-registered outcomes, which was
the one written as the least likely.

## The measurement

58 annotated studies, all of them, read by a member whose loss never saw a radiologist cell:

```
aggregate: labels 0.8658   model 0.8932
finding                labels    model  model-labels
Contusion              0.7530   0.8961       +0.1430
Synovitis              0.6870   0.7682       +0.0812
Effusion               0.8149   0.8807       +0.0658
Baker's                0.9321   0.9674       +0.0353
Medial OA              0.9085   0.9380       +0.0295
Fracture               0.8701   0.8917       +0.0215
Medial Meniscus        0.9435   0.9579       +0.0144
Lateral OA             0.7892   0.7930       +0.0039
ACL                    0.9865   0.9890       +0.0025
  --- above: the model beats its own labels, so the READER is the weak part ---
PF OA                  0.8861   0.8790       -0.0071
Lateral Meniscus       0.8422   0.8273       -0.0149
MCL                    0.9762   0.9297       -0.0465
```

The published members, measured on the same 58 studies against the same ruler, read **0.8392 against
0.8658** — the model a full 0.027 *below* its teacher, with ACL, MCL and both menisci four-of-four on
the deficit side and the deficit six summing to −0.049 macro.

Ours is **+0.0274 the other way**, with **nine of twelve** on the label side and the four-of-four
structure gone: ACL is +0.0025 and Medial Meniscus +0.0144, both surplus. Two of the four remain
deficit (Lateral Meniscus −0.0149, MCL −0.0465) and PF OA joins them.

## What that retires

`2026-08-25_the-focal-deficit.md` and `knee-487l` price the PUBLISHED baseline's bottleneck and say so in
their own caveat: *"this prices the public baseline's deficit and ASSUMES ours shares its shape."* The
assumption is now tested and it is **false**. The −0.049 macro deficit and the 4-of-4 internal-derangement
structure are properties of a model we did not fit, and neither may be quoted as ours. The two mechanisms
that page eliminated — in-plane resolution and slot coverage — were eliminated honestly and stay
eliminated; what falls is the conclusion those eliminations were pointing at.

`knee-487l` also records the instrument as **UNDELIVERABLE — "Do not re-attempt"** — on the grounds that
covering the 58 out of fold would need four more folds, which are banned. That reasoning was about the
wrong thing. What has to be unseen is not the STUDY, it is the radiologist CELLS, and `--hold-annotations`
withholds exactly those while leaving fold assignment irrelevant. One 20-minute arm and one afternoon's
entrypoint delivered what four folds were thought to cost. The asset was not dead; one route to it was.

## Why memorisation cannot be the explanation

The member trained on the images of about 46 of these 58 studies, with their REPORT labels. So the
obvious objection is that the model is reading its own training answers back.

**The direction of that bias makes the finding stronger, not weaker.** A model that has memorised a
study's report label predicts the report label, and the report label predicts the radiologist at 0.8658.
Perfect memorisation therefore scores the model at exactly the reference — it drives the gap toward
**zero**. The measured gap is +0.0274 *above* the reference, and no amount of reproducing the report
label can put a model above the report label. The surplus is the part the model gets from pixels the
report never described, and memorisation is the null it had to clear.

## What it opens

The field's own consensus in thread 735826 — *"IMO, the entire competition is the label making... those
folks did not use the reports as a starting point"* — is stated as a general truth about the competition
and could not be checked against our model until now. **It checks out for us.** On nine of twelve
findings the report reader, not the network, is the weaker half.

That is bigger than a re-labelled six. `blendlabels` was pre-registered against the six label-bottlenecked
findings the PUBLISHED split named — Baker's, Contusion, Effusion, Fracture, Medial OA, Synovitis — and
landed +0.0070 there against +0.0034 on the mean, which is why the split was called predictive. All six
sit inside our nine. **Our own split adds three more the same lever can reach: Medial Meniscus, Lateral
OA and ACL.** A better reader has a reachable surface half again as large as the one it was measured on.

It does not make the lever big. `blendlabels` is the size of the label mechanism as measured — +0.0034 on
the metric, an order of magnitude under the 1.5% bar — and nothing here changes that number. What changes
is that the mechanism is no longer competing against a story in which our model is the weak part; it is
the only side of the split that is ours to move on nine of twelve findings.

## The three that stay ours to fix

MCL at −0.0465 is the largest single gap in the table and the only one of any size. It is also, awkwardly,
the finding `blendlabels` moved most (**+0.0239\***, its largest single delta), flagged there as
*"a DEFICIT-side structure a better reader is not supposed to be able to touch."* This measurement
confirms MCL is deficit-side for our model too, so the anomaly is not resolved by the change of ruler —
it is sharpened. Lateral Meniscus and PF OA are the other two, both small.

n = 58 and the per-label standard error is 0.04–0.09 (`label_power`), so **no single row above is a
finding.** The claim is the aggregate sign and the nine/three partition, both of which are far outside
what one row's error moves.

## It does not open distillation, and it is worth saying why

"Our model beats its labels on nine of twelve" is, on its face, the textbook precondition for training the
student toward the teacher's own outputs, and the machinery is already here (`--distill-weight` +
`--teacher-targets`, gated so neither half can arrive alone). **That arm has already run and is closed by
a structural argument this measurement does not touch.** `distill_pairs` scored +0.0016 on the mean — well
inside the 0.0062 floor — and, more decisively, landed at **nearest-neighbour distance 0.9159, closer to a
seated member than any seated member sits to its own nearest.** A student fitted toward our own ensemble
converges toward our own ensemble; a rank average cannot pay for an ordering it already holds. Nothing
above changes that, because the twin problem is about where the student ENDS UP, not about whether the
teacher was better than the report.

So the correct reading is not "go distil". It is a redirection, and an uncomfortable one.

## THE RULER: the board scores against ANNOTATIONS, not against reports

Written thirty minutes after the section above, which had concluded from the same table that both
supervision levers were priced and small. **That conclusion was wrong, and it was wrong because it
assumed a fact instead of looking it up.** The fact took one forum read and no card.

The competition's own overview (topic 733343, *Courtesy of Dr. Jacob Kazam*) states it outright:

> *"Models are evaluated on twelve binary labels, each indicating the presence or absence of a specific
> finding in the imaged knee. The labels, and the criteria used by the **annotating radiologists**, are
> summarized below. In each case, ambiguous or borderline findings ("on the fence") were graded as
> **negative to favor specificity**."*
>
> *"**Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK
> radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus
> ground truth.**"*

So the board's ruler is image-read radiologist consensus — **the same KIND of label as our 58, and not
the report-derived table every number in this project is optimised against.** The reports are an input
the competition generously supplies, never the target.

That rearranges everything above:

- **`supervision_gap` is not a side instrument. It is the only direct measurement of the actual
  competition objective this project owns.** Every other number we have — every ledger row, every
  `run_comparison`, the 0.8502 control — is measured against a PROXY.
- **The proxy and the objective disagree by a lot, and we have measured it on identical predictions.**
  The same holdannot weights score **0.8487 against report labels and 0.8932 against annotation truth**.
  That is +0.045 from the ruler alone — same model, same studies, no ensembling, no TTA. It is also a
  documented mechanism for a chunk of the standing board-versus-holdout gap (0.938 against ~0.85), which
  had been carried as unexplained.
- **The label lane is NOT measured small. It is UNMEASURED.** `blendlabels`' +0.0034 is its value on the
  proxy ruler. Its value on the objective was never computed, and the two rulers are +0.045 apart. The
  earlier section's "both levers are priced" claim has to be withdrawn: only one of them was, and on the
  wrong ruler.
- **The nine/three partition gains teeth.** A finding where our model already beats the report reader is
  a finding where our TRAINING TARGET is the thing holding us back on the objective the board actually
  scores. That is nine of twelve.
- **"On the fence → negative, to favor specificity" is a calibration fact with a matching measurement.**
  Our threshold table reads 44.7% of cells positive at cut 0.50 where the adjudicated radiologist calls
  34.5%. Our targets are systematically more generous than the ground truth is defined to be. `Supervision`
  already carries the machinery that addresses exactly this — `calibrated_targets` replaces each reader
  level with the rate the radiologist actually agreed at, fitted on these 58 — and it was built before
  there was a reason to believe the annotation ruler was the one that counts.

**What this does not license is a victory lap.** Nothing here has moved the board. It says where to
measure, not that a measurement will land: n = 58, the annotated set is not a random sample of the corpus
(ACL runs 41% there against ~20% overall, per topic 734106), and an arm scored on 58 studies has a
standard error that swallows most single-finding effects. The claim is about WHICH RULER, and that claim
is documentary rather than statistical.

## The proxy ruler is not NEUTRAL toward label-side arms — it is biased against them

This is the part that costs something, because a lever was already killed on it.

The metric is per-label rank AUC. Report labels and annotation labels *rank the same studies differently* —
that is precisely what `labels 0.8658` measures. So a change that moves a model's ranking toward
annotation truth **mechanically loses AUC against report labels.** The bias is not a suspicion; its size is
bounded by the same +0.045 the two rulers already showed on identical weights. A label-side arm scored on
the proxy is scored on a ruler that pays it for agreeing with the reader it was built to correct.

`calib_span` is the arm this lands on. It measured `--calibrated-targets` on the recipe we ship, scored
**−0.0020 vs `band_span`**, and the ledger's verdict — *"THE CLAUSE DID NOT MISS, IT INVERTED"* — closed the
lever and killed `calib_dense` unlaunched by pre-registration. The calibration replaces each reader level
with the rate the radiologist actually agreed at. **Its entire purpose is to move the target away from the
report and toward the annotation.** Scoring it against report-derived holdout labels is scoring it on the
thing it deliberately walked away from. A small negative is what a correct move in that direction is
*supposed* to look like on that ruler.

**This does not revive the lever.** It says the arm never measured what it claimed to. Two things stand
against a simple "it was right all along":

- `knee-p9oc` found the remap's constants are POOLED across findings that do not agree (the 0.68 level is
  assigned 0.378 while the twelve findings run 0.000 to 0.778). That is a defect in the estimate, not in
  the ruler, and the ruler change does not repair it.
- The per-label pattern is too structured for ruler bias alone. Ruler bias should spread roughly across
  labels; `calib_span` posted Effusion −0.0212\* against PF OA +0.0170\*. Something label-specific is in
  there. Ruler bias is a *partial* alternative account of the mean, not a replacement for the mechanism.

**The decisive test is one arm and no ambiguity**: refit `calib_span` with `--hold-annotations` added and
nothing else, then read it on BOTH rulers — proxy holdout AUC and `supervision_gap` over the 58. The
pre-registration writes itself, because the two rulers now make opposite predictions for the same arm:
if the calibration is real, the proxy number stays negative or flat *and the annotation number rises*. If
both fall, the lever is dead on the objective too and stays dead. It is the first arm this project can run
where a NEGATIVE proxy result would be evidence FOR the change, and that is only sayable because the ruler
was looked up. Cost: one 20–40 minute card slot, no submission slot. Pre-registered as **knee-bs4l**, with
`holdannot_s2026` as the control rather than `native224` — both arms have to hold the annotations for the
comparison to be single-variable. The standing rule is **knee-9198**.

## Re-reading `merged_corpus` on OUR partition — the split gets SHARPER, and it costs no card

`merged_corpus` is the other arm this touches, and it touches it twice.

Its headline reading is *"the six DEFICIT findings mean +0.0061, the six SURPLUS findings mean −0.0022 …
the first mechanism on the board to move the deficit set AS A SET."* That partition came from
`2026-08-25_the-focal-deficit.md` — **the published model's deficit shape, which the table above shows is
not ours.** So the arm's strongest claim was read on the wrong six, and re-reading it is arithmetic on
numbers already in the ledger.

Our partition is three model-side (PF OA, Lateral Meniscus, MCL) against nine label-side. Re-splitting
`merged_corpus`'s recorded per-label deltas on it:

| partition | findings | deltas | mean |
|---|---|---|---|
| **ours to fix** (model-side, 3) | Lateral Meniscus, MCL, PF OA | +0.0196\*, +0.0062, −0.0044 | **+0.0071** |
| **the reader's** (label-side, 9) | ACL, Lateral OA, Medial Meniscus, Synovitis, Contusion, Medial OA, Effusion, Baker's, Fracture | +0.0161, +0.0016, −0.0024, −0.0206\*, +0.0035, +0.0023, +0.0017, +0.0004, −0.0005 | **+0.0002** |

**The separation survives the change of partition and widens** — +0.0071 against +0.0002, where the old
split read +0.0061 against −0.0022. The arm's claim was right for a reason it had not established.

Two honest cuts on that. It is **three findings**, and Lateral Meniscus +0.0196\* carries almost all of it,
so this is one starred row wearing a partition; n = 3 is not a set. And **ACL +0.0161 sits on the LABEL
side for us**, where Rijeka supervises it directly — that delta is supervision arriving, not the encoder
transferring, and it should not be counted toward the transfer claim.

The second thing it touches is the ruler. **Rijeka's ACL column is expert double-blind IMAGE reads.** Of
every supervision source in this project, exactly two are image-read rather than text-mined: the 58
annotations and Rijeka. The 58 were priced at nothing — but they are 696 cells at ~0.66% of training
weight, which is a fact about their SIZE, not about their kind. Rijeka is 736 studies, +21% of corpus, and
it moved the number. **The one axis that has ever moved this project's deficit findings is also the one
axis whose labels are the same KIND as the board's ruler.** That is convergent with `knee-ombx`'s MRNet
argument rather than new evidence for it — and it sharpens the argument's shape: MRNet is to be argued on
KIND, and the ruler finding says what "kind" is worth, which is the difference between the two rulers.

## Consequence for how arms are read

Any label-side arm from here should report BOTH numbers: its holdout AUC on the proxy, and its
`supervision_gap` aggregate against the 58 annotation cells. They are +0.045 apart and they are not
interchangeable. Running the second one requires the arm to be fitted with `--hold-annotations`, which
`holdannot` has now shown costs −0.0015 — nothing. **The honest ruler is available for the price of a
flag, and it has not been used.**

## What this does NOT say

It does not say our model is good. 0.8932 against a radiologist on 58 studies is not the board and not the
909-study holdout; the same member scores 0.8487 against report labels, −0.0015 under its control
`native224`, which is a separate and duller fact (**and a useful one: withholding the 58 annotated studies
and their 3.0 override costs nothing measurable, so the annotation upweight is not what carries the
corpus**).

It does not say the label lane is worth a slot on this evidence. It says the lane is OURS, which is the
question `holdannot` was fitted to answer and the one the deficit page had answered backwards.
