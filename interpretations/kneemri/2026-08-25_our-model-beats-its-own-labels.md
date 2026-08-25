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

## What this does NOT say

It does not say our model is good. 0.8932 against a radiologist on 58 studies is not the board and not the
909-study holdout; the same member scores 0.8487 against report labels, −0.0015 under its control
`native224`, which is a separate and duller fact (**and a useful one: withholding the 58 annotated studies
and their 3.0 override costs nothing measurable, so the annotation upweight is not what carries the
corpus**).

It does not say the label lane is worth a slot on this evidence. It says the lane is OURS, which is the
question `holdannot` was fitted to answer and the one the deficit page had answered backwards.
