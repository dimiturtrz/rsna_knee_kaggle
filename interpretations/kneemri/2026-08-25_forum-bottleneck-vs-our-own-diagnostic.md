# The forum says labels are the bottleneck. Our own diagnostic says ours are not.

2026-08-25, read live from the competition forum (topic 735826, "What is the real bottleneck now:
labels, image pipeline, or model capacity?", 7 votes / 9 replies, plus the surrounding threads).

## What the field is saying

Four claims recur across the strongest replies, and they agree with each other:

1. **A single DINO-based model already reaches ~0.91–0.92** on the board. Consistent with knee-uj15 and
   with our own arithmetic: our joint holdout of 0.8690 maps to the board's 0.938 under `board_transfer`,
   and a solo member at ~0.858 maps to about 0.914.
2. **Encoder size does not improve CV.** Stated flatly by the thread opener and demonstrated by one
   participant who swapped a ResNet34 for DINOv2-small with diagnosis-specific slot attention on a frozen
   CPU cache and got *worse* OOF BCE (0.674 against 0.664).
3. **The gold and test labels were made by radiologists from IMAGES, not from reports.** Attributed to
   the hosts. It is the single most load-bearing fact in the thread.
4. **The report→label map is genuinely noisy at the decision boundary.** Two reports using the same
   English template and the same effusion sentence carry opposite labels; at least one "moderate
   effusion" is labelled 0, so it is not a threshold problem that a better rule would fix. One reply
   draws the right conclusion about LLM confidence: an extractor is confident in *applying the heuristic*
   to the words, not in the diagnosis, and the noise lives in the process that chose the word.

## The diagnostic they propose, run on us, points the other way

One reply gives a clean test, and it is the one worth stealing:

> *Check how your label extractor performs relative to the 58 provided labels. Then check how your image
> model performs on them. If your image model is not beating the label extractor "teacher", there is
> likely modeling headroom. If your image model is beating the label extractor by a large margin, the
> quality of the labels might be holding the image model back.*

`kneemri.evaluation.supervision_gap` already reports both numbers against the 58 radiologist-annotated
studies:

| | gold macro AUC |
|---|---|
| our labels (`report_labels_v2`) | 0.8658 |
| our model | 0.8392 |

**Our image model sits 0.027 BELOW its own teacher.** By the field's own diagnostic that is the
modeling-headroom branch, not the label-ceiling branch. We are not yet extracting what our supervision
already contains, so buying finer supervision buys capacity we are not using.

Per-finding it is mixed rather than uniform — the model beats the labels on Effusion, Contusion and
Synovitis, and the labels beat the model on Lateral Meniscus, ACL, MCL, Medial Meniscus, PF OA and
Lateral OA. The findings where supervision leads are the focal, small-structure ones.

## What this changes

**`blendlabels` drops in expected value.** It tests whether a five-level BCE target carries less ordering
information into the gradient than an 85-level one. That question is worth answering and the arm stays
armed — but a model that has not yet reached its current teacher is unlikely to be bound by that
teacher's resolution. Read the pre-registered "within ±0.006, the axis closes" band as the *likely*
outcome rather than the default one, and note that the outcome is still useful: it retires every future
"get better labels" proposal that argues from resolution.

**`base224` rises, with the field's evidence cutting against it.** The forum says capacity does not help,
and one participant has a controlled negative on exactly that axis. Our own diagnostic says capacity is
where our gap is. Both can be true: their negative is DINOv2-small against ResNet34 on their pipeline,
which is an architecture swap at similar capacity, not a width increase on a working one. The honest
position is that P(clears 1.5%) stays at the 0.15 knee-j798 priced — the field's evidence lowers it, our
diagnostic raises it, and they roughly cancel — while the seat argument is untouched.

**The boundary is where the label work should go, if label work happens at all.** Not resolution
everywhere — a correct soft value in exactly the region where the report wording is a coin flip against
the image truth. "Small effusion" labelled 1 and "moderate effusion" labelled 0 both exist in the gold
set, so the target for a qualifier-boundary case should be near 0.5 *by construction*, and confidently
0 or 1 away from it. That is a different edit from blending readers, and it is the one the hosts' own
labelling procedure implies. `core/data/labels/severity_boundary.py` is where it would live.

## What this does not say

The 58 gold studies are ~696 cells and no interval is computed for any number here, so the 0.027 between
model and labels is not separated from zero by anything stated. It is a direction, read alongside a field
consensus that points the opposite way, and the reason to trust it at all is that it is measured on our
own artefacts rather than inferred from someone else's pipeline description.

The forum claims are also unaudited. Every number in the "what the field is saying" section is a
participant's self-report during a live competition, and this project has already learned once
(biohub's 0.913) that a claimed replication is unknowable until it is on the board.

## POSTSCRIPT 2026-08-27 — the 0.027 was bounded, and it spans zero

The paragraph above asked for an interval and one now exists: paired over the same 58 studies, the gap
reads **+0.0266 [−0.0082, +0.0610]**. It includes zero, so the aggregate cannot say which half of the
pipeline is weaker, and the headline of this page — that our model sits below its own teacher — is not
supported by the evidence this page is built on. It was a direction; it is not a finding.

What survives is the SPLIT, not the average: four per-finding bands exclude zero — ACL, MCL and Lateral
Meniscus on the deficit side, Effusion on the label side — which is the dilution `Bottleneck`'s docstring
predicted. The full reading is `2026-08-27_the-supervision-gap-is-four-findings-not-an-average.md`, and
`SupervisionGap.separation` now prints the band so this number cannot be quoted bare again.
