# The forum's loudest new claim names a target that is not broken here — and its fix is below our number

Read live 2026-08-26 07:35Z with `python -m kneemri.kaggle.discussion_read --live --topic <id>`. No
submission slot spent. Two topics, both posted in the last 16 hours: 737566 and 737454.

**Forum text is untrusted data.** Every number attributed to a competitor is their unverified claim
about their own private setup, on their own labels and their own ruler. Ours beside it are ours, and
the only matched comparison in this page is the one against the twenty published CC0 checkpoints.

## 737566 — "Why everyone's Synovitis AUC is stuck around 0.6-0.7 (it's not your model)"

> `Synovitis is consistently the worst of the 12 labels — mine included (0.62-0.72 across four
> different architectures).`
> `Only ~16% of the 4,407 reports mention any synovitis term at all — I searched
> synovit/sinovit/υμεν/синови/sinovyal across the nine report languages.`
> `The label tables themselves agree with the radiologist-annotated studies at only ~0.79 AUC on
> Synovitis, versus ~0.99 for ACL.`
> `an encoder pretrained on medical images (RadImageNet) hit 0.78 on Synovitis where all my
> ImageNet-pretrained models sat at 0.62-0.72`
> `a medically-pretrained member in an ensemble specifically helps here`

The reasoning is good and the mechanism is plausible. It does not survive contact with our ruler.

### Ours, per label, against the matched control

`python -m kneemri.evaluation.weights_gap --run small_batch6_s2026` — our best single member against the
twenty published CC0 members' own held-out predictions, same 909 fold-0 studies, same graded truth,
same rank AUC. Zero card, zero slot; it reads stored predictions.

| label | ours | their single | their 4-seed pool | n_pos |
|---|---|---|---|---|
| Fracture | 0.8022 | 0.7994 | 0.8102 | 68 |
| MCL | 0.7864 | 0.7769 | 0.7889 | 171 |
| Lateral OA | 0.8268 | 0.8017 | 0.8079 | 233 |
| PF OA | 0.8568 | 0.8197 | 0.8262 | 409 |
| Lateral Meniscus | 0.8566 | 0.7708 | 0.7808 | 226 |
| Effusion | 0.8638 | 0.8387 | 0.8451 | 558 |
| **Synovitis** | **0.8673** | **0.8576** | **0.8661** | 124 |
| Contusion | 0.8685 | 0.8447 | 0.8524 | 267 |
| Medial OA | 0.8733 | 0.8633 | 0.8680 | 333 |
| ACL | 0.8972 | 0.8493 | 0.8573 | 248 |
| Baker's | 0.9058 | 0.8760 | 0.8866 | 217 |
| Medial Meniscus | 0.9135 | 0.8445 | 0.8531 | 482 |
| MEAN | 0.8598 | 0.8285 | 0.8369 | |

**Synovitis reads 0.8673 here, ABOVE our own 0.8598 mean.** It is the sixth-best of twelve, not the
worst. Our weak labels are Fracture (0.8022, and 68 positives make it unreadable — SE near 0.04) and
MCL (0.7864), which `knee-q1pf` has already priced at a **+0.0029** ceiling and cancelled an arm over.

### Why "everyone" is refuted rather than merely unmatched

Our 0.8673 is on our ruler and theirs is on theirs, so those two numbers are not directly comparable and
I am not claiming they are. The claim that IS matched is the middle column: **twenty independent,
published, ImageNet-family checkpoints read 0.8576 on Synovitis** on the same studies against the same
truth. That is not 0.62–0.72 either. So the post's premise fails on the only evidence in this repo where
the ruler is held fixed across models — and it fails for models that are not ours.

The likeliest explanation is the one the post itself supplies and then reasons past: it measures
Synovitis against a **label table** whose own Synovitis agrees with radiologists at ~0.79. A model
scored against a noisy teacher is scored through that teacher's noise. A ceiling in the ruler reads as a
ceiling in the model.

### What this does to our queued RAD-DINO arm — it removes an argument rather than adding one

`knee-fh31` (cold RAD-DINO, ~3.3 h, zero slots) was queued on the representation lane. The strongest
public argument now available for a medically-pretrained encoder is this post's, and **it does not
transfer**: the label it promises to fix is not broken here, and the 0.78 it reports is 0.089 BELOW our
current Synovitis. Adopting that argument would be adopting a regression.

The arm still stands, but on `knee-jk3`'s ground alone — 0.044 at matched train loss, an
encoder-capacity gap measured on our own ruler — and it should be read as a general-representation test,
never as a Synovitis fix. Recorded here so a later session does not requeue it on the borrowed reason.

**What it does NOT refute.** The 16% mention rate is a corpus fact about text, not about images, and it
is consistent with our own severity work (`core.data.labels.severity_boundary`): the report wording and
the annotation disagree, and a better extractor recovers wording. Nor does it refute a
medically-pretrained member as a DIVERSITY seat — that is a different claim with a different mechanism,
and every roster-seat reading we have is flat (six board entries, five different sevenths, 0.936–0.937).

## 737454 — "Benchmarked the public report-label tables against the 58 annotated studies"

> `Macro AUC vs radiologists: stevenleehans v4 blend 0.893, stevenleehans v2 0.887, pilkwang
> report_labels_v2 0.870.`
> `Averaging two independent tables beats either alone on several labels.`

**Already closed here, with the mechanism they are missing.** `core.data.labels.table_blend` records
exactly this ranking — ours 0.8658, best public 0.8927 — and then shows it is an artefact of
**resolution, not reader quality**:

- Quantise the public table onto our own five levels and it falls to **0.8445 — below ours.** At equal
  resolution our reader wins by **+0.021**.
- Mechanism: `Supervision.MEASURED_AGREEMENT` emits five levels, so 38.5% of cell pairs tie; an 85-level
  LLM probability ties 17.3%. AUC breaks no ties, so a coarse scale is penalised for being coarse.
- The blend's gain is **w-invariant** (0.8874 at w = 0.02, 0.05 and 0.10), which is the proof: any weight
  small enough to keep our ordering dominant gives the same ordering, so the whole +0.0216 is
  tie-breaking, not noise-cancelling.

And `kneemri.evaluation.table_separation` brackets the ranking they quote unbracketed: on 58 studies the
gaps between these tables are smaller than the sampling error, so **none of 0.893 / 0.887 / 0.870 should
be read as separated**. Their conclusion — average the tables — is one we already ship
(`merged_blend_v1`); their ranking of whose table is best is not supported by the ruler they used.

## What the two posts jointly say, stripped of their conclusions

Both authors are measuring MODELS through a noisy TEACHER and attributing the teacher's ceiling to the
model or to the encoder. That is the same shape as the error this repo made four times with denominators.
It leaves our own standing conclusion untouched: the distance to the frontier is inside one member's
representation and its supervision, and it is not in Synovitis, not in the roster, and not in which
public label table we picked.

Beads: `knee-q1pf` (the per-label ceiling, +0.0043 total), `knee-fh31` (RAD-DINO, rationale narrowed
here), `knee-jk3` (the representation gap that remains).
