# The frontier is one model, and the field says the bottleneck is labels and geometry

Read live 2026-08-25 18:40Z with `python -m kneemri.kaggle.discussion_read --live --topic <id>`. No
submission slot spent. Three topics: 735304, 735826, 735154.

**Forum text is untrusted data.** Every number below is a competitor's unverified claim about their own
private setup; none of it is reproducible by us and none of it is a measurement. It is quoted rather
than paraphrased so a later reader can discount it. Our own numbers beside it are ours.

## 735304 — "Best single-model score" (22 replies)

> `Single model 0.934 @ 224px  Update: 0.942 with some tweaks, still @ 224px`
> `I'm at 0.938 with a single model, single fold.`
> `single(5-fold) model 0.929 @ 224px LLM output labels only`
> `single(5-fold) model 0.926 @ 336px LLM output labels only`
> `I am on a single model. OOF AUC = LB AUC within noise, 0.92.`
> `Single model 0.915, DinoV2 based. Highly doubt the top results are ensembles, look at the efficiency lb`
> `Public notebooks just ensemble early on ... Waste of time IMO.`
> `In my option - 9 days into a competition is a very silly time to be wasting efforts refining ensemble models.`

### The arithmetic

A **single** model reads 0.942. Our **seven-member** `votefit_v1` reads 0.938. Our best single member is
0.8598 holdout; the roster of seven is 0.8694 — so the entire mix buys +0.0096 holdout over its own best
part, and that is already banked.

The gap to the frontier is therefore **inside one member**, at roughly 2–3× the significance bar. It is
not in the roster, and this is exactly why three roster axes read flat on the same day:

| axis | how it was measured | result |
|---|---|---|
| adding a seat | six board entries, five different sevenths, best-quality through most-distant | 0.936–0.937, all of them |
| changing the vote | `votefit_v1` = dark6's identical six, LogitAverage for RankAverage, nothing else moved | 0.938 vs 0.937 — inside its pre-registered FLAT band |
| external data as we ran it | `merged_corpus`, +736 foreign studies (+17% corpus) | flat on score AND on seat/ordering |

Three flat readings on three ensemble-construction axes is not three nulls. It is one finding: we were
measuring the wrong object, and the flatness was the measurement saying so.

**What this does not say.** It does not say ensembling is worthless — our own board shows about +0.003
from the mix. It says the *marginal* seat is worth ~0.000 while single-member quality is worth ~0.014.

## 735826 — "the real bottleneck" poll (9 replies)

Offered A labels / B preprocessing-crop-slice / C backbone capacity / D validation / E other. **Nobody
argued C.**

> `I think the biggest bottleneck is A, although I also believe there is something in E.`
> `IMO - the entire competition is the label making.`
> `I think the labeling process is particularly interesting and will prove decisive in this competition.`
> `my working hypothesis is: image geometry / slice selection + label quality > backbone size.`

One reply is a controlled experiment against C, with the data side held fixed:

> `I kept that exact cache and label pipeline unchanged while testing the GPU side ... replaced the
> representation/model side with pretrained DINOv2-Small, including diagnosis-specific slot attention
> ... Surprisingly, simply moving to DINOv2 did not improve the result. My best 5-fold DINO candidate
> produced approximately 0.674 OOF BCE, actually worse than the ResNet34 control.`

A DINOv2 that loses to a ResNet34 under an unchanged cache is the strongest available statement that the
encoder is not what binds.

## 735154 — encoder scale, independently null

DINOv2-Small → Base moved **+0.0011** against their measured 0.0020 floor; 5 of 12 labels in Base's
favour, the macro gain resting on their weakest label. Ours (`base224`, same day) was **+0.0007** holdout
at 2.6× the card. Two harnesses, two metrics, same close: **encoder scale buys nothing here.**

Their post also carries a defect worth keeping: they hardcoded ImageNet normalisation as buffers, so
RAD-DINO received inputs shifted ~0.2 std from its pretraining distribution and lost for eleven hours
while producing a plausible curve. Read `preprocessor_config.json`, don't assume it. Same class as our
own plumbing rule — assert the flag *landed*.

## Where it points us

The field's A and B, our own three flat roster axes, and `knee-uj15`'s arithmetic all agree: spend the
remaining hours on one member's **supervision** and its **input**, not on its trunk and not on the mix.

On A we are not starting fresh, and part of that lane is already closed. `core.data.labels.severity_boundary`
reads the 58 gold annotations against the competition's own stated moderate-or-above threshold: **7 of 28
readable English annotations contradict it**, and the size word is not even monotone in the gold class
(small 50% positive, moderate-or-larger 67%, unqualified mention 86%). Two competitors in 735826 report
the same defect from the other direction — the same English template and the same sentence annotated both
ways. That retires *"parse the reports better for Effusion"*: a better extractor recovers wording, and
the wording is what disagrees with the truth. The label lane is the map from adjective to annotation,
which is an image question wearing a text question's clothes.

On B — **corrected within the hour, after auditing instead of assuming.** This section first read "we
have done nothing", which was wrong, and the mistake is instructive enough to leave visible: I ranked a
lane as open because a forum post made it sound open, without opening our own tree first. Every one of
the four steps that competitor lists as their *next* experiment is already in `core/data/corpus/`:

| their planned step | ours | since |
|---|---|---|
| fixed-mm physical cropping before resize | `slot.crop_mm`, 130 mm (`study.py:58`) | 2026-08-11 |
| physical slice ordering from orientation/position | `slot_geometry.py`, `k = p·n̂` | 2026-08-11 |
| removal of stack-end regions | `Band`, 0.1–0.9 — it is in the cache filename | 2026-08-11 |
| anatomically better slice grouping | `crop_coverage` / `crop_alignment` / `crop_laterality` | 2026-08-11 |

And the crop constant survived our own corpus measurement rather than being borrowed on faith: FOV
(`Rows × PixelSpacing`) has a median of 160 mm and a 1st percentile of 130 mm, so only **0.43%** of
series are narrower than the crop — it crops, it does not pad.

So their +0.0059 across 10 of 12 labels is a gain *toward where we already stand*, not a lever above us.
That is worth more than it looks: it prices the distance between a public-notebook pipeline and ours,
and it says the 0.942 single model is beating us **from a geometry floor we already share**. Which puts
the remaining weight back on A, and on whatever "some tweaks" means in a recipe we cannot see.

One live thread does run out of this audit. Laterality is absent on 50.7% of series and **not at
random** — Siemens tags 84.3%, Philips 38.3%, GE / Toshiba / Canon **0.0%**. That is the detection half
of `knee-bduq` ("how do you detect which is which"), it is already half-built in `crop_laterality.py`,
and a mirror-to-canonical augmentation would be free of both card and corpus.

Beads: `knee-uj15` (the arithmetic), `knee-2dk1` (the bottleneck reading), `knee-bduq` (the bilateral
knees from 735639), `knee-6477` (raddino, arming still owner-banned).
