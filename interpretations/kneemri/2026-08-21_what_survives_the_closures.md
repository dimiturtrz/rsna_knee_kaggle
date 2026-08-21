# What survives after the closures of 2026-08-21

Successor to `2026-08-20_what_is_left_after_decode_and_data.md`. That note left several axes open. Six
of them closed in the following twenty-four hours, five of them on measurements taken the same day, and
the board is now short enough to write out completely. This note does that, and then names the one arm
the eliminations select — by elimination, which is a weaker warrant than a positive result and is stated
as such.

## What closed, and on what evidence

| axis | bead | how it closed | the measurement |
| --- | --- | --- | --- |
| slot aggregation, incl. the slice-COUNT axis | knee-tqn | `depth_true` fired its own refutation clause | order-aware operator 0.8461, second-worst of four spanning **0.0019**; seat −0.0002 |
| SSL as initialisation | knee-ec0p | vehicle dead, not on its 3D content | masked reconstruction moves block 11 by **21–55%** of its norm where supervision never exceeds 0.0080; both `encoder_adapt` rates land ≈0.69, i.e. **−0.17** |
| resolution / multi-scale | knee-1nn | gated and declined for one arm's cost | `crop80_tissue`: diffuse half confirmed hard, small-structure half absent (Medial Meniscus −0.012, wrong sign); tight slots buy ≈**+0.0016** |
| crop centring | knee-f6u4 | own pre-registered clause fired | between-slot tissue-centroid disagreement **5.7 mm** p50 / 10.7 mm p90, against one encoder feature cell of **12.4 mm** |
| cross-plane attention / reformats | knee-vigl | premise refuted | slot normals p50 **87.3°** apart, 78.2% within 20° of orthogonal; one slot's stack contains the other's centre in **89.7%** of ordered pairs |
| the label axis | knee-3ns4 | not un-priced — **unscoreable** | 140 wrong cells over **57** annotated studies; the one surviving candidate's interval already spans zero unconditioned |

Two of those deserve a second look because they closed a lever that had been priced *above* the bar.

**CoPAS does not transfer, and the reason is about us rather than about them.** Its +0.031 over an
MRNet-shaped baseline decomposes in the authors' own ablation as ROI crop +0.056, cross-plane attention
+0.049, label-correlation matrix +0.013. The two large components are both bought against a full-FOV
576-matrix baseline with no crop and no view alignment. We already crop to 130 mm, and our six slots are
already the orthogonal set a reformat manufactures. We are not the baseline their delta was measured
against. This is the third external component priced high here and then found already present — the
transferable lesson is to **measure our distance from their baseline before pricing their delta.**

**The label axis is blocked by its denominator, not by its mechanism.** The miscalibration knee-bot
measured is real and large — four of five assigned levels sit outside the radiologist-agreement CI, all
too high, worst in the middle (train on 0.68 where agreement is 0.378). But every ruler for a label-side
fix is those same 57 studies, and `calib_span` already spent the obvious remap and lost 0.0020 with its
per-label prediction inverted — because it remaps along GRADE LEVEL while the over-call sits at YES
across 11 of 12 findings, an orthogonal axis. More radiologist truth would unblock this. The competition
does not supply any.

## What is left

Three items, and two of them are the same move.

1. **A trunk that sees the volume before per-slice features exist** (knee-jucq). Every encoder in 107
   runs was pretrained on natural still images, and the depth axis has only ever been handled
   *downstream* of a 2D encoder. knee-tqn's closure is explicit that it does not touch this: an
   order-blind pool and an order-aware pool both operate on features that were computed one slice at a
   time. This is the one CoPAS component our own measurements did not reach.
2. **External data** (knee-ombx) — blocked on an owner decision about whether MRNet's non-commercial
   clause is inherited by winning weights. Not a technical block.
3. **A member from a genuinely different family.** This is not a separate idea; it is items 1 and 2 read
   from the ensemble side. It arrives from two independent directions: knee-jk3's ledger, where the
   champion's entire +0.011 edge over its best single member came from ensembling rather than from any
   arm in a ten-row table of flat axes; and knee-7d1's forward test, pre-registered and still UNRUN
   because the current pool contains no two candidates the binary and ordinal rulers separate by a
   margin either can resolve — its own words, *"a reason to want a member from a genuinely different
   family, not another decode-family knob."*

## The warrant, stated honestly

Selection by elimination is weaker than selection by a positive result, and the failure mode is
obvious: the last axis standing is standing partly because nobody has shot at it yet. Three of the six
closures above were of axes that looked live until they were measured, and there is no reason this one
is different in kind.

What raises it above "the last one left" is that it is the only surviving axis whose mechanism is a
*representational* claim rather than a *routing* one. Every closed axis above changes what the model is
shown or how per-slice features are combined; none changes what a feature can encode. A 2D encoder
cannot represent a quantity that exists only across slices, and so an aggregation-stage null is exactly
what a live depth signal would produce under an encoder that discarded it — which is knee-tqn's own
harness-suspect reading, and it survives its closure.

## The gate that is running

`through_dense` is on the card now, matched to `band_dense` (0.8578) on every flag except one:
`--through-plane` puts slices *i−1, i, i+1* into the encoder's three input channels instead of three
copies of *i*. Identical parameter count, token count and FLOPs — free by construction. It is the cheap
version of the same claim: through-plane information inside one slice's feature vector, at pixel level,
over a three-slice window.

Pre-registered both ways, and both branches license the expensive version rather than only one:

- **ABOVE control by >0.0062** — the per-slice feature vector *was* the bottleneck, and a volume-
  pretrained trunk is the expensive version of a mechanism that has now paid cheaply.
- **BELOW or flat** — a 2D-pretrained colour-channel stem cannot be *asked* to mean depth, because it is
  still a 2D trunk however its inputs are reshaped. That pushes toward a trunk PRETRAINED on volumes
  rather than away from 3D.

Naming that in advance is the honest move and also the uncomfortable one: a gate whose every outcome
points the same way is not much of a gate. What it genuinely prices is the SIZE — a large positive makes
the 3D trunk a follow-on to a live axis, a flat result makes it a from-scratch bet on pretraining alone,
and those differ by more than the arm costs.

The named risk on the expensive version stands and is not small: Med3D pretrains on CT-heavy
segmentation corpora at coarse resolution, and that domain gap may exceed ImageNet's gap to knee MRI.
DINOv3 is the cautionary instance already in our ledger — a *stronger* natural-image encoder scoring
~0.04 *worse* than DINOv2, with the deficit in focal findings. Better pretraining on the wrong domain
made things worse here before.
