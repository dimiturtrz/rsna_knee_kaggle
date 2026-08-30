# Three analysis instruments this project does not use, and what each would have to beat

**2026-08-30 · field synthesis · Mahalanobis distance · KL divergence · gradient attribution (Captum)**

None of these is a training method. They are **instruments** — they change what we can SEE about a model,
not what the model is. That is the correct frame for pricing them: an instrument earns its build cost by
retiring a suspicion or by licensing a whole axis, never by improving a score directly.

Each of the three has an incumbent in this repo. An instrument that arrives without engaging its incumbent
is a rewrite, so each section below names what it would replace and what that incumbent already argues.

---

## 1. Mahalanobis distance — a different SPACE for the seat ruler

### The incumbent

`kneemri/evaluation/member_distance.py` prices member diversity as **per-label Spearman rank correlation,
averaged over the twelve targets**, reported as NEAREST (the twin detector) and MEAN (far-from-everyone).
`record_mix` greedy admission consumes it, and it is the ruler that has decided nearly every seat question
this project has settled — `raptor_distill` was declined by it (admitted 44th of 121), `distill_pairs`
before that.

### What Mahalanobis is

Distance between points measured in units of the data's own covariance:
`d(x, mu) = sqrt((x - mu)^T S^-1 (x - mu))`. Two properties matter here. It is **scale-free** per dimension,
so a target that varies a lot does not dominate. And it is **correlation-aware**: directions in which the
population already varies together are discounted, so a member is only "far" if it is far along an axis the
pool does not already span. That second property is precisely what a diversity ruler is trying to express.

### The specific gap it would close

The current ruler measures distance in **prediction space** — twelve numbers per study. But the quantity
this project has repeatedly measured as the one that transfers to the board is **decode path / pixel
group** (knee-nas5, ~2.4x the board slope; knee-for6's close reason is literally "distillation moves
ORDERING, the seat ruler prices PIXEL GROUP"). Prediction-space rank correlation is a **proxy** for that,
and knee-for6 closed a whole axis on the proxy. Mahalanobis over **embeddings** would measure representation
distance directly and with the pool's covariance divided out.

Concretely it answers a question the incumbent cannot: *is this candidate far because it reads the anatomy
differently, or merely because it is noisier?* Rank correlation cannot separate those; a noisy member and a
genuinely distinct member both decorrelate.

### The honest counter

Estimating `S^-1` needs the covariance of the pool, and with ~121 candidates over an embedding width in the
hundreds the covariance is rank-deficient — it needs shrinkage (Ledoit-Wolf) or a projection, and the choice
of shrinkage is a knob that could quietly fit the answer. Any adoption must show the new ruler reproduces
the seat decisions the board has already validated before it is trusted to overturn one.

---

## 2. KL divergence — a ruler for DISTRIBUTIONS, where we currently only have point scores

### The incumbent

There isn't a single one, which is itself the finding. Distribution questions are currently answered by
several narrow tools: `prevalence_match` (label base rates), `foreign_overlap` (train/holdout study
overlap), `level_calibration` and `--calibrated-targets` (target sharpness), `paired_floor` (noise bands).
Each measures one moment of one distribution.

### What it would buy, in order of plausible value

1. **Member disagreement as divergence, not correlation.** Two members can rank studies identically and
   still be confidently different — one saturated near 0/1, the other hedged. Rank correlation is blind to
   that by construction; symmetric KL (or Jensen-Shannon, which is bounded and a proper metric) sees it.
   This is the same complaint as section 1 from the other side, and it matters because a rank-average
   ensemble discards the confidence a KL term would be reading. Cheap: it consumes prediction arrays we
   already have on disk, no card time.
2. **Distribution shift against the competition test set.** This is the one to be careful about — the
   Kaggle test set is hidden, so no divergence to it is computable, and any claim of the form "our holdout
   matches the test distribution" is unknowable. What IS computable is divergence between our own splits,
   and between our corpus and any external corpus (MRNet, knee-ombx) before it is trusted as extra data.
   That is a real use and it is the honest version of the question.
3. **Teacher/student divergence in the distillation lane.** knee-8009's feature-distillation objective needs
   a matching loss; for score distributions KL is the standard choice and is what the pair-based `imitation`
   term is currently substituting for.

### The honest counter

KL is asymmetric and unbounded, and it blows up wherever the reference assigns near-zero mass — which is
exactly the rare-finding regime (MCL, synovitis) this project's per-label scores are weakest in. Use
Jensen-Shannon or a smoothed KL, and never report a raw KL on a thin label without its support count.

---

## 3. Gradient attribution (Captum) — and the standing argument it has to answer

### The incumbent, which is not a gap but a position

`kneemri/evaluation/attribution.py` already does attribution, by **occlusion**, and its module docstring
argues explicitly against the gradient/saliency family:

> Nothing in the supervision ever says WHERE a finding is, so a saliency map has no reason to look right
> and cannot be checked against anything. An occlusion is different: it produces a NUMBER on the held-out
> score... Saliency is deliberately not here. The sibling cardiac project keeps captum behind an optional
> extra for the same reason: the always-on diagnostic has to be the one that yields a measurement.

That argument is correct and should not be discarded. Three occlusion diagnostics exist and each returns a
delta: slot importance, border-vs-centre, depth profile.

### Where Captum would nonetheless add something

The occlusion diagnostics are **coarse by design** — they answer at the granularity of a whole acquisition
slot, a border ring, or a depth window. They cost a forward pass per occlusion, so resolution is bounded by
card time. Integrated Gradients / GradientSHAP cost roughly a fixed multiple of one backward pass regardless
of resolution, so they are the cheaper instrument *per unit of spatial detail*.

The version that survives the incumbent's argument is **not** "draw heatmaps". It is:

- **Validate saliency against occlusion, then use it where occlusion is too expensive.** The three occlusion
  measurements are ground truth that a saliency method can be scored against. A saliency method that
  reproduces the border-vs-centre and depth-profile results is then licensed to answer finer questions the
  occlusion grid cannot afford. This is the only framing in which a picture becomes a measurement.
- **Attribute over the SLOT ATTENTION rather than over pixels.** `SlotHead`'s per-diagnosis attention is a
  small, supervised-adjacent quantity, and attributing a label's logit to attention weights is checkable
  against the slot-importance occlusion directly.
- **Per-STUDY, not per-corpus — the failure-triage use.** The owner's framing was "gauging why a model does
  bad somewhere". Corpus-level attribution is what occlusion already does well. The unmet need is the
  individual failure: take the studies where the ensemble is most wrong, and ask what the model looked at.
  That is a per-study question and occlusion is too coarse and too slow for it.

### The honest counter

The literature on saliency reliability is not kind — sanity checks (Adebayo et al., randomising weights and
labels) show several popular methods produce near-identical maps for a trained and an untrained network.
Any adoption here must run that sanity check first, on our own model, and report it. `captum` should arrive
as an **optional extra**, matching the sibling cardiac project, not as a declared dependency: it must not
enter the serving path or the kernel budget.

---

## What all three share, and the order they should be built in

None of these moves a score. All three change what a subsequent decision is made on, so their value is
entirely in the decisions they would have changed. Ranked by that:

1. **Mahalanobis on embeddings** — highest, because the seat ruler is the single most decision-heavy tool in
   the repo and it currently runs on a proxy for the quantity that actually transfers. It has already closed
   at least one axis (knee-for6).
2. **Jensen-Shannon between members** — next, because it is nearly free (prediction arrays are on disk, no
   card), and it measures the confidence axis a rank-average ensemble is currently throwing away.
3. **Captum, sanity-checked, for per-study failure triage** — last, because the incumbent's argument is
   sound and the instrument only clears it in the narrow per-study framing above.

Beads: **knee-iyad** (Mahalanobis, P2), **knee-c2ew** (Jensen-Shannon, P2), **knee-l1vy** (Captum, P3).
