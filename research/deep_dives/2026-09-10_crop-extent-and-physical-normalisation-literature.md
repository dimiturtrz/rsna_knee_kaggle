# 2026-09-10 — Crop extent and physical normalisation: what the literature has and has not measured

Field synthesis for the crop-extent axis (our members decode to a fixed 130 mm
physical field by default; the live axis is extent WIDTH within an already
physical-normalised decode, not physical-vs-pixel-grid). Our measured toll: −0.0336
mean AUC (sd 0.0038, n=11, 909-study holdout, midpoint cut) — the residual AFTER
physical-mm normalisation. [CLF] = classifier endpoint, [SEG] = segmentation,
[ASSERTED] = claimed without ablation.

## The mm-normalisation-already-paid framing (correction to earlier notes)

D266's framing was wrong and is retracted: `crop160_span`'s `--crop-mm` is an extent
argument, the default is 130 mm physical (`options.py:102`, `crop_centre.py:25`), and
`Crop` has no off — so all eleven flat-toll members are decoded to a fixed 130 mm
physical field, and the axis the pool varies on is extent width within an
already-physical-normalised decode. Corrected statement: "extent width is the live
axis; physical-vs-grid was never our question."

Consequence for comparability: **none of the spacing/geometry literature measures the
residual after physical-mm normalisation.**

- **Xie 2025** (the −0.05…−0.07 external knee-MRI penalty, [CLF], 13,419 patients /
  5 centres): its pipeline crops a **256×256 PIXEL region** around a meniscus
  segmentation — pixel-grid, not physical-mm. A PRE-physical-normalisation number,
  not comparable to our −0.0336.
- **Consispace (arXiv:2606.31839) / PVS (arXiv:2411.19564)** (spacing/resampling
  strategy matters): both [SEG] — Dice endpoints, no classifier speaks either way.
- **LUMINA input-resolution ablation** [CLF]: within one population, not cross-corpus.
- **CARVE's −0.125**: protocol/depth shift, CT — a different axis.

So our −0.0336 residual cell — what a corpus crossing still costs a model that
already sees a fixed 130 mm physical field — is as empty in the literature as the
mm-vs-grid cell was. The field's published tolls are not our toll; ours is smaller in
magnitude than Xie's pixel-grid penalty, which is directionally consistent with
physical normalisation having already removed part of the gap — but that comparison
is exactly the not-comparable one: treat it as a hypothesis the extent arm can test,
not as evidence.

## (a) Train at width W₁, decode at W₂ ≠ W₁, price classifier transfer vs W — never run

The exact 2-D experiment (controlled physical field width as the dial, classifier
transfer as the endpoint) is absent from the literature. Adjacent cells:

- **Bi et al., Samba, NeurIPS 2024 — the anchor.** Breast pathology grading,
  20×↔40× magnification = a 2× physical-extent change per patch, n=2,486 train /
  1,158 test. ERM degrades badly across the extent boundary; their method recovers
  +5.2/5.6/4.1% avg ACC, grade-wise up to +17.0% [CLF, cross-domain transfer].
  Extent mismatch alone is a real classifier domain shift — but extent is a property
  of the corpus pair, not a controlled dial, so it prices the EXISTENCE of the shift,
  not its dose-response.
- **Chang, Wong, Rasiej 2019** (J Imaging Inform Med 32(6), DOI
  10.1007/s10278-019-00193-4, knee ACL): a *narrower* intercondylar FOV beat the full
  slice in-domain (96.7% acc, AUC 0.97 with dynamic sampling) [CLF, in-domain].
  Single corpus, no transfer arm — but note the direction: tighter won. This is the
  one published tension against our n=1 wider-field reading (crop160_span −0.0265 vs
  pool −0.0336, +1.87 sd, 909-study holdout, midpoint cut).
- **Chen et al. 2024** retinal CFP 30–60° → UWF 180–200°: device-confounded, grading
  ACC 0.6813 UDA vs 0.6563 supervised bound [CLF]. Direction is widen, but field and
  device change together.
- **Lee et al., arXiv:2507.19858** lung-CT anatomy-tight crop: inter-source feature
  variance −75% [CLF] — standardise-tight, the opposite direction from widening.
- Divvala 2009 context priming: in-domain only, no transfer.

Honest statement: no published study trains classifiers at controlled physical field
widths and prices transfer against train/decode-field mismatch. Our live reading and
the running arm are genuinely new measurements; the literature supplies mechanism
hypotheses, not a prior.

## (b) Fixed-wide vs jittered extent on a transfer endpoint — silent for the direct 2×2

Never run on a classifier. The neighbours both favour randomised extent:

- **Schwonberg, arXiv:2304.12122** [SEG]: synthetic→real DG; random-resized-crop is
  the strongest single geometric aug, +2–4 mIoU OOD.
- **Sun et al., ICML 2020** (arXiv:2005.10243) [CLF-det]: stronger random-crop
  augmentation improves SSL transfer.
- **KneeXNet**: random scaling 0.95–1.05 "to simulate FOV variations" — no ablation
  [ASSERTED].

So `fovjitter160` is, per this search, the first direct test of the question as posed.

## (c) Pad-learning failure mode — mechanism answered, composition unmeasured

- **Weng, arXiv:2403.06748** [SEG]: zero-pad + center-crop → a "border = background"
  shortcut; Dice decays with distance from centre; random-crop training removes it.
  Reflection/replication padding only weakens it [ASSERTED by authors].
- **Alsallakh, "Mind the Pad", arXiv:2010.02178** [CLF-det]: padding-type swap
  (circular/symmetric) mitigates the spatial bias.
- **Islam, arXiv:2101.12322**: zero-pad encodes absolute position.
- **Gichoya, Lancet Digit Health 2022** [CLF]: race predicted from CXR/CT/mammo AUC
  0.94–0.99 internal, persists under cropping.
- **Zech 2018**: the hospital-token line [CLF].

No pad-specific medical [CLF] mitigation with numbers exists. The composed failure
mode our pool faces — wide fixed field → pad-filled borders → pad learned as site cue
→ transfer toll — has every component measured separately, but the composition has
never been priced on a classifier transfer endpoint.

## Comparability caveats (carried from the correction letter)

None of these papers operates post-physical-mm-normalisation. Our −0.0336 sd 0.0038
n=11 midpoint is the residual-after-normalisation cell; every published toll here is
either pre-normalisation or a non-classifier endpoint. Comparability remains NO in
both directions — we cannot borrow their priors, and they cannot validate ours.

## Flagged unverifiable

Chang 2019's numbers carry no transfer arm (in-domain only — quoted as direction, not
as evidence against extent widening); KneeXNet's FOV-jitter claim has no ablation
[ASSERTED]; Weng's reflection/replication claim is author-asserted; Samba prices the
existence of the extent shift, not a dose-response (extent is corpus-pair-fixed, not
a dial).

Source letters: D268 (mm-already-paid correction + comparability breakdown), D269
(extent-as-shift-lever (a)/(b)/(c), Samba anchor, pad-learning literature).
