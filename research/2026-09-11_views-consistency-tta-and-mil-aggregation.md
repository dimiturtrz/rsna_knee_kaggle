# Multi-view use of a fixed corpus, test-time adaptation, and MIL aggregation (researcher, for M324 / M325)

Durable version of letter D321. Regime: 4,407 studies, 24,371 series (3-6 planes/sequences per
study), 12 study-level labels, DINOv2-class ViT trunks; stacking extra series as channels measured
flat (+0.0002); a 2x2 showed max pooling suppressing a real input change that a per-label
aggregator passes (+0.0069 interaction; MCL +0.0278, LatMen +0.0205 per-column).

ID corrections from the sweep: SAR is **arXiv:2302.12400** (not 2211.14479, a greenhouse-physics
paper); TransMIL has no canonical arXiv ID (NeurIPS 2021 proceedings only). Everything else verified.

## M324-Q1 — multi-view as SAMPLES vs as CHANNELS: measured support, and it conditions on exactly
what we have

- **Gori et al., arXiv:2505.01109** (J. Medical Imaging): 710 experiments, 4 WSI datasets, 10 MIL
  strategies — with a strong foundation-model feature extractor, simple INSTANCE-BASED MIL (bag label
  propagated to every instance, per-instance classifier, aggregate at inference) MATCHES OR BEATS
  embedding-based SOTA, new SOTA on BRACS and Camelyon16. Mechanism as they state it: label-
  propagation noise only hurts when the encoder is weak; once features are good, it is cheaper than
  the aggregation bottleneck. That is the view-as-samples arm measured positive, and the condition is
  a trunk class we already sit on.
- The canonical knee recipe IS view-as-samples: MRNet (Bien 2018, DOI 10.1371/journal.pmed.1002699)
  trains one AlexNet per plane and averages. CheXFusion (arXiv:2308.03968, CXR-LT winner): weighted
  average of single-view models beats any single view; learned fusion beats averaging. The
  channels-side counterpoint is Geras (arXiv:1703.07047, joint four-view mammography net wins) — at
  ~200k exams, a different n regime.
- NULL on the exact cell: channels-vs-samples at fixed label count, foundation trunk, small corpus —
  nobody has run it head-to-head. Our stacking-null (+0.0002) is the channels half of that cell
  already measured; the samples half is unrun here and unmeasured anywhere.

## M324-Q2 — cross-view consistency on small corpora: real anatomical-plane work exists; your exact
shape (plane AND sequence differ) is a NULL

- UMCT (Xia et al., arXiv:1811.12506 / MedIA 2020 arXiv:2006.16806): co-training between networks fed
  the three ORTHOGONAL ANATOMICAL PLANES of one volume, uncertainty-weighted pseudo-labels, measured
  at 6 labelled / 56 unlabelled cases — genuinely small-label, genuinely anatomical planes. But
  segmentation, same contrast.
- MICLe (Azizi, arXiv:2101.05224): multiple images of the SAME CASE as contrastive positives,
  +6.7%/+1.1% over SimCLR with the largest gains at small label fractions. Pretraining, not a
  consistency loss.
- **NULL, plainly**: nobody has measured a consistency/co-training loss for CLASSIFICATION where the
  views differ in plane AND pulse sequence. Whether a consistency gradient survives a T1-vs-PD-FS
  contrast change on the same meniscus is unmeasured anywhere.

## M324-Q3 — test-time adaptation under a kernel: near-NULL on medical classification, and two flags
that bite us specifically

- Medical TTA evidence is segmentation-dominated (Karani arXiv:2004.04668; Valanarasu MIDL 2024). For
  classification: DoSReMC (arXiv:2508.15452) identifies BN layers as the carrier of domain dependence
  in cross-domain mammography (but adapts with target supervision). **No measured TENT/EATA/SAR
  result on MRI or CT classification exists.**
- **Flag (a)**: TENT-family tunes normalisation statistics — DINOv2-class ViT trunks have NO BN
  layers, so the whole BN-adaptation line (TBN, MemBN, TTN) does not transfer and TENT reduces to
  LayerNorm-affine tuning, measured much weaker.
- **Flag (b)**: batch-size sensitivity is well measured and it is our case: arXiv:2208.08767's
  batch sweep shows BN adaptation degrading as test batch shrinks; "Unraveling BN" (AAAI 2024,
  arXiv:2312.09486) pins the mechanism — small batches have insufficient CLASS DIVERSITY, so
  re-estimated statistics are biased not just noisy. A kernel scoring in small batches is exactly
  this. NOTE (NeurIPS 2022) targets the temporally-correlated non-i.i.d. stream, which is what a
  kernel sees — all measured on CIFAR/ImageNet-C, none on medicine.

## M325-Q1 — input-axis ablation under two aggregators, conclusion flipped: NULL. Ours is the first.

No published study crosses the two axes: input ablations exist under ONE head (MPFuse
arXiv:2108.08136 ablates plane count under one fusion net; Yang et al. 2025, PMC12209677, ablates
planes for ACL/meniscus), aggregator comparisons exist at FIXED input. Nobody flipped an input
conclusion by changing the aggregator. **Your 2x2 is the first measured demonstration of the
interaction, in any domain** — write it down as ours.

## M325-Q2 — bag-size dilution: the literature's scaling direction is AGAINST the worry at our size —
and it re-reads your measurement

The dilution literature (DTFD-MIL arXiv:2203.12081; ACMIL arXiv:2311.07125; AEM arXiv:2406.15303;
LongMIL arXiv:2410.14195) is about bags of 10^3–10^5 patch instances; it consistently finds mean/max
competitive and dilution negligible in the SMALL-bag regime, which is where 6–12 slots live. So your
−0.0059-under-max is NOT classic bag-size dilution. The better-fitting published mechanism is
argmax-suppression: max pooling transmits only the per-bag winning instance per label channel, and a
meniscus/MCL signal that never wins the global argmax against the diffuse-abnormality signal is
invisible under max while a per-label aggregation passes it — which is your MCL +0.0278 / LatMen
+0.0205 split read mechanically. (The softmax log-gap scaling argument I would have cited is a
derivation, not a theorem — flagged as such.)

## M325-Q3 — per-label vs shared aggregation: mainstream in multi-class WSI, untested as a causal
claim in multi-label imaging

CLAM-MB (arXiv:2004.09666) uses class-specific attention branches — per-label aggregation is the
STANDARD multi-class WSI recipe; Ilse (arXiv:1802.04712) did class-specific attention on multi-label
MNIST-bags; the audio event-detection literature pools per event class routinely. **The closest
published model to this whole project turned up in this sweep: Qiu et al., Nat. Commun. 2024
(PMC11368947) — co-plane attention across MRI sequences on 12 knee-abnormality labels** — and it uses
a SHARED co-plane attention structure and does not ablate the per-label variant. NULL on the causal
attribution you asked for (gain localised to spatially-localised labels) and NULL on the
parameter-cost trap (at 12 slots × 12 labels the extra parameters are trivially small either way;
nobody has analysed it because there is nothing to analyse).

Null summary: channels-vs-samples at fixed labels (empty cell); consistency across plane+sequence
(none); TENT-family on medical classification (none); input×aggregator joint ablation (none — ours
is first); per-label aggregation's localisation mechanism (untested). Five nulls, one of them ours
to publish-shape.

— researcher
