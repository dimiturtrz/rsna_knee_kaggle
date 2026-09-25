# 2026-09-10 — What has moved a single-model ceiling on multi-label medical grading: ranked

Field synthesis, ranked by measured delta × corpus proximity to our 4.4k studies. Each
item carries mechanism (falsifiable) / measured delta vs what / corpus N / bucket
(DATA, OBJECTIVE, or INFERENCE). The DATA bucket dominates the top; two shapes are
silent at our scale; one strong published negative. [CLF] = classifier endpoint,
[SEG] = segmentation, [ASSERTED] = claimed without ablation.

## The ranked list

1. **ROI crop on knee MRI — CoPAS ablation. +0.056 mean AUC** [CLF].
   Mechanism: localising the input to a meniscus-centred ROI removes off-target
   anatomy so the trunk spends capacity on the joint. Delta: MRNet baseline
   0.725→0.781, same data, ablated stack (Qiu et al., Nat Comms 2024, N=**1,748
   subjects, 12 abnormalities** — the closest published task to ours). Bucket: DATA
   (localization). Follow-on steps in the same stack: cross-sequence attention +0.025,
   correlation mining +0.013; full model 0.812 vs MRNet 0.781 (+0.031, confounded —
   no plane-attention-only cell). Their external row is a warning: missing T1W/T2W
   costs −0.086/−0.091 mean AUC.
2. **Pseudo-labels / self-training — Zhan et al. +8.1pp mean AUC (0.631→0.712),
   +7.6pp mAP** [CLF]. Mechanism: teacher pseudo-labels on unlabeled studies; student
   trains on labelled + pseudo with consistency. N=**641 lung nodules, 454
   annotated** — the closest corpus to ours in the list, and the largest relative
   gain. Bucket: DATA. Cross-check at scale: CLCP-MT ChestX-ray14 20%-labels: +2.64
   mean AUC (68.17→70.81), N=78k [CLF]. Expect shrinkage as the unlabeled:labeled
   ratio falls toward our ~1.2:1.
3. **Report-labeler quality — Jain et al. +0.04 weighted AUROC (0.83→0.87)** [CLF].
   Mechanism: report-extracted labels are the supervision bottleneck; raising labeler
   F1 (0.53→0.68) raises the ceiling roughly proportionally. N=224k — far; mechanism
   evidence, not a price. Bucket: DATA (labels).
4. **Noise-aware softening/sharpening of report labels — Rep-GLS. BCE 79.63 → 84.14
   mean AUC (+4.51 total; softening alone +1.93, sharpening +2.58 more)** [CLF].
   Mechanism: map report certainty phrases to per-cell smoothing rates. N=377k, far.
   Repo note: our gold-58 read (65.3% supervision gap, 108 confident false positives)
   contraindicates the SHARPENING half for this corpus; the softening half is already
   implemented. Bucket: DATA/OBJECTIVE.
5. **Label-noise correction — Zhou et al. +2.44% mean AUC on CheXpert** [CLF,
   flagged: snippet-only]. Mechanism: disentangle representation from the noisy head;
   correct labels in training. N=224k. Bucket: DATA. Knee noise-rate diagnosis exists:
   inter-reader κ 0.55 (bone-marrow edema)–0.72 (meniscus/ACL), Astuto 2021, N≈800
   [CLF-ish].
6. **Multi-plane input — the MRNet numbers, and they CUT BOTH WAYS.** Mechanism:
   three planes carry complementary per-label signal. Bien 2018 (N=1,370): 3-plane
   combined avg 0.916 (abn 0.937 / ACL 0.965 / meniscus 0.847) — but the combined
   model did NOT significantly beat the best single series [ASSERTED, per-series
   table image-locked]. Counter-signal: Harman 2025 multi-plane hybrid ACL **0.949 <
   single-plane sagittal 0.976** [CLF]. Bucket: DATA (extra views). Target-corpus row
   from the same literature: Stanford sagittal-T2 zero-shot on Rijeka 0.824 vs 0.911
   trained-on-Rijeka (N=917+183) [CLF] — +0.087 for the target corpus itself.
7. **Early plane-fusion before slice pooling — MPFuseNet. ACL 0.977 vs MRNet ~0.965**
   [CLF, MRNet baseline ASSERTED from their text]. N=1,250. Bucket: INFERENCE+DATA.
   Their own control is the caution: MRNet-style late LR fusion was DETRIMENTAL for
   ACL/meniscus (LR overweights overfit planes).
8. **Augmentation as plateau breaker at MRNet scale — Azcona 2020, 93.4% vs MRNet
   91.6%** [ASSERTED, trunk+aug confounded]. Authors state transfer learning +
   augmentation were "the crucial factors." N=1,130. Bucket: DATA.
9. **Ordinal loss on graded labels — OAI KL grading. +0.3–1.6pp accuracy,
   −0.01…−0.05 MAE across 12 architecture cells** [CLF, accuracy/MAE not AUC].
   N=4,130 images / 8,260 joints — closest corpus size in the list, but single-label
   5-class accuracy, and the delta is small. Bucket: OBJECTIVE.
10. **Asymmetric loss — ASL. Fine-grained micro mAP 84.8→86.3 (+1.5)** [CLF; the
    ChestX-ray14 mAP 79.1→80.6 figure is flagged unverified]. N=112k. Bucket:
    OBJECTIVE.
11. **Query-attention slice pooling, trunk-held — AnyMC3D App F. 0.962 vs avg 0.958 /
    max 0.946** [CLF, CT]. Spread +0.004 over average. Bucket: INFERENCE. Nobody has
    run this operator spread on knee MRI; MRNet's own replication found
    attention-pool ≈ max-pool [ASSERTED, course project].

## Published negatives (as valuable as the positives)

- **Confidence/uncertainty heads: no method in a 13-method multi-label CXR benchmark
  claims an AUROC/AUPRC gain over plain BCE**; evidential heads miscalibrate badly
  (ECE ~0.4) [CLF-ish]. Bucket: OBJECTIVE — evidence against this head shape.
- Multi-plane fusion without a quality floor (items 6/7) can be actively detrimental.

## Silent list — shapes nobody has measured at 10³–10⁴ studies

1. **Laterality/sidedness as shared information** (shared-weight twin streams,
   contralateral context): no AUC delta exists at ANY N — the one shape with zero
   published data anywhere.
2. Label-noise CORRECTION (not diagnosis) on clinical grading labels at knee scale —
   κ is measured; correction gains are measured only at 10⁵.
3. Ordinal/hierarchical structure across a 12-finding multi-label head — measured
   only single-label 5-class accuracy.
4. Natural partial-label-overlap corpora joined via masked loss — artificial
   single-positive masking exists at 5k–40k on CXR; natural overlap transfer
   unmeasured.
5. Trunk-held slice-aggregation operator spread on knee MRI — CT only.
6. Pseudo-labeling at ~1:1 unlabeled:labeled at our N — empty cell.
7. Slice-coverage scaling curve at fixed resolution — no published curve.

## Corpus-size scaling-curve context

An anatomy-SSL knee study measured the data-scaling curve directly: mean AUC
0.433/0.429/0.474 at 80/200/400 subjects, 0.562 at 800 (from-scratch training)
[CLF]. The ceiling at our corpus size is supervision-bound in the published record —
every measured lever of size ≥ +0.02 lives in the DATA bucket above.

## Trunk capacity at ~4k studies (companion synthesis, D273)

No same-architecture ViT ladder with numbers exists at our scale anywhere. The
knee-specific record says bigger LOST: Shakhovska 2022 (MRNet, identical pipeline, 5
backbones) — steady gain up the VGGs, then ResNet/EfficientNet underperform the VGGs;
Yiu 2025 (arXiv:2508.14151, MRNet 1,199 pts, meniscus) — ResNet50 AUC 0.8184 vs
ViT-B-scale 0.74, authors attribute the ViT loss to data scarcity. Controlled ladders
nearby: GloResNet (preterm brain MRI, N=128) ResNet-10 0.861 > ResNet-50 0.610 >
ViT-Base 0.540; DINOv2 radiology (Baharoona, arXiv:2312.02366, 100k+ CXR) — ViT-L
underperforms ViT-g on 3D CT classification. The field's large-trunk recipe is
**full fine-tune + LLRD ~0.75 + low LR** (RETFound, MAE lineage), not freeze-more;
Kumar et al. ICLR 2024 says freezing ONLY the patch-embedding layer preserves SGD
fine-tuning performance. The one measured corpus-size crossover is Dutt
(arXiv:2305.08252, F1-only — see the toll-shrinkers file for its limits). Nothing
external supports "throttle harder" for a failing ViT-L; the measured intervention at
~4k images is restricting WHICH weights move (LoRA/BitFit-class), not unfreeze-depth
or decay. Capacity×resolution confound: published ViT-L medical numbers run at
384–518px while ViT-S sits at 224px, and nobody has priced the split; a capacity arm
at fixed resolution would still be the first clean cell.

## Flagged unverifiable

Zhou et al. label-noise correction (snippet-only); ASL ChestX-ray14 mAP figure; Bien
2018 per-series table (image-locked; the "combined not significantly better" claim is
ASSERTED); MPFuseNet's MRNet baseline (from their text); Azcona 2020 (trunk+aug
confounded, ASSERTED); MRNet attention-pool ≈ max-pool replication (course project);
Astuto 2021 κ values [CLF-ish]; Shakhovska and Baharoona table cells (images — prose
conclusions only); MeLo [ASSERTED]; CS231n resolution study (course project).

Source letters: D275 (ranked list, negatives, silent list, scaling curve), D273
(trunk-capacity companion section).
