# Slice aggregation and head architecture — topic summary

**Current verdict (2026-09-25).** The shipping aggregator is `--slice-pooling labelwise` (per-label attention over
slices). Max pooling is a disowned ruler: it suppresses real input changes that labelwise passes, with a 2x2
interaction of +0.0069 [M][vc] [AXES:61-65][axes]. On the labelwise line, every head or input add-on measured so far
reads inside the 0.0062 floor or negative on ledger75. That covers multiscale, slice_shift, fuse_blocks,
extended_slots, labelwise_patch, ROI crops and 24→36 slices [axes]. No aggregation-side lever has a mechanism left
that is plausibly worth 1.5%.

Tags: **[M]** measured by us · **[L]** literature · **[U]** unverified / asserted / derivation. Rulers are quoted
where the source names them. An unmarked number of ours is midpoint (knee-f9aao).

---

## 1. Does whole-stack reading pay, and what is the canonical operator?

- Reading the whole stack was our largest single change, +0.0147 [M] [sa].
- MRNet uses an element-wise max over ALL slices. MST uses DINOv2 plus a slice transformer. Both are literature
  [L] [sa] [ig].
- The binding constraint is supervision: 58 of 4,407 studies are gold [M] [sa] [meta].
- Single-plane sagittal ACL 0.976 beat 3-plane fusion 0.949 (Harman 2025) [L] [sa] [ceil]. So more planes is not
  automatically more signal.
- **Supersedes the "max is canonical" reading:** MRPyrNet has average beating max for meniscus, and AnyMC3D ranks
  query > avg > max [L] [qen] [qdt].

## 2. Max vs attention vs labelwise: which operator, and why?

- Literature, AnyMC3D Table 5 [L] [qdt] [qen]. Scores: query-attention 0.962, avg 0.958, max 0.946, median 0.944,
  Transformer 0.950, LSTM 0.903. The spread is 0.4–1.6%.
  - This is CT only, and the trunk is held fixed.
  - There is no knee trunk-held ablation and no per-label breakdown by operator [L] [qdt] [ceil].
  - MST: a Transformer without positional encoding wins [L] [qdt].
- Ilse ABMIL [L] [att]:
  - At bag size 100, attention reads 0.996 vs max 0.553.
  - Attention's advantage is largest when samples are few.
  - With enough data, all pooling types converge.
- Mechanism in print. CLAM says max "uses the gradient signal from only a single instance" [L] [att].
  - CLAM-MB 0.942 vs SB 0.927 vs classic MIL 0.866 (CKD WSI) [L] [att].
  - Counterpoint: FocusMIL argues max is more robust to domain shift [L] [att].
- **Ours [M] [vc]:** a 2x2 of input × aggregator.
  - Max suppresses a real input change that labelwise passes: interaction +0.0069; per column MCL +0.0278,
    LatMen +0.0205.
  - The −0.0059 under max is argmax-suppression, not bag-size dilution. Dilution is a 10³–10⁵-instance
    phenomenon [L] [vc].
  - Per the sweep, this is the first published input×aggregator interaction in any domain [L-null] [vc].
- AXES: the max-pooling ruler under four §B rows is DISOWNED; they must be re-read under labelwise [M]
  [AXES:61][axes].
  - Capacity re-opened under labelwise: `base_labelwise224` 0.8899 vs `labelwise224` 0.8832 = +0.0067 [M]
    [AXES:40][axes].
  - Pooling × count INTERACT: max wins short stacks and loses long ones [M] [AXES:92][axes].
- **Superseded:** [qen] (09-04) calls attention pooling "the one axis not varied".
  - [sm] (09-03) says attend/labelwise had already measured null over max.
  - Both are superseded by [vc] (09-11) and AXES: labelwise is now the shipping line, justified by the
    interaction, not by a main effect under max.
- Qiu/CoPAS (12 knee labels) uses SHARED co-plane attention and never ablates a per-label variant [L] [vc].
- A per-label localisation mechanism is untested as a causal claim [U] [vc].

## 3. Does the gain saturate with slice depth?

- Li MICCAI 2025, CT lymph node: 9→18→27 slices gives 81.12→82.54→81.77. It saturates at ~18. The 27-slice drop
  is within ±2.6 [L] [att].
- No published accuracy-vs-instances curve separates per-class from shared/max pooling [L-null] [att].
- Ours:
  - 12→24 slices: CLOSED, flat [M] [AXES:90][axes].
  - 24→36: −0.0224, behind at all 7 matched epochs [M] [AXES:91][axes].
  - 20 vs 12 slices under max: −0.0001, UNREAD [M] [AXES:82][axes].
  - The depth step under labelwise is UNREAD. Under the max head it calibrates flat: `native224` 0.8800 vs `band224`
    0.8801 [M] [AXES:83][axes].
- **Code/bead fact [M]:** the 12-slice cache samples a FRACTION of the acquired stack, not anatomy (knee-bps3x,
  `core/data/corpus/slot_reader.py:328`). A fixed count therefore covers different physical extents per study.
- 2026-09-01 called coverage "the one input-side cell unread" [ig]. **Superseded** by AXES:90-91: coverage is
  closed on our measurement.

## 4. Head shape: second moments, bottleneck, readout

- VARPOOL (Carmichael MICCAI 2022) is the precedent for variance pooling. No slice-axis variance pooling exists in
  the literature [L] [sm].
- Frozen ortholarge:
  - Ridge on moments 0.8249 vs SlotHead max 0.7537 (gap 0.071) [M] [sm].
  - @focal: ridge moments 0.8279 vs max 0.8146 (+0.013) [M] [sm].
  - About 85% of the gap is the head and about 15% the second moment [M] [sm].
- VARPOOL's η nonlinearity: the code default is sqrt, the paper uses log(0.01+·). B-CNN uses signed sqrt [L] [vs].
  - MPN-COV's matrix-power rationale does not transfer to per-channel std [U] [vs].
- Head bottleneck (LayerNorm→Linear 1024→256→GELU): the information-loss mechanism is backed by Gupta 2022 and
  Xue ICLR 2024 [L] [vs].
  - Proposed diagnostic: ridge(raw) vs ridge(linear proj) vs full proj vs forward.
  - Arm: `experiments/arms/moment_dense.sh` [vs].
- Readout from an earlier block (block 9 of 12): −0.0227, CLOSED [M] [AXES:94][axes].
- Label-correlation / residual head: REFUTED [M] [AXES:117][axes].
- Uncertainty heads: none of 13 multi-label CXR methods beats plain BCE on AUROC [L] [ceil].
- Frozen probe reads ortholarge +0.020 over v3large, yet ortholarge serves worse. A probe ranking is not a
  fine-tune ranking [M] [sm].
  - Knott: LP-vs-FT Spearman 0.90 vs 0.79 [L] [sm].
- LP-FT (Kumar): proposed in [sm]. **Superseded:** CLOSED as a convergence-speed effect that a long fine-tune erases
  [M] [AXES:131][axes].

## 5. Extra inputs through the head: in-plane, multiscale, multi-view, ROI

- In-plane selection (`labelwise_patch`, per-label attention over slice×patch tokens): 0.9064 vs 0.9053 ledger75 =
  +0.0011, null [M] [AXES:245][axes].
- Multiscale under labelwise: `labelwise_ms` 0.8960 ledger75 @e12 vs ctrl 0.9006 @e20 = −0.0046 [M]
  [AXES:77][axes].
- slice_shift, 12sl: +0.0035 (0.9041 vs 0.9006). `shift4_depth24` 0.9037 vs 0.9053 ledger75 = −0.0016. CLOSED [M]
  [AXES:78][axes].
- `fuse_blocks`: +0.0008 (0.9014 vs 0.9006) [M] [AXES:79][axes].
- `extended_slots`: −0.0003 (0.9045 vs 0.9048 ledger75) [M] [AXES:80][axes].
- Stacking extra series as channels: +0.0002, flat [M] [vc].
  - Gori (710 experiments) finds instance-MIL matches SOTA given a strong foundation model. That is the
    views-as-samples half, unrun here [L] [vc].
- **ROI crops [M]:** crops were APPENDED as extra sagittal views (`core/data/corpus/slots.py:389-405`) and lost.
  Scores: roi 0.8984 / roinative 0.9018 vs no-crop 0.9051, ledger75 (knee-vgz5b).
  - AXES agrees: ROI crop/upsample CLOSED, "the crop costs, and the upsample was exactly half of why" [M]
    [AXES:87][axes].
- ROI literature is all CNN-era: Astuto; KAMRNet AOLM 0.730–0.966 vs 0.687–0.953; Güngör 0.97/0.98 at 480px [L]
  [qcg] [fr].
  - No knee ROI ablation exists on a foundation trunk [L-null] [qcg].
  - MST DINOv2 meniscus 0.85±0.04 vs 3D ResNet 0.69; frozen 0.79 [L] [qcg].
  - Astuto meniscus IoU 0.49–0.61, yet AUC 0.93 [L] [qcg].
  - Whether ROI adds information or compensates for a weak trunk is unresolved [U] [qcg].
- **Contradiction:** [fr] ranks ROI/localisation as lever #1. Our appended-crop measurement and AXES:87 close it on
  our stack. The CoPAS +0.056 was bought against a no-crop baseline, and we already crop [M] [AXES:86][axes].
- Localiser routes for a real crop [qbq]:
  - MRAnnotator: CC-BY-NC, dead link.
  - CartiMorph: CC-BY-NC, 229MB, DESS shift.
  - SKM-TEA: RUA forbids redistribution [qcg].
  - IWOAI/DOSMA: GPL.
  - No attention-rollout or MIL localiser has a measured gain [L] [qbq].
  - The 9/9 staged-pipeline tally is partly comps that shipped masks [U] [qbq].
- Map-driven erasing: +0.0004 (0.9057 vs 0.9053 ledger75) [M] [AXES:246][axes].

## 6. Cross-slice / 3D modelling

- KneeXNet 2D→3D (~+0.015) [L] [qen]:
  - abnormal 0.923→0.937
  - ACL 0.896→0.912
  - meniscus 0.889→0.905
- [fr] quotes KneeXNet meniscus as 0.889→0.968 [L]. See the contradictions list.
- Li: 91.78→92.86% [L] [fr].
- CoPAS 0.812 vs MRNet 0.781 (confounded full stack) [L] [qen] [ig].
- AnyMC3D: 0.924 on 20% of the data vs 3D DenseNet 0.741 on 60% [L] [qen].
- BIPON multi-scale aggregator [L] [qen].
- Ours:
  - Through-plane neighbours show a density gradient; span loss is the named cause [M] [AXES:93][axes].
  - The pool is order-blind by construction [M] [AXES:96][axes].
  - Region/depth pooling: the patellofemoral loss was the band [M] [AXES:98][axes].
  - Registration/CoPAS CLOSED: p50 5.7 mm; `tissue224` −0.0011; `alignment_damage` +0.0103 [−0.0305, +0.0421] [M]
    [AXES:86][axes].

## 7. Per-label structure: focal/regional labels and lateral OA

- The focal/regional trade appears in published tables [L] [fr]:
  - MRNet meniscus 0.847 vs abnormal 0.937.
  - Astuto meniscus 0.93, cartilage 0.93, BME 0.83.
  - Nebelung.
- Planes: MRNet axial PD is best for meniscus, coronal T1 best for ACL [L] [fr].
- Lateral TF OA's weakness is over-determined in the literature [L] [att]:
  - It makes up 15% of patterns (Stoddart).
  - Reader κ lateral 0.741 vs medial 0.876; DCNN 0.781 vs 0.882 (Fritz).
  - Report-read lateral tibial condyle AUC 0.578 (Krakowski).
  - MOAKS tibial cartilage κ 0.36.
  - Lateral meniscal coverage 59% vs medial 50% (Bloecker).
- The coronal-band undersampling of the lateral compartment is a geometric derivation, measured nowhere [U] [att].
- Non-FS sequences are blind to BML by physics [L] [att].
- Pretraining is task-dependent [L] [fr]:
  - RadImageNet not confirmed.
  - Frees ACL 0.9969 with ImageNet.
  - SKID 0.761 vs 0.741.
- The resolution evidence is thin [L] [fr]. Ours: 224/336 CLOSED; 336→392 CLOSED [M] [AXES:74-75][axes].

## 8. Missing sequences and slot conditioning

- Taxonomy [L] [ms]: zero-fill, modality dropout, copy, synthesis, learned prompt, per-subset models.
- Grøvik 2021: AUC 0.989 vs 0.989 [L] [ms].
- Chrysochoou 2026 [L] [ms]:
  - Missing FLAIR: dropout 0.87/0.74 vs synthesis 0.85/0.68.
  - Missing T1w-pre: 0.79/0.78 vs 0.71/0.74.
- Zhang 2024: 0.861→0.842 at 80% missing [L] [ms].
- Availability is indication-driven, so it is a leak risk. Denck overlap 0.04–0.57 [L] [ms].
- Our fill rates: fluid 93–100%, T1 55–60% [M] [ms] [ig].
- CoPAS with T1W/T2W withheld: 0.812→0.721/0.726; contusion 0.821→0.698 [L] [ms] [ig].
- Recommendation was modality dropout [ms]. Ours: slot dropout KEEP at 0.40, the sole transfer carrier [M]
  [AXES:123][axes].
- Series metadata [meta]:
  - `Anatomical_Plane` is constant per study.
  - `Fat_Suppression` == `Fluid_Sensitive` on all 24,371 rows, and `Fluid_Sensitive` already routes slots [M].
  - Not a leak.
  - FiLM on an image-inferable contrast flag is inconclusive (Lemay) [L].
  - Meta-D +2.62% F1 [L]; Dohmen 96.66 vs 93.51 [L].
  - The seat is declined (knee-bc1m). Study-level series counts remain untested and have no precedent [U].

## 9. Test-time adaptation at the head

- TENT/BN adaptation does not transfer [L] [vc]: DINOv2 ViTs have no BN, and small test batches bias the re-estimated
  statistics.
- No TENT/EATA/SAR result exists on MRI/CT classification [L-null] [vc].
- Pseudo-labelling, BN/feature alignment and graph smoothing are all declined as 1.5% levers [tta].
  - Medeiros: TTA degrades MedMNIST by up to 31.6 pts [L] [tta].
  - The public best single is DINOv2-small 0.906 [L] [tta].
  - [tta]'s "0.008 gap to the leader" is stale. Read `board_read --live` instead.
- Overlap TTA: consumed [M] [AXES:199][axes]. Band TTA loses [M] [AXES:200][axes].

---

## Open questions

- Labelwise depth step (12→20 slices on the 224 cache): UNREAD. It is blocked by the cache, not by VRAM
  [AXES:83][axes].
- Resolution 420 under labelwise: UNREAD [AXES:76][axes].
- Moments pooling TRAINED is measured on the max line: `moment_dense` 0.8516 (−0.0118, width collapse), `moment_wide`
  0.8600 (−0.0034, flat) ([experiments/README.md ~15298-15390](../../experiments/README.md)). Only a labelwise × moments
  composition is unrun, with a flat prior.
- Views-as-samples (instance-MIL over series) vs channels: the samples half is unrun here and unmeasured
  anywhere [vc].
- An anatomy-indexed slice sampler to replace fraction-of-stack (knee-bps3x): no measurement yet.
- Lateral-compartment coronal coverage: nobody has counted it [att].
- Study-level series-count features: speculative, no precedent [meta].

## Sources

- [sa]: [2026-08-11_slice-aggregation-literature.md](../deep_dives/2026-08-11_slice-aggregation-literature.md)
- [ms]: [2026-09-01_missing_sequences_knee_mri.md](../deep_dives/2026-09-01_missing_sequences_knee_mri.md)
- [tta]: [2026-09-01_transductive_test_time_adaptation_audit.md](../deep_dives/2026-09-01_transductive_test_time_adaptation_audit.md)
- [meta]: [2026-09-02_series_metadata_axis_audit.md](../deep_dives/2026-09-02_series_metadata_axis_audit.md)
- [qbq]: [2026-09-02_qbq-localizer-routes-for-knee-mri.md](../deep_dives/2026-09-02_qbq-localizer-routes-for-knee-mri.md)
- [fr]: [2026-09-03_focal-regional-knee-mri-literature.md](../deep_dives/2026-09-03_focal-regional-knee-mri-literature.md)
- [qcg]: [2026-09-03_qcg-roi-gain-vs-strong-trunks.md](../deep_dives/2026-09-03_qcg-roi-gain-vs-strong-trunks.md)
- [sm]: [2026-09-03_second_moment_pooling_and_lpft_literature.md](../deep_dives/2026-09-03_second_moment_pooling_and_lpft_literature.md)
- [vs]: [2026-09-03_variance_scale_and_head_bottleneck_literature.md](../deep_dives/2026-09-03_variance_scale_and_head_bottleneck_literature.md)
- [qdt]: [2026-09-04_qdt-slice-fusion-mechanism.md](../deep_dives/2026-09-04_qdt-slice-fusion-mechanism.md)
- [qen]: [2026-09-04_qen-cross-slice-modelling.md](../deep_dives/2026-09-04_qen-cross-slice-modelling.md)
- [att]: [2026-09-11_attention-pool-depth-saturation-and-lateral-oa-reads.md](../deep_dives/2026-09-11_attention-pool-depth-saturation-and-lateral-oa-reads.md)
- [vc]: [2026-09-11_views-consistency-tta-and-mil-aggregation.md](../deep_dives/2026-09-11_views-consistency-tta-and-mil-aggregation.md)
- Also cited: [ig] [2026-09-01_input_side_knee_mri_gains.md](../deep_dives/2026-09-01_input_side_knee_mri_gains.md),
  [ceil] [2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md](../deep_dives/2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md)
- Axis index: [axes] = [docs/AXES.md](../../docs/AXES.md). Row numbers are file lines as of 2026-09-25.

[sa]: ../deep_dives/2026-08-11_slice-aggregation-literature.md
[ms]: ../deep_dives/2026-09-01_missing_sequences_knee_mri.md
[tta]: ../deep_dives/2026-09-01_transductive_test_time_adaptation_audit.md
[meta]: ../deep_dives/2026-09-02_series_metadata_axis_audit.md
[qbq]: ../deep_dives/2026-09-02_qbq-localizer-routes-for-knee-mri.md
[fr]: ../deep_dives/2026-09-03_focal-regional-knee-mri-literature.md
[qcg]: ../deep_dives/2026-09-03_qcg-roi-gain-vs-strong-trunks.md
[sm]: ../deep_dives/2026-09-03_second_moment_pooling_and_lpft_literature.md
[vs]: ../deep_dives/2026-09-03_variance_scale_and_head_bottleneck_literature.md
[qdt]: ../deep_dives/2026-09-04_qdt-slice-fusion-mechanism.md
[qen]: ../deep_dives/2026-09-04_qen-cross-slice-modelling.md
[att]: ../deep_dives/2026-09-11_attention-pool-depth-saturation-and-lateral-oa-reads.md
[vc]: ../deep_dives/2026-09-11_views-consistency-tta-and-mil-aggregation.md
[ig]: ../deep_dives/2026-09-01_input_side_knee_mri_gains.md
[ceil]: ../deep_dives/2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md
[axes]: ../../docs/AXES.md

## MST-style slice transformer: already answered by our own order-aware operators (2026-09-28)

A 2026-09-28 web sweep re-surfaced MST (DINOv2 + a slice transformer). What it adds over our heads is
letting slices see each other along depth. We have measured that twice. `depth` pooling (DepthAttention,
span 3) read `depth_true` 0.8461, the second-worst of four operators spanning 0.0019
(`experiments/README.md` §"the pool is order-blind", ~:11258). `slice_shift` (gated neighbour tokens
inside the last blocks) is +0.0035 on the 12-slice labelwise line and −0.0016 on the 24-slice shipping
line (docs/AXES.md, slice_shift row). So do not queue a slice transformer as new. Re-open it only with a
mechanism neither of those tested, such as long-range attention across the whole stack with positional
encoding. Note that MST's own ablation found no positional encoding worked best.

Deep dive: [2026-09-28 gap audit and frontier sweep](../deep_dives/2026-09-28_gap-audit-and-frontier-sweep.md).
