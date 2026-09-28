# Trunks, pretraining and transfer — topic summary

**Current verdict.** The solo checkpoint stays on a natural-image DINOv2 trunk read per slice. Every alternative the deep dives proposed has been closed in `docs/AXES.md` §A: DINOv3, OrthoFoundation, OrthoDiffusion, RadImageNet, RAD-DINO, MRI-CORE, 3D-native encoders, CNNs, V-JEPA, LP-FT and LoRA.
The literature predicted this. In-domain SSL gains over ImageNet init vanish at ~800 labelled subjects, and we hold ~4,400. 2D slice-pooled readers match or beat 3D ones. Probe-best trunks invert under fine-tuning.
What is still open is only a cheap, mechanism-bearing test: an error-correlation read of a 3D member, or a small SSL-continuation smoke on our own pixels. Neither is priced to clear the 1.5% bar.

Tags: **[M]** measured by us · **[L]** literature · **[U]** unverified or inferred.
Ruler: a number is on the repo-default midpoint cut unless it says `ledger75`. The deep dives almost never name a cut.

---

## 1. Which trunk family carries the solo?

- **[M]** The stack is DINOv2-specific. It carries +0.054 on a ViT and 0.000 on a CNN. AXES §A headline (README:17307).
- **[M]** Capacity pays only up to base. `base_labelwise224` 0.8899 vs `labelwise224` 0.8832 = +0.0067, just above the 0.0062 floor. `large_labelwise224_half` reads **0.8936 ledger75** vs base **0.8950 ledger75** = −0.0014, for 3.1× the parameters and 3.1× the wall-clock. On midpoint, large "wins". AXES §A capacity row (knee-ss5rd).
  - This supersedes the 09-02 reading "capacity is locally negative": v3large_dense 0.8258 vs base224 0.8502 vs ortho_dense 0.8452 ([capacity_ceiling](../deep_dives/2026-09-02_capacity_ceiling_trunk_family_qad.md)).
- **[L]** Scale helps only large targets (Jitsev). Transfusion finds minimal benefit (Raghu). VGG19 beat ConvNeXt/DINOv2 on lumbar spine. At 4,407 studies we sit in the small regime; small diverse families would be expected to give +1-2 points ([capacity_ceiling](../deep_dives/2026-09-02_capacity_ceiling_trunk_family_qad.md)).
- **[M]** DINOv3 is **CLOSED**. "DINOv3 is weak" was an adaptation finding (AXES §A DINOv3 row).
  - Frozen probes: DINOv2-L 0.7852, DINOv3-L 0.7446, OrthoFoundation-L 0.7387 ([sota_pretrained](../deep_dives/2026-09-02_sota_pretrained_knee_mri_models_2025_2026.md)).
  - [trunk_vs_readout](../deep_dives/2026-09-02_trunk_vs_readout_quality_medical_volumes_qal.md) argues the −0.0406 DINOv3 probe deficit is a **readout confound**, not lost signal **[L]**. Muthyala CXR: CLS 0.500-0.524 vs patch-local ~1.0. AnyMC3D: query attention 0.962 vs max 0.946; DINOv2 and DINOv3 tie there.
- **[M]** CNNs are **CLOSED as a class**: ConvNeXt, ResNet, RadImageNet DenseNet, EfficientNetV2-L null. The CNN fills no per-label ViT blind spot; MCL is −0.0344 with a CI that excludes zero (AXES §A CNN rows, knee-b39kg).
  - ConvNeXt rate ladder: 4e-5 → 0.7927, 1.2e-4 → 0.7993, 4e-4 → 0.8186 ([qj](../deep_dives/2026-09-01_qj_convnext_rate_radimagenet_plumbing.md)).
  - **[L]** ConvNets are more annotation-efficient than ViTs (Taher, MedIA 2025). ViTs lean more on AugReg (Steiner). That doc reads the 0.033 ConvNeXt gap as partly a corpus-size artefact **[U]** ([cnn_vit_aug](../deep_dives/2026-09-01_cnn_vit_augmentation_sample_efficiency.md)).
- **[L]** Zhou et al.: ConvNets and ViTs use the same LR (1e-4). No published "10-30×" CNN rule exists ([qj](../deep_dives/2026-09-01_qj_convnext_rate_radimagenet_plumbing.md)).
- **Solo implication:** keep DINOv2-base at 224/12 slices. Width above base is retired unless a new mechanism arrives (AXES §A).

## 2. Knee- or medical-domain foundation trunks

### OrthoFoundation-L (DINOv3-L continued on knee)
- **[L]** MIT licence. Trained on 1,251,655 knee images (357,670 X-ray + 893,985 MRI; public ~600k). Reported ACL AUROC 98.49, PCL 84.31, MCL 93.34, cartilage 93.38 ([modern_knee](../deep_dives/2026-09-01_modern_knee_mri_models_beyond_mrnet.md)).
  - The 09-01 TL;DR calling it "the next encoder experiment" is **superseded**. It is declined in the 09-02 docs.
- **[M]** `ortho_dense_s2026` 0.8452 vs the DINOv2-base control 0.8502 → declined ([modern_3d](../deep_dives/2026-09-02_modern_3d_knee_mri_models_beyond_mrnet.md), [qea](../deep_dives/2026-09-04_qea-volume-native-ssl-knee-mri-literature.md)).
- **[M]** Probe/FT inversion: `ortholarge` probe is +0.0203 over `v3large`. The FT members `ortho_dense` 0.8452 and `ortho_pool_read` 0.8464 sit below the ~0.855 seats ([qdk](../deep_dives/2026-09-04_qdk-continued-ssl-probe-ft-inversion.md)).
- **[L]** Preprocessing: ImageNet mean/std, 224 px, patch 16, 4 storage tokens, student weights. A gap that survives these settings is real ([ortho_preproc](../deep_dives/2026-09-02_orthofoundation_preprocessing_qaf_answer.md)).
- **[M]** `ortho_lora` 0.8349 vs `ortho_dense` 0.8452 = −0.0103. LoRA is **CLOSED** (AXES §A LoRA row).

### OrthoDiffusion (3D diffusion U-Nets)
- **[L]** Licence verified MIT via HF `lanstat0123` (correction dated 2026-09-07). The 09-02 docs' "licence unknown" is superseded ([modern_knee](../deep_dives/2026-09-01_modern_knee_mri_models_beyond_mrnet.md)).
- **[L]** Paper recipe: t = 200/150/50, blocks mid_0/mid_2, SAP pooling, concat fusion, **0.908** macro AUROC on 8 labels. Only 5 of the 8 labels map to RSNA. Li et al. brain DDPM: middle blocks, t≈40-110, CATS fusion, LSE pooling, fine-tune the backbone ([diffusion_features](../deep_dives/2026-09-02_diffusion_feature_extraction_for_knee_mri_classification.md)).
- **[M]** Frozen LP-FT stage one read 0.7537 and was declined ([modern_knee](../deep_dives/2026-09-01_modern_knee_mri_models_beyond_mrnet.md)). Seat value +0.0001 ([qea](../deep_dives/2026-09-04_qea-volume-native-ssl-knee-mri-literature.md)).
- **[M]** **CLOSED at Stage II, 2026-09-25**: `orthodiff_ft_s2026` **0.8054 ledger75** vs control **0.9006 ledger75** (−0.0952). Plateau by e4 of 6. The probe trails on all 12 labels (AXES §A OrthoDiffusion row). This supersedes every "most actionable 3D candidate" line ([foundation_web](../deep_dives/2026-09-02_foundation_models_for_knee_mri_web_search.md), [sota_pretrained](../deep_dives/2026-09-02_sota_pretrained_knee_mri_models_2025_2026.md)).

### Radiology-general trunks
- **[L]** RadioDino (Snarci) is DINO ViT-B/16, S/16, S/8 trained on RadImageNet. MRI is ~70% of studies and knee is a listed region. BiomedCLIP has no MRI/MSK evaluation. Recommendation: RadioDino, with s8 as a patch-size arm ([qdp](../deep_dives/2026-09-04_qdp-biomedclip-vs-radiodino-trunk-choice.md)).
  - The licensed-trunks doc instead lists RadioDino-b16 as ~840k **chest X-rays** ([licensed_trunks](../deep_dives/2026-09-04_licensed-medical-trunk-encoders-for-kaggle.md)). See Contradictions.
- **[L]** Licences:
  - RadioDino HF: `cc-by-4.0` ([qdq](../deep_dives/2026-09-04_qdq-radiodino-shippability-radimagenet-license.md)).
  - OmniRad README says CC BY 4.0 but its `config.json` says `cc-by-nc-4.0` → rejected ([qdr](../deep_dives/2026-09-04_qdr-omnirad-vs-radiodino-trunk-choice.md)).
  - MedSAM mirror `flaviagiammarino/medsam-vit-base` is Apache, uses identity normalisation and 1024-px position embeddings ([licensed_trunks](../deep_dives/2026-09-04_licensed-medical-trunk-encoders-for-kaggle.md)).
- **[M/U]** Packaging: RadioDino b16/s16/s8 are timm-keyed but their config says `ViTModel`, which silently loads a random trunk. BiomedCLIP is nested under `visual.trunk.*`. RadImageNet mirrors wrap keys in `_orig_mod.`/`backbone.`. MRI-CORE is SAM-keyed and GitHub-only ([qdo](../deep_dives/2026-09-04_qdo-medical-trunk-key-naming-audit.md)).
- **[L]** Read MedSAM from the raw 768-d tokens, not the 256-ch neck. SAMantics neck probe gives 11.06% vs DINOv2 77.43% on IN1k. FODA-PG drops the neck. Raptor: SAM 0.850 vs MedSAM 0.872 ([sam_neck](../deep_dives/2026-09-04_sam-neck-vs-raw-tokens-for-classification.md)).
- **[M]** RAD-DINO is **CLOSED**: 0.8141 vs its exact `base_dense_s2026` control 0.8512 = −0.037. Only the init differs between the two arms (AXES §A RAD-DINO row). MRI-CORE is **CLOSED** at 0.8390 (AXES §A).
- **[L]** No permissive knee-MRI encoder exists ([licensed_trunks](../deep_dives/2026-09-04_licensed-medical-trunk-encoders-for-kaggle.md)).

### RadImageNet DenseNet-121
- **[L]** MIT. torchvision `features` keys under `backbone.0.*`, 725 entries. Preprocessing conventions conflict: caffe BGR, [-1,1] BGR, or ImageNet RGB. The Nilay ensemble +0.031 has no DenseNet ablation, so it is not attributable to that branch ([radimagenet_plumbing](../deep_dives/2026-09-01_radimagenet_densenet_plumbing_feasibility.md)).
- **[L]** Mei 2022 (2D single-slice MRNet): ACL 0.97 vs 0.91 (+4.8%), meniscus 0.96 vs 0.92 (+4.5%).
  - Frees 2025 finds no confirmed gain; its best ACL model, 0.9969, is ImageNet-init.
  - Juodelyte 2023: ImageNet wins 6 of 7 datasets; knee is comparable.
  - There is no multi-slice or 3D comparison ([qri](../deep_dives/2026-09-04_qri-radimagenet-vs-imagenet-knee-mri.md)).
- **[M]** The 0.6716 collapse at backbone LR 4e-4 (frozen ridge probe 0.8197) was first diagnosed as an LR problem, with a proposed fix of LP-FT at 1e-5 ([qbm](../deep_dives/2026-09-02_qbm-gentle-finetune-radnet.md)).
  - **Superseded** by [frozen_bn](../deep_dives/2026-09-04_frozen-batchnorm-drift-in-transfer-learning.md): `num_batches_tracked = 226711` shows the BN running stats were overwritten, so 0.6716 is **void**. Fixes: `.eval()`, `norm_eval`, or AdaBN.
- **[M]** **CLOSED**; this closes the conv trunk lane empty (AXES §A RadImageNet row, README:6685).

## 3. Continued SSL / in-domain pretraining

- **[L]** Zhang 2024 (fastMRI ~10k exams → MRNet, 2D-slice SSL), best in-domain MTL vs ImageNet: **+0.073 @80, +0.046 @200, +0.002 @400, −0.005 @800** subjects. MAE in-domain 0.624 @800. The objective must be anatomy-aware ([qej](../deep_dives/2026-09-04_qej-in-domain-mri-pretraining-for-knee-classification.md)).
  - AXES §C "external data for SSL pretraining" logs this as a **published wash** at our ~4,400 studies (README:7436).
- **[L]** CXR scaling (Sharma): MI2 beats open weights at 30k, RAD-DINO at 100k. On 32×H100: 30k = 0.33 h … 3.5M = 40.83 h. More in-domain data **hurt** on out-of-domain VinDR. All positive results are objective-matched ([ssl_continuation](../deep_dives/2026-09-01_ssl_continuation_from_strong_init.md), [continued_ssl](../deep_dives/2026-09-04_continued-ssl-on-in-domain-medical-corpus.md)).
- **[L]** Cost: ExPLoRA gives +1.2 pp at ~100 GPU-h and helps from 24 GPU-h. No DINO continuation shows a gain below 12 GPU-h. DIET-CP (<10 min) gives +4.81 LP F1 on MedMNIST under a different objective. Advice: a 30-minute probe smoke before any run ([qep](../deep_dives/2026-09-04_qep-continued-pretraining-dinov2-single-gpu.md)).
- **[L/U]** Base-size continuation: ~200-400 GPU-h (1-2 weeks) on the 5090. It is not literature-supported, and licence is the binding constraint: RSNA pixels are shippable, OAI depends on the DUA, fastMRI is out. 150k slices is a floor ([qdn](../deep_dives/2026-09-04_qdn-knee-continuation-cost-at-base-size.md)).
- **[L]** 3D/volume-native knee SSL: the only paper is the ISMRM 2025 #4014 abstract, with no numbers ([qea](../deep_dives/2026-09-04_qea-volume-native-ssl-knee-mri-literature.md)).
- **[M]** AXES §A: continued multi-crop SSL is **CLOSED at this corner, not as a class**. RAD-DINO is exactly that mechanism and loses −0.037. AXES §C masked adaptation / SSL continuation is **CLOSED**: every masked adaptation switched attention off (README:11389).
- **[L]** Joint vs sequential co-training with a foreign corpus: no direct precedent. Use dataset-specific batches, masked losses and no size reweighting (Yan 2021). Sequential beat joint by +0.367 MCC when the batch was pooled 80:1. **[M]** Sequential MRNet pretraining here was flat-to-negative ([qct](../deep_dives/2026-09-04_qct-joint-vs-sequential-cotraining.md)).
- **Solo implication:** continuation spends card on a ~0 published delta at our labelled scale. Treat it as an instrument only.

## 4. Fine-tuning recipe: probe vs FT, LP-FT, rates

- **[L]** Probe/FT rank inversion is the known protocol mismatch. Knott: LP r≈0.90, FT r≈0.79. A 10%-shot FT predicts FT rank better than the probe does ([qch](../deep_dives/2026-09-03_qch-probe-vs-finetune-inversion.md), [qdk](../deep_dives/2026-09-04_qdk-continued-ssl-probe-ft-inversion.md)).
- **[L]** Kumar LP-FT: +1% ID / +10% OOD vs FT; features change ~20× less. LLRD 0.65-0.85 moves IN1k ~0.3-0.6 pp. Surgical FT depends on the shift type ([ood_vit](../deep_dives/2026-09-05_ood-vit-finetune-schedules.md)).
- **[L]** There is no LP-FT precedent with a nonlinear/attention head or a slice-pooled model. Stage-1 length is a heuristic ([qbn](../deep_dives/2026-09-02_qbn-lpft-nonlinear-head.md), [qci](../deep_dives/2026-09-03_qci-lpft-schedule.md)).
- **[M]** LP-FT is **CLOSED**: its ID gain is a convergence-speed effect that a long fine-tune erases (AXES §C, README:7841). Warm start is flat at two rates (README:5308). So the LP-FT recommendations in qdk, qci and ood_vit are **superseded by measurement**.
- **[M]** Readout block depth: block 9 of 12 reads −0.0227 (AXES §B, README:1200).
- **[M]** Ramp/curriculum family: +0.0077 and +0.0068 on the disagreed recipe. It fails to transfer to the dense corner: `ramp_dense_gate` 0.8579 vs 0.8634 = −0.0055. 0 of 4 clear 0.015, 0 board seats, ~11.25 card-h ([qbw](../deep_dives/2026-09-02_qbw-ramp-family-payoff-and-cost.md)).

## 5. 3D-native vs 2D slice-pooled

- **[L]** MRNet (2D AlexNet + max): 0.937 / 0.965 / 0.847. Mead review: 2D+3D 0.934, 2D 0.931, 3D 0.871. MST: DINOv2 + slice transformer 0.85 vs 3D ResNet 0.69 ([3d_path](../deep_dives/2026-09-01_3d_knee_mri_path.md)).
- **[L]** Namiri: 2D 92% / κ0.83 vs 3D 89% / κ0.83, p=0.27. AnyMC3D: 2D 0.894 vs MedicalNet 0.783 / VoCo 0.793. MedicalNet (MIT) and SwinUNETR (Apache) are shippable. No 2D-vs-3D decorrelation evidence exists → **axis CLOSES** ([ads](../deep_dives/2026-09-04_ads-volumetric-native-encoders.md)).
- **[M]** Fusion is saturated: attend 0.8476, mean 0.8473, depth 0.8461, max 0.8457 (span 0.0019 < 0.0062). 20 vs 12 slices −0.0001. extended_slots +0.0002 ([ads](../deep_dives/2026-09-04_ads-volumetric-native-encoders.md)). Slice count 24→36 −0.0224 (AXES §B).
- **[M]** V-JEPA2 `vjepa2l` 0.8343, −0.017 vs DINOv2 base ([3d_path](../deep_dives/2026-09-01_3d_knee_mri_path.md)). **CLOSED** as a tie on frozen features (AXES §A).
- **[L]** 3D literature: MR-Transformer 0.89-0.91 OAI / 0.82 MOST, 3DMeT 0.63-0.82, MedNet-FS ACL 0.76/0.62 ([modern_3d](../deep_dives/2026-09-02_modern_3d_knee_mri_models_beyond_mrnet.md)).
- **[L]** MARS: 3D Swin, SSL on 336,476 MRI volumes including knee, Apache-2.0, 78.3 MiB. Hard 1 mm isotropic 96³ contract, which destroys thin-slice anisotropy ([mars](../deep_dives/2026-09-06_efficiency-formula-and-mars-census.md)).

## 6. Efficiency track (runtime as a trunk constraint)

- **[L]** The first reading of the formula was `(1−AUC) + Runtime/32400`, inferred from a leaderboard join ([efficiency_frontier](../deep_dives/2026-09-01_efficiency_frontier_knee_mri.md)).
  - **Superseded** by the official page: `AUC/(Benchmark−maxAUC) + Runtime/32400`. 1 h ≈ 0.05 AUC-equivalent. The public LB shows rank only ([efficiency_formula](../deep_dives/2026-09-06_efficiency-formula-and-mars-census.md)).
- **[L]** Levers: FP16, TensorRT (~2×), a single student, and IO (pylibjpeg-rle ~20× faster than pydicom). No AUC-vs-escalation curve exists for cascades ([efficiency_frontier](../deep_dives/2026-09-01_efficiency_frontier_knee_mri.md)).
- **[M]** Efficiency lane **CLOSED**: 0.001 AUC = 70.9 s (AXES §D).
- **Solo implication:** one DINOv2-base forward per slice at 224 is already the efficient shape.

---

## Open questions

1. Error correlation of a cheap MedicalNet-ResNet10 member vs the best 2D member. This is the only named re-opener of the 3D axis ([ads](../deep_dives/2026-09-04_ads-volumetric-native-encoders.md)).
2. SSL continuation on RSNA pixels only (shippable), with a DINO objective and a 30-min probe smoke. Unpriced; the prior is ~0 ([qep](../deep_dives/2026-09-04_qep-continued-pretraining-dinov2-single-gpu.md), [qej](../deep_dives/2026-09-04_qej-in-domain-mri-pretraining-for-knee-classification.md)).
3. RadImageNet re-run with BN frozen. The only reading (0.6716) is void; the AXES closure rests on README:6685, not on this doc chain ([frozen_bn](../deep_dives/2026-09-04_frozen-batchnorm-drift-in-transfer-learning.md)).
4. RadioDino (RadImageNet-trained, cc-by-4.0) vs RAD-DINO (CXR). AXES closes RAD-DINO at −0.037. Whether a RadImageNet-trained DINO was ever run separately is not stated in these docs.
5. MedSAM raw-token read: no internal ablation exists ([sam_neck](../deep_dives/2026-09-04_sam-neck-vs-raw-tokens-for-classification.md)).
6. Joint co-training with MRNet under masked heads is untested; sequential was flat ([qct](../deep_dives/2026-09-04_qct-joint-vs-sequential-cotraining.md)).

## Contradictions between deep dives

- [qbm](../deep_dives/2026-09-02_qbm-gentle-finetune-radnet.md) vs [frozen_bn](../deep_dives/2026-09-04_frozen-batchnorm-drift-in-transfer-learning.md): qbm blames the LR for RadImageNet's 0.6716; frozen_bn voids it as BN drift.
- [radimagenet_plumbing](../deep_dives/2026-09-01_radimagenet_densenet_plumbing_feasibility.md) vs [licensed_trunks](../deep_dives/2026-09-04_licensed-medical-trunk-encoders-for-kaggle.md): the RadImageNet licence is MIT in one and "unclear / by request, not mountable" in the other.
- [qj](../deep_dives/2026-09-01_qj_convnext_rate_radimagenet_plumbing.md) vs [qdo](../deep_dives/2026-09-04_qdo-medical-trunk-key-naming-audit.md): "no HF mirror" vs two HF mirrors found (convergedmachine, Lab-Rasool).
- [qdp](../deep_dives/2026-09-04_qdp-biomedclip-vs-radiodino-trunk-choice.md) vs [licensed_trunks](../deep_dives/2026-09-04_licensed-medical-trunk-encoders-for-kaggle.md): RadioDino-b16 is RadImageNet (MRI-dominant) in one and ~840k chest X-rays in the other.
- [modern_knee](../deep_dives/2026-09-01_modern_knee_mri_models_beyond_mrnet.md) vs [qdn](../deep_dives/2026-09-04_qdn-knee-continuation-cost-at-base-size.md): OrthoFoundation is initialised from DINOv3-L in one and from DINOv2 weights in the other.
- [modern_knee](../deep_dives/2026-09-01_modern_knee_mri_models_beyond_mrnet.md) vs [sota_pretrained](../deep_dives/2026-09-02_sota_pretrained_knee_mri_models_2025_2026.md)/[modern_3d](../deep_dives/2026-09-02_modern_3d_knee_mri_models_beyond_mrnet.md): OrthoFoundation is "next experiment" in one and DECLINED in the others. The OrthoDiffusion licence is "unknown" in the 09-02 docs and MIT in the corrected 09-01 doc.
- [trunk_vs_readout](../deep_dives/2026-09-02_trunk_vs_readout_quality_medical_volumes_qal.md) vs [sota_pretrained](../deep_dives/2026-09-02_sota_pretrained_knee_mri_models_2025_2026.md): the DINOv3 probe deficit is a readout confound in one and real signal loss in the other.
- [qri](../deep_dives/2026-09-04_qri-radimagenet-vs-imagenet-knee-mri.md) vs [qdp](../deep_dives/2026-09-04_qdp-biomedclip-vs-radiodino-trunk-choice.md): qdp cites Mei +4.8/+4.5 as confirming radiology transfer; qri shows two replications failed to confirm it.
- [qdn](../deep_dives/2026-09-04_qdn-knee-continuation-cost-at-base-size.md) vs [qej](../deep_dives/2026-09-04_qej-in-domain-mri-pretraining-for-knee-classification.md): knee continuation is "plausible, one-card tractable" in one and "~0 gain at our labelled scale, not well-supported" in the other.
- [efficiency_frontier](../deep_dives/2026-09-01_efficiency_frontier_knee_mri.md) vs [efficiency_formula](../deep_dives/2026-09-06_efficiency-formula-and-mars-census.md): `(1−AUC)+Runtime/32400` vs the official `AUC/(Benchmark−maxAUC)+Runtime/32400`.
- [cnn_vit_aug](../deep_dives/2026-09-01_cnn_vit_augmentation_sample_efficiency.md) vs [qej](../deep_dives/2026-09-04_qej-in-domain-mri-pretraining-for-knee-classification.md): the corpus is "~1,000 studies" in one and ~4,400 studies in the other.

## Sources

- [2026-09-01_3d_knee_mri_path](../deep_dives/2026-09-01_3d_knee_mri_path.md)
- [2026-09-01_modern_knee_mri_models_beyond_mrnet](../deep_dives/2026-09-01_modern_knee_mri_models_beyond_mrnet.md)
- [2026-09-01_qj_convnext_rate_radimagenet_plumbing](../deep_dives/2026-09-01_qj_convnext_rate_radimagenet_plumbing.md)
- [2026-09-01_radimagenet_densenet_plumbing_feasibility](../deep_dives/2026-09-01_radimagenet_densenet_plumbing_feasibility.md)
- [2026-09-01_ssl_continuation_from_strong_init](../deep_dives/2026-09-01_ssl_continuation_from_strong_init.md)
- [2026-09-01_cnn_vit_augmentation_sample_efficiency](../deep_dives/2026-09-01_cnn_vit_augmentation_sample_efficiency.md)
- [2026-09-01_efficiency_frontier_knee_mri](../deep_dives/2026-09-01_efficiency_frontier_knee_mri.md)
- [2026-09-02_capacity_ceiling_trunk_family_qad](../deep_dives/2026-09-02_capacity_ceiling_trunk_family_qad.md)
- [2026-09-02_diffusion_feature_extraction_for_knee_mri_classification](../deep_dives/2026-09-02_diffusion_feature_extraction_for_knee_mri_classification.md)
- [2026-09-02_foundation_models_for_knee_mri_web_search](../deep_dives/2026-09-02_foundation_models_for_knee_mri_web_search.md)
- [2026-09-02_modern_3d_knee_mri_models_beyond_mrnet](../deep_dives/2026-09-02_modern_3d_knee_mri_models_beyond_mrnet.md)
- [2026-09-02_orthofoundation_preprocessing_qaf_answer](../deep_dives/2026-09-02_orthofoundation_preprocessing_qaf_answer.md)
- [2026-09-02_qbm-gentle-finetune-radnet](../deep_dives/2026-09-02_qbm-gentle-finetune-radnet.md)
- [2026-09-02_qbn-lpft-nonlinear-head](../deep_dives/2026-09-02_qbn-lpft-nonlinear-head.md)
- [2026-09-02_sota_pretrained_knee_mri_models_2025_2026](../deep_dives/2026-09-02_sota_pretrained_knee_mri_models_2025_2026.md)
- [2026-09-02_trunk_vs_readout_quality_medical_volumes_qal](../deep_dives/2026-09-02_trunk_vs_readout_quality_medical_volumes_qal.md)
- [2026-09-02_qbw-ramp-family-payoff-and-cost](../deep_dives/2026-09-02_qbw-ramp-family-payoff-and-cost.md)
- [2026-09-03_qch-probe-vs-finetune-inversion](../deep_dives/2026-09-03_qch-probe-vs-finetune-inversion.md)
- [2026-09-03_qci-lpft-schedule](../deep_dives/2026-09-03_qci-lpft-schedule.md)
- [2026-09-04_ads-volumetric-native-encoders](../deep_dives/2026-09-04_ads-volumetric-native-encoders.md)
- [2026-09-04_continued-ssl-on-in-domain-medical-corpus](../deep_dives/2026-09-04_continued-ssl-on-in-domain-medical-corpus.md)
- [2026-09-04_frozen-batchnorm-drift-in-transfer-learning](../deep_dives/2026-09-04_frozen-batchnorm-drift-in-transfer-learning.md)
- [2026-09-04_licensed-medical-trunk-encoders-for-kaggle](../deep_dives/2026-09-04_licensed-medical-trunk-encoders-for-kaggle.md)
- [2026-09-04_qct-joint-vs-sequential-cotraining](../deep_dives/2026-09-04_qct-joint-vs-sequential-cotraining.md)
- [2026-09-04_qdk-continued-ssl-probe-ft-inversion](../deep_dives/2026-09-04_qdk-continued-ssl-probe-ft-inversion.md)
- [2026-09-04_qdn-knee-continuation-cost-at-base-size](../deep_dives/2026-09-04_qdn-knee-continuation-cost-at-base-size.md)
- [2026-09-04_qdo-medical-trunk-key-naming-audit](../deep_dives/2026-09-04_qdo-medical-trunk-key-naming-audit.md)
- [2026-09-04_qdp-biomedclip-vs-radiodino-trunk-choice](../deep_dives/2026-09-04_qdp-biomedclip-vs-radiodino-trunk-choice.md)
- [2026-09-04_qdq-radiodino-shippability-radimagenet-license](../deep_dives/2026-09-04_qdq-radiodino-shippability-radimagenet-license.md)
- [2026-09-04_qdr-omnirad-vs-radiodino-trunk-choice](../deep_dives/2026-09-04_qdr-omnirad-vs-radiodino-trunk-choice.md)
- [2026-09-04_qea-volume-native-ssl-knee-mri-literature](../deep_dives/2026-09-04_qea-volume-native-ssl-knee-mri-literature.md)
- [2026-09-04_qej-in-domain-mri-pretraining-for-knee-classification](../deep_dives/2026-09-04_qej-in-domain-mri-pretraining-for-knee-classification.md)
- [2026-09-04_qep-continued-pretraining-dinov2-single-gpu](../deep_dives/2026-09-04_qep-continued-pretraining-dinov2-single-gpu.md)
- [2026-09-04_qri-radimagenet-vs-imagenet-knee-mri](../deep_dives/2026-09-04_qri-radimagenet-vs-imagenet-knee-mri.md)
- [2026-09-04_sam-neck-vs-raw-tokens-for-classification](../deep_dives/2026-09-04_sam-neck-vs-raw-tokens-for-classification.md)
- [2026-09-05_ood-vit-finetune-schedules](../deep_dives/2026-09-05_ood-vit-finetune-schedules.md)
- [2026-09-06_efficiency-formula-and-mars-census](../deep_dives/2026-09-06_efficiency-formula-and-mars-census.md)
- `docs/AXES.md` §A-§D rows cited inline (not a deep dive).

## KneePreM (arXiv 2609.31461, 2026-09-25): barred twice, and weak on its own numbers

- **[L] What it is:** a 3D U-Net masked autoencoder pretrained on 19,011 unlabelled OAI knee MRI series (4,791 participants).
- **[L] Reported numbers:**
  - fastMRI+ classification: ROC AUC 0.722, using 50% of the training data.
  - APM eight-target classification: 0.740, using 70% of the data.
  - Compared against random init and SuPreM. Weights and code are not stated on the abstract page.
- **Barred for shipping:** OAI is under an NDA/DUA, not public domain ([external data and licensing summary](external-data-and-licensing.md)). An OAI-derived trunk therefore falls under the 09-15 ruling that anything derived from a foreign corpus is instrument-only.
- **Weak as an instrument:** it is the same class as the OrthoDiffusion 3D U-Net trunk. That trunk was CLOSED at Stage II: 0.8054 against 0.9006 on ledger75 (docs/AXES.md), and it was fine-tuned. KneePreM's own AUCs of ~0.72-0.74 give no reason to expect it to do better.
- **Verdict:** do not queue it. Re-open only if weights ship under a licence the owner clears AND a reported knee number is within ~0.02 of our trunk.

Deep dive: [2026-09-28 gap audit and frontier sweep](../deep_dives/2026-09-28_gap-audit-and-frontier-sweep.md).
