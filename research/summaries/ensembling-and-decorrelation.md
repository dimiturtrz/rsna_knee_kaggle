# Ensembling and decorrelation — topic summary

**Current verdict (2026-09-25).** The ensemble axis is spent: worth +0.011 and closed on the board
([AXES §G](../../docs/AXES.md)); seat count plateaus from three, a learned combiner loses to the flat
vote, diversity as a selection rule is refuted. The literature agrees: in the high-capacity regime,
diversity bought on purpose costs solo quality, and simple means match stacking once members correlate.
The deliverable is now a **solo checkpoint**, so what survives from this topic is its implications for
one model (§7), not seat levers.

Legend: **[ours]** measured in this repo · **[lit]** published · **[unverified]** asserted/prose-only or
extrapolated. Rulers are quoted where the source names one; an unlabelled repo number is midpoint
(AXES §I).

---

## 1. Can a member be decorrelated without paying solo quality?

- **[lit] No, not near SOTA solo.** Every published decorrelation family moves the trade-off, none removes
  it. NCL/GNCL: diversity rises monotonically with lambda while member accuracy falls; for large-capacity
  members optimal lambda is 0-0.7. DNCC members sit 60-90% under a 94.26% ensemble. EDDE/DivBO get more
  diversity per unit accuracy (EDDE 74.38% vs 72.17%; DivBO's single learner is *worse* than plain BO).
  DICE: 5 DICE nets match 7 independent. Gradient/feature decorrelation (ADP/TRS) buys robustness, not
  clean accuracy. Joint training collapses (learner collusion, Jeffares 2023)
  ([decorrelation_without_solo_cost](../deep_dives/2026-09-01_decorrelation_without_solo_cost.md) §2).
- **[lit] Large-scale answer:** Abe et al. 2024 — encouraging predictive diversity *harms* large-member
  ensembles; ImageNet ResNet-18 standard 71.49% vs strong diversity (gamma -0.9) 70.16% (~1.3 pt drop)
  ([same](../deep_dives/2026-09-01_decorrelation_without_solo_cost.md) §3).
- **[ours] The trade-off is a wall on our pool.** `seat_value --against honest12` over 25 members: 3 of 25
  DISTINCT members clear the 0.8521 size bar, all at rank corr 0.88-0.89; everything genuinely
  decorrelated is 0.02-0.03 under; `focus_mcl` (rank 0.0445, solo 0.5332) is distinct because broken.
  Board: `distinct14` read **0.937** = control `honest12` 0.937, inside the pre-registered 0.936-0.938
  FLAT band ([same](../deep_dives/2026-09-01_decorrelation_without_solo_cost.md) §6-7). Midpoint ruler
  (not stated; repo default).
- **[ours] Apparent exception, not a rescue.** `vjepa_warm` is more distant than the LoRA rows *and*
  stronger solo — but seated at **-0.0007**, band spanning zero
  ([qdl](../deep_dives/2026-09-04_qdl-geometry-diversity-vs-objective-diversity.md) §4). AXES: V-JEPA
  CLOSED, "the finding is the DISTANCE, not the score" (§A).
- **[ours/AXES]** Objective diversity is not a seat lever (`dissent_seat`, rank/dissent 38th of 66;
  AXES §D, README:463). Diversity as a SELECTION rule refuted (AXES §G, README:5489, :3274).

## 2. Does a different trunk family decorrelate, and at what solo price?

- **[lit] Heterogeneous families can add without a solo penalty — at mid-quality.** COVID-19 chest CT
  (arXiv 2603.14621): DINOv2 ViT-B/14 best single macro F1 **0.8969**; 9-model ensemble (+RadImageNet
  DenseNet-121 + 7 MIL CNNs) **0.9280** (+0.031). Reports ensemble gain, **no correlation coefficients**
  ([qi](../deep_dives/2026-09-01_qi_cross_family_decorrelation_medical_volumes.md);
  [cnn-vs-vit](../deep_dives/2026-09-04_cnn-vs-vit-ensemble-decorrelation-medical-imaging.md)).
  ChestX-ray14 CNN+ViT+hybrid AUROC 0.842 → 0.854 (+0.012); stroke CT +0.46 pp
  ([cnn-vs-vit](../deep_dives/2026-09-04_cnn-vs-vit-ensemble-decorrelation-medical-imaging.md)).
- **[lit] General vision:** CNN+ViT ensembles more diverse than intra-family (Filipiuk & Singh 2022);
  ConvNeXt-B + ViT-B beats ConvNeXt-B + ConvNeXt-XL — but diversity does not help when one member is
  >~5 pp weaker ([cnn-vs-vit](../deep_dives/2026-09-04_cnn-vs-vit-ensemble-decorrelation-medical-imaging.md)).
- **[lit] MRI family ranking is unsettled.** Lumbar stenosis: VGG19+LR best external (acc 0.9556), DINOv2
  features generalise worse (LightGBM 0.6222); knee OA: CVAE 0.670 / DenseNet-121 0.656 / ResNet-50 0.651
  ([qi](../deep_dives/2026-09-01_qi_cross_family_decorrelation_medical_volumes.md)). No knee-MRI study
  reports pairwise error correlation across families — **[unverified]** any family ranking for decorrelation.
- **[lit] Corpus-only diversity does not buy prediction diversity.** Juodelyte (TMLR 2023): ImageNet vs
  RadImageNet representations diverge after fine-tuning but **predictions stay similar**. Khan (ML4H 2023)
  is the only same-arch/same-SSL corpus-varied ensemble (ViT masked MAE + Medical MAE 81.0 vs natural MAE
  79.3) — no seat gain reported ([qeo](../deep_dives/2026-09-04_qeo-corpus-diversity-ensemble.md) §1).
- **[ours] RadioDino solo cost is modality mismatch, not capacity:** `radiodino_s16` -0.0394 vs `native224`
  0.8502; base width -0.0241 — widening did not close it; consistent with Krones 2026 (mismatched medical
  SSL hurts vs ImageNet by -0.005 to -0.017)
  ([qeo](../deep_dives/2026-09-04_qeo-corpus-diversity-ensemble.md) §2).
- **[ours/AXES] Every off-DINO family we trained is CLOSED:** ConvNeXt, ResNet, RadImageNet DenseNet,
  Swin, ViT-in21k, RAD-DINO (-0.037 vs its exact control), OrthoDiffusion Stage II (0.8054 ledger75 vs
  0.9006), architecture monoculture as a seat lever "negative six of six" (AXES §A, README:10988). The CNN
  per-label complement: `resnet_224_long` behind on 9 of 12, ahead nowhere significantly, 50/50 average
  **-0.0025 [-0.0067, +0.0019]** vs DINO alone (AXES §A, knee-b39kg). The headline: the stack is
  DINOv2-specific, +0.054 on a ViT, 0.000 on a CNN (AXES §A, README:17307).

## 3. How weak can a member be and still pay?

- **[lit] The "~5 pp" boundary is one paper, absolute, accuracy-scale.** Filipiuk & Singh: MLP-Mixer L,
  5 pp below B, makes L+B "slightly better" (no numeric delta); its diversity mostly comes from its higher
  error rate. No AUC conversion, no leave-one-out ablation exists
  ([qef](../deep_dives/2026-09-04_qef-weak-member-ensemble-boundary.md) §1-2).
- **[lit] Ensemble size:** a weak member's damage and a decorrelated member's benefit are both capped by
  1/N (Durrant & Lim 2020; Wood et al. 2023 bias-variance-diversity)
  ([qef](../deep_dives/2026-09-04_qef-weak-member-ensemble-boundary.md) §3). Rank-combiner tolerance of a
  miscalibrated weak member: **[unverified]** conjecture.
- **[lit] Complementarity > accuracy in principle** (Zhou & Feng, Deep Forest). Closest knee-MRI case:
  MobileNetV2 96.85% + Swin 98.12% hybrid 99.94% — but one hybrid network, MRI *plane classification*.
  MURA calibrated ensemble explicitly **excluded** the weakest backbone (ConvNet AUC 0.88)
  ([qem](../deep_dives/2026-09-04_qem-non-dino-trunk-and-weak-member-ensemble.md)).
- **[ours] RadioDino trigger case:** `radiodino_s16` ~0.8105 vs `band_dense` 0.8578, a 4.7-point gap —
  right on the only published boundary; recommendation was an empirical leave-one-out gate
  ([qef](../deep_dives/2026-09-04_qef-weak-member-ensemble-boundary.md)). Operational gate proposed:
  marginal > +0.0005, per-label error corr < 0.95, per-label blind-spot proof if solo > 0.03 below best
  ([qem](../deep_dives/2026-09-04_qem-non-dino-trunk-and-weak-member-ensemble.md)) — **not literature-derived**.
- **[ours/AXES] Superseded by measurement:** 0 of 12 per-finding columns in the pool clear their own 95%
  paired interval over the shipping solo (AXES §G, `2026-09-18_the_pool_holds_no_per_finding_specialist.md`).
  AXES re-opener for off-axis members: "must be COMPETITIVE first", within ~0.01 of the champion (§A).

## 4. Combiners: stacking vs averaging

- **[lit] Once members correlate, a mean matches stacking.** Nakata 2023 liver US: best single 0.719;
  stacking ST9 0.776; weighted average WAV7 **0.783**; soft vote SV16 0.776; 112 of 120 pairs ≥ 0.4
  correlation, ResNet50-ResNet101 0.746. Mohtasham 2026: >0.75 correlation "add redundancy and weaken
  ensemble gains". Müller 2022 (12 pooling functions, 4 datasets): "simple statistical pooling functions
  are equal or often even better"
  ([stacking_vs_averaging](../deep_dives/2026-09-02_stacking_vs_averaging_correlated_members_qag_answer.md);
  [qel](../deep_dives/2026-09-04_qel-ensemble-gain-and-combiners.md)).
- **[lit]** The only medical case where a learned combiner clearly helped: Hussain 2021 MURA per-body-part
  routing, because base models were genuinely **specialised**
  ([qel](../deep_dives/2026-09-04_qel-ensemble-gain-and-combiners.md)).
- **[ours]** `competence_axis`: 81.7-82% of what separates 105 members is one scalar
  ([stacking_vs_averaging](../deep_dives/2026-09-02_stacking_vs_averaging_correlated_members_qag_answer.md)).
- **[ours/AXES]** Learned combiner LOSES to the flat vote, re-read after its bug fix (§G, README:7543,
  :11041). Averaging is a CONTRACTION — the pool cannot produce a new board point at any seating (§G,
  README:15590). Serve-time column mixing NULL (§H).
- **[lit] Rank vs probability averaging:** used by SIIM-ISIC 2020 winners (18 models), delta **never
  reported** ([past_ensemble_plateau](../deep_dives/2026-09-01_past_ensemble_plateau_mechanisms.md) §3).

## 5. How big is an ensemble gain, and what is left after saturation?

- **[lit] Gain scales inversely with single-model strength.** Mammography +0.03 AUC (0.88→0.91;
  0.95→0.98); MURA 0.92→0.93; Müller stacking +13% F1 ISIC / +12% DRD, +1-2% on easy sets. MRNet
  itself is not a model ensemble; MRNet leaderboard single-model top 0.917
  ([qel](../deep_dives/2026-09-04_qel-ensemble-gain-and-combiners.md)).
- **[lit/field] This competition:** public CoAtNet 384 / Swin-base 384 / EffNetV2-L 480 three-family blend
  worth **~+0.001 AUC** on the live board; rival self-reported singles at **0.942** vs our six-member
  ensemble 0.939 (later 0.943-0.944) — "the entire mix was buying roughly what one better single member
  would" ([qel](../deep_dives/2026-09-04_qel-ensemble-gain-and-combiners.md);
  [qem](../deep_dives/2026-09-04_qem-non-dino-trunk-and-weak-member-ensemble.md)). Board ruler.
- **[ours]** Our offline +0.016 over best member is "well inside the published range"
  ([qel](../deep_dives/2026-09-04_qel-ensemble-gain-and-combiners.md)); the board figure the AXES row keeps
  is +0.011, spent (§G).
- **[lit] Post-ensemble levers, ranked by published delta**
  ([qbo](../deep_dives/2026-09-02_qbo-what-moves-after-ensemble-saturation.md)):
  1. external labelled data — MRNet ACL 0.824 → 0.911 (+0.087) with external-site labels; +3.5% / +2.5%
     cross-dataset DA;
  2. input geometry — CoPAS ROI crop +0.056 mean AUC (0.725→0.781); adjacent vs equal-spaced slices +0.018
     fold-0 OOF (competitor forum);
  3. label engineering — +0.04 AUROC for +0.15 labeler F1 (CheXpert); AUC-margin >+2%;
  4. TTA / multi-view — multi-view +1.0% ACL, per-plane +2-3%; standard TTA can **hurt** (Medeiros 2026,
     up to -31.6 pp on MedMNIST).
  The project's readings: external-corpus seat axis flat at one/two/four/six seats; crop already at
  130 mm; label ceiling small; planes already fused by `SlotHead`. The one unpriced cell then was slice
  coverage (`stack224`).
- **[lit] Earlier, softer ranking** put per-source threshold calibration first (+0.14 F1 on a
  per-source-F1 metric, arXiv 2603.14621) and said TTA "almost always helps" over an ensemble at a few
  parts per thousand ([past_ensemble_plateau](../deep_dives/2026-09-01_past_ensemble_plateau_mechanisms.md)).
  **Superseded** for this task: AUC reads order only, so calibration/thresholds are "worth zero by
  construction" (AXES §H, knee-e8a); overlap TTA already consumed, band TTA loses to its single band
  (AXES §H). Prefer [qbo](../deep_dives/2026-09-02_qbo-what-moves-after-ensemble-saturation.md).
- **[ours/AXES] Later closures of qbo's list:** external data as labelled augmentation CLOSED
  (`noisy_student` 0.8548); training on Rijeka/MRNet/fastMRI/SKM-TEA BARRED by licence, not refuted;
  label table/correction/masking CLOSED (AXES §E).

## 6. Pricing members you cannot hold out; foreign members

- **[lit] Full-fit members are priced by convention.** Standard Kaggle: price via OOF/fold-0 sibling,
  refit, assume ordering transfers — **not formally measured** anywhere (Caruana 2004 does not isolate it;
  Bates 2023 proves CV estimates a different quantity). Scaling-law bound: refit gain low thousandths at
  k=5, so the `honest12` 0.937 → `raptorweight_v1` 0.944 gap "is not mostly a refit effect"
  ([qek](../deep_dives/2026-09-04_qek-pricing-full-fit-ensemble-members.md)) — **[unverified]** estimate.
  **[ours/AXES]** `--fold -1` refit measured at ~+0.004, spread as wide as the mean (§F, README:17492),
  consistent with the bound.
- **[ours] Foreign census:** ≥7 structurally distinct permissive families beyond the seated CoAtNet
  (`raptor-knee-widedense`, CC0). Pillar-0 is the only true-3D checkpoint (ECL-2.0; gold-58 macro 0.797,
  per-label 0.630 MCL to 0.965 Effusion; held-out claim **prose-only**); `raptor-knee-arms` adds
  ConvNeXt-v2-B / EffNetV2-L on existing decode; DINOv3-derived CC0 metadata is overridden by the DINOv3
  License, DINOv2 derivatives are CC-BY-NC-4.0
  ([qdu](../deep_dives/2026-09-04_qdu-foreign-architecture-census.md)).
- **[ours]** `raptor_v4_swa` (CoAtNet) is **98.6% in-sample** (`foreign_overlap`), cannot be priced offline
  ([qem](../deep_dives/2026-09-04_qem-non-dino-trunk-and-weak-member-ensemble.md)).
- **[ours/AXES]** Foreign WEIGHTS as members CLOSED; cross-program equal-weight join negative (§E,
  README:9094, :8940). Foreign teacher/distillation bounded below the gap (+0.0056 under the 0.0062 floor)
  and "DESTROYS distance instead of importing it" (§E). EfficientNetV2-L NULL (§A).

## 7. What this implies for a SOLO checkpoint

- **Buy quality, not distance.** Every doc's conclusion reduces to: in this regime, ensemble the strongest
  independent models; forced decorrelation costs solo (§1). For a solo, a decorrelation loss is pure cost —
  AXES §D states it directly: "a loss that buys diversity and costs solo quality buys nothing at all".
- **The ensemble ceiling ≈ one better single.** Field evidence (§5) says a cross-family blend is ~+0.001 and
  rival singles already reach 0.942; the +0.011 ensemble axis is roughly the size of the solo gap to close.
  The spent ensemble is therefore a **target for the solo**, not a lever.
- **Distillation is the bridge the docs name** (EnD2/Hydra compress an ensemble's disagreement into one
  student; [decorrelation_without_solo_cost](../deep_dives/2026-09-01_decorrelation_without_solo_cost.md)
  §2.6) — but the repo measures foreign-teacher headroom at 0.002 and no teacher above its student
  (AXES §E). Self-ensemble distillation from our own pool is not evaluated in these 13 docs.
- **Weight-space averaging is the solo form of ensembling**: one-shot basin soup net +0.0014 (r1), must be
  the LAST operation; uniform top-K soup negative (AXES §F). This is what replaced seats.
- **Complement-by-label is ruled out for the solo too**: no pool member beats the shipping solo on any
  column (AXES §G), and the CNN complement fails on MCL (AXES §A) — so no per-label routing inside one model
  is licensed by pool evidence.
- **qbo's post-saturation list is the solo list** (external labels, input geometry, labels) — and each is now
  closed, barred, or already consumed in AXES §B/§E; slice coverage / `multiscale` under labelwise is the
  one geometry cell AXES still lists UNREAD.

---

## Open questions

- AXES is internally inconsistent: §A closes "does the CNN fill the ViT's blind spots" card-free
  (knee-b39kg), while "Open, and unread" still lists "ResNet-vs-DINO decorrelation read — free, unrun".
- Is geometry diversity (2D vs 3D) genuinely cheaper than objective diversity? One point (`vjepa_warm`,
  seat -0.0007); qdl's proposed 2-geometry sub-ensemble was never run, and OrthoDiffusion (the named second
  point) closed at Stage II. Moot for a solo unless it transfers into one model.
- No published AUC-scale weak-member boundary, and no measured rank-vs-probability averaging delta.
- The fold-0 → full-fit ordering-transfer assumption remains unmeasured anywhere.
- Pillar-0's held-out claim is prose-only; unverifiable from the artefact.
- Author attribution of arXiv 2603.14621 differs across docs ("Nilay et al." vs "Thapar et al.") —
  unverified which is correct.

## Sources

- [2026-09-01_decorrelation_without_solo_cost](../deep_dives/2026-09-01_decorrelation_without_solo_cost.md)
- [2026-09-01_past_ensemble_plateau_mechanisms](../deep_dives/2026-09-01_past_ensemble_plateau_mechanisms.md)
- [2026-09-01_qi_cross_family_decorrelation_medical_volumes](../deep_dives/2026-09-01_qi_cross_family_decorrelation_medical_volumes.md)
- [2026-09-02_qbo-what-moves-after-ensemble-saturation](../deep_dives/2026-09-02_qbo-what-moves-after-ensemble-saturation.md)
- [2026-09-02_stacking_vs_averaging_correlated_members_qag_answer](../deep_dives/2026-09-02_stacking_vs_averaging_correlated_members_qag_answer.md)
- [2026-09-04_cnn-vs-vit-ensemble-decorrelation-medical-imaging](../deep_dives/2026-09-04_cnn-vs-vit-ensemble-decorrelation-medical-imaging.md)
- [2026-09-04_qdl-geometry-diversity-vs-objective-diversity](../deep_dives/2026-09-04_qdl-geometry-diversity-vs-objective-diversity.md)
- [2026-09-04_qef-weak-member-ensemble-boundary](../deep_dives/2026-09-04_qef-weak-member-ensemble-boundary.md)
- [2026-09-04_qek-pricing-full-fit-ensemble-members](../deep_dives/2026-09-04_qek-pricing-full-fit-ensemble-members.md)
- [2026-09-04_qel-ensemble-gain-and-combiners](../deep_dives/2026-09-04_qel-ensemble-gain-and-combiners.md)
- [2026-09-04_qem-non-dino-trunk-and-weak-member-ensemble](../deep_dives/2026-09-04_qem-non-dino-trunk-and-weak-member-ensemble.md)
- [2026-09-04_qeo-corpus-diversity-ensemble](../deep_dives/2026-09-04_qeo-corpus-diversity-ensemble.md)
- [2026-09-04_qdu-foreign-architecture-census](../deep_dives/2026-09-04_qdu-foreign-architecture-census.md)
- Index: [docs/AXES.md](../../docs/AXES.md) §A, §D, §E, §F, §G, §H, §I, "Open, and unread"
