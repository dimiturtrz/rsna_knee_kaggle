# Competition landscape and the public frontier — topic summary

**Current verdict.** Board read 2026-09-25T05:52Z — ours 0.946 rank 97/4318, leader 0.959; best public notebook 0.943 (search snippet).
No public source holds a licence-clean mechanism worth +1.5% on our stack ([09-25 sweep](../deep_dives/2026-09-25_frontier-sweep-for-a-0p95-mechanism.md), measured-by-us + literature).
Every public idea we could test either was already ours or tested flat. The only playbook items never run are auxiliary segmentation, which is barred by the mask licence, and a diverse ensemble, which the solo directive excludes ([09-18](../deep_dives/2026-09-18_kaggle-playbook-vs-our-ledger.md)).
The gap to the leader is not explained by anything published.

Tags: **[M]** measured by us · **[L]** literature or host text · **[U]** unverified (forum claim, search snippet, secondary source).
Every board position or score below other than the header line is historical, dated at its doc, and is **not current**.

---

## 1. Competition facts (host)

- **[L]** Task and metric: 12 binary labels scored by macro AUC. The eval set is annotated by expert radiologists. Data spans 12 languages, 16 sites and five continents, with >5,000 exams. Prize is $77,000 plus an efficiency award ([08-11](../deep_dives/2026-08-11_public-notebooks.md)). The metric definition is covered in [evaluation-metrics-and-transfer.md](evaluation-metrics-and-transfer.md).
- **[U] Site count:** Kimi's landscape doc says **19 sites** ([09-01](../deep_dives/2026-09-01_rsna_2026_challenge_landscape.md)), which contradicts the 16 in 08-11. Unresolved; read the host page.
- **[L] Deadline:** 08-11 said 22 Oct 2026. **Superseded** by 09-18's "deadline 2026-11-05" ([09-18](../deep_dives/2026-09-18_kaggle-playbook-vs-our-ledger.md)). Prefer the later date, but check the rules page.
- **[L] Test set:** ~1300 studies with no report column. 819,640 DICOMs, ~570 GB. 9 h GPU kernel cap ([09-03](../deep_dives/2026-09-03_public_rsna_knee_baseline_audit.md); the 9 h cap is also in [09-01](../deep_dives/2026-09-01_rsna_2026_challenge_landscape.md)).
- **[L] Train set:** 4407 studies, of which 58 are gold (radiologist-adjudicated). Report↔gold agreement is ~82% ([09-03](../deep_dives/2026-09-03_public_rsna_knee_baseline_audit.md)); 09-01 puts disagreement at ~18% ([09-01](../deep_dives/2026-09-01_rsna_2026_challenge_landscape.md)).
- **[U] Ground truth:** GT comes from 2 MSK radiologists plus an adjudicator, with ~82% agreement (raahimnawaz, cited in [09-24](../deep_dives/2026-09-24_all-lanes-web-sweep.md)).
- **[L] Gold-58 is now public** (forum 740162) ([09-10 round2](../deep_dives/2026-09-10_round2-practitioner-and-access-terrain.md)).
- **[L] Efficiency LB:** formula and exchange rate are in [evaluation summary §6](evaluation-metrics-and-transfer.md). In 08-11 it had 905 teams, and the best accuracy entry (0.940) sat at efficiency rank 638 ([08-11](../deep_dives/2026-08-11_public-notebooks.md)). The lane is CLOSED for us per [AXES:203](../../docs/AXES.md).
- **[U] Rerun hazards** ([09-23](../deep_dives/2026-09-23_forum_sweep.md)):
  - 741916: a DICOM fix may have failed.
  - 742122: mixed slice sizes break the hidden rerun.

## 2. Host rulings on data and tooling

- **[L] Host 733965 (Chen):**
  - Non-commercial (NC) licences are acceptable.
  - A hosted LLM API used for labels is not private sharing ([09-13](../deep_dives/2026-09-13_host_rulings_forum_sweep.md); earlier the same topic in [08-23](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)).
  - **Contradiction:** [09-03](../deep_dives/2026-09-03_public_rsna_knee_baseline_audit.md) calls hosted-LLM report processing "likely a Data-Security violation". The later, host-sourced 09-13 supersedes it.
- **[L] Other topics:**
  - 734109: KneeCoT is prohibited (by agreement).
  - 733652: redirected.
  - 737950 and 738156: unanswered at 09-13 ([09-13](../deep_dives/2026-09-13_host_rulings_forum_sweep.md)).
  - 08-23 recorded the registration-gated questions as unanswered; 09-13 supersedes that.
- **[L] 09-13 verdict table:**
  - KMAR-50K: ship yes.
  - CC0 tables: yes.
  - Rijeka: open (ND licence vs rule 2.5).
  - MRNet and SKM-TEA: points against.
  - fastMRI: no.
  - KneeCoT: prohibited.
- **[L] Licence notes from 09-10 round2:** the MRNet mirror's licence is "Unknown", and fastMRI has a licence conflict ([09-10 round2](../deep_dives/2026-09-10_round2-practitioner-and-access-terrain.md)).
- **[U] 741819:** free self-serve data passes. The owner's 09-15 ruling is stricter and governs ([09-23](../deep_dives/2026-09-23_forum_sweep.md)). AXES records the barred corpora at [AXES:162-163](../../docs/AXES.md).
- **[M] Instrument-only site readings:** a second site +0.0043, a third −0.0024 ([09-13](../deep_dives/2026-09-13_host_rulings_forum_sweep.md)). The corpus axis "saturates at two sites" per [AXES:151](../../docs/AXES.md).

## 3. What public notebooks are, and what transferred

- **[L] Early landscape (08-11):**
  - Notebooks: pilkwang baseline (268 votes), prvsiyan rule reader, ryanholbrook efficiency LB, fleongg DINOv2-base soup, "0-899-let-me-cook" (0.899).
  - Their model design had converged with ours. The differentiator was a multilingual rule reader: 52% of reports contain no common English word ([08-11](../deep_dives/2026-08-11_public-notebooks.md)).
- **[M] Their ideas under our harness, all within noise or negative** ([08-11](../deep_dives/2026-08-11_public-notebooks.md)):
  - 32-slice inference: −0.009 to −0.010.
  - Max pooling: +0.0018.
  - Label-count loss: −0.0022.
  - Weight floor: +0.0019.
  - Label fusion: 0.8966 vs 0.8982.
  - Their table: −0.0008.
  - Only our own bugs transferred (augmentation fix, +0.0110).
- **[U→M] Ensembling is worth ~+0.001 on this board.** hyakumanben's three-family blend (CoAtNet / Swin / EffNetV2) states it ([08-31](../deep_dives/2026-08-31_public-frontier-sweep.md)). This matches our own flat seating. AXES rates the ensemble axis at "+0.011 and spent", out of scope for a solo ([AXES:182-193](../../docs/AXES.md)).
- **[M] The frontier's best single, CoAtNet, IS `raptor_v4_swa`:** 0.923 solo on the same day as our best single ([08-31](../deep_dives/2026-08-31_public-frontier-sweep.md)).
- **[U→M] Corpus volume:** the public rebuild 3,155→4,349 gave +0.0131 (92.7% of bootstraps). Gains land on thin findings: LM +0.071, Fracture +0.057. The labelled corpus is exhausted on both sides ([08-31](../deep_dives/2026-08-31_public-frontier-sweep.md)). Data volume is "the only axis that ever cleared the bar", on a concave slope ([AXES:150](../../docs/AXES.md)).
- **[L→M] Models-tab probes** ([09-01](../deep_dives/2026-09-01_rsna_2026_challenge_landscape.md), [09-03](../deep_dives/2026-09-03_public_rsna_knee_baseline_audit.md)):

  | Model | Score |
  |---|---|
  | DINOv2-small | 0.906 |
  | DINOv2-base | 0.861 |
  | EffNet-B3 | 0.701 |
  | dinov2-reg-base | 0.696 |
  | MiniLM | 0.656 |
  | EffNet-B0 | 0.622 |
  | DINOv2-large | 0.500 |

  - **Contradiction:** 09-01 reads DINOv2-large's 0.500 as "capacity doesn't buy" / overfitting. 09-03 calls it degenerate, not confirmed. Prefer 09-03.
  - Our own capacity verdict is at [AXES:40](../../docs/AXES.md): base pays +0.0067 under labelwise; large is flat at −0.0014.

## 4. Forum claims and competitor ablations

- **[U] Topic 735304 single-model claims:** 0.934→0.942 @224, 0.938, 0.929, 0.926, 0.92, 0.915 DinoV2 ([08-25](../deep_dives/2026-08-25_competitor-single-model-scores.md)).
- **[M] The marginal seat is worth ~0.000, while single-member quality is worth ~0.014.** Three roster axes are flat (0.936–0.937; 0.938 vs 0.937; merged_corpus flat) ([08-25](../deep_dives/2026-08-25_competitor-single-model-scores.md)).
- **[U] 735826 poll:** leaders credit labels and geometry; nobody argued capacity. **[U] 735154:** Small→Base +0.0011; ours base224 +0.0007 [M] ([08-25](../deep_dives/2026-08-25_competitor-single-model-scores.md)).
- **[M] Laterality and geometry were already ours** ([08-25](../deep_dives/2026-08-25_competitor-single-model-scores.md)):
  - 130 mm crop; FOV median 160 mm.
  - Laterality resolves 98.4%, agreement 0.986.
  - 7 of 28 gold annotations contradict the severity threshold.
- **[U] 737597 ablation** ([08-26](../deep_dives/2026-08-26_the-ablation-table-a-competitor-published.md)):
  - Setup: EffNet-B0 at 0.7815 OOF / 0.870 public.
  - Center-9 beats equal-spaced by +0.018 (fold 0).
  - EMA +0.003, mixup +0.002, 60 epochs +0.001, ASL −0.139.
- **[M] Our versions of those levers:**
  - Band has an interior optimum: 0.20–0.80 0.8525, 0.10–0.90 0.8565, 0.00–1.00 0.8427.
  - Mixup +0.0015; mixup is closed at [AXES:138](../../docs/AXES.md).
  - EMA closed at [AXES:129](../../docs/AXES.md).
- **[U/M] 737454 label tables:** stevenleehans v4 0.893 (ours 0.8927), pilkwang 0.870 (ours 0.8658). Blend 0.8912. Three leaked tables score 1.0000 ([08-26](../deep_dives/2026-08-26_the-ablation-table-a-competitor-published.md)). Label-table choice is closed, with v2 winning ([AXES:153](../../docs/AXES.md)).
- **[U] Claims from 09-23** ([09-23](../deep_dives/2026-09-23_forum_sweep.md)):
  - 44→80 slices took 0.917→0.932. Ours, 24→36, was −0.0224 [M] ([AXES:91](../../docs/AXES.md)).
  - RadImageNet +0.020. Ours: RadImageNet DenseNet closed ([AXES:48](../../docs/AXES.md)).
  - "Fast Parent 0.957" is an in-sample gold58 score, **not a board score**.

## 5. Latest public frontier (09-25; supersedes 08-31)

- **[U] Named notebooks** ([09-25](../deep_dives/2026-09-25_frontier-sweep-for-a-0p95-mechanism.md)):
  - Speedy Raptors 0.943 (snippet): the Hisernberg lineage, already ours.
  - TranBaDat 0.939 (0.813 vs 0.809).
  - Shiven 0.935 (+0.0010, −0.0008, −0.0054).
  - JunhaoLi 0.664.
  - **Supersedes** 08-31's "No public notebook above 0.926 publishes its method" ([08-31](../deep_dives/2026-08-31_public-frontier-sweep.md)).
- **[M] ROI tested:** 0.8984 / 0.9018 vs 0.9051 ledger75 ([09-25](../deep_dives/2026-09-25_frontier-sweep-for-a-0p95-mechanism.md)). The ROI crop is closed ([AXES:87](../../docs/AXES.md)).
- **[L] Seg aux loss is +0.01–0.03 on LOG LOSS**, not AUC. It survives only as an instrument ([09-25](../deep_dives/2026-09-25_frontier-sweep-for-a-0p95-mechanism.md)).
- **Survivors of 09-25:** registered slice selection (knee-bps3x, P ≈ 0.03–0.05, 4–6 h) and seg aux as an instrument. Slot registration is otherwise CLOSED on two instruments ([AXES:86](../../docs/AXES.md)).

## 6. Published per-label SOTA against ours (annotation ruler)

- **[M vs L] Gold-58 ruler** (58 studies, 696 cells, SE 0.04–0.06) ([08-23](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)):

  | Label | Ours | Published |
  |---|---|---|
  | LM | 0.6497 | 0.850 OrthoDiffusion |
  | ACL | 0.8640 | 0.970 |
  | MCL | 0.8798 | 0.919/0.782 |
  | MM | 0.8498 | 0.872 |
  | Synovitis | 0.7378 | DL 0.76, radiologist 0.77 |
  | Contusion | 0.8421 | 0.821 |
  | Medial OA | 0.9504 | 0.93 |
  | Effusion | 0.9453 | 0.864 |
  | Lateral OA / PF OA / Baker's / Fracture | 0.7099 / 0.8005 / 0.9547 / 0.8861 | none found |

  - The ceiling if every gap closed is (0.20+0.10)/12 = +0.025. LM alone is +0.017.
- **[L] CoPAS** ([08-24](../deep_dives/2026-08-24_published_knee_mri_multilabel_sota.md)):
  - 12 labels, AUC 0.812 vs MRNet 0.781.
  - Supplementary Table 10 has no clean CPA step; "+0.049" was a mis-subtraction.
  - CSA +0.025, CRM +0.013; the crop substitutes for attention.
  - **Contradiction:** the ceiling-interventions doc quotes CoPAS ROI **+0.056** as a lever ([09-10 ceiling](../deep_dives/2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md)). AXES:86 says CoPAS's +0.056 does not transfer, because it was measured against a no-crop baseline.
- **[L] Per-pathology literature** ([09-02 pathology](../deep_dives/2026-09-02_pathology_specific_sota_knee_mri_2025_2026.md)):
  - ACL: Xue 0.99 (six-class); MRNet 0.965 internal / 0.911 Rijeka.
  - OrthoDiffusion macro 0.908.
  - Meniscus: MRNet 0.847.
  - No BML, synovitis or Baker's models exist. Only ~5 of 12 labels map to external data ("cupboard is bare").
  - [M] merged_blend +0.0063.
  - OrthoDiffusion as a trunk is closed at −0.0952 ([AXES:41](../../docs/AXES.md)).
- **[L] Capacity and single-model levers:**
  - Bigger lost (Yiu 0.8184 vs 0.74).
  - Ordinal +0.3–1.6pp, ASL +1.5, AnyMC3D 0.962/0.958/0.946 ([09-10 ceiling](../deep_dives/2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md)).
  - Ordinal on our harness "buys nothing the roster can see" ([AXES:111](../../docs/AXES.md)).

## 7. Mechanism hunts: is there a +1.5% lever?

- **[L] 09-02 Q-AJ:** no clean published >1.5% mechanism ([09-02 qaj](../deep_dives/2026-09-02_qaj_one_unpriced_1p5pct_mechanism.md)).
  - Best bet named: domain SSL continuation (Azizi +1.1% mean AUC, SKID +0.020, PRISM +2.27% accuracy).
  - Zhang 2024: fastMRI SSL only matched ImageNet.
  - AXES closes in-domain SSL / masked adaptation ([AXES:142](../../docs/AXES.md), [AXES:50](../../docs/AXES.md)).
- **[L] Top-1% separators in closed RSNA contests** ([09-02 qbp](../deep_dives/2026-09-02_qbp-top-1pct-vs-median-in-closed-medical-imaging-competitions.md)):
  - Two-stage localize (9/9).
  - RNN/attention aggregation + metric-aware fusion (7/9).
  - Pseudo-labels (4/9).
  - Aux segmentation (4/9; RSNA 2023 1st +0.01–0.03).
  - Diverse ensemble (9/9).
  - External data (2/9).
  - [M] stack224 −0.0033 and cover_labelwise −0.0014 vs base224.
- **[M] Playbook vs ledger** ([09-18](../deep_dives/2026-09-18_kaggle-playbook-vs-our-ledger.md)):
  - Pseudo-labels flat.
  - Aux objectives negative ([AXES:113](../../docs/AXES.md)).
  - Distillation inside 0.0053.
  - Refit ~+0.005.
  - Aggregator span 0.0019.
  - Aux SEGMENTATION never run (licence-blocked mask source).
- **[L] 09-10 practitioner terrain** ([09-10 round2](../deep_dives/2026-09-10_round2-practitioner-and-access-terrain.md)):
  - arXiv 2606.11606: CLS pooling sits at chance on small findings and patch-local is ≈1.0 (gap up to +0.243). [M] Label-specific in-plane selection is null, +0.0011 ([AXES:245](../../docs/AXES.md)).
  - Winners hand-fix labels.
  - nnU-Net took 7th/1,147.
  - Breast pooling ~+0.02 CV [U].
- **[L] 09-10 ceiling-breakers** ([09-10 unseen](../deep_dives/2026-09-10_unseen-ceiling-breakers-discovery.md)):
  - Bernhardt: 4× more efficient relabelling; a 40%-noisy val set reversed a ranking (46.5% vs 65.8%).
  - MRNet→Rijeka 0.824→0.911.
  - Xie AOLM +0.01–0.04.
  - Negatives: CurBench, CleanPatrick 22%.
- **[M] Unseen candidates mapped onto our measurements** ([09-10 mapped](../deep_dives/2026-09-10_unseen-candidates-mapped-onto-our-measurements.md)):
  - merged3 −0.0024.
  - Label cleaning ceiling +0.0053.
  - half_data −0.0234, so the next doubling ≈ +0.009.
  - ortholarge vs v3large +0.0202/+0.0203.
  - Proxy vs annotation +0.045 on identical weights.
- **[M/L] 09-24 all-lanes sweep** ([09-24](../deep_dives/2026-09-24_all-lanes-web-sweep.md)):
  - Baseline 0.9053 ledger75.
  - Candidates ranked at P ≤ ~0.10: AUC-margin FT, WiSE, Rep-GLS, multiscale, NVUM.
  - Since measured: AUC-margin FT −0.0096 ([AXES:107](../../docs/AXES.md)), WiSE monotone down ([AXES:81](../../docs/AXES.md)), multiscale −0.0046 ([AXES:77](../../docs/AXES.md)).
  - Rep-GLS is "already the label" ([AXES:125](../../docs/AXES.md)).
  - **Contradiction:** 09-10 ceiling says Rep-GLS sharpening is contraindicated by our gold read; 09-24 ranks it at P ~0.10. AXES sides with neither as a lever.
- **[L] Round 3** ([09-10 round3](../deep_dives/2026-09-10_round3-local-unrun-and-evaluation-economics.md)):
  - ~76 unrun local arms (an over-count).
  - Roelofs 2019 finds no widespread Kaggle overfitting.
  - Physics augmentation: silent in the literature. [M] ghosting is OPEN, +0.0057 directional ([AXES:126](../../docs/AXES.md)).

## 8. Historical board positions (dated, NOT current)

- **08-23:** leader 0.952, ours 0.934; best notebook 0.924 ([08-23](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)).
- **08-25:** votefit_v1 0.938 ([08-25](../deep_dives/2026-08-25_competitor-single-model-scores.md)).
- **09-02:** 0.944 rank 33/2919, leader 0.952 ([09-02 qbp](../deep_dives/2026-09-02_qbp-top-1pct-vs-median-in-closed-medical-imaging-competitions.md)).
- **09-18:** 0.946 (tripleseat_v2), ~0.011 behind ([09-18](../deep_dives/2026-09-18_kaggle-playbook-vs-our-ledger.md)).
- **09-23:** top 0.958, #20 0.953 ([09-23](../deep_dives/2026-09-23_forum_sweep.md)).

## Open questions

- Is the site count 16 or 19? Is the deadline 22 Oct or 2026-11-05? Read the host page, not these docs.
- What does the leader's +0.013 consist of? Nothing public explains it (09-25).
- Is there any licence-clean mask source for aux segmentation? Until one exists it is instrument-only (09-18, 09-25).
- Registered slice selection (knee-bps3x): priced, unrun.
- Should a diverse ensemble be priced against the gap despite the solo directive? 09-18 asks that it be restated whenever the gap is priced.

## Sources

- [2026-08-11_public-notebooks.md](../deep_dives/2026-08-11_public-notebooks.md)
- [2026-08-23_published-per-label-auc-vs-ours.md](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)
- [2026-08-24_published_knee_mri_multilabel_sota.md](../deep_dives/2026-08-24_published_knee_mri_multilabel_sota.md)
- [2026-08-25_competitor-single-model-scores.md](../deep_dives/2026-08-25_competitor-single-model-scores.md)
- [2026-08-26_the-ablation-table-a-competitor-published.md](../deep_dives/2026-08-26_the-ablation-table-a-competitor-published.md)
- [2026-08-31_public-frontier-sweep.md](../deep_dives/2026-08-31_public-frontier-sweep.md)
- [2026-09-01_rsna_2026_challenge_landscape.md](../deep_dives/2026-09-01_rsna_2026_challenge_landscape.md)
- [2026-09-02_qaj_one_unpriced_1p5pct_mechanism.md](../deep_dives/2026-09-02_qaj_one_unpriced_1p5pct_mechanism.md)
- [2026-09-02_qbp-top-1pct-vs-median-in-closed-medical-imaging-competitions.md](../deep_dives/2026-09-02_qbp-top-1pct-vs-median-in-closed-medical-imaging-competitions.md)
- [2026-09-02_pathology_specific_sota_knee_mri_2025_2026.md](../deep_dives/2026-09-02_pathology_specific_sota_knee_mri_2025_2026.md)
- [2026-09-03_public_rsna_knee_baseline_audit.md](../deep_dives/2026-09-03_public_rsna_knee_baseline_audit.md)
- [2026-09-10_round2-practitioner-and-access-terrain.md](../deep_dives/2026-09-10_round2-practitioner-and-access-terrain.md)
- [2026-09-10_round3-local-unrun-and-evaluation-economics.md](../deep_dives/2026-09-10_round3-local-unrun-and-evaluation-economics.md)
- [2026-09-10_unseen-candidates-mapped-onto-our-measurements.md](../deep_dives/2026-09-10_unseen-candidates-mapped-onto-our-measurements.md)
- [2026-09-10_unseen-ceiling-breakers-discovery.md](../deep_dives/2026-09-10_unseen-ceiling-breakers-discovery.md)
- [2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md](../deep_dives/2026-09-10_single-model-ceiling-interventions-medical-grading-ranked.md)
- [2026-09-13_host_rulings_forum_sweep.md](../deep_dives/2026-09-13_host_rulings_forum_sweep.md)
- [2026-09-18_kaggle-playbook-vs-our-ledger.md](../deep_dives/2026-09-18_kaggle-playbook-vs-our-ledger.md)
- [2026-09-23_forum_sweep.md](../deep_dives/2026-09-23_forum_sweep.md)
- [2026-09-24_all-lanes-web-sweep.md](../deep_dives/2026-09-24_all-lanes-web-sweep.md)
- [2026-09-25_frontier-sweep-for-a-0p95-mechanism.md](../deep_dives/2026-09-25_frontier-sweep-for-a-0p95-mechanism.md)
- Cross-check: [docs/AXES.md](../../docs/AXES.md)
