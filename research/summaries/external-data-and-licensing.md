# External data and licensing — topic summary

> **BINDING RULING (owner, overrides every deep dive below).**
> 1. External data other than KneeCoT was ruled usable on **2026-09-14**. That ruling was **NARROWED on 2026-09-15**.
> 2. Anything derived from a **foreign corpus** is **instrument-only and never shipped**. This covers Rijeka (KneeMRI), MRNet, fastMRI / fastMRI+, and SKM-TEA, whether used for training, fine-tuning, distillation, or as a teacher or member.
> 3. **Measurement-only use is clean.** Examples: deriving an augmentation constant from KMAR-50K, or pricing a mechanism offline.
> 4. **KneeCoT is never allowed.**
>
> Any deep dive that calls a foreign corpus "shippable", "permitted" or "Kaggle-OK" is **overruled** by this.

**Verdict (2026-09-25).** External data is effectively a closed lever for the shipped entry.

- Data volume is the only axis in the project that ever cleared the bar. It is concave (β 0.22), and the reachable gain saturates at a second site ([AXES:150-151](../../docs/AXES.md)).
- Every foreign-corpus route that scored is barred by the ruling. The ones that did not score (SSL at our size, distillation, members) are closed on the ledger anyway ([AXES:157-163](../../docs/AXES.md)).

Tags: **[ours]** = measured by us, **[lit]** = literature, **[unv]** = unverified / forum / estimate. Numbers use the midpoint ruler unless a ruler is named.

---

## 1. Licences, per corpus (latest source wins)

| Corpus | Licence / terms | Can train-and-ship? | Source |
|---|---|---|---|
| MRNet (Stanford) | Research Use Agreement: non-commercial, no redistribution, no derivatives (RUA §1/§4/§6) | **No** (ruling) | [qbu](../deep_dives/2026-09-02_qbu-mrnet-rijeka-licence.md) |
| Rijeka KneeMRI | CC BY-NC-ND (Zenodo 14789903); trained weights = Adapted Material | **No** (ruling; ND) | [qbu](../deep_dives/2026-09-02_qbu-mrnet-rijeka-licence.md) |
| fastMRI | NYU DUA §2, §5(c), §11 | **No**: "do not ship" | [qbt](../deep_dives/2026-09-02_qbt-fastmri-plus-licence-and-loss.md) |
| fastMRI+ annotations | MIT (boxes), but the images sit under the fastMRI DUA | **No** | [corpora-survey](../deep_dives/2026-09-01_knee-mri-corpora-survey.md), [qbt](../deep_dives/2026-09-02_qbt-fastmri-plus-licence-and-loss.md) |
| SKM-TEA | Non-commercial, no derivatives | **No** | [shippable-survey](../deep_dives/2026-09-11_shippable-corpus-licence-survey.md) |
| OAI | NDA DUC / DUA §4(c); **not** public domain | **No** | [shippable-survey](../deep_dives/2026-09-11_shippable-corpus-licence-survey.md) |
| KMAR-50K | CC BY 4.0 | Measurement use only (AXES:126) | [corpora-survey](../deep_dives/2026-09-01_knee-mri-corpora-survey.md) |
| TCIA Sarcoma, M4Raw | CC BY 3.0 / CC BY 4.0, unlabelled for our findings | Licence-clean, no labels | [shippable-survey](../deep_dives/2026-09-11_shippable-corpus-licence-survey.md) |
| KneeCoT | — | **Never** (owner, forum 734109 refused) | [host_rulings](../deep_dives/2026-09-13_host_rulings_forum_sweep.md), [AXES:163](../../docs/AXES.md) |

Notes on the table:

- **[lit]** No labelled knee-MRI corpus is shippable under this competition's winner-licence obligation ([shippable-survey](../deep_dives/2026-09-11_shippable-corpus-licence-survey.md)).
  - Kaggle rule 2.6a (external data reasonably accessible) passes. The exposure is rule **2.5** and the winner's duty to publish weights (the competition rules (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules):187-196).
  - Evidence: the 09-07 correction in [access_audit](../deep_dives/2026-09-02_external_knee_mri_data_access_audit.md); [AXES:162](../../docs/AXES.md).
- **[unv]** The US Copyright Office's May 2025 view on weights trained from licensed data is unsettled ([shippable-survey](../deep_dives/2026-09-11_shippable-corpus-licence-survey.md)).
- **[lit]** Licensed *models*, as distinct from corpora:
  - HAI-DEF §1.1(h) says outputs are not Model Derivatives ([medgemma_licence](../deep_dives/2026-09-03_medgemma_licence_and_domain_for_knee_mri.md)).
  - Trunk licences are covered in [licensed-trunks](../deep_dives/2026-09-04_licensed-medical-trunk-encoders-for-kaggle.md) and [qdq](../deep_dives/2026-09-04_qdq-radiodino-shippability-radimagenet-license.md).
- **[ours]** Host rulings found in the forum sweep ([host_rulings](../deep_dives/2026-09-13_host_rulings_forum_sweep.md)). The owner's 09-15 ruling is **narrower** than what these threads allow:
  - Thread 733965: NC licences OK.
  - Thread 734109: KneeCoT refused.
  - Threads 733652, 737950 and 738156: unanswered.
  - Also unanswered: 741819 (OAI, 09-17) and 741212 (winner obligations for a public checkpoint).

## 2. What is on disk, and what is tainted

- **[ours]** Local copies ([access_audit](../deep_dives/2026-09-02_external_knee_mri_data_access_audit.md)):

  | Corpus | Size on disk |
  |---|---|
  | MRNet | 7.3 GB |
  | Rijeka | 3.1 GB |
  | fastMRI | 194 GB |
  | SKM-TEA | 19 GB |
  | KMAR | 67 GB |

- **[ours]** Tainted lineages, which are instrument-only:
  - The `merged_blend_cf` family, `merged3_corpus` and `mrnet_warm`.
  - 47 kernels seat a merged member (same doc). `merged_blend_cf` appeared in more than 30 submissions ([qbu](../deep_dives/2026-09-02_qbu-mrnet-rijeka-licence.md)).
  - CC0 public-notebook members: raptor, mattiaangeli, pilkwang (same doc).
- **[ours]** fastMRI+ mapping ([label-mapping](../deep_dives/2026-09-01_fastmri-plus-label-mapping.md)):
  - Scale: 974 studies, 16,167 rows, and `study_level` is "No" on 16,154 of them.
  - Boxes per label:

    | Label | Boxes |
    |---|---|
    | ACL | 1,443 |
    | MCL | 296 |
    | Effusion | 1,311 |
    | Baker's | 864 |
    | Meniscus | 5,658 |
    | OA | 4,586 |
    | Contusion + Fracture | 1,060 |
    | Synovitis | 0 |

  - Slice indices map through a 35→12 decimation.
  - [corpora-survey](../deep_dives/2026-09-01_knee-mri-corpora-survey.md), [using_external](../deep_dives/2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score.md) and [qxd](../deep_dives/2026-09-04_qxd-external-knee-mri-corpora.md) quote **1,172** studies instead. The 974 figure is the one counted on disk.
- **[lit]** No public knee MRI has synovitis or Baker's labels, and no public knee MRI corpus comes paired with reports ([power-law](../deep_dives/2026-09-11_data-volume-power-law-and-knee-mri-corpora.md)).

## 3. What external data bought (measured), and what the ledger closed

- **[ours]** Supervised corpus merge ([status_quo](../deep_dives/2026-09-02_external_data_and_foundation_model_status_quo.md)):
  - `merged_blend` gains +0.0063 on the mean, with LatMen +0.0196 and ACL +0.0161.
  - AXES:151: saturates at two sites. `mergedk` NULL. "The gain is the SITE, not the supervision."
  - `slots.py:341-344`: a second site adds +0.0043; a third costs −0.0024 ([host_rulings](../deep_dives/2026-09-13_host_rulings_forum_sweep.md)).
- **[ours]** Label-overlap cap: ~+0.0075, on 3 labels only. Merged seats in an ensemble peak at +0.006 and cost −0.004 at six seats ([qbv](../deep_dives/2026-09-02_qbv-layer-transfer-partial-label-overlap.md)).
- **[ours]** Oversampling the corpus is flat (AXES:152).
- **[ours]** External labelled augmentation is CLOSED: `noisy_student` scored 0.8548 and `pool_specialise` was flat (AXES:157, README:12485, :15254).
- **[ours]** Format mismatch, i.e. transfer from external MAE / encoders, lost ([format_mismatch](../deep_dives/2026-09-02_external_data_format_mismatch_models_training.md)):

  | Run | Score |
  |---|---|
  | mae_matched | 0.6735 |
  | encoder_adapt_thawed | 0.7162 |
  | encoder_adapt_lower | 0.7309 |

  - Frozen probes: 0.7852 / 0.7446 / 0.7387 ([status_quo](../deep_dives/2026-09-02_external_data_and_foundation_model_status_quo.md)).
  - The scheme priors in that doc (A +0.002–0.005, B +0.005–0.012, C +0.010–0.020) are **[unv]** estimates.
- **[ours]** External SSL: +0.073 at 80 subjects, −0.005 at 800 (AXES:159, README:7436). This matches Zhang ([kmar50k](../deep_dives/2026-09-01_kmar50k_ssl_transfer.md)):
  - Gains by size: +0.073 / +0.046 / +0.002 / −0.005.
  - SKID: SSL 0.869 vs supervised 0.894.
- **[ours]** Foreign teacher / distillation is bounded. Headroom is 0.002, and the honest term +0.0056 sits below the 0.0062 floor. Distillation destroys member distance (AXES:160, README:8424, :14113, :16623).
- **[ours]** Foreign weights as members: CLOSED (AXES:161, README:9094, :8940).
  - radiodino_s16 −0.0394 ([qeo](../deep_dives/2026-09-04_qeo-corpus-diversity-ensemble.md)).
  - RAD-DINO −0.037 (AXES:49).
  - MRI-core 0.8390 (AXES:55).
  - OrthoDiffusion 0.8054 **ledger75** vs 0.9006 (AXES:41).
- **[ours]** Generated images: never run. Rijeka ACL and fastMRI+ boxes are barred for training (AXES:158).
- **[ours]** Training on Rijeka / MRNet / fastMRI / SKM-TEA is **BARRED, not refuted**. Measurement-only use stays clean (rules :144; docs/EXTERNAL_CORPORA.md; README:15671; AXES:162).
- **[ours]** The one clean live use is ghosting augmentation: `--ghosting 0.040`, derived from 109 paired KMAR-50K volumes. It is +0.0057 and directional (AXES:126).

## 4. What the literature says transfers

- **[lit]** Pretraining source ([corpus_transfer](../deep_dives/2026-09-01_external_knee_mri_corpus_transfer.md)):
  - Zhang at 800 subjects: scratch 0.562, ImageNet 0.824, SSL 0.819 / 0.815. ImageNet already captures most of the value.
  - SKID MRNet→KneeMRI: +0.020.
- **[lit]** fastMRI+ box supervision ([box-transfer](../deep_dives/2026-09-01_fastmri-plus-box-supervision-transfer.md)):
  - In-corpus: +0.01–0.03.
  - Cross-corpus: −0.009 / −0.016 / +0.017 / −0.006, which is roughly zero.
  - The doc states its noise band inconsistently (±0.005 in one place, ±0.02 in another).
- **[lit]** fastMRI transfer mechanisms ([fastmri-transfer](../deep_dives/2026-09-11_fastmri-transfer-mechanisms.md)):
  - Tran: 0.941 / 0.860.
  - Germann −0.07; Lu −0.097.
  - PU mAP 0.530 vs 0.501.
  - The doc concludes offline use only.
  - [qbt](../deep_dives/2026-09-02_qbt-fastmri-plus-licence-and-loss.md): Nischaydnk's +0.01–0.03 does not transfer.
- **[lit]** Matched vs mismatched corpora ([qeo](../deep_dives/2026-09-04_qeo-corpus-diversity-ensemble.md)):
  - Krones (mismatched modality): −0.005 to −0.017.
  - Matched: +0.024.
- **[lit]** Harmonisation is priced at ~0 ([cross-vendor](../deep_dives/2026-09-05_cross-vendor-domain-shift-prior-art-cardiac-seg.md)):
  - Nyul −0.007.
  - Ensemble on unseen vendors +0.000 / +0.006.
- **[lit]** External cohorts rarely hold their ranking ([external-cohort](../deep_dives/2026-09-10_external-cohort-ranking-does-not-reliably-survive.md)):
  - Expected drop 0.03–0.10.
  - Yu: 81% of algorithms drop, 24% substantially.
  - MRNet→Rijeka: −0.11.
- **[lit]** Continued SSL on fastMRI ([continuation](../deep_dives/2026-09-02_fastmri_continuation_arithmetic.md)):
  - 973 train volumes / 34,742 slices.
  - DINOv2 issue #204: kNN collapses from 80% to <10% after 20–30k images.
  - The positive regime is 50k–300k images at 25–100 GPU-h.
- **[lit]** Power law ([power-law](../deep_dives/2026-09-11_data-volume-power-law-and-knee-mri-corpora.md)):
  - Zhang β 0.21–0.25.
  - Gaining +0.010 from 0.944 needs 2.2–2.7× the data (~9,700–11,800 studies).
  - That is unreachable if the ceiling gap c ≥ 0.046. The refit ceiling is 0.885.
- **[unv]** Design proposals that are now moot for shipping:
  - [merging_schemes](../deep_dives/2026-09-02_external_mri_data_merging_schemes.md): masked BCE.
  - [scheme_b_dataloader](../deep_dives/2026-09-02_scheme_b_dataloader_design_local_corpora.md): mix 40/25/15/10/10.
  - [using_external](../deep_dives/2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score.md): scheme B +0.005–0.012; pool of 3,300 labelled / 122k unlabelled.
  - [synthesis_3d](../deep_dives/2026-09-02_synthesis_mrnet_3d_paths_and_new_data.md): bets A / B / C. Bet B (OrthoDiffusion) is CLOSED per AXES:41.
  - [qxd](../deep_dives/2026-09-04_qxd-external-knee-mri-corpora.md): AH-Knee, 3,662 studies.

---

## Contradictions and supersessions

1. **OAI licence.** [access_audit](../deep_dives/2026-09-02_external_knee_mri_data_access_audit.md) and [format_mismatch](../deep_dives/2026-09-02_external_data_format_mismatch_models_training.md) call OAI "public domain". [corpora-survey](../deep_dives/2026-09-01_knee-mri-corpora-survey.md) and [shippable-survey](../deep_dives/2026-09-11_shippable-corpus-licence-survey.md) show an NDA DUC with no redistribution. The later docs win.
2. **MRNet licence.** [qxd](../deep_dives/2026-09-04_qxd-external-knee-mri-corpora.md) says "CC-BY, Kaggle-OK", citing a broken link. [qbu](../deep_dives/2026-09-02_qbu-mrnet-rijeka-licence.md) and [fastmri_mrnet_audit](../deep_dives/2026-09-02_fastmri_mrnet_competition_use_licence_audit.md) show an RUA, non-commercial, with no derivatives.
3. **fastMRI risk.** [fastmri_mrnet_audit](../deep_dives/2026-09-02_fastmri_mrnet_competition_use_licence_audit.md) rates it "moderate". [qbt](../deep_dives/2026-09-02_qbt-fastmri-plus-licence-and-loss.md) says DUA §5(c) means do not ship.
4. **Legal blocker.** [scheme_b_dataloader](../deep_dives/2026-09-02_scheme_b_dataloader_design_local_corpora.md) ("blocker no longer legal") and [using_external](../deep_dives/2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score.md) ("MRNet permitted") are contradicted by [host_rulings](../deep_dives/2026-09-13_host_rulings_forum_sweep.md) plus the 09-15 ruling and AXES:162 (BARRED).
5. **Host threads vs owner ruling.** [host_rulings](../deep_dives/2026-09-13_host_rulings_forum_sweep.md) reads the threads as allowing NC corpora to ship. The owner's 09-15 narrowing says instrument only. The ruling wins.
6. **fastMRI+ study count.** [label-mapping](../deep_dives/2026-09-01_fastmri-plus-label-mapping.md) counts 974 studies. [corpora-survey](../deep_dives/2026-09-01_knee-mri-corpora-survey.md), [using_external](../deep_dives/2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score.md) and [qxd](../deep_dives/2026-09-04_qxd-external-knee-mri-corpora.md) say 1,172.
7. **MRNet labels.** [merging_schemes](../deep_dives/2026-09-02_external_mri_data_merging_schemes.md) lists a "cartilage" label. MRNet has only ACL, meniscus and abnormal ([label-mapping](../deep_dives/2026-09-01_fastmri-plus-label-mapping.md), [status_quo](../deep_dives/2026-09-02_external_data_and_foundation_model_status_quo.md)).
8. **Noise band.** [box-transfer](../deep_dives/2026-09-01_fastmri-plus-box-supervision-transfer.md) contradicts itself on its own noise band (±0.005 vs ±0.02).

## Open questions

- Forum threads 741819 (OAI) and 741212 (winner obligations for a public checkpoint) are unanswered. Neither can re-open shipping without the owner.
- Are there more clean *measurement-only* uses, following the pattern of the ghosting constant (AXES:126)? For example, deriving physical augmentation constants from KMAR-50K, TCIA Sarcoma or M4Raw.
- Licence-clean unlabelled volume (M4Raw, TCIA) for in-domain SSL is unpriced. The prior is poor at our size: −0.005 at 800 subjects (AXES:159).

## Sources

- [2026-09-01_external_knee_mri_corpus_transfer](../deep_dives/2026-09-01_external_knee_mri_corpus_transfer.md)
- [2026-09-01_fastmri-plus-box-supervision-transfer](../deep_dives/2026-09-01_fastmri-plus-box-supervision-transfer.md)
- [2026-09-01_fastmri-plus-label-mapping](../deep_dives/2026-09-01_fastmri-plus-label-mapping.md)
- [2026-09-01_kmar50k_ssl_transfer](../deep_dives/2026-09-01_kmar50k_ssl_transfer.md)
- [2026-09-01_knee-mri-corpora-survey](../deep_dives/2026-09-01_knee-mri-corpora-survey.md)
- [2026-09-02_external_data_and_foundation_model_status_quo](../deep_dives/2026-09-02_external_data_and_foundation_model_status_quo.md)
- [2026-09-02_external_data_format_mismatch_models_training](../deep_dives/2026-09-02_external_data_format_mismatch_models_training.md)
- [2026-09-02_external_knee_mri_data_access_audit](../deep_dives/2026-09-02_external_knee_mri_data_access_audit.md)
- [2026-09-02_external_mri_data_merging_schemes](../deep_dives/2026-09-02_external_mri_data_merging_schemes.md)
- [2026-09-02_fastmri_continuation_arithmetic](../deep_dives/2026-09-02_fastmri_continuation_arithmetic.md)
- [2026-09-02_fastmri_mrnet_competition_use_licence_audit](../deep_dives/2026-09-02_fastmri_mrnet_competition_use_licence_audit.md)
- [2026-09-02_qbt-fastmri-plus-licence-and-loss](../deep_dives/2026-09-02_qbt-fastmri-plus-licence-and-loss.md)
- [2026-09-02_qbu-mrnet-rijeka-licence](../deep_dives/2026-09-02_qbu-mrnet-rijeka-licence.md)
- [2026-09-02_qbv-layer-transfer-partial-label-overlap](../deep_dives/2026-09-02_qbv-layer-transfer-partial-label-overlap.md)
- [2026-09-02_scheme_b_dataloader_design_local_corpora](../deep_dives/2026-09-02_scheme_b_dataloader_design_local_corpora.md)
- [2026-09-02_synthesis_mrnet_3d_paths_and_new_data](../deep_dives/2026-09-02_synthesis_mrnet_3d_paths_and_new_data.md)
- [2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score](../deep_dives/2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score.md)
- [2026-09-03_medgemma_licence_and_domain_for_knee_mri](../deep_dives/2026-09-03_medgemma_licence_and_domain_for_knee_mri.md)
- [2026-09-04_licensed-medical-trunk-encoders-for-kaggle](../deep_dives/2026-09-04_licensed-medical-trunk-encoders-for-kaggle.md)
- [2026-09-04_qdq-radiodino-shippability-radimagenet-license](../deep_dives/2026-09-04_qdq-radiodino-shippability-radimagenet-license.md)
- [2026-09-04_qeo-corpus-diversity-ensemble](../deep_dives/2026-09-04_qeo-corpus-diversity-ensemble.md)
- [2026-09-04_qxd-external-knee-mri-corpora](../deep_dives/2026-09-04_qxd-external-knee-mri-corpora.md)
- [2026-09-05_cross-vendor-domain-shift-prior-art-cardiac-seg](../deep_dives/2026-09-05_cross-vendor-domain-shift-prior-art-cardiac-seg.md)
- [2026-09-10_external-cohort-ranking-does-not-reliably-survive](../deep_dives/2026-09-10_external-cohort-ranking-does-not-reliably-survive.md)
- [2026-09-11_data-volume-power-law-and-knee-mri-corpora](../deep_dives/2026-09-11_data-volume-power-law-and-knee-mri-corpora.md)
- [2026-09-11_fastmri-transfer-mechanisms](../deep_dives/2026-09-11_fastmri-transfer-mechanisms.md)
- [2026-09-11_shippable-corpus-licence-survey](../deep_dives/2026-09-11_shippable-corpus-licence-survey.md)
- [2026-09-13_host_rulings_forum_sweep](../deep_dives/2026-09-13_host_rulings_forum_sweep.md)
- [docs/AXES.md](../../docs/AXES.md) rows A:41, A:49, A:55, D:126, E:150–163

- [2026-09-28_forum-and-noisy-loss-sweep.md](../deep_dives/2026-09-28_forum-and-noisy-loss-sweep.md) — forum threads 743416/743420 on external-data rules

- [2026-09-25_frontier-sweep-for-a-0p95-mechanism.md](../deep_dives/2026-09-25_frontier-sweep-for-a-0p95-mechanism.md) — licence-gated segmentation masks
