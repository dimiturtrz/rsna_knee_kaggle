# Labels and label noise — topic summary

**Verdict (2026-09-25).** The training labels are extracted from reports. They are noisy in one direction, with over-calls outnumbering misses (108 confident false positives vs 32 misses). True cell noise is about 2.34%.

Every correction we measured on this axis is closed. The one survivor is `blend_hybrid_fx`, worth +0.0053 on the mean, and choosing a label table is also CLOSED (v2 wins) ([AXES:153-156](../../docs/AXES.md)). The literature agrees that at 2–3% noise the correction gain is below our noise floor. The one untested mechanism the field still names is *manufacturing* labels (pseudo-labels, label blends), which is different from correcting them.

Tags: **[ours]** = measured by us, **[lit]** = literature, **[unv]** = unverified or forum claim. Numbers use the midpoint ruler unless a ruler is named. gold-58 = 58 radiologist image-read studies = 696 cells. Most gold-58 numbers carry a per-label SE of about 0.04–0.12, so read them as coarse.

---

## 1. The rulers: gold-58 and its denominator

- **[ours]** gold-58 is 58 studies with all 12 labels, radiologist image-read, which gives 696 cells. The other 4,349 of the 4,407 studies have only report text ([missing_gold_labels](../deep_dives/2026-09-02_missing_gold_labels_and_medical_vlp_axis.md)).
- **[ours]** `report_labels_v2.csv` (and v3) hold **57** of the 58 gold studies. The missing one, `…73527530686853911124431549317032662220`, is gold-positive for ACL, PF OA, Effusion, Contusion and Fracture. v4 has it. So any "gold-58" number quoted against v2 is really on 57 studies ([58-vs-57](../deep_dives/2026-09-04_58-vs-57-study-denominator-mismatch.md); [state_of_labels](../deep_dives/2026-09-03_state_of_labels_and_model_axes.md); [label_tables_audit](../deep_dives/2026-09-02_rsna_label_tables_pseudo_label_audit.md), which found the same UID missing from all six `blend_*` files).
- **[ours]** The report ruler flatters us. On the report ruler Synovitis reads 0.873. On the annotation ruler it reads **0.7378** (−0.135). Per-label SE on gold-58 is 0.04–0.06 ([published-per-label](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)).
- **[lit / derived]** The two docs give different SEs for these per-label AUCs:
  - At 27 positives / 31 negatives, the Hanley SE is ≈ **0.12** at AUC 0.80 ([vlm-montage](../deep_dives/2026-09-11_vlm-montage-prompting-and-small-gold-validation.md)).
  - [report-error-gradient](../deep_dives/2026-09-11_report-error-gradient-vs-gold58-ordering.md) quotes ≈ **0.055** at 35/23 and AUC 0.85.
  - Different n+ and AUC explain part of the gap, but the two do not reconcile cleanly (see Contradictions).
- **[lit]** The test labels are most likely radiologist-adjudicated. RSNA says "annotated by expert radiologists", and its usual protocol uses ≥3 annotators plus consensus ([qao](../deep_dives/2026-09-02_test_labels_report_derived_or_radiologist_reviewed_qao.md); [qap](../deep_dives/2026-09-02_qap_adjudication_protocol_and_58_gold_pricing.md)). Neither doc states the reader count.
- **[lit]** Eval-label noise: with a one-sided positive noise rate ρ, AUC_obs = (1−ρ)·AUC* + ρ/2. A few percent of *test*-label noise could hide the 0.944→0.954 gap on its own. Train-label noise is different: class-conditional noise with ρ₊+ρ₋<1 leaves the Bayes-optimal ranking unchanged (Menon 2015) ([label-noise-ceiling](../deep_dives/2026-09-11_label-noise-ceiling-and-report-vs-pixels-error.md)).

## 2. How wrong are the labels, and where?

- **[ours]** The extractor disagrees with gold on **145 of 696 cells (20.83%)**. That rate is over cells; the taxonomy percentages below are over the 145.
  - Some errors are re-extractable, for example 75 hedged-negation cases.
  - Others are irreducible: the finding is not in the report, or the radiologist's criteria differ.
  - Fixing the re-extractable ones first would drop residual noise from ~21% to ~9% of cells ([extractor_taxonomy](../deep_dives/2026-09-02_extractor_error_taxonomy_58_gold.md)).
- **[ours]** Asymmetry on gold: **108 confident false positives vs 32 misses**. cleanlab flags beat random by **4.05×** (precision 0.640 vs 0.158, AUC 0.7580) ([qeb](../deep_dives/2026-09-04_qeb-label-noise-treatment-literature.md)).
- **[ours]** Corpus noise rate:
  - Flagged: 1,932 / 52,872 = 3.65%.
  - At precision 0.640 that is ~1,236 true errors, i.e. **2.34%** true cell noise.
  - Worst label is ACL: 5.33% flagged, ~3.4% true.
  - The noise is closest to **pairflip**, not symmetric ([qeh](../deep_dives/2026-09-04_qeh-noise-rate-curve-extrapolation.md)).
- **[ours]** Per-label failure modes on gold ([state_of_labels](../deep_dives/2026-09-03_state_of_labels_and_model_axes.md)):
  - Effusion: **70% false-positive rate** on gold negatives ("small effusion" read as positive).
  - Synovitis: **52% false-negative rate** on gold positives.
  - The severity boundary is a named, reproducible defect (`core/data/labels/severity_boundary.py`) ([rival_single_models](../deep_dives/2026-08-26_rival_single_models_and_the_label_bottleneck.md)).
- **[ours]** Gold-58 column ordering (worst first) ([report-error-gradient](../deep_dives/2026-09-11_report-error-gradient-vs-gold58-ordering.md)):

  | Column | gold-58 AUC | n+ |
  |---|---|---|
  | Synovitis | 0.7515 | 27 |
  | Lateral OA | 0.7872 | 11 |
  | Effusion | 0.8522 | 35 |
  | Lateral Meniscus | 0.8832 | 23 |
  | Fracture | 0.8931 | 18 |
  | Contusion | 0.8961 | 19 |
  | PF OA | 0.9035 | 21 |
  | MCL | 0.9229 | 9 |
  | Medial OA | 0.9535 | 15 |

  - **[lit]** The published report-error gradient (Synovitis κ 0.42, cartilage sensitivity 24–42%) matches the top of that ordering.
  - Effusion does not fit. Literature expects it CLEAN (κ≈0.90), yet it ranks third-worst. That is unresolved.
- **[ours vs unv]** Forum claim: Synovitis is "stuck at 0.6–0.7 for everyone". It is not our weak label. Our weak labels are Fracture (0.8022, SE≈0.04) and MCL (0.7864); the MCL ceiling is priced at **+0.0029** (`knee-q1pf`) ([synovitis-is-not-our-weak-label](../deep_dives/2026-08-26_synovitis-is-not-our-weak-label.md)).
- **[ours]** The model beats its own labels. On gold, the model trained on report labels out-predicts those labels by **0.047 AUC** over 696 cells. Per label: Synovitis +0.1075, Fracture +0.0715, LatMen +0.0522, Baker's +0.0462 ([state_of_labels](../deep_dives/2026-09-03_state_of_labels_and_model_axes.md)).

## 3. Against the published frontier

- **[ours vs lit]** Most of our annotation-ruler labels have no published per-label AUC to compare against: Lateral OA 0.7099, PF OA 0.8005, Baker's 0.9547, Fracture 0.8861 ([published-per-label](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)).
- **[lit]** CoPAS (Nat Commun 2024) is 12 labels on 1,748 subjects at overall AUC 0.812. Its plane-aware fusion adds +0.031 over MRNet on its own ruler. The ruler differs, so the absolutes do not transfer ([published_sota](../deep_dives/2026-08-24_published_knee_mri_multilabel_sota.md)).
- **[lit]** MRNet has Fleiss κ 0.508 / 0.800 / 0.745 and model AUC 0.937 / 0.965 / 0.847. Its lowest-κ task still reaches 0.937, so κ does not cap AUC tightly ([label-noise-ceiling](../deep_dives/2026-09-11_label-noise-ceiling-and-report-vs-pixels-error.md)).

## 4. Label correction — CLOSED ([AXES:154](../../docs/AXES.md))

- **[ours]** One correction survived: `blend_hybrid_fx` = `Fracture := monotone_rescale(Fracture + 0.25·Contusion)`. It adds **+0.0632 on Fracture and +0.0053 on the 12-label mean**, which is under the 0.0062 floor ([state_of_labels](../deep_dives/2026-09-03_state_of_labels_and_model_axes.md); [qeb](../deep_dives/2026-09-04_qeb-label-noise-treatment-literature.md)).
- **[ours]** Status per AXES:154: closed at 0.1%, and built on the wrong table. Relabelling is FLAT and pruning COSTS (README:8381, :7639, :7681, :7732).
  - Earlier docs treated the axis as live: [qao](../deep_dives/2026-09-02_test_labels_report_derived_or_radiologist_reviewed_qao.md) ("a live lever"), [qap](../deep_dives/2026-09-02_qap_adjudication_protocol_and_58_gold_pricing.md) and [extractor-repair](../deep_dives/2026-09-11_extractor-repair-structured-noise-and-selective-repair.md). **AXES and the ledger supersede them.**
- **[ours]** `errors_dense` prunes 5,360 cells (10.1%).
  - [confident_learning](../deep_dives/2026-09-02_label_noise_confident_learning.md) records it as run at 0.8420.
  - [qeb](../deep_dives/2026-09-04_qeb-label-noise-treatment-literature.md) (09-04T1535Z) retracts that: "not yet run … appears premature".
  - AXES:154 now records pruning as a cost. Use AXES.
- **[lit]** How big could correction be at our noise rate?
  - HLC curve extrapolated to 2.34%: **< 0.0023 AUC** (MS-COCO analogue, likely < 0.001) and **< 0.0058** (VOC analogue, likely < 0.003) ([qeh](../deep_dives/2026-09-04_qeh-noise-rate-curve-extrapolation.md)).
  - HLC and Burgert test only at 10–70% noise, and gains shrink as noise falls ([qeg](../deep_dives/2026-09-04_qeg-multilabel-noise-correction.md)).
- **[lit]** No published result shows +1.5% from correction at ~3k studies.
  - BoMD gains 1–3% AUC, but on 86k–171k images ([qln](../deep_dives/2026-09-04_qln-label-noise-correction-medical-imaging.md)).
  - No confident-learning result on multi-label radiology moved held-out AUC by >0.01 ([confident_learning](../deep_dives/2026-09-02_label_noise_confident_learning.md)).
  - LNMBench finds methods degrade under real-world noise (same doc).
- **[lit]** No head-to-head of prune vs relabel vs down-weight with the detector held fixed has been published ([qeb](../deep_dives/2026-09-04_qeb-label-noise-treatment-literature.md)).
  - The "~5–10% crossover" is folklore (same doc).
  - The cost ratio k (false correction vs missed error) has never been measured ([label-error-detectors](../deep_dives/2026-09-05_label-error-detectors-vs-confident-learning.md)).
  - SimiFeat and SEMD rank errors better than CL on CIFAR (same doc).
- **[lit]** Relabel form: interpolation ŷ = α·p̂ + (1−α)·ỹ is the norm. Prevalence shift next to AUC is essentially never reported ([qee](../deep_dives/2026-09-04_qee-relabel-posterior-vs-interpolation.md)).
- **[lit]** Small mechanistic repairs: there is no published evidence that a sub-2% repair beats count-scaling. MEE harm (Yuan 2025) is measured at 40% noise ([qh](../deep_dives/2026-09-01_qh_small_label_defect_repair_evidence.md)).

## 5. A better labeler (parser / reader repair)

- **[lit]** A labeler upgrade carries downstream only where extraction was badly broken.
  - Jain 2021: labeler F1 0.55→0.73 moved AUROC 0.83→**0.87 (+0.04)**, concentrated in Enlarged Cardiomediastinum (+0.12–0.16) ([extractor-repair](../deep_dives/2026-09-11_extractor-repair-structured-noise-and-selective-repair.md)).
  - A text-only CheXbert re-label gave +0.00 ([visualchexbert](../deep_dives/2026-09-06_visualchexbert-mechanism-and-noisy-label-pricing.md)).
  - Wollek: rule labeler 0.858 → DL labeler 0.939 (+0.081) vs manual 0.934 ([qec](../deep_dives/2026-09-04_qec-parser-vs-label-noise.md)).
  - No paper compares parser repair head-to-head against a label-noise method (same doc).
- **[lit]** VisualCheXbert uses the image model as a SOFT relabeler whose targets are probabilities. Our extractor is already an LLM re-read, so expect a small prior ([visualchexbert](../deep_dives/2026-09-06_visualchexbert-mechanism-and-noisy-label-pricing.md)).
  - This reconciles with qec: Wollek's gain came from replacing a weak *rule* labeler.
- **[ours]** Label TABLE choice is CLOSED ([AXES:153](../../docs/AXES.md)). Ours is the worst of five community tables; v2 wins (README:14520, :17444).
  - The five tables agree unanimously on 93.9% of cells, and four of them are LLM extractions of the same reports ([multi-source](../deep_dives/2026-09-11_multi-source-label-merging-and-agreement-estimators.md)).
  - [label_tables_audit](../deep_dives/2026-09-02_rsna_label_tables_pseudo_label_audit.md) inventories them: blends, steven_v4, gxpixel, rayanbabur, flight0234, lixin73, stevenleehans, yunusgmsoy.
- **[ours]** Coarse quantisation costs AUC through ties. The public table quantised onto our five levels falls to **0.8445**, below ours ([synovitis-is-not-our-weak-label](../deep_dives/2026-08-26_synovitis-is-not-our-weak-label.md)).
- **[ours]** Hedge-aware soft targets are already the label ([AXES:125](../../docs/AXES.md)). Grades: absent 0.08, unclear 0.28, possible 0.68, present 0.82, severe 0.94.

## 6. LLM / VLM readers — CLOSED ([AXES:156](../../docs/AXES.md): "three findings, not twelve", README:15857)

- **[ours]** Qwen3.8-27B NVFP4 gate on 696 cells ([qwen38_gate](../deep_dives/2026-09-03_qwen38_reader_gate_result.md)):

  | Reader | Wrong | FP | Misses | Silent |
  |---|---|---|---|---|
  | Qwen3.8 | 122 | 48 | 74 | 30.2% |
  | Existing local reader | 121 | 83 | 38 | 21.0% |
  | Lexicon (684 cells) | 140 | 108 | 32 | — |

  - Qwen3.8 is better calibrated but does **not** beat the local reader on net errors.
  - v2 leaves **27.3%** of (study, finding) pairs UNK ([audit_protocol](../deep_dives/2026-09-03_label_audit_protocol_with_qwen38.md)).
  - API gotcha: `enable_thinking: False`.
- **[lit]** MedGemma licence: HAI-DEF §1.1(h) says "Outputs are not deemed Model Derivatives", so MedGemma outputs are usable as training labels ([medgemma_licence](../deep_dives/2026-09-03_medgemma_licence_and_domain_for_knee_mri.md)).
  - The shortlist withdrew MedGemma as a generator, then partly retracted that the same day.
  - Its hallucination numbers come from free-text generation and do not price constrained extraction. The "no vLLM checkpoint" finding applies only to the 27B, not the 4B ([vlm_shortlist](../deep_dives/2026-09-03_local_vlm_shortlist.md)).
- **[lit]** For VLM image reads, prefer fewer tiles at higher resolution over dense montages. Grading prompts under-call.
  - Nobody has published per-class leaderboard probing by constant substitution (the Whitehill oracle family is the nearest) ([vlm-montage](../deep_dives/2026-09-11_vlm-montage-prompting-and-small-gold-validation.md)).

## 7. Loss-side treatments (masking, correction, dependency, uncertainty)

- **[ours]** Label MASKING is CLOSED: both selectors cost, non-monotonically ([AXES:155](../../docs/AXES.md), README:14976, :15045).
- **[lit]** Forward loss correction with a *known* per-label 2×2 T exists (Natarajan 2013, Patrini 2017) ([loss-correction](../deep_dives/2026-09-11_loss-correction-with-measured-noise-matrix.md)).
  - Applying it per column in the multi-label case "follows from linearity" but is unpublished and untested.
  - FN-only noise is PU learning. FN-dominant noise is a monotone distortion, so per-label AUC ranking is largely preserved.
- **[lit]** SPML / G-AN is best on small CheXpert subsets of 5k–40k ([qln](../deep_dives/2026-09-04_qln-label-noise-correction-medical-imaging.md)).
- **[lit]** For multi-source merging, Raykar 2010 EM gives per-source sensitivity and specificity. Our asymmetric-confidence encoding is a hand approximation of it. Four correlated extractors break the triplet estimators ([multi-source](../deep_dives/2026-09-11_multi-source-label-merging-and-agreement-estimators.md)).
- **[lit]** Label-dependency modelling adds 0.4–2.0 pp macro AUC in CXR, mostly bundled with other changes ([qld](../deep_dives/2026-09-04_qld-label-dependency-macro-auc-medical-imaging.md)).
  - **[ours]** Our label-correlation residual head is REFUTED ([AXES:117](../../docs/AXES.md), README:2485).
- **[lit]** A separate confidence head loses to ensembles. EDL has ECE ~0.4, and the aleatoric/epistemic split is degenerate ([uncertainty_heads](../deep_dives/2026-08-31_report_label_noise_and_uncertainty_heads.md)).
- **[lit]** Pricing the supervision axis ([supervision_pricing](../deep_dives/2026-09-01_supervision_axis_pricing_for_multi_label_medical_auc.md)):
  - VoG curriculum: +0.0245 on elbow X-ray.
  - Label smoothing for uncertainty labels: ≈+0.01–0.02 (estimated).
  - Modality dropout: 0.000 AUC.
  - Denoising ranking by measured delta ([qbr](../deep_dives/2026-09-02_qbr-label-denoising-with-58-gold-studies.md)): better labeler > OOF relabel > sample selection > smoothing. 58 gold is "almost certainly too few" for CL transition matrices.

## 8. Manufacturing labels (pseudo-labels, blends, self-distillation)

- **[lit]** Semi-supervised pseudo-labelling on ChestX-ray14 at 20% labelled gains **+2.64%** mean AUC (70.81 vs 68.17), closing about 1/3 of the gap to full supervision ([pseudo_label_partial](../deep_dives/2026-09-02_label_noise_pseudo_label_partial_labels_literature.md)).
- **[lit]** Medical VLP (MedCLIP) decouples image and text and is data-efficient at 20k pairs. It is not priced for us ([missing_gold_labels](../deep_dives/2026-09-02_missing_gold_labels_and_medical_vlp_axis.md)).
- **[unv]** Forum: "labels are still my greatest lever, especially pseudo labeling and blending different labels". Two entries trained on LLM output labels only reach 0.926–0.929 five-fold. The best single claim is 0.943 against our 0.946 champion. A quoted "0.949 small ResNet" is not in the thread and was withdrawn ([rival_single_models](../deep_dives/2026-08-26_rival_single_models_and_the_label_bottleneck.md), 09-16 re-read).
- **[ours]** AXES:157: `noisy_student` scored 0.8548 and `pool_specialise` was FLAT. That run used *external* data as labelled augmentation, not in-domain pseudo-labels on the 4,349 report-only studies.
- **[ours]** In-domain soft pseudo-labels HAVE run: `--teacher-targets` from our own out-of-fold predictions over the 4,349 studies, `--distill-weight 0.7`. The leak-free student `seat_clean_distill` reads 0.8634 vs `band_dense` 0.8578 = **+0.0056**, under the 0.0062 floor; gen-2 teachers add +0.0007 ([experiments/README.md ~8566, ~14050](../../experiments/README.md); knee-zwcyk). The ensemble-teacher headroom is 0.002 on the board (AXES:160). So pseudo-labelling is measured and sub-floor, not unrun.

---

## 9. Report text as a training signal

- **Measured by us:** an auxiliary cosine loss toward whole-report embeddings lost monotonically, −0.0063 ([text_supervision_mechanism_audit](../deep_dives/2026-09-01_text_supervision_mechanism_audit.md)). See also [the report lane is a partial shortcut](../../interpretations/kneemri/2026-09-17_the_report_lane_is_a_partial_shortcut.md).
- **Literature:** almost all published text gains come from contrastive image–report *pretraining* (ConVIRT, CheXzero, GLoRIA, BioViL). Little evidence supports a text alignment loss during supervised fine-tuning on good labels ([same](../deep_dives/2026-09-01_text_supervision_mechanism_audit.md)).
- **Public 0.906 baseline:** the multilingual MiniLM report embedding is concatenated to the vision features and trained with plain BCE. There is no coupling loss. The model is trained vision-only first, then with text. `test.csv` has no `Report` column, so text is used at training time only. The repo is a starter skeleton; the 0.906 is from the Models tab and is not reproduced ([public_baseline_text_branch_objective](../deep_dives/2026-09-02_public_baseline_text_branch_objective.md)).
- **Circularity:** when the labels were extracted from the same reports, a text teacher mostly relearns the extractor. The documented precedent is a *better labeler*, such as VisualCheXbert over CheXpert/CheXbert (Jain et al. 2021), not text-to-vision distillation. No paper was found that beats the extractor with a report-trained text teacher ([text_as_privileged_information_qac](../deep_dives/2026-09-02_text_as_privileged_information_qac.md)). That points back to §5 (a better labeler) and §4 (correction, CLOSED).

## Contradictions and supersessions

1. **errors_dense.** [confident_learning](../deep_dives/2026-09-02_label_noise_confident_learning.md) says it ran and scored 0.8420. [qeb](../deep_dives/2026-09-04_qeb-label-noise-treatment-literature.md) says it had not run. AXES:154 now says pruning costs.
2. **Is correction live?** [qao](../deep_dives/2026-09-02_test_labels_report_derived_or_radiologist_reviewed_qao.md), [qap](../deep_dives/2026-09-02_qap_adjudication_protocol_and_58_gold_pricing.md) and [extractor-repair](../deep_dives/2026-09-11_extractor-repair-structured-noise-and-selective-repair.md) say "live lever". [state_of_labels](../deep_dives/2026-09-03_state_of_labels_and_model_axes.md), [qeh](../deep_dives/2026-09-04_qeh-noise-rate-curve-extrapolation.md) and AXES:154 say closed. AXES wins.
3. **Qwen3.8.** [vlm_shortlist](../deep_dives/2026-09-03_local_vlm_shortlist.md) calls it the "best single-model candidate". [qwen38_gate](../deep_dives/2026-09-03_qwen38_reader_gate_result.md) finds it does not beat the local reader on net errors (122 vs 121).
4. **MedGemma.** [vlm_shortlist](../deep_dives/2026-09-03_local_vlm_shortlist.md) withdraws it and then partly retracts the withdrawal inside the same file.
5. **SE of a gold-58 AUC.** [vlm-montage](../deep_dives/2026-09-11_vlm-montage-prompting-and-small-gold-validation.md) gives ≈0.12. [report-error-gradient](../deep_dives/2026-09-11_report-error-gradient-vs-gold58-ordering.md) gives ≈0.055, with different n+ and AUC.
6. **Is pseudo-labelling measured?** [rival_single_models](../deep_dives/2026-08-26_rival_single_models_and_the_label_bottleneck.md) (09-16) calls it untested. It is measured: own-OOF self-distillation +0.0056, sub-floor (§8).
7. **Test-label protocol.** [qao](../deep_dives/2026-09-02_test_labels_report_derived_or_radiologist_reviewed_qao.md) and [qap](../deep_dives/2026-09-02_qap_adjudication_protocol_and_58_gold_pricing.md) agree the test is radiologist-read. Neither gives a reader count. `README:15833` says the opposite ("the scored target is THE READER"), but its evidence is two flat board entries whose treatment touched 7.8% of the defect. That is under the board's 0.001 resolution under either hypothesis, so it cannot decide the question; the primary source stands ([scored-target](../deep_dives/2026-09-28_scored-target-and-loss-side-noise.md)). The AXES correction closure stands on its other grounds.
8. **Labeler gains.** [visualchexbert](../deep_dives/2026-09-06_visualchexbert-mechanism-and-noisy-label-pricing.md) and [qec](../deep_dives/2026-09-04_qec-parser-vs-label-noise.md) reconcile: the gain comes from replacing a *weak rule* labeler, which we do not have.

## Open questions

- Why does Effusion rank third-worst on gold when the literature expects it clean? Candidates: a threshold mismatch, since the annotator wants moderate/large ([report-error-gradient](../deep_dives/2026-09-11_report-error-gradient-vs-gold58-ordering.md)).
- Is test-label noise masking the gap? AUC_obs = (1−ρ)·AUC* + ρ/2 ([label-noise-ceiling](../deep_dives/2026-09-11_label-noise-ceiling-and-report-vs-pixels-error.md)).
- ~~Per-column forward correction with a gold-measured T is unpublished and untested~~ — RETIRED on theory 2026-09-28. A per-label T is class-conditional noise, which leaves the ranking unchanged (Menon 2015), so its AUC ceiling is ~0 ([scored-target](../deep_dives/2026-09-28_scored-target-and-loss-side-noise.md)).

## Sources

- [2026-08-23_published-per-label-auc-vs-ours](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)
- [2026-08-24_published_knee_mri_multilabel_sota](../deep_dives/2026-08-24_published_knee_mri_multilabel_sota.md)
- [2026-08-26_rival_single_models_and_the_label_bottleneck](../deep_dives/2026-08-26_rival_single_models_and_the_label_bottleneck.md)
- [2026-08-26_synovitis-is-not-our-weak-label](../deep_dives/2026-08-26_synovitis-is-not-our-weak-label.md)
- [2026-08-31_report_label_noise_and_uncertainty_heads](../deep_dives/2026-08-31_report_label_noise_and_uncertainty_heads.md)
- [2026-09-01_qh_small_label_defect_repair_evidence](../deep_dives/2026-09-01_qh_small_label_defect_repair_evidence.md)
- [2026-09-01_supervision_axis_pricing_for_multi_label_medical_auc](../deep_dives/2026-09-01_supervision_axis_pricing_for_multi_label_medical_auc.md)
- [2026-09-02_extractor_error_taxonomy_58_gold](../deep_dives/2026-09-02_extractor_error_taxonomy_58_gold.md)
- [2026-09-02_label_noise_confident_learning](../deep_dives/2026-09-02_label_noise_confident_learning.md)
- [2026-09-02_label_noise_pseudo_label_partial_labels_literature](../deep_dives/2026-09-02_label_noise_pseudo_label_partial_labels_literature.md)
- [2026-09-02_missing_gold_labels_and_medical_vlp_axis](../deep_dives/2026-09-02_missing_gold_labels_and_medical_vlp_axis.md)
- [2026-09-02_qap_adjudication_protocol_and_58_gold_pricing](../deep_dives/2026-09-02_qap_adjudication_protocol_and_58_gold_pricing.md)
- [2026-09-02_qbr-label-denoising-with-58-gold-studies](../deep_dives/2026-09-02_qbr-label-denoising-with-58-gold-studies.md)
- [2026-09-02_rsna_label_tables_pseudo_label_audit](../deep_dives/2026-09-02_rsna_label_tables_pseudo_label_audit.md)
- [2026-09-02_test_labels_report_derived_or_radiologist_reviewed_qao](../deep_dives/2026-09-02_test_labels_report_derived_or_radiologist_reviewed_qao.md)
- [2026-09-03_label_audit_protocol_with_qwen38](../deep_dives/2026-09-03_label_audit_protocol_with_qwen38.md)
- [2026-09-03_local_vlm_shortlist](../deep_dives/2026-09-03_local_vlm_shortlist.md)
- [2026-09-03_medgemma_licence_and_domain_for_knee_mri](../deep_dives/2026-09-03_medgemma_licence_and_domain_for_knee_mri.md)
- [2026-09-03_qwen38_reader_gate_result](../deep_dives/2026-09-03_qwen38_reader_gate_result.md)
- [2026-09-03_state_of_labels_and_model_axes](../deep_dives/2026-09-03_state_of_labels_and_model_axes.md)
- [2026-09-04_58-vs-57-study-denominator-mismatch](../deep_dives/2026-09-04_58-vs-57-study-denominator-mismatch.md)
- [2026-09-04_qeb-label-noise-treatment-literature](../deep_dives/2026-09-04_qeb-label-noise-treatment-literature.md)
- [2026-09-04_qec-parser-vs-label-noise](../deep_dives/2026-09-04_qec-parser-vs-label-noise.md)
- [2026-09-04_qee-relabel-posterior-vs-interpolation](../deep_dives/2026-09-04_qee-relabel-posterior-vs-interpolation.md)
- [2026-09-04_qeg-multilabel-noise-correction](../deep_dives/2026-09-04_qeg-multilabel-noise-correction.md)
- [2026-09-04_qeh-noise-rate-curve-extrapolation](../deep_dives/2026-09-04_qeh-noise-rate-curve-extrapolation.md)
- [2026-09-04_qld-label-dependency-macro-auc-medical-imaging](../deep_dives/2026-09-04_qld-label-dependency-macro-auc-medical-imaging.md)
- [2026-09-04_qln-label-noise-correction-medical-imaging](../deep_dives/2026-09-04_qln-label-noise-correction-medical-imaging.md)
- [2026-09-05_label-error-detectors-vs-confident-learning](../deep_dives/2026-09-05_label-error-detectors-vs-confident-learning.md)
- [2026-09-06_visualchexbert-mechanism-and-noisy-label-pricing](../deep_dives/2026-09-06_visualchexbert-mechanism-and-noisy-label-pricing.md)
- [2026-09-11_extractor-repair-structured-noise-and-selective-repair](../deep_dives/2026-09-11_extractor-repair-structured-noise-and-selective-repair.md)
- [2026-09-11_label-noise-ceiling-and-report-vs-pixels-error](../deep_dives/2026-09-11_label-noise-ceiling-and-report-vs-pixels-error.md)
- [2026-09-11_loss-correction-with-measured-noise-matrix](../deep_dives/2026-09-11_loss-correction-with-measured-noise-matrix.md)
- [2026-09-11_multi-source-label-merging-and-agreement-estimators](../deep_dives/2026-09-11_multi-source-label-merging-and-agreement-estimators.md)
- [2026-09-11_report-error-gradient-vs-gold58-ordering](../deep_dives/2026-09-11_report-error-gradient-vs-gold58-ordering.md)
- [2026-09-11_vlm-montage-prompting-and-small-gold-validation](../deep_dives/2026-09-11_vlm-montage-prompting-and-small-gold-validation.md)
- [docs/AXES.md](../../docs/AXES.md) rows C:117, D:125, E:153–157
- [2026-09-01_text_supervision_mechanism_audit](../deep_dives/2026-09-01_text_supervision_mechanism_audit.md)
- [2026-09-02_public_baseline_text_branch_objective](../deep_dives/2026-09-02_public_baseline_text_branch_objective.md)
- [2026-09-02_text_as_privileged_information_qac](../deep_dives/2026-09-02_text_as_privileged_information_qac.md)

## 2026-09-28: the loss side of label noise was never tried

Every label-noise lever so far changed the targets, or masked them (masking is loss-side, and CLOSED). No noise-ROBUST loss (ELR, GCE, SCE) has run. Follow-up [scored-target](../deep_dives/2026-09-28_scored-target-and-loss-side-noise.md): the test is radiologist-read, so ELR aims at the right target. Read it on gold-58 as well as ledger75; T-correction is dead on ranking theory. ELR is unrun; it is priced at P ~0.08 in knee-bpuei. Deep dive: [2026-09-28 gap audit and frontier sweep](../deep_dives/2026-09-28_gap-audit-and-frontier-sweep.md).
