# Evaluation metrics, rulers and offline→board transfer — topic summary

**Current verdict.** The good ruler is **ledger75**, with a noise floor of **0.0062**. On ledger75 the offline→board slope is about **+1.10**, with residual sd about **0.0013–0.0017** (CLAUDE.md, 2026-09-17).
**A bare number in this repo is midpoint unless labelled** ([AXES:215](../../docs/AXES.md)).
Transfer constants are **READ** from `python -m kneemri.kaggle.board_check` / `board_transfer`, and **never quoted**. **+4.67, +2.14, +1.55 and +1.82 were all wrong quotes**, and so is any constant in the deep dives below ([AXES:216](../../docs/AXES.md)).
The metric is unweighted macro ROC AUC over 12 labels. It is rank-based, so calibration is worth zero by construction.

Tags: **[M]** measured by us · **[L]** literature or host text · **[U]** unverified.

---

## 1. The metric

- **[L] Definition.** Macro ROC AUC over 12 binary labels: one AUC per label, then an unweighted mean ([09-04 qmw](../deep_dives/2026-09-04_qmw-rsna-knee-2026-metric-definition.md)).
  - Sources: the sahil4818 boilerplate; community notebooks (beicicc, romanrozen, salemali7); konradcinkusz's repo, which says it verified against the Data tab.
  - No official page was readable as plain HTML because the page is JS-rendered, so this is **[U]** against a primary host text.
- **[M] Rank invariance.** A monotone transform of the champion's table scored the champion's exact number ([09-04 qmw](../deep_dives/2026-09-04_qmw-rsna-knee-2026-metric-definition.md)).
  - So Platt scaling or isotonic regression moves nothing unless it reorders predictions within a label.
  - Calibration and thresholds are "worth zero by construction" ([AXES:202](../../docs/AXES.md)).
  - Serve-time column mixing is NULL ([AXES:201](../../docs/AXES.md)).
- **[L] RSNA precedent differs** ([09-04 qmw](../deep_dives/2026-09-04_qmw-rsna-knee-2026-metric-definition.md)):
  - 2023 Abdominal Trauma: weighted log loss, weights 1/2/4/6 plus any_injury 6.
  - 2024 Lumbar Spine: sample-weighted log loss, weights 1/2/4, plus any_severe_spinal.
  - 2026 Knee is the first of the three to use AUC with equal label weight.
  - Consequence: every label is worth 1/12, and rare labels (Fracture, Baker's) count as much as Effusion.
- **[L] Board resolution is 0.001, the competition's own** ([AXES:212](../../docs/AXES.md)). Board blur on tied pairs is 0.0020 ([AXES:213](../../docs/AXES.md)).

## 2. Rulers, floors and denominators

- **[M] Noise floors** ([AXES:210-211](../../docs/AXES.md)):
  - Seed spread: **0.0062**, the binding bar used here.
  - Paired holdout floor (Hanley-McNeil): 0.0053.
- **[M] Midpoint vs ledger75** ([AXES:215](../../docs/AXES.md)):
  - The midpoint (0.5-cut) ruler is "seven times blinder" than ledger75.
  - The cut alone moves the honest12 roster by +0.026 (CLAUDE.md, knee-f9aao).
  - Every scalar should carry statistic, population and cut, e.g. `solo@ledger75`.
- **[M] Gold denominator trap: 58 vs 57** ([09-04 58v57](../deep_dives/2026-09-04_58-vs-57-study-denominator-mismatch.md)):
  - `report_labels_v2.csv` and `v3` hold 4,406 studies and lack gold study `…73527530686853911124431549317032662220`.
  - `v4`, `local_reader_annotated.csv` and the qwen38 annotated file hold all 58.
  - Any "on 58 gold" score against v2 is on **57** unless the intersection is explicit.
  - Label-table choice is closed with v2 winning ([AXES:153](../../docs/AXES.md)), so this trap is live wherever v2 is the ruler.
- **[M] Probe records are off-ruler.** `feature_probe` writes a binarised truth, so a probe can never be recut to ledger75 ([AXES:224-230](../../docs/AXES.md)).
- **[L] Resolution of the gold-58 ruler** ([09-11](../deep_dives/2026-09-11_broad-macro-gains-high-baseline-and-concordance-rulers.md)):
  - Hanley–McNeil at AUC 0.9, 27 pos / 31 neg: SE 0.043, 95% half-width **±0.085**. D324's ±0.19–0.24 was wrong and was corrected.
  - N=4,407 gives ±0.012.
  - 0 errors in 58 bounds error at ≤5% (rule of three).
  - A stratified audit of ~400 cells bounds error at (k+3)/400.
- **[L] Residual ruler error binds before sampling error** ([09-11](../deep_dives/2026-09-11_broad-macro-gains-high-baseline-and-concordance-rulers.md)):
  - Northcutt 2021: ≥3.3% test-label error, and close-model rankings flip at ~6% mislabel.
  - Beyer 2020 finds the same.
  - Bernhardt: a 40%-noisy validation set reversed a ranking (46.5% vs 65.8%) ([09-10 unseen](../deep_dives/2026-09-10_unseen-ceiling-breakers-discovery.md)).
- **[M] Proxy vs annotation gap.** Identical weights score +0.045 higher on the proxy (report) ruler than on the annotation ruler ([09-10 mapped](../deep_dives/2026-09-10_unseen-candidates-mapped-onto-our-measurements.md)).
- **[L] Concordance-built rulers need independent errors** ([09-11](../deep_dives/2026-09-11_broad-macro-gains-high-baseline-and-concordance-rulers.md)):
  - Hui–Walter requires conditional independence of errors.
  - Albert & Dodd 2004: dependence cannot be diagnosed from the tables' own agreement, so the check must be external.
  - Sorin 2025: cross-modality concordance, 96.3% accepted, residual ≲0.75%.
  - Funk 2015: same-method NLP agreement reached precision ≤18%.
  - Five same-report, same-LLM-family tables sit nearer Funk than Sorin, so unanimous-wrong is ~ρ·ε, not ε⁴.
- **[L] Evaluation economics** ([09-10 round3](../deep_dives/2026-09-10_round3-local-unrun-and-evaluation-economics.md)):
  - Roelofs 2019: no widespread overfitting across 100+ competitions.
  - Mania 2019: model similarity mitigates holdout reuse.
  - Recht 2019: rankings largely preserved on fresh test sets.
  - The external-cohort flips counterpoint is about cohort shift, not reuse.
  - EACL 2026 (2512.08121): gold 25→51,000 plateaus after ~1–2k. Gallas 2006 MRMC; Ju 2022.

## 3. Offline→board transfer

- **Binding (CLAUDE.md 2026-09-17):**
  - On ledger75 the slope is about +1.10, flat from the whole record into our own band.
  - Residual sd is about 0.0013–0.0017.
  - The midpoint "steepening" in the top band is an artefact of the midpoint ruler.
  - Read the current value with `board_check --live --ruler ledger75`; never carry it forward from prose.
- **[M] 09-18 playbook doc:** "CV-LB trust: board_transfer, ledger75; residual sd 0.0013" ([09-18](../deep_dives/2026-09-18_kaggle-playbook-vs-our-ledger.md)). Consistent with the binding value.
- **[M, SUPERSEDED] Q-TS (09-04) built on a slope of 0.82** ([09-04 qts](../deep_dives/2026-09-04_qts-offline-leaderboard-transfer-slope.md)).
  - It also said the top five span +0.0047 offline vs +0.0005 on the board.
  - That 0.82 is a quoted older constant, on the midpoint ruler with in-sample rows still in the fit. **Do not use it; read `board_check`.**
  - Its literature half stands [L]:
    - Recht CIFAR-10.1 `1.62·acc − 65.5%` and ImageNetV2 `1.11·acc − 20.2%`: slopes >1, meaning no adaptive overfitting.
    - Roelofs names the plateau signature.
    - Akhtar ICML 2026 defines stagnation, saturation and the saturation index.
  - Its three candidate mechanisms: distribution gap, ceiling/saturation, holdout overfit.
  - Its quintile diagnostic was proposed and then run (see §4).
- **[M, SUPERSEDED] Q-HC (09-04) quotes band slopes:** hard band (Q4-Q5) ~0.73–0.75, easy band (Q1) ~0.44, with a "0.009 gap" to a 0.953 leader ([09-04 qhc](../deep_dives/2026-09-04_qhc-hard-band-and-empirical-ceiling.md)).
  - All three are midpoint-era quoted constants. Read the bands from the tool.
  - Its "saturation, not distribution gap" diagnosis was made on that ruler.
  - CLAUDE.md warns that band pair counts and floors are quantile artefacts and must not be quoted.
- **[M] Generalisation toll** ([AXES:217-221](../../docs/AXES.md)):
  - Flat at −0.0336 ± 0.0038 across eleven members.
  - Offline ordering transfers at Spearman ρ = 0.8818.
  - Every axis was closed in-domain while the board is out-of-domain to it.
  - 09-24 restates the flat shift as −0.034 across 11 members ([09-24](../deep_dives/2026-09-24_all-lanes-web-sweep.md)).
- **[M] AXES top-band residual row reads "0.0022 ... ledger75"** ([AXES:214](../../docs/AXES.md)).
  - CLAUDE.md gives 0.0022 as the **midpoint** detrended top-band sd. The ledger75 figures are 0.0013 (top band) and 0.0017 (whole record).
  - The AXES label is inconsistent with CLAUDE.md. Prefer CLAUDE.md's ledger75 0.0013–0.0017.

## 4. Ceilings: is the gap headroom or noise?

- **[L] No published estimator turns a submission record into a ceiling** ([09-04 qhc](../deep_dives/2026-09-04_qhc-hard-band-and-empirical-ceiling.md)). The adjacent tools:
  - Dawid-Skene over entries. Overestimates, because entries are correlated.
  - Bayes-error estimators (Berisha 2019, Chen 2025, Ushio 2026). These estimate error rates, not multi-label AUC.
  - An inter-rater agreement bound from gold (Bucarelli CVPR 2023). Recommended as the cheapest credible read.
- **[L] Hard-band losses** ([09-04 qhc](../deep_dives/2026-09-04_qhc-hard-band-and-empirical-ceiling.md)):
  - RAL: 0.351 mAP / 0.837 mAUC on CXR-LT.
  - ASL.
  - AUC-margin (CheXpert5000).
  - Joint-ensemble diversity collapses by learner collusion (Jeffares 2023).
  - [M] Both focal ([AXES:112](../../docs/AXES.md)) and AUC-margin, single-stage and two-stage, are CLOSED ([AXES:107](../../docs/AXES.md)).
- **[L] Per-finding agreement and AUC ceilings** ([09-05](../deep_dives/2026-09-05_knee-mri-column-ceilings.md)):
  - MRNet Fleiss κ: abnormal 0.508, ACL 0.800, meniscus 0.745.
  - Kim 2023 (Baker's-cyst cohort, selection-inflated) Cohen κ: ACL 0.925, MCL 0.899, MM 0.922, LM 0.938, effusion 0.917, OA 0.830.
  - Xie 2025 κ 0.921–0.988.
  - MRNet AUC: 0.937 / 0.965 / 0.847. Rijeka ACL 0.824 without retraining, 0.911 retrained. Partial ACL as low as 0.58.
  - Takeaway: no clean literature substitute for a per-column ceiling.
  - [U] MCL (κ 0.90–0.97) is more likely model- or definition-limited. Meniscus and partial ACL are more likely label-limited.
  - [U] "Baker's board AUC ~0.992" is not a board number we can see per column. Treat it as unverified.
- **[L] High-baseline regime** ([09-11](../deep_dives/2026-09-11_broad-macro-gains-high-baseline-and-concordance-rulers.md)):
  - **No published +0.01–0.03 broad macro gain from a ≥0.93 baseline.** Gains follow delta ∝ (1 − baseline): ELNet, TransMed and Belton multi-plane on MRNet.
  - Azizi SSL +0.011 from ~0.88.
  - Resolution plateaus at 256–448. [M] Our 336→392 was −0.0002, as the curve predicts ([AXES:75](../../docs/AXES.md)).
  - LibAUC won CheXpert by ~+0.001–0.005.
  - At 0.93–0.99, published single-model interventions read as ±0.005 noise.
- **[M] Per-label published gap on the annotation ruler.** The ceiling is +0.025 macro if LM and ACL reached the literature; see [competition summary §6](competition-landscape-and-frontier.md) ([08-23](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)).

## 5. Label dependency under macro AUC

- **[L] Label-dependency modelling moves macro AUC modestly, ~0.4–2.0 pp, and always bundled with other changes** ([09-04 qld](../deep_dives/2026-09-04_qld-label-dependency-macro-auc-medical-imaging.md)):
  - Srivastava 2025: sparse L×L coupling `z′ = z + α·σ(z)·A`, 0.849 ChestX-ray14.
  - Wu ICLR 2020: +0.4% AUC, +1.0% under 20% label noise.
  - Pan 2026: 0.79→0.81.
  - Gains concentrate on rare and subtle labels.
  - No evidence at ~3k studies; all cited work uses 80k–170k images.
- **[M] Our versions are closed:**
  - Label-correlation / residual head: REFUTED for zero card ([AXES:117](../../docs/AXES.md)).
  - Serve-time column mixing: NULL ([AXES:201](../../docs/AXES.md)).
  - The surviving blend `Fracture + 0.25·Contusion` that Q-LD addressed is superseded by that null.

## 6. Efficiency metric

- **[L] Formula, quoted from the host page** ([09-06](../deep_dives/2026-09-06_efficiency-formula-and-mars-census.md)): `Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`, minimised.
  - Benchmark ≈ 0.5.
  - maxAUC is endogenous: the best private AUC over all submissions.
  - 32400 s = the 9 h kernel cap.
  - The public efficiency LB is rank-only and updates daily.
- **[M] Exchange rate.** At maxAUC 0.95, 1 h of runtime ≈ 0.05 AUC-equivalent ([09-06](../deep_dives/2026-09-06_efficiency-formula-and-mars-census.md)).
  - Later priced as 0.001 AUC = 70.9 s, and the lane was CLOSED ([AXES:203](../../docs/AXES.md)).
- **[L] MARS census** (same doc): 3D Swin, SSL on 336,476 MRI volumes including knee, Apache-2.0, 78.3 MiB. Hard-3D 1 mm isotropic 96³ input, which destroys our slice anisotropy.

## 7. Instruments (see what a model does, not change it)

- **[L/proposal] Three instruments proposed as rulers, not levers** ([08-30](../deep_dives/2026-08-30_analysis-instruments-distance-divergence-attribution.md)):
  - Mahalanobis on embeddings, replacing prediction-space Spearman seat distance. Needs Ledoit-Wolf shrinkage and must reproduce validated seat decisions first. knee-iyad.
  - Jensen-Shannon between members: free, and sees the confidence that rank correlation cannot. knee-c2ew.
  - Captum saliency, sanity-checked (Adebayo) against occlusion, for per-study triage only, as an optional extra. knee-l1vy.
  - The doc's KL-to-test caveat: divergence to the hidden test set is unknowable.
- **[M] JS distance is now in use.** AXES quotes js values, e.g. orthodiff 0.0418 vs roster 0.0150 ([AXES:41](../../docs/AXES.md)). Seat questions are out of scope for a solo ([AXES:182](../../docs/AXES.md)).

## Open questions

- There is no per-label inter-rater κ on gold-58 in any doc. It is the "cheapest credible ceiling" (Q-HC) and is still unread.
- The AXES:214 label (0.0022 "ledger75") disagrees with CLAUDE.md (0.0022 midpoint; 0.0013 ledger75 top band). Fix the index row.
- Which live scripts still score gold on v2 (57 studies) rather than v4 (58)?
- A PPI-style CI for AUC combining 4,407 machine labels and 58 gold is NULL in the literature (09-11).
- Mahalanobis and Captum instruments: unbuilt.

## Sources

- [2026-08-23_published-per-label-auc-vs-ours.md](../deep_dives/2026-08-23_published-per-label-auc-vs-ours.md)
- [2026-08-30_analysis-instruments-distance-divergence-attribution.md](../deep_dives/2026-08-30_analysis-instruments-distance-divergence-attribution.md)
- [2026-09-04_qmw-rsna-knee-2026-metric-definition.md](../deep_dives/2026-09-04_qmw-rsna-knee-2026-metric-definition.md)
- [2026-09-04_qts-offline-leaderboard-transfer-slope.md](../deep_dives/2026-09-04_qts-offline-leaderboard-transfer-slope.md)
- [2026-09-04_qhc-hard-band-and-empirical-ceiling.md](../deep_dives/2026-09-04_qhc-hard-band-and-empirical-ceiling.md)
- [2026-09-04_58-vs-57-study-denominator-mismatch.md](../deep_dives/2026-09-04_58-vs-57-study-denominator-mismatch.md)
- [2026-09-04_qld-label-dependency-macro-auc-medical-imaging.md](../deep_dives/2026-09-04_qld-label-dependency-macro-auc-medical-imaging.md)
- [2026-09-05_knee-mri-column-ceilings.md](../deep_dives/2026-09-05_knee-mri-column-ceilings.md)
- [2026-09-06_efficiency-formula-and-mars-census.md](../deep_dives/2026-09-06_efficiency-formula-and-mars-census.md)
- [2026-09-10_round3-local-unrun-and-evaluation-economics.md](../deep_dives/2026-09-10_round3-local-unrun-and-evaluation-economics.md)
- [2026-09-10_unseen-candidates-mapped-onto-our-measurements.md](../deep_dives/2026-09-10_unseen-candidates-mapped-onto-our-measurements.md)
- [2026-09-10_unseen-ceiling-breakers-discovery.md](../deep_dives/2026-09-10_unseen-ceiling-breakers-discovery.md)
- [2026-09-11_broad-macro-gains-high-baseline-and-concordance-rulers.md](../deep_dives/2026-09-11_broad-macro-gains-high-baseline-and-concordance-rulers.md)
- [2026-09-18_kaggle-playbook-vs-our-ledger.md](../deep_dives/2026-09-18_kaggle-playbook-vs-our-ledger.md)
- [2026-09-24_all-lanes-web-sweep.md](../deep_dives/2026-09-24_all-lanes-web-sweep.md)
- Cross-check: [docs/AXES.md](../../docs/AXES.md)

- [2026-09-30_single-model-thread-reread-and-biohub-transfer.md](../deep_dives/2026-09-30_single-model-thread-reread-and-biohub-transfer.md) — selection rule, public vs private, the report-label ruler cannot see pseudo-label training (knee-p0s4f)

- [2026-09-28_scored-target-and-loss-side-noise.md](../deep_dives/2026-09-28_scored-target-and-loss-side-noise.md) — gold-58 as a ruler
