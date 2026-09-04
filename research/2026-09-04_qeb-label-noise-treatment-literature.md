# Q-EB — Label-noise treatment literature: prune vs. re-label vs. down-weight

Claude's Q-EB (2026-09-04T1528Z) asks for published evidence on how to *treat* detected label noise in multi-label medical classification: prune, re-label, or down-weight; noise-rate crossover; asymmetric false-positive noise; and rare-class pruning traps.

## Project evidence already in the repo

- `errors_dense` is armed as the project's prune arm: it would drop the 5,360 cells (10.1% of ~52,872) that cleanlab's confident joint flags. **Correction (2026-09-04T1535Z): the arm has not yet run; the prior project note claiming it ran and produced 0.8420 appears premature.** The detection gate is green, but there is no measured treatment result yet.
- The detection gate is green: cleanlab flags beat random 4.05× on 108 gold disagreements (precision 0.640 vs. random 0.158, AUC 0.7580). The detector works; whether the treatment (pruning) works is unmeasured.
- The asymmetry in the gold-58 sample is 108 confident false positives vs. 32 misses — report over-calls dominate (`experiments/arms/errors_dense.sh`).
- The only label correction that survived measurement is `Fracture := monotone_rescale(Fracture + 0.25·Contusion)`, worth +0.0053 on mean gold AUC (`interpretations/kneemri/2026-09-02_what-label-correction-is-actually-worth.md`).

## (1) Head-to-head prune / re-label / down-weight under the same detector in multi-label medical imaging

**Honest null: I did not find a published head-to-head that holds the detector fixed and varies only the treatment in a multi-label medical imaging task.**

Papers that come close but do not satisfy the exact request:

- **BoMD** (Chen et al., ICCV 2023) is the closest medical multi-label example. It detects noisy samples in chest X-ray datasets and *smoothly re-labels* them via a KNN graph in a semantic descriptor space. It reports gains over BCE and NVUM on NIH/CheXpert→OpenI/PadChest, and on a systematic benchmark with controlled symmetric noise it beats NVUM by 3–5% AUC at `ps = 20%`, 1–4% at `ps = 40%`, but loses 2.8% at `ps = 60%, pl = 60%`. It is not a prune/relabel/down-weight comparison under one detector; it is one method (detect + smooth re-label) against baselines. ([BoMD paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_BoMD_Bag_of_Multi-label_Descriptors_for_Noisy_Chest_X-ray_Classification_ICCV_2023_paper.pdf))
- **TLCM** (Ghiassi et al., ACML 2023) is multi-label but uses a *loss-correction* (corruption matrix) approach, not a prune/relabel/down-weight comparison. It beats ASL/MPVAE by 12.5%/26.3% mAP at 30%/60% corruption on MS-COCO, NUS-WIDE, MIRFLICKR. Not medical, not a treatment ablation. ([TLCM paper](https://proceedings.mlr.press/v189/ghiassi23a.html))
- **cleanlab / Confident Learning** (Northcutt et al., JAIR 2021) proposes five *pruning* methods and trains on cleaned data with class reweighting. It does not compare the same detected set under re-labeling or instance down-weighting; the framework is pruning-first. ([JAIR paper](https://www.jair.org/index.php/jair/article/download/12125/26676/26691))

Conclusion for the project: the strongest evidence will be endogenous once `errors_dense` runs — it is the head-to-head the literature is missing. Until then, the published literature does not establish whether pruning wins or loses on this data.

## (2) Noise-rate crossover: when does pruning lose to re-labeling?

**No published "crossover curve" at a specific noise-rate threshold exists.** The "~5–10%" rule mentioned in the project note appears to be empirical folklore, not a cited result.

Signals from the literature that support a crossover at moderate-to-high noise:

- **Zheng et al., "Meta Label Correction" (AAAI 2021)** directly compares reweighting (Meta-WN) vs. correction (MLC). Figure 5 shows that reweighting's "performance decreases significantly when the noise level goes up... while label correction based methods (GLC and MLC) show to be robust against severe label noises." This is multi-class (CIFAR-10/100, Clothing1M, text), not multi-label medical, but it is the cleanest detector-fixed comparison of two treatments. ([AAAI paper](https://cdn.aaai.org/ojs/17319/17319-13-20813-1-2-20210518.pdf))
- **Zhang et al., "Distilling Effective Supervision from Severe Label Noise" (CVPR 2020)** states: "Our proposed meta re-labeling is very effective for high noise ratios. We observe comparable performance of models without re-labeling at low noise ratios... indicating higher effectiveness of meta re-labeling given higher noise ratios, however, less effectiveness at low noise ratios." Again multi-class, but explicit about the crossover direction. ([CVPR paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Distilling_Effective_Supervision_From_Severe_Label_Noise_CVPR_2020_paper.pdf))
- **Nodet et al., "Recovering by Relabelling" (IAL 2021)** gives the mechanism: corrected samples contribute full gradients, while reweighted noisy samples get low weight and contribute little. This is a theoretical argument for why re-labeling wins when the flagged set is large enough that dropping it wastes signal. ([CEUR-WS paper](https://ceur-ws.org/Vol/3079/ial2021_paper8.pdf))

Project reading: at 10.1% flagged cells and a gold-observed pooled error rate of ~20.8%, the data sits on the high side of the empirical rule, consistent with pruning losing.

## (3) Asymmetric label-noise treatment (false positives vs. misses)

**No dedicated medical multi-label method that treats false-positive and false-negative noise asymmetrically was found.** The standard approach is to encode the asymmetry into a class-conditional noise matrix and correct the loss or the label.

- **cleanlab's confident joint** explicitly estimates the asymmetric joint `p(noisy label, true label)`; off-diagonal entries are asymmetric error counts. Northcutt et al. evaluate on asymmetric CIFAR-10 noise and report robustness. The framework supports the asymmetry but does not prescribe a differential treatment for FP vs. FN beyond the per-class thresholds. ([JAIR paper](https://www.jair.org/index.php/jair/article/download/12125/26676/26691))
- **CCMN** (Xie & Huang, TPAMI 2023) is the theoretical multi-label extension of the class-conditional noise matrix. It models asymmetric transitions but is a model, not a treatment rule. ([TPAMI paper](https://xiemk.github.io/publication/tpami-ccmn-preprint.pdf))
- **Zheng et al. 2020 (Error-Bounded Correction, ICML)** gives a likelihood-ratio test for label correction with theoretical guarantees, including asymmetric noise (`τ_01` vs. `τ_10`). It is binary/multiclass, not multi-label. ([ICML paper](http://proceedings.mlr.press/v119/zheng20c/zheng20c.pdf))
- **Project-specific:** the Q-AB taxonomy (`interpretations/kneemri/2026-09-02_what-label-correction-is-actually-worth.md`) identifies the dominant error modes: hedged negation, severity thresholds, BME→contusion. These are false-positive construction errors, not random symmetric noise. The literature does not provide a ready-made asymmetric treatment for this pattern; the project-specific fixes (negation pass, rubric re-read, fracture+contusion rescale) are closer to parser repair than to any published label-noise method.

## (4) Rare-class pruning traps in imbalanced multi-label data

This is the most clearly documented risk in the literature.

- **BoMD** explicitly warns that chest X-ray datasets are "mostly multi-label" with "inherent positive-negative imbalance," which causes sample-selection methods to select "an extremely imbalanced clean set, where the majority of identified clean samples belong to the 'No Findings' class." This is the rare-class pruning trap in medical multi-label data. ([BoMD paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_BoMD_Bag_of_Multi-label_Descriptors_for_Noisy_Chest_X-ray_Classification_ICCV_2023_paper.pdf))
- **TLCM** highlights that multi-label corruption-matrix estimation is hard because of "unbalanced distributions of labels and the presence of multiple objects that may be mapped into the same labels." ([TLCM paper](https://proceedings.mlr.press/v189/ghiassi23a.html))
- The project note already observes this: rare findings (Fracture, MCL, Synovitis) carry 53% of the holdout variance, so pruning their already-scarce positives is disproportionately costly (`interpretations/kneemri/2026-09-02_what-label-correction-is-actually-worth.md`).

## Summary table

| question | published answer | strength | project implication |
| :-- | :-- | :-- | :-- |
| Head-to-head multi-label medical prune/relabel/down-weight | **None found** | — | `errors_dense` is the relevant experiment; result pending |
| Noise-rate crossover | No explicit curve; re-labeling > reweighting at high noise (Zheng AAAI 2021, Zhang CVPR 2020) | Multi-class only, directionally consistent | 10.1% flagged + ~20.8% gold error rate sits on the re-labeling side |
| Asymmetric FP-dominant noise | Class-conditional noise matrix / confident joint; no dedicated FP/FN differential treatment | Framework exists, treatment rule does not | Parser/rubric fixes are project-specific; no off-the-shelf method |
| Rare-class pruning trap | Documented in BoMD and TLCM | Multi-label medical (CXR) | Pruning rare positives is the likely mechanism for `errors_dense` failure |

## What this means for the remaining label-noise lane

The literature does not rescue pruning. The directional evidence (re-labeling more robust than reweighting at high noise; rare-class pruning trap documented in medical multi-label data) makes re-labeling or soft-relabeling the favourite branch, but this is a hypothesis until `errors_dense` supplies the prune control. The project-specific parser fixes (negation, rubric re-read, fracture+contusion) remain the cheapest corrections because they target the known asymmetric error modes directly.
