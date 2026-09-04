# Q-HC — Hard-band training methods and empirical ceiling estimation

2026-09-04

## Bottom line for Claude

- **For moving the hard band (Q4-Q5) in a multi-label medical-imaging ensemble, the most direct evidence belongs to asymmetric losses.** ASL (Ridnik et al. ICCV 2021) and especially RAL (Park et al. ICCVW 2023) are designed for positive-negative imbalance and long-tailed multi-label medical data. RAL reports 0.351 mAP / 0.837 mAUC on CXR-LT and a Top-5 finish in the CVAMD 2023 competition, with explicit hard-negative regularization via a Hill term.
- **Disagreement-weighted distillation has medical evidence, but mostly in segmentation, not multi-label classification.** The kidney-segmentation KD-Ω result is +0.92–1.04 Dice; no comparable multi-label chest-x-ray result surfaced.
- **Diversity objectives (DICE, NCL) have clean general-ensemble theory but no proven multi-label medical hard-case gains, and a 2023 NeurIPS paper warns that naive joint/diversity training of deep ensembles collapses via "learner collusion."**
- **Selective prediction / abstention is Bayes-optimal in theory but is not established as a training signal for hard-case improvement in medical imaging.**
- **There is no published estimator that reads a benchmark's submission record and emits an empirical ceiling.** The closest tools are: (1) Dawid-Skene consensus treating entries as noisy annotators (cheap, but only on local holdout where labels exist); (2) Bayes-error estimation from data (Berisha et al. 2019, Chen et al. 2025, Ushio et al. 2026 — needs features/distances); (3) inter-rater agreement bounds from multiple human labels (Bucarelli et al. CVPR 2023 — you already own the 58 gold studies).
- **Practical ceiling read:** use the 58 radiologist-reviewed studies to compute a per-label inter-rater agreement ceiling. That is the cheapest, most credible answer to "is 0.009 headroom or noise?"

## Q-HC-1 — Training methods that move the hard band

Claude's diagnostic shows the hard band (Q4-Q5) transfers at ~0.73–0.75, while the easy band (Q1) transfers at ~0.44. The question is which published method has evidence for improving Q4-Q5 specifically.

### 1.1 Focal / asymmetric losses on multi-label chest-x-ray-style tasks

**ASL — Asymmetric Loss for Multi-Label Classification (Ridnik et al., ICCV 2021)** [[arXiv:2009.14119](https://arxiv.org/abs/2009.14119)] [[ICCV page](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html)]

- Designed for multi-label positive-negative imbalance.
- Mechanism: separate focusing parameters for positive and negative samples; down-weights easy negatives; hard-thresholds negatives below a probability clip; discards possibly mislabeled samples.
- Datasets: MS-COCO, Pascal-VOC 2007, NUS-WIDE, Open Images.
- Reports mAP gains over BCE and focal loss; does not publish a hard-case/easy-case decomposition, but the gradient design explicitly targets the hard positives and suppresses easy negatives.
- Easy to implement; drop-in replacement for BCE in multi-label training.

**RAL — Robust Asymmetric Loss for Multi-Label Long-Tailed Learning (Park et al., ICCVW 2023)** [[PDF](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Park_Robust_Asymmetric_Loss_for_Multi-Label_Long-Tailed_Learning_ICCVW_2023_paper.pdf)]

- Directly targets long-tailed multi-label medical imaging.
- Mechanism: extends ASL to a polynomial expansion, then adds a Hill-loss term that regularizes gradients from hard negative samples so the loss is less hyper-parameter sensitive.
- Dataset: CXR-LT (377,110 chest X-rays, 26 findings, heavy long tail).
- Backbone: ConvNeXT-B, 384×384 (development), 1024×1024 (test).
- Reported scores: **0.351 mAP, 0.837 mAUC, 0.256 mF1**, Top-5 in CVAMD 2023 competition.
- Ablations in the paper show RAL outperforms BCE, focal loss, LDAM, and ASL on CXR-LT under noise (Gaussian blur, salt-pepper, speckle), with a ~13% AUC margin over BCE/ASL in noisy conditions.

**AUC margin loss (De Gruyter 2023)** [[PDF](https://www.degruyter.com/document/doi/10.1515/cdbme-2023-1165/pdf)]

- Case study on CheXpert5000: limited, imbalanced, noisy medical diagnosis.
- Directly optimizes AUC margin rather than cross-entropy.
- Not as widely validated as ASL/RAL, but it is explicitly motivated by the same saturation/imbalance regime.

**Bottom line on losses:** ASL and RAL are the only methods in this list with direct multi-label medical-imaging validation and a mechanism that up-weights the hard stratum. RAL is the closer match to a long-tailed 12-label AUC task.

### 1.2 Disagreement-weighted distillation

**Disagreement-Guided Knowledge Distillation for Kidney Segmentation (MDPI 2026)** [[paper](https://www.mdpi.com/2076-3417/16/11/5573)]

- KD-Ω: distills from a large teacher ensemble into a compact student, weighting samples by teacher disagreement.
- Gains: +0.92% Dice on KiTS19, +1.04% Dice on an in-house dataset; +1.3% Dice in a strong-teacher setup.
- Domain: segmentation, not multi-label classification.

**Ensemble distillation in medical imaging generally**

- Noothout et al., "Knowledge distillation with ensembles of convolutional neural networks for medical image segmentation" (PMC 2022) [[paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9142841/)] surveys ensemble distillation for segmentation.
- For multi-label chest-x-ray classification, disagreement-weighted distillation is much less studied. The mechanism is plausible, but I did not find a published result with hard/easy stratification on a multi-label medical classification task.

**Bottom line on distillation:** plausible, but the medical evidence is in segmentation. If you want to try it, you would be transferring a method rather than replicating a published hard-band win.

### 1.3 Deep-ensemble diversity objectives (NCL, DICE)

**DICE — Diversity in Deep Ensembles via Conditional Redundancy Adversarial Estimation (Ramé & Cord, ICLR 2021)** [[arXiv:2101.05544](https://arxiv.org/abs/2101.05544)]

- Reduces spurious correlations among ensemble members by adversarially preventing features from being conditionally predictable from each other.
- Result: an ensemble of 5 networks trained with DICE matches an ensemble of 7 independently trained networks on CIFAR-10/100.
- Domain: general image classification. No multi-label medical result found.

**NCL — Negative Correlation Learning**

- Liu & Yao (1999) introduced NCL for regression ensembles; later generalized to deep ensembling (Buschjäger et al. 2020; Shi et al. 2018 for crowd counting).
- Mechanism: add a penalty on correlated errors among ensemble members.
- Medical imaging: not found for multi-label classification.

**Caution: joint training of deep ensembles can collapse**

Jeffares et al., "Joint Training of Deep Ensembles Fails Due to Learner Collusion" (NeurIPS 2023) [[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/2bde8fef08f7ebe42b584266cbcfc909-Paper-Conference.pdf)]

- Shows that directly optimizing an ensemble-diversity/joint objective leads to "learner collusion": members bias their predictions in opposite directions to artificially inflate diversity, then cancel out in the aggregate.
- Collusion inflates training diversity but generalizes poorly.
- Practical implication: diversity objectives that change the training objective risk degenerating unless carefully constrained (e.g., β ≤ 1 in their augmented objective).

**Bottom line on diversity objectives:** DICE/NCL are principled but unproven on your task, and naive joint/diversity training has a known failure mode. They are higher risk than ASL/RAL.

### 1.4 Selective prediction / abstention as a training signal

**Reliable Multilabel Classification with Partial Abstention (Nguyen et al., AAAI 2020 / JAIR 2021)** [[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5972/5828)] [[JAIR](https://www.jair.org/index.php/jair/article/download/12610/26733/28714)]

- Extends multilabel classification to allow per-label abstention.
- Provides Bayes-optimal reject rules and learns abstention functions.
- Strong theory, but the experiments are on general multi-label datasets, not medical imaging, and the paper does not show that abstention training improves performance on the non-abstained hard cases.

**Bottom line on abstention:** theoretically attractive, not established as a hard-band training lever in medical imaging.

### 1.5 Summary table

| Method | Evidence on hard band | Multi-label medical data | Size of published effect | Risk |
|---|---|---|---|---|
| ASL | implicit (hard-thresholds easy negatives) | indirect (general multi-label) | mAP gains on COCO/VOC/Open Images | low |
| RAL | explicit (Hill term on hard negatives) | **yes — CXR-LT** | 0.351 mAP / 0.837 mAUC, Top-5 CVAMD 2023 | low |
| AUC margin loss | implicit (optimizes AUC) | CheXpert5000 case study | single dataset | medium |
| Disagreement-weighted KD | explicit (disagreement weighting) | segmentation only | +0.92–1.3% Dice | medium-high (domain transfer) |
| DICE / NCL | implicit (diversity → coverage) | none found | 5 nets ≈ 7 independent nets on CIFAR | high (collusion) |
| Selective abstention | theoretical | none found | — | high (unproven as training signal) |

**Recommendation:** start with RAL (or ASL if you want a simpler drop-in). It is the only method in the list with both a hard-negative mechanism and a validated medical multi-label competition result.

## Q-HC-2 — Empirical ceiling estimation

Claude asks whether there is a published estimator that turns a benchmark's submission record plus label noise into "the reachable maximum is X."

**Honest answer: no.** Akhtar et al. define saturation and a saturation index, but they do not output a numerical ceiling. The literature offers three adjacent tools, none of which reads a submission record directly.

### 2.1 Dawid-Skene consensus: treat entries as noisy annotators

**Idea:** Fit a Dawid-Skene model where each submission/entry is an annotator with its own confusion matrix, and infer a latent "true" label per case. Score the inferred labels against the ground truth (on a local holdout) to get an empirical ceiling under the assumption that entries are independent noisy observers.

**What it gives:** a "wisdom of the ensemble" ceiling — better than any single entry if entries make different errors.

**Cost:** cheap. Fitting DS on N entries × M cases is polynomial time; there are standard implementations.

**Limitations:**
- Requires ground-truth labels to evaluate the consensus, so it cannot be run on the Kaggle public test set directly.
- Assumes entries are independent; in practice, entries share members, data, and code, so errors are correlated and DS will overestimate the ceiling.
- Does not distinguish "irreducible label noise" from "systematic errors all entries share."

### 2.2 Bayes error estimation from data

**Berisha et al., "Learning to Benchmark: Determining Best Achievable Misclassification Error from Training Data" (2019)** [[arXiv:1909.07192](https://arxiv.org/abs/1909.07192)]

- Estimates the exact Bayes misclassification error rate from data without learning a Bayes-optimal classifier.
- Uses an ensemble of ε-ball estimators + Chebyshev approximation.
- Theory: O(N^{-1}) MSE under smoothness assumptions.
- Multi-label AUC: not directly applicable; it is for misclassification error.

**Chen et al., "An efficient Bayes error rate estimation method" (Machine Learning 2025)** [[Springer](https://link.springer.com/article/10.1007/s10994-025-06761-w)]

- More recent, code available, improves computational efficiency.

**Ushio et al., "Practical estimation of the optimal classification error with soft labels and calibration" (ICLR 2026)** [[PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/039ef7b5decebab3145f7691b4708fe5-Paper-Conference.pdf)]

- Uses soft labels and calibration; code available.
- Still binary/multiclass error, not multi-label AUC ceiling.

**Cost:** moderate. Needs features or pairwise distances for all cases. For multi-label AUC, you would need to adapt the estimator per label or use a ranking-based Bayes-error bound, which is not standard.

### 2.3 Inter-rater agreement bound

**Bucarelli et al., "Leveraging Inter-Rater Agreement for Classification in the Presence of Noisy Labels" (CVPR 2023)** [[CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Bucarelli_Leveraging_Inter-Rater_Agreement_for_Classification_in_the_Presence_of_Noisy_CVPR_2023_paper.html)]

- Estimates the noise distribution from multiple raters and uses it to bound classifier performance.
- The inter-rater agreement sets a practical ceiling: a model cannot reliably exceed the agreement rate of the experts it is trained to predict.

**What you already own:** 58 radiologist-reviewed gold studies. Computing Cohen/Fleiss κ or per-label agreement among the radiologist calls gives a direct, cheap empirical ceiling.

**Cost:** near-zero. You already have the labels.

### 2.4 Which ceiling read to trust

| Method | Inputs | Output | Cost | Credibility for your gap |
|---|---|---|---|---|
| Dawid-Skene consensus | Entry predictions + local labels | Consensus score | cheap | medium (correlated entries) |
| Bayes error estimation | Features / distances | BER per class | moderate | medium (needs adaptation to AUC) |
| Inter-rater agreement | Multiple human labels per case | Agreement ceiling | near-zero | **high** |

**Recommendation:** compute the inter-rater agreement ceiling from the 58 gold studies. If radiologist agreement is ~0.96 AUC-equivalent, then 0.953 is likely near the noise ceiling and the 0.009 gap is mostly irreducible. If agreement is higher (e.g., 0.98+), there is still headroom.

## Q-HC-3 — What this means for the next arm

Given the diagnostic (saturation, not distribution gap) and the literature:

1. **If the 58-study inter-rater ceiling is ≥ ~0.96:** the 0.009 gap to the leader is largely noise. Stop chasing it with ensemble polish; spend the card on an instrument or a genuinely different axis.
2. **If the ceiling is ≥ ~0.97:** there is measurable headroom. The hard-band slope of 0.73 means a Q4-Q5-targeted method buys ~0.75 of its offline delta. RAL is the cheapest, most evidence-backed lever to try.
3. **Do not start with DICE/NCL or abstention:** they lack medical multi-label hard-band evidence and/or have known failure modes.
