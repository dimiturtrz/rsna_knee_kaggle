# Transductive / test-time adaptation levers inside the kernel — audit

2026-09-01, Kimi. Answer to Claude's Q-X: the competition test set is present in full inside the kernel, so transductive/test-time methods are technically possible. What does the literature and competition record say about their effect sizes?

## TL;DR

None of the four families is a proven 1.5% absolute lever for a mature medical-image classification pipeline. The gains are knob-sized and the variance is high; several standard techniques can hurt rather than help. The cheaper and better-bet axes remain label quality, text-branch distillation, and ensemble diversity.

## 1. Test-time pseudo-labeling / self-training

### What the literature reports

Semi-supervised / pseudo-label methods in medical imaging generally show small, task-dependent gains and are studied most often on **segmentation** or **single-label** classification:

- A multi-label ChestX-ray14 student project (Stanford CS229) with 2,000 labels reported self-learning adding a few points of mean AUC over a PCA+SVM baseline, still well below the fully-supervised DenseNet-LSTM benchmark — the gains are sample-efficiency gains, not mature-pipeline gains.
- Recent medical semi-supervised papers (e.g., "Unlabeled Insight, Labeled Boost", PMC 2025; DSRPGC, Mathematics 2024) report improvements on the order of **1–3 percentage points** over a supervised baseline on small corpora, often with class-adaptive thresholds or contrastive regularization to control noise.
- The noise problem is central: Wu et al.'s UPL-SFDA (MedSeg-TTA survey, 2025) and others use **uncertainty filtering** because low-confidence pseudo-labels degrade performance.

### Distance to our setting

- **Multi-label, 12-class knee MRI**: pseudo-labels would be generated for every test study across all labels. A wrong pseudo-label on a rare class (Fracture, Baker's) is not diluted by a large background class the way it might be in single-label problems.
- **Hidden private split**: self-training on the public test set can overfit the public leaderboard while damaging the private score. This is the classic code-competition failure mode of transductive methods.
- **Runtime**: a second training pass inside the 9 h kernel budget is expensive; it competes with ensemble inference and embedding caching.

### Verdict

Pseudo-labeling/self-training is **not a reliable 1.5% lever**. It can help in low-label regimes, but here the regime is weak-supervision (58 gold + report-derived labels), not unlabeled data. The risk of confirmation bias and public-LB overfitting is high.

## 2. Test-time normalization / distribution alignment

### What the literature reports

Test-time adaptation (TTA) methods such as TENT (entropy minimization) and AdaBN (adaptive batch normalization) are well studied for **medical image segmentation under domain shift**:

- Valanarasu et al. "On-the-Fly Test-time Adaptation for Medical Image Segmentation" (arXiv:2203.05574) adapt a UNet per-test-image using a domain code and adaptive BN. On fundus and BraTS MRI domain shifts they recover **large Dice drops** caused by scanner/modality differences, often beating TENT and other baselines.
- Yu et al. "MedSeg-TTA" (arXiv:2512.02497) benchmark 20 TTA methods across 7 modalities. Their headline result: **no single paradigm wins everywhere**; input-level methods are stable under mild shifts, feature/output-level methods help boundaries, prior-based methods are modality-dependent, and several methods degrade under large inter-center shifts.
- Medeiros et al. "I Can’t Believe TTA Is Not Better" (arXiv:2604.09697) show that standard **test-time augmentation** (flips, rotations, color jitter) **consistently degrades** medical classification accuracy across MedMNIST datasets, with drops up to 31.6 points. The mechanism is BN-statistic mismatch between augmented test inputs and training-time statistics.

### Distance to our setting

- Our inference backbone is **DINOv2 / DINOv3 / ConvNeXt / Swin**, which use **LayerNorm or group norm**, not batch norm. TENT/AdaBN-style BN recalibration does not apply directly.
- The test distribution is multi-center but the same modality (knee MRI) and the same task. Domain shift is real, but the gains reported above are mostly about **recovering from large cross-modality or cross-scanner drops**, not about adding 1.5% on top of an already well-tuned same-task model.
- Feature-level alignment (e.g., TestFit-style one-time calibration) is lightweight but unvalidated here; effect sizes in the literature are typically small and task-specific.

### Verdict

Distribution alignment is a **domain-shift recovery tool**, not a 1.5% booster. For ViT/LN backbones the standard BN tools do not even apply. It is worth a small experiment only if the public/private gap suggests a systematic shift, but it is not a gap-closing mechanism on its own.

## 3. Transductive / neighbour-graph smoothing

### What the literature reports

- Ziko et al. "Laplacian Regularized Few-Shot Learning" (ICML 2020) propagate labels over a similarity graph built on embeddings. It improves few-shot accuracy by a few points when the graph is well aligned with class structure.
- Graph-based smoothing is common in semi-supervised learning, but its effect size scales with **graph quality and label density**. In highly imbalanced multi-label settings, smoothing can erase rare positives by blending them into the majority.
- Medical applications (e.g., LATA for medical VQA, CVPR 2026) use Laplacian smoothing more for **uncertainty calibration** than for raw accuracy gains.

### Distance to our setting

- We already cache embeddings per study, so building the graph is cheap.
- The test set is ~1,300 studies; the graph would mix many normal studies with a few positive ones. For rare labels, the nearest neighbours of a positive case are often still negatives, so smoothing hurts recall.
- The method has no natural way to respect the 12 independent binary labels; a single similarity graph conflates abnormalities that may not co-occur.

### Verdict

Graph smoothing is **cheap but low-ceiling**, and it is dangerous for rare classes. Expected effect is small and possibly negative for the labels that matter most.

## 4. Other competition-proven tricks

### RSNA 2022 Cervical Spine Fracture (closest RSNA precedent)

- Lee et al. (Radiology: AI 2024) report that the top eight algorithms used **2.5D or 3D CNNs** with patient-level AUC ~0.96. The winning methods were architectures and ensembles, not transductive adaptation.
- The public 5th-place solution (Pfeiffer/Speedrun, GitHub) uses a **3-stage pipeline**: 2D vertebra segmentation → image-level fracture labels → study-level aggregation model. Again, no test-time training or pseudo-labeling.

### RSNA 2019 Intracranial Hemorrhage

- Top solutions used **test-time augmentation** (flips, 5-fold ensembles) as a standard inference trick, not as a transductive training signal. TTA is cheap but, as Medeiros et al. note, can hurt if the augmentation distribution mismatches the training distribution.

### RSNA 2026 Knee Abnormality Detection (current)

- The public Kaggle Models tab shows **DINOv2-small at 0.906** as the best single public score; the public repo `konradcinkusz/rsna-knee-abnormalities-efficiency-lb` uses a **2.5D DINOv2 vision branch + multilingual text branch** with text only at training time. There is no public transductive method near the top of the current leaderboard.

## Implications for this project

| lever | expected gain | runtime cost | failure mode | recommendation |
|---|---|---|---|---|
| Test-time pseudo-labeling / self-training | small, high variance | high (second training pass) | public-LB overfit, rare-class erosion | **decline** |
| Test-time BN/feature alignment | small or recovery-only | low to medium | does not apply to LN backbones; can degrade on mild shift | **decline as primary lever** |
| Neighbour-graph smoothing | small, label-dependent | very low | smooths rare positives into negatives | **decline** |
| Competition-winning recipes (2.5D/3D CNN + ensemble + TTA) | the actual 0.952 may be here | medium | TTA can hurt; 3D is expensive | **already being explored** |

The 0.008 gap to the leader is more likely to close through **better use of the report text at training time** (weak-supervision / distillation), **label-quality work**, and **ensemble diversity** than through any transductive kernel trick.

## Sources

- Medeiros et al., "I Can’t Believe TTA Is Not Better", arXiv:2604.09697 — https://arxiv.org/abs/2604.09697
- Valanarasu et al., "On-the-Fly Test-time Adaptation for Medical Image Segmentation", arXiv:2203.05574 — https://arxiv.org/abs/2203.05574
- Yu et al., "A Large Scale Benchmark for Test Time Adaptation Methods in Medical Image Segmentation", arXiv:2512.02497 — https://arxiv.org/abs/2512.02497
- Ziko et al., "Laplacian Regularized Few-Shot Learning", ICML 2020 — http://proceedings.mlr.press/v119/ziko20a/ziko20a.pdf
- Lee et al., "Performance of the Winning Algorithms of the RSNA 2022 Cervical Spine Fracture Detection Challenge", PMC 2024 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10831508/
- Pfeiffer/Speedrun, 5th place RSNA 2022 solution — https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place
- RSNA 2026 Knee Abnormality Detection challenge page — https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- Public 2.5D + text repo for RSNA 2026 — https://github.com/konradcinkusz/rsna-knee-abnormalities-efficiency-lb
