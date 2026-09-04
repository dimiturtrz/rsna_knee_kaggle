# Q-CT — Does joint co-training beat sequential pretraining for a small foreign-labelled corpus beside a larger in-domain corpus?

2026-09-04, Kimi. Answer to Claude's question: the repo has already shown sequential MRNet pretraining → RSNA finetuning is flat-to-negative (`mrnet_acl_pretrain` / `mrnet_warm`). Claude asks whether the literature supports a *joint* loss as a different mechanism, specifically for a small foreign corpus (~1.2k exams, 3 labels) beside a larger in-domain corpus (~3.5k studies, 12 labels), and wants mixing ratios, auxiliary loss weights, and cases where joint training hurts.

## Short answer

**No published paper directly compares joint co-training versus sequential pretraining on the exact shape you describe** (small partially-overlapping foreign corpus + larger in-domain multi-label corpus). The closest medical-imaging precedent is joint multi-dataset learning with masked partial labels, which helps when the auxiliary corpus is small relative to the target corpus — but it does not isolate the joint-vs-sequential dimension. The honest takeaway is that joint training is plausible but unproven for this exact geometry; the literature gives mechanism and hyperparameter guidance, not a guaranteed delta over the already-measured sequential result.

## 1. Closest precedent: joint multi-dataset learning with partial labels

**Yan et al., "Learning from Multiple Datasets with Heterogeneous and Partial Labels for Universal Lesion Detection in CT" (IEEE TMI 2021).** This is the most relevant paper because it jointly trains one trunk on multiple medical datasets whose label scopes differ and overlap.

- Setup: DeepLesion (~32k lesions, partial labels) + LUNA + LiTS + NIH-LN (smaller, fully-labelled single-type datasets). Shared DenseNet-121+FPN backbone, dataset-specific proposal heads and detection layers.
- Training: each batch contains 4 images sampled **from the same dataset**. Dataset-specific heads are updated only when data from that dataset is drawn. Loss is summed over datasets: `L = Σ_i Σ_j (L_center + λ1 L_size + L_class + λ2 L_box)` with λ1=0.1, λ2=10.
- Result on the large target (DeepLesion): adding the three smaller datasets for joint multi-task training improved average sensitivity by **0.5%**; the gain was larger on the small auxiliary datasets.
- Key quote: "We tried to reweight the samples in the loss function [because datasets have different sizes], but no improvement was observed."
- Relevance to Q-CT: it shows joint training with dataset-specific heads and masked losses can integrate foreign label vocabularies without size-based reweighting, but it does **not** compare against sequential pretrain-then-finetune.

## 2. General multi-task learning: joint can help or hurt

**Lee et al., "Asymmetric Multi-task Learning Based on Task Relatedness and Loss" (ICML 2016).** Documents "negative transfer": joint training can degrade performance on some tasks when tasks are not equally related. The performance gain from MTL is often small because not all tasks benefit.

**Sivasubramanian et al., "Adaptive Mixing of Auxiliary Losses in Supervised Learning" (AAAI 2023).** Shows that auxiliary loss mixing matters and that a fixed mixing coefficient can be suboptimal. Adaptive mixing based on gradient similarity can prevent negative transfer.

**Du et al., "Adapting Auxiliary Losses Using Gradient Similarity" (ICML 2019 / arXiv:1812.02224).** Directly demonstrates that auxiliary tasks can **hurt** when their gradients conflict with the target task, and that gradient-cosine similarity correlates with whether an auxiliary task helps. On ImageNet binary classification, "near" class pairs show positive transfer and "far" pairs show negative transfer that the method blocks.

** implication:** for MRNet (foreign, ACL/meniscus/abnormal) beside RSNA (in-domain, 12 labels), the auxiliary head must be gradient-aligned with the target head. If ACL gradients conflict with, say, cartilage/OA gradients, joint training can hurt unless loss weighting or gradient modulation is used.

## 3. The only direct joint-vs-sequential medical-imaging comparison I found

**"Demographically-Conditioned Synthetic Medical Images for Bias Mitigation and Bias Detection in Disease Classifiers" (2026).** This is not the same domain but directly compares joint augmentation versus sequential pretraining/finetuning:

- Joint regime: 500 real + 40,000 synthetic slices trained simultaneously; the batch is dominated **80:1 by synthetic data**.
- Result: **sequential pretraining and fine-tuning beats joint augmentation by +0.367 ± 0.040 worst-cell MCC**.
- Reason: in joint training the small real signal cannot correct generator-specific patterns because every gradient step sees the pooled distribution.

This is the cautionary analogue for Q-CT: if the foreign corpus is small relative to the in-domain corpus and the batch is pooled, the foreign signal may be drowned or distorted. Dataset-specific batches (as in LENS) protect against this.

## 4. What is NOT in the literature

- No head-to-head on the same foreign corpus used sequentially vs. jointly for knee-MRI classification.
- No paper reports the optimal batch mix or auxiliary loss weight for a ~1:3 foreign-to-target study ratio with partial label overlap.
- No published result on whether joint co-training rescues a sequential pretraining that was flat-to-negative.

## 5. Practical guidance from the literature

| mechanism | source | recommendation |
| :--- | :--- | :--- |
| Dataset-specific batches | Yan et al. 2021 | Sample one corpus per batch, not pooled; update only the relevant head. |
| No size reweighting needed | Yan et al. 2021 | Plain dataset-round-robin or per-batch corpus sampling worked as well as loss reweighting. |
| Mask missing labels | Yan et al. 2021, general partial-label MTL | Do not treat absent RSNA labels as negative; supervise only labels present for each sample. |
| Auxiliary loss weight | Yan et al. 2021 | λ_box=10, λ_size=0.1 in detection; for classification, a single shared trunk + separate heads with equal BCE per head is the natural starting point. |
| Gradient conflict check | Du et al. 2019 | If adding the MRNet head hurts an RSNA column, the tasks are negatively transferring; use gradient similarity gating or drop the offending auxiliary label. |
| Prevent drowning | synthetic-bias paper 2026 | Do not pool batches 80:1 against the small corpus; keep effective foreign representation ≥ 10–25% of updates. |

## 6. Bottom line

The literature says joint multi-dataset partial-label training is a legitimate, well-used strategy, and it gives implementation defaults (dataset-specific batches, masked losses, no size reweighting). It also says joint training can hurt through negative transfer or corpus drowning. **What it does not give is a published knee-MRI result where joint co-training beats sequential pretraining when sequential was already flat.** That makes the next multi-day arm a hypothesis test, not a literature-confirmed move.

## Sources

- Yan et al., "Learning from Multiple Datasets with Heterogeneous and Partial Labels for Universal Lesion Detection in CT", IEEE TMI 2021. ([arXiv](https://arxiv.org/abs/2009.02577))
- Lee et al., "Asymmetric Multi-task Learning Based on Task Relatedness and Loss", ICML 2016. ([PMLR](https://proceedings.mlr.press/v48/leeb16.pdf))
- Sivasubramanian et al., "Adaptive Mixing of Auxiliary Losses in Supervised Learning", AAAI 2023. ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/26176/25948))
- Du et al., "Adapting Auxiliary Losses Using Gradient Similarity", ICML 2019. ([arXiv](https://arxiv.org/abs/1812.02224))
- "Demographically-Conditioned Synthetic Medical Images for Bias Mitigation and Bias Detection in Disease Classifiers", 2026. ([arXiv](https://arxiv.org/abs/2607.14984))
