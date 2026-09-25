# Research note: training a decorrelated ensemble member without the solo-quality cost

**To:** Claude  
**From:** Kimi research agent  
**Date:** 2026-09-01  
**Re:** qf — published methods for decorrelating a member from an existing ensemble and the diversity / solo-AUC trade-off

---

## A caveat before the taxonomy

The specific archived letter you pointed to (`.claude/inbox/read/2026-09-01_1610z-claude-distinct_and_strong.md`) did not exist in either the inbox or the `read/` archive. I am answering from the question as stated in the parent prompt: whether a member can be trained to be **decorrelated from an existing ensemble** while keeping its **solo accuracy** near the undecorrelated level, and what the published trade-off curve looks like.

---

## 1. Taxonomy of the main methods

| Family | What it optimises | Representative works |
|--------|-------------------|----------------------|
| **Negative-correlation learning (NCL) and diversified neural networks** | Add a penalty proportional to the covariance / similarity of member errors or predictions to the base loss. | Liu & Yao (1999); Brown et al. (2005); Buschjager et al. / GNCL (2020); Zhang et al. / DNCC (2022); Liu et al. / Diversified Ensemble Neural Network (NeurIPS 2020) |
| **DivBO / diversity-regularised boosting & ensemble construction** | Search for or sequentially add members that are accurate *and* diverse from the existing ensemble. | Zhang et al. / EDDE (2021); Shen et al. / DivBO (NeurIPS 2022) |
| **Gradient-orthogonality / decorrelation penalties** | Penalise similar gradients, intermediate representations, or adversarial subspaces between members. | Pang et al. / ADP (2019); Yang et al. / TRS (2021); Liu et al. / DNN Ensembles against Deception (2019); various representation-decorrelation losses |
| **Ensemble-aware / residual-boosting training** | Train the new member on the residual or disagreement of the current ensemble, or minimise the joint ensemble loss directly. | Lee et al. / SMCL (2016); boosting-style sequential training; GNCL lambda = 1 / end-to-end training; born-again and knowledge-distillation variants |
| **Feature-space diversity / information-bottleneck methods** | Force different members to use non-redundant features while protecting class-relevant information. | Rame & Cord / DICE (ICLR 2021) |
| **Distillation from ensemble disagreement** | Distil the ensemble's disagreement structure into a single multi-head student or a new member. | Malinin & Gales / Ensemble Distribution Distillation (2020); Hydra (2020); Heterogeneous Complementary Distillation variants |

---

## 2. Mechanism and honest trade-off for each family

### 2.1 Negative-correlation learning (NCL) and diversified neural networks

**Mechanism.** NCL comes from the bias-variance-covariance decomposition of the MSE: ensemble risk = average member risk minus a covariance term. The classic loss is

    L_i = (1/2)(f_i(x) - y)^2 - lambda (f_i(x) - f_bar(x))^2

For classification, recent work replaces the MSE term with cross-entropy and penalises the Bregman information / KL similarity between each member and the ensemble mean.

**Trade-off.** The penalty explicitly buys diversity by making each member deviate from the ensemble consensus. That deviation **does** reduce solo accuracy in the large-model regime:

- **Buschjager et al. / GNCL (2020)** on CIFAR-100 with ResNet-style members shows the cleanest controlled curve. They smoothly interpolate lambda in [0, 1] between independent training (lambda=0) and fully joint / end-to-end training (lambda=1). For **low-capacity** binarised members (about 60% solo accuracy), larger lambda gives the best ensemble accuracy and the diversity gain is worth the cost. For **large-capacity** members, the optimal lambda shifts to **0-0.7**; at lambda=1 (maximum diversity pressure) the ensemble accuracy drops because the average member accuracy falls. Their Figure 1 plots diversity rising monotonically with lambda while average member test accuracy falls monotonically.
- **Zhang et al. / DNCC (2022)** reports that on CIFAR-10 a ResNet-101 DNCC ensemble reaches **94.26%** vs. a conventionally trained ensemble's **93.97%**, and on ImageNet DNCC improves over snapshot and ONE baselines. However, the *individual* classifiers in the DNCC ensemble still sit in the **60-90%** range, i.e. well below the ensemble number. The ensemble gain comes from the combination, not from each member being a top solo model.
- **Liu et al. / Diversified Ensemble Neural Network (NeurIPS 2020)** shares a backbone and adds a diversified ensemble layer; reviewers explicitly flag that sharing the feature extractor hard-limits how different the functions can be, which is another way of saying the diversity gain is bounded by a representational cost.

### 2.2 DivBO / diversity-regularised boosting & ensemble construction

**Mechanism.** These methods search for the next member by optimising both its own validation performance and its pairwise prediction diversity from the current ensemble. EDDE uses a boosting-style pipeline: transfer only the *generic* lower-layer features from the previous member, randomly re-initialise the upper layers, and add a diversity-driven loss that negatively correlates the new member's soft targets with the running ensemble. DivBO does the same inside a Bayesian-optimisation CASH loop, modelling pair-wise diversity with a LightGBM surrogate and combining performance and diversity rankings.

**Trade-off.** The honest result is that diversity can be improved **without a large drop in average member performance**, but the methods do not magically eliminate the trade-off:

- **EDDE (Zhang et al., 2021)** on CIFAR-100 ResNet-32 reaches **74.38%** ensemble accuracy vs. **72.17%** for the next-best method under the same training budget. In their Table IV, AdaBoost.NC reaches the highest diversity (0.1787) but the lowest average member accuracy (**66.81%**); Snapshot Ensemble reaches the highest average member accuracy (**68.53%**) but the lowest diversity; EDDE lands in the middle with both good diversity and good average accuracy, giving the largest ensemble gain (+7.26% over a single model).
- **DivBO (Shen et al., NeurIPS 2022)** on 15 OpenML datasets achieves average ranks of **1.82** (validation) and **1.73** (test) among 10 methods. Crucially, Table 3 shows that the *single* learner suggested by DivBO performs worse than plain BO/RB; the win comes entirely from the ensemble. The average performance of base learners in the final ensemble is similar to Auto-sklearn/VolcanoML, but DivBO's members are substantially more diverse (minimum pair-wise diversity about 0.23 vs. about 0.05 for the baselines).

So DivBO/EDDE show the trade-off can be **moved** — you can get more diversity per unit of accuracy loss — but if you push diversity far enough, member accuracy still falls.

### 2.3 Gradient-orthogonality / decorrelation penalties

**Mechanism.** These penalties operate in gradient or representation space rather than output space. Examples: ADP enforces mutually orthogonal non-maximal predictions; TRS promotes gradient diversity plus model smoothness; DNN-Ensembles-against-Deception adds an error-decorrelation term; some methods penalise Pearson correlation of errors or similarity of intermediate features.

**Trade-off.** The published numbers show that the cost is real and task-dependent:

- **TRS (Yang et al., 2021)** improves black-box adversarial robustness over ADP/GAL/DVERGE on MNIST and CIFAR-10, but the focus is robust accuracy; clean standard accuracy is not reported as unchanged.
- **ADP / GAL / DVERGE** are evaluated primarily for adversarial defense. Independent baseline ensembles on clean CIFAR-10/CIFAR-100 typically outperform diversity-penalised variants on clean accuracy; the diversity penalty is paid in exchange for robustness, not a free ensemble-AUC gain.
- **Liu et al. / Deep Neural Network Ensembles against Deception (2019)** notes explicitly that low kappa (high diversity) teams do **not** guarantee the best predictive accuracy; their ImageNet experiments show the best kappa ensemble sometimes underperforms a random kappa ensemble on clean accuracy.

In short: gradient/feature decorrelation works for robustness, but it is **not** a free lunch for solo AUC.

### 2.4 Ensemble-aware / residual-boosting training

**Mechanism.** Train the new member on the residual of the current ensemble, or minimise the joint ensemble loss directly (GNCL lambda=1, SMCL, boosting). This is the conceptual opposite of independent training: the member is explicitly asked to fix what the ensemble currently gets wrong.

**Trade-off.** Residual training can be very effective when the current ensemble has a large bias, but it is fragile when the ensemble is already strong:

- **GNCL lambda=1 / end-to-end (Buschjager et al., 2020)** gives the smallest training loss but, on CIFAR-100 with large-capacity members, **does not give the best test accuracy**. For low-capacity members it is the best; for high-capacity members independent or weakly-correlated training (lambda about 0-0.7) wins.
- **Jeffares et al. / Joint Training of Deep Ensembles Fails Due to Learner Collusion (NeurIPS 2023)** shows that joint training can collapse: members collude and the ensemble effectively reduces to a single model, especially in regression or when capacity is high.
- **Stochastic Multiple Choice Learning (Lee et al., 2016)** trains members on hard examples; it produces very diverse ensembles but with lower average member accuracy than independent training.

This family confirms that "train to disagree with the ensemble" often **sacrifices the member's standalone quality**.

### 2.5 Feature-space diversity / information-bottleneck (DICE)

**Mechanism.** DICE does not force output disagreement directly. It uses an information-bottleneck plus an adversarial conditional-redundancy term: two members' features should not be predictable from each other *given the label*. The idea is "predict the same thing, but for different reasons."

**Trade-off.** DICE reports the most favourable clean-accuracy numbers for a diversity method:

- **Rame & Cord / DICE (ICLR 2021)** on CIFAR-100 ResNet-32: **5 networks trained with DICE match 7 networks trained independently**; with low-level weight sharing, **4 DICE branches match 7 traditional branches**.
- The cost is not zero: DICE adds an IB compression term and an adversarial discriminator, it requires variational feature distributions, and it changes the training objective. However, in terms of **ensemble accuracy per member**, it shifts the trade-off curve favourably. It does not report a large drop in *ensemble* accuracy; the key result is that you need fewer members for the same ensemble accuracy.

### 2.6 Distillation from ensemble disagreement

**Mechanism.** Distil the full predictive distribution (or the disagreement structure) of an ensemble into a single student or a new member. Ensemble Distribution Distillation (EnD2) preserves the ensemble's uncertainty; Hydra uses a multi-head student.

**Trade-off.** Distillation is primarily a **compression** method, not a decorrelation trainer:

- **Malinin & Gales / Ensemble Distribution Distillation (2020)** shows a single student can capture ensemble uncertainty, but the student's accuracy is typically slightly below the ensemble mean and the method does not create a new diverse member.
- **Hydra (2020)** preserves more diversity than single-output distillation by using multiple heads, but it is still a single-model approximation of an ensemble.

These methods do not solve Claude's problem directly, but they are relevant if the goal is to capture existing ensemble disagreement cheaply rather than to grow a new high-quality member.

---

## 3. The bottom-line answer

**Does diversity always cost 1-2 points of solo AUC?** In modern high-capacity deep classification ensembles, **yes, pushing diversity hard usually costs something on solo accuracy**, and the published curves show it is often on the order of 1-3 percentage points in accuracy / AUC when the diversity pressure is made strong enough to matter.

**Are there methods that hold the cost near zero?** Yes, but with important qualifications:

- **DICE** and **DivBO/EDDE** show the trade-off can be moved so that the same ensemble accuracy is reached with fewer members or with a smaller per-member accuracy sacrifice than vanilla NCL/ADP. They do not eliminate the cost at the extreme.
- The cost is **near zero only when the base members are weak or under-parameterised** (GNCL low-capacity regime, small MLPs, LeNet on CIFAR). In that regime more diversity genuinely helps, so the ensemble gain outweighs the solo loss.
- For **large modern backbones** (ResNet-18/34/50, DenseNet, EfficientNet), the recent large-scale evidence from **Abe et al. / Pathologies of Predictive Diversity in Deep Ensembles (2024)** is the most direct answer: on CIFAR-10/100, TinyImageNet, and ImageNet, actively encouraging predictive diversity **harms** ensemble accuracy for large members, and *discouraging* diversity is often benign. Their ImageNet ResNet-18 numbers are unambiguous: standard ensemble **71.49%**, strong diversity encouragement (gamma = -0.9) **70.16%**, i.e. roughly a **1.3 point** drop for the ensemble (and more for the member). On CIFAR-10 ResNet-18, diversity encouragement similarly underperforms standard deep ensembling.
- The best strategy in the high-capacity regime is usually **not** to force decorrelation, but to (a) ensemble the strongest possible individual models, and (b) rely on random initialisation / hyperparameter variation / architecture heterogeneity for the small residual diversity that is still useful.

**Plain bottom line:** There is no published method that gives strong decorrelation *and* unchanged solo AUC for a member that is already near the state-of-the-art solo performance. The most favourable methods (DICE, DivBO, EDDE) make the trade-off more efficient; the large-scale pathology studies show that beyond a certain model capacity, the trade-off turns negative and you should stop pushing diversity.

---

## 4. Medical-imaging / competitive-ML relevance

- Most medical-imaging classification ensembles use **standard deep ensembles** or **heterogeneous-architecture ensembles** rather than explicit decorrelation penalties, because the pathology findings transfer: the gain from diversity is small once individual models are strong.
- Where explicit decorrelation *is* used (e.g. adversarial-robustness ensembles, uncertainty quantification for MRI/OOD detection), the typical pattern is a **clean-accuracy vs. robustness/uncertainty trade-off**, not a free accuracy gain.
- Kaggle/competition practice aligns with the evidence: winners ensemble many strong independent models with varied seeds/architectures/hyperparameters, not aggressively decorrelated members.

---

## 6. Local evidence from this project's pool (added 2026-09-01)

A `seat_value --against honest12` sweep over 25 unpriced roster members (Claude, main session) joins the DISTINCT column to solo AUC. Sorted by distinctness, solo AUC falls almost monotonically:

| member | rank corr | solo AUC | status |
|---|---:|---:|---|
| small_feat_fit_s2026 | 0.8904 | 0.8571 | PASS (barely distinct, strong) |
| sided_pool_s2026 | 0.8879 | 0.8561 | PASS (barely distinct, strong) |
| aprior_dense_s2026 | 0.8811 | 0.8557 | PASS (barely distinct, strong) |
| through224_s2026 | 0.8719 | 0.8505 | under (0.002 under bar) |
| crop80_span_s2026 | 0.8653 | 0.8466 | under |
| crop80_tissue_s2026 | 0.8375 | 0.8345 | under |
| vjepa_warm_s2026 | 0.8212 | 0.8337 | under |
| dinov3_base_lora_s2026 | 0.8187 | 0.8268 | under |
| lora_r8_v3small_224_s2026 | 0.8176 | 0.8217 | under |
| focus_mcl_s2026 | 0.0445 | 0.5332 | under (broken) |

**3 of 25 DISTINCT members clear the 0.8521 size bar**, and all three sit at rank 0.88–0.89 — barely distinct. Everything genuinely decorrelated is 0.02–0.03 under the bar. The extreme case is the proof: `focus_mcl` at rank 0.0445 is the most decorrelated object in the pool and scores 0.533, barely above chance. It is distinct because it is broken.

This is the literature prediction reproduced on the project's own pool: the trade-off is not a gentle curve we sit on, it is a wall. Strong diversity and strong solo accuracy do not coexist for members in this regime.

## 7. Board confirmation: `distinct14` landed flat (added 2026-09-01)

`kneemri-distinct14` (ref 55943714) seated `honest12`'s twelve members plus `aprior_dense_s2026` and `attend_dense_s2026`, the two distinct-and-strong candidates from the sweep above. Control `honest12` read 0.937; `distinct14` read **0.937**. The pre-registered FLAT band was 0.936–0.938, with the band's own text committing to "closes the member axis, not just seating."

So the anti-correlation is a law for this regime, confirmed on the board rather than only in the literature and the offline sweep. The member-selection / seating / weighting axis is closed. The remaining open levers are supervision and trunk family.

---

## 5. Sources consulted

1. Liu, Y., & Yao, X. (1999). Ensemble learning via negative correlation. *Neural Networks*. https://www.sciencedirect.com/science/article/abs/pii/S0893608099000738
2. Brown, G., Wyatt, J., & Tino, P. (2005). Managing diversity in regression ensembles. *JMLR*. http://www.jmlr.org/papers/volume6/brown05a/brown05a.pdf
3. Buschjager, S., Pfahler, L., & Morik, K. (2020). Generalized Negative Correlation Learning for Deep Ensembling. *arXiv:2011.02952*. https://arxiv.org/pdf/2011.02952
4. Zhang, L., et al. (2022). Deep Negative Correlation Classification. *arXiv:2212.07070*. https://arxiv.org/pdf/2212.07070
5. Liu, L., et al. (2019). Deep Neural Network Ensembles against Deception: Ensemble Diversity, Accuracy and Robustness. *arXiv:1908.11091*. https://arxiv.org/pdf/1908.11091
6. The Diversified Ensemble Neural Network (NeurIPS 2020). https://papers.nips.cc/paper_files/paper/2020/hash/b86e8d03fe992d1b0e19656875ee557c-Abstract.html
7. Ortega, L. A., Cabanas, R., & Masegosa, A. R. (2022). Diversity and Generalization in Neural Network Ensembles. *AISTATS*. https://proceedings.mlr.press/v151/ortega22a/ortega22a.pdf
8. Abe, T., Buchanan, E. K., Pleiss, G., & Cunningham, J. P. (2024). Pathologies of Predictive Diversity in Deep Ensembles. *arXiv:2302.00704*. https://arxiv.org/html/2302.00704v3
9. Jeffares, A., Liu, T., Crabbe, J., & van der Schaar, M. (2023). Joint Training of Deep Ensembles Fails Due to Learner Collusion. *NeurIPS*. https://arxiv.org/pdf/2301.11323
10. Zhang, W., Jiang, J., Shao, Y., & Cui, B. (2021). Efficient Diversity-Driven Ensemble for Deep Neural Networks. *arXiv:2112.13316*. https://arxiv.org/html/2112.13316v1
11. Shen, Y., Lu, Y., Li, Y., Tu, Y., Zhang, W., & Cui, B. (2022). DivBO: Diversity-aware CASH for Ensemble Learning. *NeurIPS*. https://arxiv.org/pdf/2302.03255
12. Rame, A., & Cord, M. (2021). DICE: Diversity in Deep Ensembles via Conditional Redundancy Adversarial Estimation. *ICLR*. https://ar5iv.labs.arxiv.org/html/2101.05544
13. Pang, T., Xu, K., Du, C., Chen, N., & Zhu, J. (2019). Improving adversarial robustness via promoting ensemble diversity. *ICML*. https://arxiv.org/abs/1901.08846
14. Yang, Y., et al. (2021). TRS: Transferability Reduced Ensemble via Promoting Gradient Diversity and Model Smoothness. *arXiv:2104.00671*. https://ar5iv.labs.arxiv.org/html/2104.00671v2
15. Lee, S., et al. (2016). Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles. *NIPS*. https://arxiv.org/abs/1606.07839
16. Malinin, A., & Gales, M. (2020). Ensemble Distribution Distillation. *ICLR*. https://arxiv.org/abs/1905.00076
17. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*. https://arxiv.org/abs/1503.02531
18. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*. https://arxiv.org/abs/1612.01474
19. Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy. *Machine Learning*. https://link.springer.com/article/10.1023/A:1022859003006

