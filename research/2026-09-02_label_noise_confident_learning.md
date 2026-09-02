# Q-XI — Label noise / confident learning: field effect size, threshold rule, and the 58-gold bound

Claude asked three ranked questions about whether confident learning is worth a card for the RSNA knee MRI task.

- Context: `ortho_dense` finished at **0.8452** vs `base224` **0.8502** (-0.0050, inside the 0.0062 floor). `v3large_dense` is still running. If it is also flat, the only large axis left is supervision quality.
- The project has 4,407 report-derived label rows and 58 radiologist-annotated studies (696 cells).

## (1) Confident learning / label-noise estimation on multi-label medical imaging — real effect size?

### What exists

- **Confident Learning (CL)** is a formal framework for estimating label noise from out-of-sample predicted probabilities. It estimates the joint distribution of noisy and true labels via a *confident joint* and per-class self-confidence thresholds (Northcutt et al. 2021, *JAIR* / arXiv:1911.00068).
- **Multi-label support** exists in the reference implementation: `cleanlab.multilabel_classification` provides `find_label_issues`, `get_label_quality_scores`, and `get_label_quality_scores_per_class` with a one-vs-rest confident joint of shape `(K, 2, 2)` ([cleanlab multilabel_classification.rank](https://docs.cleanlab.ai/v2.7.1/cleanlab/multilabel_classification/rank.html), [filter](https://docs.cleanlab.ai/v2.7.1/cleanlab/multilabel_classification/filter.html)). The multi-label issue thread ([cleanlab #55](https://github.com/cleanlab/cleanlab/issues/55)) tracks that this is supported.
- **Class-conditional multi-label noise (CCMN)** is the theoretical extension of the noise matrix to multi-label learning (Xie & Huang, *TPAMI* 2023; [PubMed](https://pubmed.ncbi.nlm.nih.gov/34995182/), [PDF](https://xiemk.github.io/publication/tpami-ccmn-preprint.pdf)).

### What is missing

- I could **not** find a published confident-learning result on a **multi-label radiology corpus** where training-label correction moved **held-out AUC by >0.01** against report-derived labels. The cleanlab/cleanlab-style results I can reach are on single-label natural-image benchmarks (ImageNet, CIFAR).
- Medical label-noise papers report gains, but usually on:
  - cleaned **test sets** or re-annotated evaluation sets,
  - **segmentation IoU** (e.g. the retinal CL segmentation paper [PMC12293725](https://pmc.ncbi.nlm.nih.gov/articles/PMC12293725/)),
  - methods that modify the loss or use consistency, not confident-learning label surgery.
- Zhou et al. 2023 report medical image classification gains from a disentangled-distribution + consistency method: **CheXpert mean AUC +2.44%**, **KaggleDR+ Q.W. Kappa +5.65%**, **FGADR +4.66%** ([Beihang page](https://research.buaa.edu.cn/en/publications/combating-medical-noisy-labels-by-disentangled-distribution-learn/)). This is a *different mechanism* (not CL), but it establishes that label noise is worth points in medical imaging.
- Lin et al. 2025 in *npj Digital Medicine* report that **10% label noise** decreased classification accuracy by **4.0% (CFP)** and **4.4% (OCT)** ([Nature](https://www.nature.com/articles/s41746-024-01424-x)). This bounds the *cost* side, not the *fix* side.
- **LNMBench** (Ma et al., arXiv:2512.09315, 2025/2026) benchmarks 10 label-noise methods on 7 medical datasets / 6 modalities / 3 noise patterns and finds that existing methods degrade substantially under high or real-world noise ([arXiv](https://arxiv.org/abs/2512.09315)). This is a cautionary signal: the fix is not guaranteed to transfer cleanly.

### Honest null

> A principled multi-label confident-learning implementation exists (cleanlab), but I cannot cite a field result that shows it improves held-out AUC on a 12-label radiology task with report-extracted labels and severe imbalance. The mechanism is plausible; the effect size on this exact shape is **unmeasured in the literature I can reach**.

If the bead gate already shows that cleanlab flags beat random on the 108 known disagreements, the project-specific evidence is stronger than the literature.

## (2) Per-cell reweighting vs dropping suspect cells — what is the principled threshold rule?

### The CL threshold rule (not tuned)

Northcutt et al. define the **confident class threshold** for class `j` as the expected (average) self-confidence for class `j`:

> threshold_j = average predicted probability of class `j` over all examples labeled as class `j`

This is Equation 2, Section 3.1 of Northcutt et al. 2021 and is implemented as `cleanlab.count.get_confident_thresholds` ([cleanlab count docs](https://docs.cleanlab.ai/v2.7.1/cleanlab/count.html)).

- It is **per-class**, not global.
- It is **computed from the noisy labels and out-of-sample probabilities**, not from the holdout AUC.
- For multi-label, cleanlab treats each of the K classes as a binary one-vs-rest problem and computes a 2×2 confident joint per class.

### What CL does with the threshold

- An example is counted as "confidently true label = k" if its predicted probability for class k is **≥ threshold_k**.
- The **confident joint** counts examples in (noisy label, true label) bins.
- From this joint, CL estimates the noise matrix and identifies label issues.

### Prune vs reweight

- **Confident Learning itself is a pruning framework**: it identifies and removes noisy examples. The DCAI course notes state CL is "based on the principles of pruning noisy data, counting to estimate ..., re-weighting examples by the estimated latent prior" ([MIT DCAI](https://dcai.csail.mit.edu/2024/label-errors/)).
- **cleanlab's `LearningWithNoisyLabels`** is the loss-reweighting variant; it reweights examples by the estimated latent prior `p(y)` / inverse noise matrix.
- I did **not** find a clean literature comparison that says "reweighting beats pruning by X" under class imbalance in multi-label medical imaging. The principled starting point from the CL literature is **prune-by-class or prune-by-noise-rate**, using the class-conditional self-confidence threshold.

### Recommendation for the project

If a warm-start reweighted arm is cheap, the safest literature-backed sequence is:

1. Use `cleanlab.multilabel_classification.filter.find_multilabel_issues_per_class` with `filter_by='prune_by_noise_rate'` and the default thresholds.
2. Measure the flagging gate against the 108 known disagreements (already done and passed).
3. For the arm, compare two treatments at the same threshold:
   - **Drop** the flagged (study, finding) cells.
   - **Reweight** the flagged cells by the estimated inverse noise rate.
4. The threshold is fixed by CL theory; do not fit it to the holdout score.

## (3) Bounding the noise rate from the 58 gold labels

Source: `interpretations/kneemri/2026-09-02_report-to-label_audit.md`.

### Denominator

The correct denominator is all **58 studies × 12 labels = 696 cells**, not only cells where the extractor fired. A rate conditioned on "extractor fired" is a conditional error rate, not a corpus-wide noise rate.

### Pooled estimate

| quantity | value |
| :-- | ---: |
| cells | 696 |
| missed (report negative, gold positive) | 54 |
| denied (report positive, gold negative) | 91 |
| total label errors | 145 |
| pooled label-error rate | **20.83%** |
| 95% exact (Clopper-Pearson) interval | **[17.82%, 23.85%]** |
| 95% Wilson interval | [17.98%, 24.01%] |

### Per-label label-error rates (58 cells each)

| label | errors / 58 | rate | 95% exact CI |
| :-- | ---: | ---: | :--- |
| ACL | 10 | 17.2% | [8.6%, 27.6%] |
| MCL | 9 | 15.5% | [6.9%, 25.9%] |
| Medial Meniscus | 8 | 13.8% | [5.2%, 22.4%] |
| Lateral Meniscus | 13 | 22.4% | [12.1%, 32.8%] |
| Medial OA | 9 | 15.5% | [6.9%, 25.9%] |
| Lateral OA | 9 | 15.5% | [6.9%, 25.9%] |
| PF OA | 13 | 22.4% | [12.1%, 32.8%] |
| Effusion | 17 | 29.3% | [17.2%, 41.4%] |
| Synovitis | 17 | 29.3% | [17.2%, 41.4%] |
| Baker's | 6 | 10.3% | [3.4%, 19.0%] |
| Contusion | 23 | 39.7% | [27.6%, 51.7%] |
| Fracture | 11 | 19.0% | [8.6%, 29.3%] |

### Largest defensible claim

- The pooled rate is bounded away from zero: with 95% confidence, **at least 17.8%** of the (study, finding) cells in the gold-58 sample carry a report-vs-radiologist disagreement.
- This is **not** the same as the rate over the full 4,407-study corpus. The 58 studies were selected for having all 12 image-read labels; they may not be a random sample of the corpus. The bound is on the **audit sample**, and extrapolation to the corpus is an assumption.
- The 58-gold sample cannot tightly bound a per-label rate for rare findings (e.g. MCL has only 9 gold positives), but it can bound the **pooled** rate.

### What this says about the 87% aleatoric mass

The aleatoric term in the ensemble uncertainty decomposition is defined *given the labels*. If ~18–24% of the gold-58 cells are label errors, then some fraction of the "aleatoric" mass is annotation noise, not imaging ambiguity. The 58-gold sample is too small to split the 87% cleanly, but it is large enough to say the split is **non-zero** and **material**.

## Ranking by decision impact

I agree with Claude's ordering:

1. **Field effect size (Q1)** — needed to decide whether the arm is exploration or replication. The honest null is the key output.
2. **Threshold rule (Q2)** — needed to avoid tuning the threshold on the holdout. The CL self-confidence rule is the principled answer.
3. **58-gold bound (Q3)** — bounds the problem size, but does not select a threshold or arm. It is a supporting number.

## Sources

- Northcutt et al., "Confident Learning: Estimating Uncertainty in Dataset Labels", *JAIR* 2021. ([JAIR](https://www.jair.org/index.php/jair/article/download/12125/26676/26691) / [arXiv:1911.00068](https://arxiv.org/abs/1911.00068))
- Xie & Huang, "Class-Conditional Multi-Label Noise", *TPAMI* 2023. ([PubMed](https://pubmed.ncbi.nlm.nih.gov/34995182/) / [PDF](https://xiemk.github.io/publication/tpami-ccmn-preprint.pdf))
- cleanlab multi-label docs: [rank](https://docs.cleanlab.ai/v2.7.1/cleanlab/multilabel_classification/rank.html), [filter](https://docs.cleanlab.ai/v2.7.1/cleanlab/multilabel_classification/filter.html), [count](https://docs.cleanlab.ai/v2.7.1/cleanlab/count.html).
- Zhou et al., "Combating Medical Noisy Labels by Disentangled Distribution Learning", 2023. ([Beihang](https://research.buaa.edu.cn/en/publications/combating-medical-noisy-labels-by-disentangled-distribution-learn/))
- Ma et al., "LNMBench: A Benchmark for Medical Label-Noise Learning", arXiv:2512.09315. ([arXiv](https://arxiv.org/abs/2512.09315))
- Lin et al., "Impact of label noise on deep learning in medical image classification", *npj Digital Medicine* 2025. ([Nature](https://www.nature.com/articles/s41746-024-01424-x))
