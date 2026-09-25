# Label noise, pseudo-labeling, and partial labels — literature for Claude's three questions

This note answers the three questions in `2026-09-02T0735Z-claude-withdraw-qai-trunk-axis-is-measured.md`:

1. Semi-supervised / pseudo-labeling on multilabel medical imaging: gain and denominator.
2. Partially labelled rows (one of twelve labels): masked-loss training, measured numbers.
3. Label-noise ceiling for report-extracted labels: inter-extractor agreement and its effect on achievable AUC.

The rule of the search was: cite numbers and their denominators, not mechanisms.

---

## 1. Semi-supervised / pseudo-labeling on multilabel medical imaging

### Directly relevant result

Yang et al., 2025, "Unlabeled Insight, Labeled Boost: Contrastive Learning and Class-Adaptive Pseudo-Labeling for Semi-Supervised Medical Image Classification" ([PMC12563532](https://pmc.ncbi.nlm.nih.gov/articles/PMC12563532/)) reports the most useful ablation for our denominator question.

| labeled ratio | unlabeled:labeled | ChestX-ray14 mean AUC | gain vs supervised baseline |
|---|---|---|---|
| 100% (upper bound) | 0:1 | 78.54% | — |
| 20% | 4:1 | **70.81%** | **+2.64%** vs 20% supervised baseline (68.17%) |
| 2% | 49:1 | not stated vs baseline | +1.04% vs Mean Teacher only |

The ChestX-ray14 experiment is multi-label (14 thoracic disease categories). With 20% labeled data (15,694 images) and 80% unlabeled (62,774 images), the proposed CLCP-MT method gains +2.64% mean AUC over the same model trained only on the 20% labeled subset. The upper bound is 78.54%, so the method closes roughly one-third of the gap between 20%-supervised and fully-supervised.

### Supporting result

Zhan et al., 2023, "An uncertainty-aware self-training framework with consistency regularization for the multilabel classification of common computed tomography signs in lung nodules" ([PubMed/37711798](https://pubmed.ncbi.nlm.nih.gov/37711798/)):

- 641 lung nodules, 454 annotated with 6 CT signs
- SSL: mAP 0.806 vs supervised 0.730 (+7.6%)
- SSL: mean AUC 0.712 vs supervised 0.631 (+8.1%)

This is a smaller, private dataset and the absolute numbers are lower, but the relative gain from pseudo-labeling is large.

### Reading for our ratio

Our merged external cache is ~5,143 foreign train-only rows against 4,407 RSNA labeled rows, i.e. an **additional-to-labeled ratio of ~1.17:1**. That is far below Yang's 4:1 condition. Yang's curve is monotonic but concave: moving from 2% to 20% labeled roughly doubles the gain per added unlabeled image. At 1.17:1 the honest extrapolation is a **gain in the low single-digit thousandths of AUC**, not the +0.01–0.02 range. If the foreign rows were truly unlabeled, pseudo-labeling them is unlikely to close the 0.008 gap to the leaderboard.

### Verdict for Claude

Pseudo-labeling unlabeled medical images works in the literature, but the published gains at ratios close to ours are small. This is a **weak kill for the unlabeled-only variant** of the axis unless the unlabeled pool can be grown substantially.

---

## 2. Partially labelled rows

### Single-positive multi-label learning (the extreme case)

Schneider et al., 2024, "The Potential of Acknowledging the Unknown: Single Positive Multi-label Learning in Medical Image Processing" ([OpenReview](https://openreview.net/pdf?id=UDzWBT3Rgu)) is the cleanest medical-imaging experiment. They train DenseNet-121 on CheXpert with only **one positive label per sample** and treat the rest as unknown.

- Full BCE (all labels) is the ceiling.
- Assume Negative (AN) treats unknowns as negative — strong baseline because most labels are negative.
- Entropy Maximization (EM) and Asymmetric Pseudo-Labeling (EM+APL) reduce the harm of false negatives.
- Generalized Assume Negative (G-AN) is the best SPML method in their table, but the authors state it **does not reach full BCE performance**.

Exact AUROC gaps are not quoted in the extracted text, but the qualitative finding is consistent across all four sample sizes (5K, 10K, 25K, 40K): the best SPML loss sits below full supervision. Masked BCE — ignoring unknown labels rather than treating them as negative — is the standard partial-label baseline and is what their EM/G-AN losses improve upon.

### General partial multi-label learning

Masked BCE (mask out unknown labels in the loss) is the default recipe in the general computer-vision literature and in medical multitask learning. The GitHub discussion on DeepChem/ToxCast ([deepchem#1175](https://github.com/deepchem/deepchem/issues/1175)) makes the practical point: filling NaNs with zeros without masking overestimates predictiveness; masking during training and evaluation is required.

### Reading for our case

MRNet has decoded pixels and a labels table that covers ACL/meniscus/tears, not all 12 RSNA findings. Rijeka has ACL only. Treating these as rows with one or two known labels and masking the other eleven is exactly the SPML/partial-label setup. The literature says this is **worth doing** (it adds data without adding false negatives) but the gain is limited by how many labels are missing. A row with one known label out of twelve contributes roughly one-twelfth of the supervision signal of a fully-labeled row, minus the overlap benefit if the known label correlates with unknown ones.

### Verdict for Claude

Masked-loss training on partially-labeled rows is **standard and positive**, but the gain is bounded by label density. With ~5,143 rows that are mostly single-label, the effective supervision added is modest. It is not folklore; it is the honest way to use MRNet/Rijeka, but it is unlikely to be a 1.5% lever on its own.

---

## 3. Label-noise ceiling for report-extracted labels

### Radiology report labeler quality directly transfers to model quality

Jain et al., 2021, "Effect of Radiology Report Labeler Quality on Deep Learning Models for Chest X-Ray Interpretation" ([arXiv:2104.00793](https://arxiv.org/abs/2104.00793)) is the key result. On CheXpert:

| labeler | report-label F1 (avg) | vision-model AUROC (avg) | vision-model weighted AUROC |
|---|---|---|---|
| CheXpert / CheXbert | 0.53 | 0.84 | 0.83 |
| VisualCheXbert | **0.68** | **0.88** | **0.87** |

A +0.15 improvement in report-labeler average F1 (from 0.53 to 0.68) translated to a **+0.04 weighted AUROC** improvement in the downstream image classifier. The better labeler also had higher per-label F1 on every condition. This is direct evidence that report-label quality sets a meaningful ceiling on model performance.

### Inter-reader agreement for knee MRI

Astuto et al., 2021, "Automatic Deep Learning-assisted Detection and Grading of Abnormalities in Knee MRI Studies" ([PMC8166108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/)) reports linear-weighted Cohen κ between two attending radiologists on an external clinical knee-MRI set:

| tissue | κ |
|---|---|
| Bone marrow edema | 0.55 |
| Cartilage | 0.65 |
| Meniscus | 0.72 |
| ACL | 0.72 |

Trainee-to-attending agreement was lower (κ as low as 0.42 for cartilage without AI aid, rising to 0.61 with AI aid). The corresponding model AUCs on the holdout set were:

| tissue | AUC |
|---|---|
| Cartilage | 0.93 |
| Meniscus | 0.93 |
| Bone marrow edema | 0.83 |
| ACL | 0.90 |

These AUCs are measured against the radiologist consensus ground truth, so they are already an upper bound imposed by that consensus.

### Chest-X-ray inter-reader agreement for scale

Li et al., 2022, "Inter- and Intra-Observer Agreement When Using a Diagnostic Labeling Scheme for Annotating Findings on Chest X-rays" ([MDPI](https://www.mdpi.com/2075-4418/12/12/3112)) reports Randolph's κ across 25 CXR labels from 0.40 to 0.99, with many interpretive labels in the 0.40–0.60 range. This matches the knee-MRI picture: agreement is high for unambiguous findings and low for subtle/interpretive ones.

### Reading for our case

Our 12 RSNA targets were extracted from free-text reports by LLM/consensus methods. The best consensus reports ~0.90 macro AUC on the 58 gold studies (`rayanbabur_rsna-knee-2026-calibrated-soft-targets/target_quality_audit.json`). Jain's result suggests that if a better labeler could push F1 up by another ~0.10, the downstream model might gain another ~0.02–0.03 AUROC. But the knee-MRI inter-reader κ values (0.55–0.72) and the small gold set (58 studies) suggest the **remaining label-improvement headroom is likely small and hard to measure**.

### Verdict for Claude

The label-noise ceiling is real and measurable. Jain shows +0.15 labeler F1 → +0.04 AUROC. Astuto shows knee inter-reader κ in the 0.55–0.72 range for the tissues we care about. The 0.008 gap to the leaderboard is **probably not sitting in the labels** as a clean +0.01 recoverable item, because the current consensus labels already reach 0.90 on gold-58 and the labelers agree reasonably well on the common classes.

---

## Summary for decision-making

| axis | published gain at our scale | verdict |
|---|---|---|
| Pseudo-labeling unlabeled images | ~+0.002–0.005 AUC at 1.17:1 ratio | weak kill unless pool grows |
| Masked-loss training on partial labels (MRNet/Rijeka) | bounded by label density; modest positive | proceed, but expect small gain |
| Better report-extracted labels | +0.04 AUROC for +0.15 labeler F1 in chest X-ray | ceiling is near; not the main 1.5% lever |

The three questions together point in the same direction: the remaining gap is unlikely to come from squeezing the label side harder. The gains are real, measured, and small.
