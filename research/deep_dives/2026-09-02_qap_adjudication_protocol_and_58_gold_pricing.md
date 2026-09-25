# Q-AP — adjudication protocol, the 58-gold pricing, and the cross-standard literature

Claude's follow-up to Q-AO asks three things: the RSNA knee challenge's test-set adjudication protocol, a published case where report-derived train labels were corrected and measured against a radiologist-adjudicated test set, and the competition metric definition.

## 1. Test-set annotation protocol

### Direct source

RSNA's launch article for the challenge says:

> "The dataset used to assess model performance was annotated by expert radiologists." ([RSNA News, 2026-08-05](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri))

The AuntMinnie reprint says the same:

> "The dataset used to assess model performance was annotated by expert radiologists." ([AuntMinnie, 2026-08-06](https://www.auntminnie.com/imaging-informatics/artificial-intelligence/news/15831808/rsna-launches-2026-knee-abnormality-detection-ai-challenge))

This is a statement about the **test set**, not the training reports. It does not give the number of readers or the consensus rule.

### RSNA's general protocol

Kitamura et al. 2024 (*Lessons Learned in Building Expertly Annotated Multi-Institutional Datasets for AI Competitions*, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11140499/)) describes RSNA's standard:

> "A minimum of three annotators are assigned to the test set, and a consensus method for labeling and region of interest is established."

Concrete precedents from the same paper:

| Challenge | Test-set ground truth |
|---|---|
| Pediatric bone age | Averaged over 6 radiologist labelers (continuous target) |
| Pneumonia detection | Intersection of annotators' bounding boxes |

So the knee challenge's test set is almost certainly a multi-reader consensus, but the exact number of readers and the consensus rule for this specific competition are **not stated in any public source found so far**.

## 2. Competition metric

The task is 12-label binary classification. The evaluation metric is the **macro-averaged AUC-ROC across the 12 findings**. This is stated in multiple public notebooks and the Kaggle competition page snippets:

> "The final score is, in other words, the macro-averaged AUC ROC." ([Kaggle — RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection))

Because it is macro-averaged, the per-finding noise ceiling matters independently per column. Astuto et al.'s knee kappa numbers (BME 0.55, cartilage 0.65, meniscus 0.72, ACL 0.72) are the right ceiling to keep in mind.

## 3. Cross-standard published case: report-derived train → adjudicated test

### Closest match: Jain et al. 2021

[Jain et al., *Effect of Radiology Report Labeler Quality on Deep Learning Models for Chest X-Ray Interpretation*, 2021](https://arxiv.org/html/2104.00793v3) is the exact shape of the experiment we are considering:

- **Training labels:** automatically extracted from radiology reports using different report labelers (CheXpert, CheXbert, VisualCheXbert).
- **Test labels:** radiologist-annotated X-rays (the CheXpert test set of 500 studies).
- **Question:** does improving the quality of the report-derived training labels improve the score on the radiologist-adjudicated test set?

**Result:** yes. On the radiologist-labeled test set, a DenseNet-121 trained on VisualCheXbert labels scored **weighted-average AUROC 0.87** (95% CI 0.86–0.89), versus **0.83** (95% CI 0.81–0.84) for the same model trained on CheXpert labels — a **+0.04 AUROC** gain from a better report labeler alone.

| Labeler | Weighted avg AUROC on radiologist test |
|---|---|
| CheXpert U-Zeros | 0.83 |
| CheXpert U-Ones | 0.76 |
| CheXbert U-Zeros | 0.83 |
| CheXbert U-Ones | 0.79 |
| **VisualCheXbert** | **0.87** |

Important caveat: this is **not** image-based label correction. It is report-labeler improvement. It tells us that cleaning the report-derived training signal can buy material gains on an adjudicated test set, but it does not bound how much an image-based correction could buy.

### Analog: CXR-LT 2026

The [CXR-LT 2026 challenge overview](https://arxiv.org/html/2604.15555v1) explicitly uses the same split standard:

> "For the training set, labels were automatically extracted from radiology reports ... For the development and test sets, we used manual radiologist annotations to ensure high-quality evaluation."

But the paper does **not** report an experiment that measures the gain from correcting the noisy training labels. It is a benchmark description, not a label-correction ablation.

### Bottom line on the literature question

A direct paper that trains on report-derived labels, corrects them toward image-derived labels, and measures the gain on a radiologist-adjudicated test set was **not found**. Jain 2021 is the strongest indirect evidence: cleaning the report-derived training signal gave +0.04 AUROC on an adjudicated test set.

## 4. What this means for the label-correction axis

- The test set is very likely radiologist-adjudicated (RSNA statement + RSNA's standard protocol).
- The metric is macro AUC, so per-finding noise ceilings bite independently.
- There is no off-the-shelf number that prices image-based correction, but the closest analog says the axis is live: a cleaner training signal bought +0.04 AUROC on an adjudicated test.
- The cheap experiment proposed in the Q-AO note remains the right first move: use the 58 gold studies to build an image-report disagreement signal, add it as sample weight or auxiliary loss, and measure whether it helps or hurts the holdout. If it helps, the test labels are likely image-aligned and the correction axis is worth building out. If it hurts, the test labels are likely report-aligned and the axis should be declined.
