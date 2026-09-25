# Missing gold labels and medical vision-language pretraining axis

2026-09-02, Kimi. Survey triggered by two things: the public competition skeleton says scores above ~0.9 likely require a strategy for missing gold labels, and the project has 4,407 studies with free-text reports but only 58 studies with explicit 12-label annotations.

## The setup

- `train.csv` has 4,407 studies. Only 58 carry any of the 12 abnormality labels; those 58 have all 12 labels filled in.
- Every study has a free-text `Report` in one of nine languages.
- The project already derives labels from reports (`report_labels_v2.csv`) and trains on those derived labels.
- The competition itself suggests deriving labels from report text as the intended way to use the unlabeled majority.

So the question is not "can we use reports?" but "can we use them *better* than the current extraction pipeline?" and "what else can we do with the 4,349 studies that have no gold labels?"

## What the field has measured

### 1. Medical vision-language pretraining (MedCLIP-style)

[MedCLIP: Contrastive Learning from Unpaired Medical Images and Text](https://ar5iv.labs.arxiv.org/html/2210.10163) (Wu et al., EMNLP 2022) is the most cited recent medical VLP work. Key design choices:

- **Decouples image and text** so image-only, text-only and paired datasets can all be used. This is exactly the shape of the RSNA knee data: all studies have images, all have text, only a few have explicit labels.
- Uses **MetaMap + UMLS** to extract medical entities from reports and align them with diagnosis labels, building a soft semantic similarity matrix.
- Replaces InfoNCE with a **semantic matching loss** that treats reports describing the same symptoms as positives even when they come from different patients, eliminating false negatives.
- Reports **>10% accuracy improvement on zero-shot** classification and **>2% on retrieval** over ConVIRT and GLoRIA.
- Shows strong **data efficiency**: with 20K pretraining samples it beats GLoRIA trained on ~200K paired samples.

Limitations for this project:

- MedCLIP is evaluated on chest X-ray (MIMIC-CXR, CheXpert). Transfer to knee MRI is not proven.
- It requires a text encoder and medical NER/UMLS pipeline; the current project has no text encoder.
- The gains are measured on zero-shot and frozen-probe classification. Full fine-tuning gains may be smaller.

Still, it is the canonical reference for "contrastive image-report pretraining as a pretraining stage" (mechanism (a) in Q-Y), which the project's own auxiliary-loss refutation does not test.

### 2. Semi-supervised / pseudo-label learning with missing labels

Positive-unlabeled and semi-supervised methods are the obvious tool for 58 labels + 4,349 unlabeled studies.

- [Huynh et al., *Artificial Intelligence in Medicine* 2023](https://www.sciencedirect.com/science/article/abs/pii/S016926072200013X): semi-supervised learning for medical image classification with imbalanced data. Cited 151 times; relevant to the extreme label imbalance here.
- [Ihler et al., CVPR 2024W](https://openaccess.thecvf.com/content/CVPR2024W/DCAMI/papers/Ihler_Distribution-Aware_Multi-Label_FixMatch_for_Semi-Supervised_Learning_on_CheXpert._CVPRW_2024_paper.pdf): Distribution-Aware Multi-Label FixMatch for CheXpert. Directly addresses multi-label semi-supervised medical classification with class imbalance.
- [Kiryo et al., NIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/6f7c766c7b6c843b217a49f9ee7f5d49-Abstract.html): positive-unlabeled learning with non-negative risk estimator. A foundational method when most data are unlabeled and only positives are observed.

These are more generic than MedCLIP and have not been benchmarked on knee MRI, but the problem shape matches.

### 3. What the competition says

A [Kaggle discussion note](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733652) states that the competition explicitly suggests deriving labels for the unlabeled studies from the report text, and that the reports are in nine languages. The current project already does this, so the next question is whether a more sophisticated NLP extraction (multilingual NER, negation handling, section extraction) improves the derived labels.

## Candidate mechanisms, ranked by cost

| Mechanism | Cost | Evidence strength | What it would look like here |
|---|---|---|---|
| **Improve report-to-label extraction** | Low (NLP only) | Moderate | Replace the current report parser with a multilingual clinical NER/negation model; validate against the 58 gold labels. |
| **MedCLIP-style pretraining on image-report pairs** | Medium (needs text encoder + pretraining run) | Moderate (chest X-ray only) | Pretrain a DINOv2/Swin trunk with a report contrastive objective on the 4,407 studies, then fine-tune on derived labels. |
| **Semi-supervised pseudo-labeling (FixMatch/PU)** | Medium (training-run cost) | Low-moderate for knee MRI | Use the 58 gold + derived labels as seed, generate pseudo-labels for the rest, train with consistency regularization. |
| **Use gold 58 as a calibration/validation set** | Very low | Strong | Since the 58 are the only clean labels, use them to calibrate per-label thresholds and estimate report-label noise. |

## Implications

- The **cheapest and highest-confidence move** is to treat the 58 gold-labeled studies as a clean calibration/validation set. They are too few to train on, but they are enough to estimate the noise in the report-derived labels and to tune per-label decision thresholds.
- The **next cheapest** is to improve the report parser. The competition explicitly expects this, and the reports are multilingual. A parser that handles negation, uncertainty, and section structure (Findings vs Impression) is likely to raise label quality.
- **MedCLIP-style pretraining** is the mechanism behind the public baseline's text branch (Q-Z) and the open axis in Q-Y. It is not refuted by the project's auxiliary-loss experiment, which tested mechanism (b). It is expensive enough that it should be gated by the OrthoFoundation probe and the metadata seat first.

## Recommended next step

File a bead to audit the current report-to-label pipeline against the 58 gold labels: compute per-label precision/recall of the derived labels, identify systematic error modes (negation, false positives, missed synonyms), and decide whether an improved parser is worth the work. This is a desk task that costs no card time.

Bead filed: `knee-skcf`.
