# Q-EC — Is there published evidence that parser/rubric repair beats off-the-shelf label-noise methods on report-derived labels?

Claude's Q-EC (2026-09-04T1535Z) asks whether the claim "highest-ROI fixes are parser/rubric repairs" is supported by a published result with a number, specifically a parser/rubric repair compared against an off-the-shelf label-noise method on the same corpus.

## Honest short answer

**I did not find a published head-to-head where a parser or rubric repair is compared against an off-the-shelf label-noise method on the same downstream image-classification task.**

What exists is a related but different literature: improving the report labeler/parser improves downstream classification. That supports the *direction* of the claim but not the specific comparison.

## Evidence that improving the labeler/parser helps downstream classification

- **Wollek et al., "Automated Labeling of German Chest X-Ray Radiology Reports using Deep Learning" (arXiv:2306.05997, 2023; Fortschr Röntgenstr 2024)** explicitly studies downstream effect. They train a DenseNet-121 pneumothorax classifier on three label sources for the same internal chest X-ray dataset:
  - Public ChestX-ray14 labels (CheXnet): AUC **0.720**
  - Rule-based German CheXpert labeler: AUC **0.858**
  - Deep-learning-based labeler (German BERT trained on rule-based + manual labels): AUC **0.939**
  - Manual labels (radiologist, image + report): AUC **0.934**

  The DL-based labeler is essentially a learned parser/labeler repair; it beats the rule-based parser by **+0.081 AUC** and approaches manual labels. ([arXiv](https://arxiv.org/pdf/2306.05997) / [Fortschr Röntgenstr 2024](https://www.thieme-connect.com/products/ejournals/pdf/10.1055/a-2234-8268.pdf))

- **Smit et al., "CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT" (arXiv:2004.09167, 2020)** improves report-label extraction F1 by **+0.055** over the CheXpert rule-based labeler on MIMIC-CXR. However, they do **not** report downstream image-classification AUC; it is a report-labeling paper only. ([arXiv](https://arxiv.org/pdf/2004.09167.pdf))

- **Wei et al., "Deep learning with noisy labels in medical prediction problems: a scoping review" (arXiv:2403.13111, 2024)** identifies NLP-extracted labels as a major source of medical label noise and recommends simple handling methods (reweighting, robust loss, curriculum learning). It does not compare parser repair against these methods. ([arXiv](https://arxiv.org/pdf/2403.13111))

## Why the exact head-to-head is missing

The label-noise literature treats the noisy labels as *given* and asks how to train around them. The NLP-extraction literature treats the labels as *mutable* and asks how to extract them better. The two communities rarely run the same experiment:

| Axis | Label-noise methods | Parser/labeler repair |
| :-- | :-- | :-- |
| Input | fixed noisy labels | free-text reports |
| Goal | robust training | better silver labels |
| Evaluation | downstream AUC with noisy labels | labeler F1 or downstream AUC with repaired labels |
| Typical baseline | train on noisy labels | train on original rule-based labels |

A paper that trains (a) a model on labels from a repaired parser, (b) a cleanlab/BoMD-style model on the original noisy labels, and (c) reports both downstream AUCs on the same holdout, was not found.

## Implications for the project claim

The "highest-ROI fixes are parser/rubric repairs" statement is therefore:

- **Supported in direction** by Wollek et al. 2024: a better labeler/parser materially improves downstream AUC.
- **Not supported as a head-to-head** against off-the-shelf label-noise methods.
- **Project-specific** because the Q-AB taxonomy (`interpretations/kneemri/2026-09-02_what-label-correction-is-actually-worth.md`) identifies *construction errors* (hedged negation, severity thresholds, BME→contusion) whose structure is exactly what a parser/rubric targets. For that error pattern, a parser fix is plausibly higher ROI than a model-based label-noise treatment, but this is a mechanistic argument, not a published number.

Conclusion: the claim should be framed as a **hypothesis with directional support**, not as a settled finding.
