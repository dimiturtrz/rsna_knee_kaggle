# Stacking vs averaging when ensemble members are correlated — answer to Claude's Q-AG

Claude asked for published numbers where a stacked ensemble is compared against a plain average over members with measured high inter-member correlation, with the delta and the correlation at which it was measured. The goal is to decide whether a fitted combiner is worth a slot when `competence_axis` says 81.7-82% of what separates 105 members is one scalar.

## Strongest direct evidence: Nakata et al., 2023 — liver ultrasound (medical imaging)

Nakata et al. trained 16 CNNs on 26,440 B-mode ultrasound images of focal liver lesions and compared soft voting (SV), weighted average voting (WAV), weighted hard voting (WHV), and stacking (ST) with LightGBM as meta-learner. They also report a full pairwise correlation matrix of predicted labels.

Key numbers:
- **Best single model:** ResNeXt101, accuracy **0.719**.
- **Best stacking:** ST9 (top 9 CNNs), accuracy **0.776** → **+0.057** over single best.
- **Best weighted average:** WAV7 (top 7 CNNs), accuracy **0.783** → **+0.064** over single best.
- **Plain soft voting:** SV16 (all 16 CNNs), accuracy **0.776**.
- **Weighted hard voting:** WHV13, accuracy **0.779**.

Correlation structure:
- 120 pairwise correlations among 16 CNNs.
- 8 pairs in the 0.3 range; **112 pairs ≥ 0.4**.
- All pairs below 0.95; the highest within-family values include ResNet50–ResNet101 at **0.746** and EfficientNetB5–EfficientNetB6 at **0.613**.

Interpretation: in this medical-imaging task with moderate-to-high member correlation, **stacking did not beat a weighted average**. The fitted meta-learner captured the same +0.057 that simple statistical pooling already captured, and the weighted average did slightly better. This is direct support for the "gains collapse as correlation rises" hypothesis.

Source: Nakata N et al., *Ensemble Learning of Multiple Models Using Deep Convolutional Neural Networks for Image Classification of Liver Masses in Ultrasound Images*, Sci Rep 2023, PMID 36906418, PMCID [PMC9854883](https://pmc.ncbi.nlm.nih.gov/articles/PMC9854883/).

## Supporting evidence from biomedical stacking with explicit correlation threshold

Mohtasham et al. (2026) built a diversity-guided stacking framework for COVID-19 mortality prediction and explicitly state:

> "Traditional stacking often selects base models with low prediction correlation to ensure each contributes complementary information. **Highly correlated models (>0.75) add redundancy and weaken ensemble gains.**"

They filtered out base-model pairs above 0.75 correlation. Their heterogeneous stacks could still gain (best: RF + XGB with NN meta-learner, accuracy **0.914** vs best individual RF **0.892**, a **+0.022** delta), but they note that ensembles of highly correlated models "yielded limited gains, indicating performance saturation."

This is not imaging, but it is biomedical and it gives an explicit redundancy threshold: **>0.75 correlation is where stacking starts to weaken**.

Source: Mohtasham F et al., *Hybrid feature-selection and diversity-guided stacking ensemble for clinical prediction*, BMC Med Inform Decis Mak 2026, PMCID [PMC13089710](https://pmc.ncbi.nlm.nih.gov/articles/PMC13089710/).

## Cross-modality medical-imaging survey: Müller et al., 2022

Müller et al. evaluated Augmenting, Stacking, and Bagging across four medical imaging datasets (histology, COVID X-ray, ISIC dermoscopy, diabetic retinopathy fundus). They tested 12 pooling functions for stacking, ranging from unweighted mean to SVM and logistic regression.

Key findings:
- Stacking achieved up to **+13% F1-score gain** over the best single model on the hardest datasets.
- However: "**simple statistical pooling functions are equal or often even better than more complex pooling functions** like Support Vector Machines."
- The best pooling functions were frequently Mean (unweighted or weighted) or Soft Majority Vote.

They do not publish a pairwise correlation matrix, but their result pattern is consistent with Nakata: when gains are available, simple averaging often captures them as well as a learned combiner.

Source: Müller D, Soto-Rey I, Kramer F, *An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks*, arXiv:2201.11440, 2022 (189 cites).

## Bottom line for our roster

The published pattern is:
1. **Stacking gains are real when members are diverse.**
2. **When pairwise prediction correlations climb above ~0.75, stacking gains saturate** and simple/weighted averaging becomes competitive or superior.
3. In the strongest direct medical-imaging comparison (Nakata), the fitted combiner did **not** beat weighted averaging, and both were within ~0.007 of each other while the single-best model was ~0.06 behind.

If our 105 members are as redundant as `competence_axis` suggests, the literature says the stacking lane is likely to return only a few points at best — probably not the 1.5% we are looking for.
