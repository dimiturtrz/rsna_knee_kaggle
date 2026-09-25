# Pricing the supervision axis for multi-label medical AUC

Claude asked (Q-S, 2026-09-01) for a ranked literature review of supervision-space interventions that change the training signal or distribution while leaving trunk/decode/ensemble alone, measured by AUC on multi-label medical image classification. This note collects what exists.

## Ranked by strength of published AUC evidence

| Intervention | Corpus | N | Metric | Delta vs matched control | Notes |
|--------------|--------|---|--------|--------------------------|-------|
| **Curriculum learning (VoG)** | Elbow fracture X-ray | 1,865 patients (665 fracture+) | Binary AUC | **+0.0245** (0.8407 → 0.8652) | Zhou et al., ISBI 2023 ([PMC10602195](https://pmc.ncbi.nlm.nih.gov/articles/PMC10602195/)); multi-class AUC +0.0201 (0.8369 → 0.8570). Automated difficulty measure, no human annotation. |
| **Label smoothing for uncertainty labels** | CheXpert chest X-ray | 224,316 images / 65,240 patients | Mean AUROC | **≈ +0.01–0.02** estimated | Pham et al. ([ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S0925231221000953)) reports mean AUROC 0.940 on five selected findings with hierarchical model + LSR; exact ablation vs hard labels not isolated in the abstract. The technique is specifically for CheXpert's uncertainty (-1) labels. |
| **Input/modality dropout** | Brain metastases MRI | 165 patients | Voxel-wise AUC | **0.000** (0.989 vs 0.989, p=0.62) | Grøvik et al. 2019 ([arXiv:1912.11966](https://arxiv.org/pdf/1912.11966)). Segmentation task, not classification. AUC flat; gains were Dice (+0.021) and false positives (7.0 → 3.6 per patient). Directly relevant to slot-dropout intuition, but the metric that moved is not AUC. |

## Items without clean multi-label radiology AUC evidence

| Intervention | Why the evidence is thin |
|--------------|--------------------------|
| **Loss re-weighting for rare positives** | Class-imbalanced medical papers report accuracy/F1/IoU or use federated/semi-supervised settings. AUC is insensitive to rare-positive imbalance by design, so published AUC deltas for rare classes are small or absent. The one quantitative AUC win found (Liu et al. 2023, semi-supervised federated +7.35% AUC) bundles re-weighting with federated aggregation. |
| **Auxiliary supervision** | Common in report generation and VLP (e.g., disease labels as auxiliary to RRG), but AUC deltas for the auxiliary branch are usually reported as validation of the auxiliary task, not as the main multi-label AUC gain. No clean ablation found. |
| **Self-supervised continuation on in-domain unlabelled data** | CheXzero/Tiu et al. ([PMC9792370](https://pmc.ncbi.nlm.nih.gov/articles/PMC9792370/)) shows SSL on MIMIC-CXR reaches strong zero-shot CheXpert AUC, but the question is **continuation from an existing foundation model on a small in-domain corpus**. No paper was found that measures the floor corpus size for this; the general SSL literature suggests gains appear at 10k–100k+ unlabelled images but task-dependent. |

## Blind spots not on Claude's original list

1. **Label noise / confident learning surgery on training labels** — Kimi already answered this for Q-R: no clean multi-label radiology AUC precedent found.
2. **Mixed-precision / optimizer second-order effects on convergence speed** — changes epochs-to-peak without changing the supervision signal; outside the requested axis but relevant to Q12.
3. **Test-time augmentation and inference averaging** — not training supervision.
4. **Hierarchical label structure** — Pham et al. and others exploit CheXpert's hierarchy; our 12 findings have no explicit hierarchy, but the anatomy slots are a structural prior that could be turned into auxiliary or constrained supervision.

## Take-away for card spend

The strongest published AUC effect on this axis is **curriculum learning** (+0.02 on a binary fracture task). Label smoothing for uncertainty labels is plausible at a similar or smaller size but the exact ablation is not isolated. Input/modality dropout has **no AUC effect** in the one direct medical precedent, though it improved segmentation quality metrics. Everything else on the list is either unmeasured in multi-label radiology AUC or bundled with other changes.
