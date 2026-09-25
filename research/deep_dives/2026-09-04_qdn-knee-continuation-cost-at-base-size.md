# Q-DN — Knee-domain continuation at ViT-B size: cost, evidence, and shippable data

**Date:** 2026-09-04  
**Trigger:** Claude question Q-DN on whether a ViT-B knee continuation is the highest-value open lever.  
**Scope:** Literature survey only; no code changes.

## Key finding

The lever is plausible but empirically unsettled. No published work directly ablates *continuation gain* against *trunk size* on the same corpus and task. Cost estimates from analogous medical-SSL recipes suggest ~200–400 GPU-hours on an RTX 5090 for a ~1M-slice DINOv2 continuation. Licence constraints, not compute, are the binding uncertainty.

## 1. Does continuation pay more at base size?

### Direct evidence: none

- **OrthoFoundation** is the only public knee-focused foundation model. It initializes DINOv2 weights and continues self-supervised pretraining on 1.2M knee X-ray + MRI images (OAI + fastMRI + internal PUTH). It reports only a **DINOv3-Large** backbone and does not compare ViT-B vs. ViT-L on the same knee corpus ([arXiv:2601.18250](https://arxiv.org/abs/2601.18250)).
- **CheXFound** pretrains both ViT-B and ViT-L on CXR-207K / 744K / 987K. At every data scale ViT-L outperforms ViT-B, and both improve monotonically with more data ([arXiv:2502.05142v2](https://arxiv.org/html/2502.05142v2), Table X). This is evidence for **data and model scaling**, not for a larger continuation *delta* at base size.
- A plant-image SSL paper states that "dataset relevance is a more critical factor than model capacity" ([arXiv:2604.27538v2](https://arxiv.org/html/2604.27538v2)), but this is a different domain and not a controlled ablation.

### Interpretation

The intuition that smaller models benefit more from domain-specific pretraining is grounded in general transfer-learning experience but is **not proven for this exact setting**. The base-size continuation cell is unpriced.

## 2. Cost estimates

### Anchor recipes

| Recipe | Model | Data | Hardware | GPU-hours | Source |
|--------|-------|------|----------|-----------|--------|
| CheXFound | ViT-L/16 | ~987K CXRs | 8× A100 40GB, 100 epochs | 384–768 | [arXiv:2502.05142v2](https://arxiv.org/html/2502.05142v2), §IV-A1 |
| Curia-2 | ViT-B | 400M radiology images | 32× A100 64GB | 1,820 | [arXiv:2604.01987v1](https://arxiv.org/html/2604.01987v1), §2.3 |
| Curia-2 | ViT-L | 400M radiology images | 64× A100 64GB | 3,270 | [arXiv:2604.01987v1](https://arxiv.org/html/2604.01987v1), §2.3 |
| DINOv2 distillation | ViT-B | LVD-142M natural images | A100 | 5,300 | [MODEL_CARD.md](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) |
| DINOv2 distillation | ViT-L | LVD-142M natural images | A100 | 8,000 | [MODEL_CARD.md](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) |

### Scaling to one RTX 5090

A ~1M-slice knee continuation at 224² is closest to CheXFound in scale. Assuming ~2× FP16 throughput of an A100, a ViT-B/L run lands in the **200–400 GPU-hour** range, or roughly **1–2 weeks of wall time on a single card** for a full 100-epoch recipe.

### Method

The standard recipe is **DINOv2-style continued self-distillation** from a natural-image teacher (DINOv2 or DINOv3 initialization), with multi-crop augmentation and EMA teacher. This is what OrthoFoundation and CheXFound both use.

## 3. Shippable unlabelled knee MRI corpora

| Corpus | Approx. MRI slices | Shippability | Notes |
|--------|-------------------|--------------|-------|
| fastMRI | 300k (OrthoFoundation count) | **No** — internal instrument only per NYU DSA (bead knee-nxthh) | Already excluded |
| OAI | 200k (OrthoFoundation count) | **DUA-dependent** — requires checking current data-use agreement for model-weight redistribution | Largest public knee MRI corpus |
| MRNet | ~tens of thousands | Unknown — no public licence text found | Too small to be primary corpus |
| SKM-TEA | 155 scans | Unknown licence; far too small | Not viable for SSL |
| RSNA competition pixels | competition training set | **Yes for competition** — Kaggle rules require model weights to be public ([RSNA Knee rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)) | Safest shippable source |
| Internal hospital cohort | variable | Not available | OrthoFoundation used PUTH internal data |

**Conclusion:** RSNA pixels are the only corpus with explicit public-weight terms. OAI is viable only after reading the DUA. fastMRI is out.

## 4. Floor corpus size

No knee-MRI ablation exists. The nearest evidence:

- CheXFound shows monotonic gains from CXR-207K → 744K → 987K, with no saturation at 987K ([arXiv:2502.05142v2](https://arxiv.org/html/2502.05142v2), Table X).
- OrthoFoundation used ~500k public knee MRI slices plus 200k internal slices and saw meaningful gains.

**150k slices** is roughly one-third of OrthoFoundation's public MRI corpus and 70% of CheXFound's smallest subset. It is likely **on the increasing part of the curve** but well below the saturated regime. Combining RSNA + OAI to reach 300–500k slices would materially improve expected return.

## Decision implications

1. The base-size continuation hypothesis is **not literature-supported**; it is a bet.
2. The compute cost is **one-card tractable** (~1–2 weeks).
3. The licence check is the **blocking task** before any run can be armed.
4. Corpus size should be maximized within shippable sources; **150k is a floor, not a target**.
