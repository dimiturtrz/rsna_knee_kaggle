# Q-DL — Does input geometry buy ensemble diversity without the quality cost of objective diversity?

**Date:** 2026-09-04
**Question (Claude):** `vjepa_warm` is more distant from the DINOv2 pool than the LoRA rows and also stronger solo, breaking the usual "distance costs accuracy" rule. Is this a known effect, what mechanism explains it, what second geometry point could be added, and does geometry diversity pay at the ensemble level?

## TL;DR

The exact "geometry diversity is cheaper than objective diversity" pattern is **not a named effect**, but **2D/3D complementary ensembles** are a well-reported recipe in medical imaging. The best-supported mechanism is that the two geometries make **different errors** because one sees through-plane context and the other does not. For a second geometry point, **OrthoDiffusion** is the only public, knee-specific, multi-label-diagnosis 3D foundation model. The ensemble-level evidence is discouraging: `vjepa_warm` seated at **-0.0007**, consistent with the rule that a single distant member rarely moves the mean of an already-strong ensemble.

## 1. Named/reported effect

No single term exists for "input-geometry diversity per unit solo accuracy." It is an instance of **multi-view / multi-planar / 2D–3D complementary ensembles**.

| paper | finding | relevance |
|---|---|---|
| AlzhiNet (Akindele et al., 2024) [arXiv:2410.02714](https://arxiv.org/abs/2410.02714) | 2D-CNN + 3D-CNN hybrid outperforms each alone; calls the representations **complementary**. | Direct classification precedent for 2D+3D geometry fusion. |
| Thapar et al. (CVPRW 2026) [arXiv:2603.14621](https://arxiv.org/abs/2603.14621) | Ensemble of DINOv2 ViT + RadImageNet DenseNet + CNN MIL improves COVID-19 CT F1. | Cross-architecture/objective diversity helps, but not geometry-only. |
| Baskota et al. (2026) [Frontiers in AI](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1812932/full) | Disagreement-based weighted ensemble of Swin UNETR + SegResNet for 3D glioma segmentation. | Disagreement between 2D-like and 3D-like architectures is treated as signal. |
| NVIDIA safety-diversity (Filipiuk & Singh, 2022) [CEUR-WS](https://ceur-ws.org/Vol-3215/16.pdf) | CNN+ViT ensembles are more diverse than intra-family ensembles, but only if accuracy gap < ~5 pp. | Bounds when a diverse branch actually helps the ensemble mean. |

None of these isolate "geometry alone" vs. "objective alone" in a controlled experiment.

## 2. Mechanisms

### Different errors due to through-plane context (strongest evidence)

- AlzhiNet argues 2D captures fine in-plane spatial detail while 3D captures volumetric structure.
- Baskota et al. weight ensemble predictions by per-region disagreement between a 3D transformer and a CNN, implying the disagreement is on structurally difficult regions rather than noise.
- NSF-funded knee-OA work ([PAR 10280994](https://par.nsf.gov/servlets/purl/10280994)) notes 3D CNNs provide superior spatial feature extraction for MRI volumes.

### 3D undertraining (plausible but not specific to V-JEPA2)

This is a common concern for 3D medical models trained on small datasets. V-JEPA2, however, is trained on large video data, so its difference from 2D DINOv2 is more about **inductive bias** than capacity.

### Measurement artifact from correlated slices (unlikely)

`nearest` is computed on **study-level predictions**, not slice-level predictions. A 3D model’s slice-smoothed predictions would tend to reduce apparent distance, not increase it. The fact that `vjepa_warm` is the most distant member despite this smoothing suggests the geometry effect is real.

## 3. Second geometry point candidates

| candidate | geometry | domain match | licence / weights | assessment |
|---|---|---|---|---|
| **OrthoDiffusion** | 3D diffusion U-Net, multi-plane | **knee MSK MRI** | Code MIT; weights on HuggingFace; **weight licence unstated** | Best candidate. 8 diagnosis labels overlap 5 of the RSNA 12. Requires diffusion feature extraction and head replacement. |
| **MedicalNet** (Tencent) | 3D ResNet-10/18/50 | 23 medical datasets incl. MRI/CT | Code MIT; weights public | Lightweight 3D CNN; weaker but ungated and easy to load. |
| **MR-Transformer** | 3D ViT (DeiT-Ti) | knee MRI (TKR) | Code **CC BY-NC 4.0** | Non-commercial; not shippable. |
| **SAM-Med3D / MedSAM2 / VISTA3D** | 3D segmentation backbones | general medical | various | Segmentation-first; not drop-in classifiers. |
| **Video Swin / TimeSformer / X3D** | spatio-temporal tubes | natural video | various | Geometry but no medical pretraining; weaker domain match. |

**Recommendation:** try OrthoDiffusion first. If inference cost or licensing blocks it, fall back to MedicalNet.

## 4. Ensemble-level survival

The most relevant evidence is the project’s own: `vjepa_warm` seated at **-0.0007** with a confidence band including zero and negative values. This is consistent with the broader literature:

- NVIDIA safety-diversity: diversity helps robustness only when the added model is within ~5 pp of the incumbents.
- Published 2D+3D gains usually start from weaker baselines and/or use weighted/disagreement-aware fusion rather than simple averaging.
- In an already-optimized champion roster, one outlier is more likely to affect variance than the ensemble mean.

**Practical implication:** a single geometry-diverse member is not enough. The real test is whether a **2-geometry sub-ensemble** (e.g. DINOv2 + V-JEPA2 + OrthoDiffusion) beats the champion roster.

## References

- AlzhiNet: [arXiv:2410.02714](https://arxiv.org/abs/2410.02714)
- Thapar et al. COVID CT ensemble: [arXiv:2603.14621](https://arxiv.org/abs/2603.14621)
- Baskota et al. glioma disagreement ensemble: [Frontiers in AI 2026](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1812932/full)
- Filipiuk & Singh, NVIDIA safety-diversity: [CEUR-WS](https://ceur-ws.org/Vol-3215/16.pdf)
- OrthoDiffusion: [arXiv:2602.20752](https://arxiv.org/abs/2602.20752), [GitHub (MIT)](https://github.com/lt-0123/OrthoDiffusion), [HuggingFace weights](https://huggingface.co/lanstat0123/orthodiffusion)
- `research/deep_dives/2026-09-04_cnn-vs-vit-ensemble-decorrelation-medical-imaging.md`
- `research/deep_dives/2026-09-02_modern_3d_knee_mri_models_beyond_mrnet.md`
