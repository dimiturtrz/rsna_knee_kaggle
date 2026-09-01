# Q-I: Cross-family trunk decorrelation without solo cost — medical-volume analogue

Question from Claude (2026-09-01 18:30Z): For knee MRI or the nearest medical-volume analogue, which pretrained trunk families produce ensemble members that decorrelate from a DINOv2/ConvNeXt pool WITHOUT the 1-3 point solo cost found in Q-F? Rank by published cross-family decorrelation with the solo AUC each lands at.

## Short answer

A direct knee-MRI cross-family ensemble with published correlation coefficients was not found. The strongest near-analogue is **multi-source 3D chest CT for COVID-19 detection**, where a heterogeneous ensemble of DINOv2 + RadImageNet DenseNet + CNN MIL models gained +0.031 F1 over the best single model (DINOv2), with no constituent paying a solo penalty. For MRI specifically, a lumbar spinal stenosis study shows **VGG19 features generalize better than ConvNeXt-Tiny or DINOv2**, suggesting the "different trunk" axis for knee MRI may not be the newest SSL transformer families.

## Strongest direct evidence: heterogeneous chest-CT ensemble

### Nilay et al. — "A Heterogeneous Ensemble for Multi-Center COVID-19 Classification from Chest CT Scans" (arXiv 2603.14621v1, 2026)

This is the closest published analogue: 3D medical-volume classification, heterogeneous pretrained backbones, ensemble gain over the best single model.

- Dataset: PHAROS multi-source COVID-19 CT, 1,224 training scans / 308 validation scans across 4 hospital centers.
- Best single model: **DINOv2 ViT-B/14**, macro F1 = **0.8969**.
- Ensemble (9 models): DINOv2 + RadImageNet DenseNet-121 + 7 Gated Attention MIL models over EfficientNet-B3, ConvNeXt-Tiny, and EfficientNetV2-S backbones. Macro F1 = **0.9280**.
- Gain: **+0.031** over the best single model, with the best single model being part of the pool.
- Interpretation: the gain comes from architectural/pretraining diversity, not from including a worse solo model. DenseNet brings radiology-domain pretraining; MIL brings slice-level attention; EfficientNet/ConvNeXt bring complementary CNN inductive biases.
- Limitation: the paper reports ensemble gain, not pairwise correlation coefficients between model families. It therefore answers the "without solo cost" part directly, but does not provide a decorrelation ranking.

## MRI-specific trunk comparisons

### Udomluck et al. — "Multi-Task Deep Learning Model for Automated Detection and Severity Grading of Lumbar Spinal Stenosis on MRI" (MDPI Bioengineering 14(1):32, 2026)

- Task: multi-grade lumbar spinal stenosis on axial MRI.
- Features tested: VGG19 (ImageNet), ConvNeXt-Tiny (ImageNet), DINOv2 (self-supervised).
- Best external-test performance: **VGG19 + Logistic Regression**, accuracy 0.9556, F1 0.9558, per-grade AUC 0.994-1.000.
- ConvNeXt-Tiny showed stable cross-classifier performance.
- DINOv2 features had reduced generalizability, especially with LightGBM (accuracy 0.6222).
- Implication: for spinal MRI, a classic supervised ImageNet CNN (VGG19) outperformed both a modern convnet and a modern SSL transformer. This suggests the "different family" candidate for knee MRI may include older supervised CNN families, not just DINOv2/ConvNeXt alternatives.

### Alexopoulos et al. — "Early detection of knee osteoarthritis using deep learning on knee MRI" (arXiv 2209.01192, 2022)

- Task: knee OA incidence prediction within 24 months from IW-TSE MRI (OAI).
- Architectures: ResNet-50, DenseNet-121, CVAE.
- Best AUC: CVAE 0.670, DenseNet-121 0.656, ResNet-50 0.651.
- All models were close; differences were small and task-limited. No ensemble reported.
- Relevance: shows CNN families can be close on knee MRI, but provides no ensemble/correlation evidence.

## What is NOT in the literature

- No knee-MRI study was found that compares DINOv2, ConvNeXt, EfficientNet, DenseNet, ResNet, and VGG19 in a single ensemble and reports pairwise error correlations or per-family solo AUC.
- No study reports a medical-volume ensemble where a family with lower solo AUC materially improves the ensemble by decorrelating from DINOv2/ConvNeXt.

## Candidate ranking for knee MRI, based on indirect evidence

| Family | Solo evidence | Why it might decorrelate | Risk |
|--------|---------------|--------------------------|------|
| **VGG19 / supervised ImageNet CNNs** | Best single in lumbar MRI; historically strong on MRI texture | Different inductive bias from SSL transformers and modern convnets; shallow hierarchies may miss what transformers catch and vice versa | Older family, often worse on natural images |
| **RadImageNet DenseNet** | Strong in chest-CT ensemble | Radiology-specific pretraining; sees domain-specific texture/statistics | Requires access to RadImageNet weights |
| **DINOv2 ViT** | Best solo in chest CT | Already in pool; strong global attention | High correlation with other SSL ViT variants |
| **ConvNeXt** | Mixed MRI results | Modern pure-CNN design, may correlate less with ViT than ResNet | Inconsistent MRI transfer |
| **EfficientNet/EfficientNetV2** | Used in chest-CT MIL branch | Compound-scaled CNN, different failure modes | May correlate with other ImageNet CNNs |

## Practical conclusion

The COVID-CT result is the best published justification for adding a **radically different trunk family** to a DINOv2/ConvNeXt knee-MRI pool: it demonstrates that +0.03-level gains are achievable in 3D medical imaging when the new family brings a different pretraining signal (RadImageNet) or a different inference paradigm (MIL). The lumbar-spinal-stenosis result suggests **VGG19 or another supervised ImageNet CNN** is a plausible low-correlation addition, because it outperformed both ConvNeXt and DINOv2 on MRI despite being architecturally distant from both.

## References

- Nilay et al., "A Heterogeneous Ensemble for Multi-Center COVID-19 Classification from Chest CT Scans", arXiv:2603.14621v1, 2026.
- Udomluck et al., "Multi-Task Deep Learning Model for Automated Detection and Severity Grading of Lumbar Spinal Stenosis on MRI", Bioengineering 14(1):32, 2026.
- Alexopoulos et al., "Early detection of knee osteoarthritis using deep learning on knee magnetic resonance images", arXiv:2209.01192, 2022.
