# Q-AD — Capacity ceiling repriced: does the chest-CT trunk-family gain survive at small scale?

Claude asked four questions after `v3large_dense` finished at **0.8258**, well below `base224` (0.8502) and `ortho_dense` (0.8452). The measurement says trunk capacity is locally negative, so the Q-I chest-CT precedent (+0.031 from a heterogeneous ensemble) needs to be re-examined.

## The precedent: Nilay et al. chest-CT COVID-19 ensemble

Nilay et al., "A Heterogeneous Ensemble for Multi-Center COVID-19 Classification from Chest CT Scans" (CVPRW 2026 / arXiv:2603.14621) is the +0.031 source:

- Dataset: PHAROS multi-source COVID-19 CT, 1,224 training / 308 validation scans across 4 centers.
- Best single model: **DINOv2 ViT-B/14**, macro F1 = **0.8969**.
- 9-model ensemble: DINOv2 + RadImageNet DenseNet-121 + 7 Gated Attention MIL models over EfficientNet-B3, ConvNeXt-Tiny, EfficientNetV2-S. Macro F1 = **0.9280**.
- Gain: **+0.031** over the best single model, with the best single model inside the pool.

## (1) Scale breakdown in the precedent

**The paper does not vary backbone size at fixed pretraining corpus.** It uses one model per family/size:

- DINOv2 ViT-B/14 (~86M params, self-supervised on natural images)
- RadImageNet DenseNet-121 (~8M params, supervised on radiology images)
- EfficientNet-B3 (~12M), ConvNeXt-Tiny (~28M), EfficientNetV2-S (~21M) in the MIL branch

So the +0.031 is a **mixed effect of architecture, pretraining corpus, and inference paradigm** (slice-level sigmoid averaging vs scan-level MIL), not a pure "add capacity" or "add corpus diversity" result. If the question is "does a bigger trunk of the same pretraining help?", the paper cannot answer it. That is the first honest limitation.

## (2) Corpus vs capacity, separated

There is work that holds architecture approximately fixed and varies pretraining corpus or scale:

- **Truong et al., "How Transferable are Self-supervised Features in Medical Image Classification Tasks?"** (arXiv:2108.10048, 2021) evaluates SimCLR/SwAV/DINO self-supervised ImageNet features against supervised ImageNet features on the **same architectures** for medical tasks (sentinel lymph node, diabetic retinopathy, chest X-ray). Self-supervised features beat supervised features, with gains up to 14.79% Kappa, 5.4–9.4% AUC in linear evaluation on small data subsets. This separates **pretraining objective** from capacity, but it does not separate corpus domain from objective.
- **Jitsev et al., "Effect of Pre-Training Scale on Intra- and Inter-Domain Full and Few-Shot Transfer Learning for Natural and Medical X-Ray Chest Images"** (arXiv:2106.00116, 2021) systematically varies **network size** and **source data scale/domain** (ImageNet-1k/21k vs large medical chest X-ray datasets) and transfers to medical targets. Key finding: for **inter-domain natural→medical transfer**, larger scale helps on **larger targets** in the full-shot regime, but not on **smaller targets** or in the few-shot regime ([arXiv](https://arxiv.org/abs/2106.00116)).

Neither paper is an exact match for "same size, different corpus" in knee MRI, but Jitsev et al. is the closest published shape. It suggests that at small target sizes, extra pretraining scale (and by extension extra trunk capacity) may not pay.

## (3) The small-trunk end

There is direct evidence that small, diverse CNN families can be close and that ensembles of them help:

- **MRISeqClassifier** (PMC12150705, 2025) compares AlexNet, GoogleNet, ResNet18, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny, EfficientNetV2-Small, MobileNetV3-Small, and VGG11 on MRI sequence classification. A **voting ensemble reaches 94.62% accuracy** vs the best single model at 93.17% (EfficientNetV2-Small). The gain is modest (+1.45 points) but real, and it comes from combining small models with different inductive biases.
- **Udomluck et al.** (Bioengineering 2026) found **VGG19 + logistic regression outperformed ConvNeXt-Tiny and DINOv2** on lumbar spinal stenosis MRI, suggesting older supervised CNN families can be the strongest single model on MRI.
- **DenseViT-OCT** (MDPI 2026) reports ResNet18, DenseNet121, and VGG all in the 92–93% range on retinal OCT severity classification, close to modern ConvNeXt/EfficientNet/MaxViT.

This supports two claims relevant to Q-AD:

1. **Small trunks are not automatically worse on medical images.** VGG11/ResNet18/EfficientNet-B0 are competitive.
2. **Family diversity can survive at small scale**, but the gain is modest (+1–2 points) rather than +3.

The concern that small trunks converge to similar representations is not strongly supported in the literature I can reach; the MRISeqClassifier voting ensemble and the lumbar-spine VGG19 result both show that different families make different errors.

## (4) Sample-size dependence

- **Raghu et al., "Transfusion: Understanding Transfer Learning for Medical Imaging"** (arXiv:1902.07208, 2019) is the landmark result. On CheXpert (~224k studies) and a retinal fundus dataset, they found that ImageNet pretraining provided **minimal performance benefit** over training from scratch, and that **lightweight models performed on par with large ImageNet-pretrained architectures**. The main effect of pretraining was faster convergence, not higher final accuracy.
- Jitsev et al. 2021 adds the scale dimension: larger pretraining scale only helps on **larger target datasets** for natural→medical transfer.

With **4,407 studies**, the RSNA knee-MRI corpus is much smaller than CheXpert and far smaller than the datasets in Jitsev et al. that benefited from large-scale pretraining. This places the project in the regime where extra trunk capacity is least likely to pay and where the gains from pretraining corpus are muted.

## Implications for the trunk-family plan

- The Q-I precedent's +0.031 cannot be treated as a capacity effect. It is an ensemble-diversity effect across different architectures, pretraining corpora, and inference paradigms.
- `v3large_dense` underperforming `ortho_dense` and `base224` is consistent with Raghu/Jitsev: at this corpus size, more capacity does not help.
- The small-trunk end is viable: ResNet18/VGG11/EfficientNet-B0 have published medical-transfer numbers and can form a diverse pool, but the expected ensemble gain is likely +1–2 points, not +3.
- The cheaper lever is not a bigger trunk but **more diverse families at the same small scale**, provided the family is not just a variant of the DINOv2/ConvNeXt already in the pool.

## Honest gaps

- I have not found a paper that fixes trunk size **exactly** and varies only pretraining corpus (ImageNet supervised vs RadImageNet vs self-supervised) in a medical imaging task.
- I have not found a knee-MRI study comparing family diversity at small scales.
- The +0.031 chest-CT gain may be partly due to multi-center domain shift, which is a different problem from the RSNA single-corpus setting.
