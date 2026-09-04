# CNN vs ViT ensemble error decorrelation in medical imaging (Q-DI)

Question from Claude: is there a published, measured relationship between architecture family and ensemble error decorrelation in medical imaging — specifically, does adding a CNN branch to a transformer pool produce measurable decorrelation?

## General-vision evidence

The strongest direct measurement is *Exploring Diversity in Neural Architectures for Safety* (NVIDIA, 2022) [1]. On ImageNet-C, they compute error consistency and diversity metrics for CNN, ViT, and MLP-Mixer ensembles. Key finding: **CNN+ViT ensembles are more diverse than intra-architecture ensembles**, and that diversity improves robustness — e.g. ConvNeXt-B + ViT-B outperforms ConvNeXt-B + ConvNeXt-XL even though ViT-B is less accurate. High diversity does **not** help, however, when one model is much weaker (>~5 pp accuracy gap).

## Medical-imaging evidence

| paper | task | ensemble | single best | gain | CNN ablation? |
|-------|------|----------|-------------|------|---------------|
| Thapar et al. 2026 [2] | COVID-19 CT | DINOv2 ViT + RadImageNet DenseNet-121 + 7 CNN MIL | F1=0.8969 | F1=0.9280 (+0.031) | No |
| Ashraf et al. 2024 [3] | ChestX-ray14 | CNN + ViT + hybrid | AUROC=0.842 | AUROC=0.854 (+0.012) | No |
| Brain-stroke CT [4] | binary CT | scratch CNN + scratch ViT fusion | acc=99.31% | acc=99.77% (+0.46pp) | Yes |

No medical-imaging paper found reports per-case error correlation (Q-statistic, kappa, or disagreement rate) between CNN and ViT trained on the same target, nor a leave-one-branch-out ablation isolating the CNN contribution.

## Practical rule of thumb

Cross-architecture diversity pays when the candidate branch is within a few percentage points of the incumbents. A branch >5 pp weaker tends to add noise rather than value, even if its errors look different.

## Recommendation

Run RadImageNet DenseNet-121 as an instrument, then measure its solo AUC and its per-case error correlation with the ViT pool before deciding whether to queue VGG-16 / ResNet-50.

## Sources

[1] Filipiuk & Singh, *Exploring Diversity in Neural Architectures for Safety*. IJCAI-ECAI AISafety 2022. https://ceur-ws.org/Vol-3215/16.pdf  
[2] Thapar et al., *A Heterogeneous Ensemble for Multi-Center COVID-19 Classification from Chest CT Scans*. CVPRW 2026. https://arxiv.org/abs/2603.14621  
[3] Ashraf et al., *SynthEnsemble: A Fusion of CNN, Vision Transformer, and Hybrid Models for Multi-Label Chest X-Ray Classification*. arXiv:2311.07750. https://arxiv.org/abs/2311.07750  
[4] Deep ensemble learning with convolutional and transformer models for brain stroke classification. *Scientific Reports* 2026. https://link.springer.com/article/10.1007/s44163-026-01077-7
