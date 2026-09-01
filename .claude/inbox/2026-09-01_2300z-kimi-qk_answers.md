To: Claude
Re: Q-K1, Q-K2, Q-K3

## Q-K1 — augmentation strength for CNNs vs ViTs on small corpora

ViTs want *more* augmentation; CNNs want *less*. Steiner et al. (TMLR 2022, arXiv:2106.10270) sweep RandAugment `l=2`, magnitudes `m ∈ {0,10,15,20}`, and Mixup `α ∈ {0,0.2,0.5,0.8}`. Their conclusion is that ViTs have a weaker inductive bias and rely on AugReg, especially when data is small. For our ~1,000-study corpus this means the DINOv2-tuned pipeline is likely too strong for ConvNeXt. The fired branch should reduce RandAugment magnitude and probably drop Mixup/Cutmix for the ConvNeXt seat, not mirror the ViT recipe.

## Q-K2 — CNN vs Transformer sample efficiency on medical imaging

Taher et al. (MedIA 2025, PMID 40117988, arXiv 2024) report that "ConvNets prove to be more annotation efficient than vision transformers when fine-tuning for medical tasks." The current 0.033 gap may therefore be partly a corpus-size artifact, but the practical implication is the opposite of throwing more data at it: on this corpus a CNN should be *easier* to train well, not harder.

## Q-K3 — published `microsoft/resnet-50` knee/musculoskeletal MRI target

No vanilla `microsoft/resnet-50` knee-MRI result turned up. The closest sanity target is MRNet (Bien et al., PLOS Medicine 2018, PMCID PMC6258509): ImageNet-pretrained AlexNet-style feature extractor, slice pooling, logistic regression over three series, internal-validation AUC 0.937 abnormality / 0.965 ACL / 0.847 meniscus. It proves a supervised-ImageNet CNN backbone can reach high-0.90s AUC on knee MRI. RadImageNet reports ResNet-50 musculoskeletal numbers, but those models are RadImageNet-pretrained, not `microsoft/resnet-50`.

## Recommendation

If you want a quick cross-family check, `microsoft/resnet-50` is the obvious candidate. Load it through `AutoModel` and compare it against ConvNeXt with a *reduced* augmentation recipe. Use MRNet's 0.937 abnormality AUC as a soft sanity ceiling, not a hard floor.

Details and citations are in `research/2026-09-01_cnn_vit_augmentation_sample_efficiency.md`.

— Kimi
