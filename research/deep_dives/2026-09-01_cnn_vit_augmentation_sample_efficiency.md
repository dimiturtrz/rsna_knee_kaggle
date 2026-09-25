# CNN vs ViT augmentation and sample efficiency for small medical imaging corpora

Context: Claude asked three things about whether the fired branch's augmentation clause should treat ConvNeXt like DINOv2, whether corpus size explains the ConvNeXt-vs-seat gap, and whether `microsoft/resnet-50` has a published knee-MRI sanity target.

## Q-K1: Do CNNs want more or less augmentation than ViTs on ~1,000 medical studies?

Steiner et al., *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers* (TMLR 2022, arXiv:2106.10270), is the cleanest source. They sweep RandAugment layers `l=2`, magnitudes `m ∈ {0,10,15,20}`, and Mixup alpha `α ∈ {0,0.2,0.5,0.8}` over ImageNet-21k pre-trained ViTs fine-tuned on ImageNet-1k and smaller datasets.

Key claim:

> "Vision Transformers have a weaker inductive bias than ConvNets ... and thus rely more on data augmentation and regularization."

For medium datasets, almost every AugReg combination helps. For small datasets, the right AugReg can match a 10× data increase. The implication for us is that a DINOv2-tuned augmentation pipeline is likely *too strong* for ConvNeXt. The branch should reduce RandAugment magnitude / drop Mixup-Cutmix rather than mirror the ViT recipe.

## Q-K2: CNN vs Transformer sample efficiency on medical imaging

Taher et al., *Large-scale benchmarking and boosting transfer learning for medical image analysis* (MedIA 2025, PMID 40117988, arXiv 2024), directly compares ConvNets and ViTs fine-tuned on medical tasks.

Key claim:

> "ConvNets prove to be more annotation efficient than vision transformers when fine-tuning for medical tasks."

Our ~1,000-study corpus sits in exactly the regime where this matters. The 0.033 gap between ConvNeXt and the seat bar is therefore plausibly partly a corpus-size artifact, not only an architecture deficiency.

## Q-K3: Published `microsoft/resnet-50` or supervised-ImageNet CNN knee/musculoskeletal MRI target

Bien et al., *MRNet: Deep-learning-assisted diagnosis for knee magnetic resonance imaging abnormalities* (PLOS Medicine 2018, PMCID PMC6258509), is the canonical knee-MRI CNN result:

- Abnormality AUC 0.937
- ACL tear AUC 0.965
- Meniscal tear AUC 0.847

Internal validation: 120 exams from 1,370 Stanford knee MRIs. MRNet uses an ImageNet-pretrained AlexNet-style feature extractor + slice pooling + logistic regression over three series. It is not ResNet-50, but it establishes that a supervised-ImageNet CNN backbone can reach high-0.90s AUC on knee MRI.

I have not found a published *vanilla* `microsoft/resnet-50` knee-MRI number. RadImageNet reports ResNet-50-based musculoskeletal results, but those models are RadImageNet-pretrained, not supervised-ImageNet. The cheap next cross-family candidate is therefore `microsoft/resnet-50`, with MRNet as the nearest comparable sanity target.

## Take-aways

1. Treat augmentation as family-specific: ConvNeXt probably wants *less* augmentation than the DINOv2/DINOv3 members, not more.
2. The small corpus (~1,000 studies) favors CNN sample efficiency; the current gap may shrink with more data, but that is not the fastest experiment.
3. If the project wants a ResNet-50 knee-MRI sanity target, MRNet is the best published anchor; no vanilla ResNet-50 knee-MRI result is known to this search.
