# Q-EM — which ONE non-DINO trunk, and does the family gain survive a weaker seat?

2026-09-04

## Bottom line

- **CoAtNet is the only candidate with a measured score on this exact task.** The public RSNA frontier reports a CoAtNet scoring 0.914 LB solo and 0.942 after tweaks; a CoAtNet/Swin/EffNetV2 three-family blend bought ~+0.001. For Swin, EfficientNetV2, and ConvNeXt the published evidence is on single findings, radiographs, or MRI plane classification — not on twelve-label knee MRI pathology at ~4,400 studies.
- **There is no head-to-head paper comparing these four trunks on multi-label knee MRI.** Anyone who picks one is extrapolating from a different task or a different modality.
- **A weaker-but-complementary member can help in principle**, and there is knee-MRI evidence that a weaker local feature extractor plus a stronger global model beats the stronger model alone. But there is **no published medical-imaging case** that isolates a weaker *separate* model improving a stronger homogeneous ensemble with solo numbers reported.
- **The only defensible floor is a marginal-contribution gate.** There is no universal AUC threshold. Seat the new trunk only if a block-CV test shows the ensemble improves by more than the noise floor (pre-register >+0.0005) and its errors are not just correlated copies of the existing ensemble.

## Candidate trunks

### CoAtNet — measured on this competition

The public RSNA Kaggle leaderboard is the only place where a non-DINO trunk has a number on the exact twelve-label task:

- `hyakumanben2025/rsna-knee-dinosaur-v5-repro` (the raptor public lineage) reports **CoAtNet alone at 0.914 LB**, and **0.942 after tweaks**, still at 224 px (`research/2026-08-31_public-frontier-sweep.md`).
- The same public notebook reports a three-family blend of **CoAtNet 384 / Swin-base 384 / EffNetV2-L 480** worth only **~+0.001 AUC** on the live board.
- A Kaggle dataset publishes trained CoAtNet weights for twelve knee-MRI findings: [`dreaddevelopment/raptor-knee-widedense`](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense).

Our own `raptor_v4_swa` is a CoAtNet and is already in the champion, but it is **98.6% in-sample** (`foreign_overlap`), so it cannot be priced offline. A CoAtNet trained fresh on our corpus would be the directly comparable seat.

### Swin Transformer — strong on knee MRI, but not on multi-label pathology

- Tanwar et al. (2025) built MV2SwinNet for knee MRI: **MobileNetV2 alone 96.85%**, **Swin Transformer alone 98.12%**, and the hybrid **99.94%** accuracy on MRNet plane-orientation classification [[PMC12385416](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385416/)]. This is the cleanest knee-MRI ablation showing that a weaker local model plus a stronger global model outperforms the stronger single model — but the task is plane classification, not pathology.
- Sezen et al. used a transformer-based model on MRNet for ACL tear / meniscus tear / general abnormality and reported an **average AUC of 0.905** [[PMC12385416](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385416/) Table 1]. It is not specified as Swin, and it is single-task, not multi-label.

### EfficientNetV2 — proven on meniscus tear, single-finding

- Güngör et al. (2025) used **EfficientNetV2-M** for meniscal tear classification on 642 knee MRI studies, achieving AUC **0.97 sagittal / 0.98 coronal** on cropped ROIs [[PMC11792105](https://pmc.ncbi.nlm.nih.gov/articles/PMC11792105/)].
- Chou et al. (2023) used **EfficientNet-B7** in a clinical decision-support system for meniscal injury, reaching AUC **0.984 sagittal / 0.972 coronal** [[PMC12385416](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385416/) Table 1].
- Both are single-finding, cropped, and much smaller datasets. There is no published EfficientNetV2 result on the twelve-label knee MRI task.

### ConvNeXt — radiographs and localization, not MRI multi-label

- KL-FuseNet (Alkhatatbeh et al. 2026) fuses **ConvNeXt-Base** with ResNet-50 for **Kellgren-Lawrence grading on knee radiographs** (X-ray), not MRI [[PMC12988769](https://pmc.ncbi.nlm.nih.gov/articles/PMC12988769/)].
- Wang et al. (2023) integrated ConvNeXt into a YOLOv5 C3 module for meniscus tear **localization**, improving mAP from 82.5% to 84.8% [[PMC12385416](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385416/)].
- No multi-label knee MRI classification result.

## Does the ensemble gain survive when the heterogeneous member is weaker?

### Theoretical answer: yes, if complementarity is high enough

Zhou & Feng, in the Deep Forest paper, state the standard ensemble-learning result directly:

> "Combining only accurate learners is often inferior to combining some accurate learners with some relatively weaker ones, because the complementarity is more important than pure accuracy." [[Zhou & Feng, Deep Forest, arXiv:1702.08835](https://ar5iv.org/html/1702.08835)]

This is the theoretical reason a weaker conv member *could* help a stronger DINO ensemble.

### Knee-MRI evidence that complementarity can beat the stronger single

Tanwar et al.'s ablation is the closest measured case on knee MRI:

| Model | Accuracy |
|-------|---------:|
| MobileNetV2 (weaker, local) | 96.85% |
| Swin Transformer (stronger, global) | 98.12% |
| MobileNetV2 + Swin hybrid | **99.94%** |

The hybrid is **1.82 points better** than the stronger single model. The weaker member supplied local detail that the stronger global model missed. The caveat: this is a single hybrid network, not an ensemble of two independently trained models, and the task is MRI plane classification rather than pathology classification.

### No published medical-imaging case isolates a weaker *separate* model helping a stronger ensemble

The public RSNA three-family blend (CoAtNet/Swin/EffNetV2) is flat, but its members are all strong; it does not test the "weaker member" scenario. MURA's calibrated ensemble explicitly **excluded** the weakest backbone (ConvNet AUC 0.88) because adding it did not help [[Hussain et al., Sci Rep 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8079683/)].

So the honest answer to question 2 is: **it is possible, but not guaranteed, and the published literature does not give a medical-imaging example with the solo numbers you want.** You would be running the experiment that produces that evidence.

## Is there a rule of thumb for how weak is too weak?

**No universal AUC floor exists.** The standard method is Caruana et al.'s *ensemble selection*: greedily add the model whose inclusion most improves the ensemble on a held-out metric, and stop when no addition helps [[Caruana et al., ICML 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)]. This is exactly how AutoGluon, H2O AutoML, and auto-sklearn build their final blends; the gate is **marginal cross-validated contribution**, not a fixed accuracy threshold.

A practical pre-registered gate for the arm:

1. Train the candidate non-DINO member with the same train/val split and holdout as the existing DINO members.
2. Compute a **block-5 or block-10 cross-validated marginal contribution**: `AUC(ensemble + candidate) − AUC(ensemble)`.
3. Seat only if the marginal gain is **> +0.0005** on the holdout (half the current board gap).
4. Also require **per-label error correlation with the current ensemble < 0.95**; otherwise the candidate is a correlated copy, not a diversity source.
5. If the candidate's solo AUC is **> 0.03 below the best member**, demand a per-label breakdown showing it fixes a blind spot on a focal finding (Lateral Meniscus, Fracture, Lateral OA, etc.).

That is not a literature-derived threshold; it is a conservative operational rule that treats the candidate as guilty until proven innocent.

## What I would do

- Run **one** non-DINO arm, not four.
- Make it **CoAtNet**, because it is the only trunk with a board-measured score on the exact task and because the public frontier already calibrated its solo strength.
- Pre-register the marginal-contribution gate above before the arm starts.
- If CoAtNet fails the gate, stop. The other three have even less direct evidence.

## Sources

- CoAtNet public frontier / raptor lineage — internal: `research/2026-08-31_public-frontier-sweep.md`
- Public trained CoAtNet weights — [Kaggle dataset `dreaddevelopment/raptor-knee-widedense`](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)
- MV2SwinNet knee MRI ablation — [Tanwar et al., PLOS ONE 2025 / PMC12385416](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385416/)
- EfficientNetV2 meniscus tear — [Güngör et al., Knee Surgery Sports Traumatology Arthroscopy 2025 / PMC11792105](https://pmc.ncbi.nlm.nih.gov/articles/PMC11792105/)
- KL-FuseNet ConvNeXt on knee radiographs — [Alkhatatbeh et al., Ann N Y Acad Sci 2026 / PMC12988769](https://pmc.ncbi.nlm.nih.gov/articles/PMC12988769/)
- Deep Forest complementarity quote — [Zhou & Feng, arXiv:1702.08835](https://ar5iv.org/html/1702.08835)
- Ensemble selection gate — [Caruana et al., ICML 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
