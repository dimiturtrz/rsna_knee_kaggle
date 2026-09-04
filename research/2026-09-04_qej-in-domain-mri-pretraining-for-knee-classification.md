# Q-EJ — is in-domain knee-MRI pretraining worth a card week?

Claude asked four questions about pretraining on unlabelled knee MRI
(fastMRI / SKM-TEA) for a downstream knee-MRI classification task.  The
core concern is whether the gain over an ImageNet / DINOv2 initialisation
is large enough to justify a multi-day card commitment.

## Directly relevant paper

The strongest direct evidence I found is:

**Zhang et al., *Self-Supervised Learning for Medical Image Data with
Anatomy-Oriented Imaging Planes*, Medical Image Analysis 2024.**
([arXiv:2403.16499](https://arxiv.org/abs/2403.16499))

They pretrain on **fastMRI knee** (~10,000 clinical exams, sagittal T2W
fat-suppressed) and evaluate downstream classification on the **Stanford
knee MRI dataset** (MRNet; 1,250 exams, sagittal T2W fat-suppressed).  The
downstream labels are the same three findings as MRNet: general
abnormality, ACL tear, and meniscal tear.  Reported metric is mean AUC.

This is almost exactly the experiment Claude is considering, except they
use an AlexNet backbone and 2D slices rather than a modern ViT.

## 1. Published gain over ImageNet initialization

Table 5 of Zhang et al. reports mean AUC for different pretraining
methods and different amounts of downstream labelled data:

| pretrain method | 80 subjects (10%) | 200 (25%) | 400 (50%) | 800 (100%) |
|-----------------|-------------------|-----------|-----------|------------|
| None (scratch)  | 0.433 | 0.429 | 0.474 | 0.562 |
| ImageNet        | 0.682 | 0.737 | 0.799 | **0.824** |
| Models Genesis  | 0.631 | 0.722 | 0.707 | 0.699 |
| APP             | 0.654 | 0.713 | 0.719 | 0.723 |
| Jigsaw puzzle   | 0.632 | 0.703 | 0.770 | 0.793 |
| Slice ordering  | 0.625 | 0.740 | 0.773 | 0.792 |
| MAE (in-domain) | 0.548 | 0.583 | 0.618 | 0.624 |
| BYOL (in-domain)| 0.577 | 0.641 | 0.690 | 0.753 |
| SimCLR (in-domain)| 0.698 | 0.761 | 0.774 | 0.800 |
| Rel. location   | 0.740 | 0.781 | 0.791 | 0.815 |
| Rel. orientation| 0.703 | 0.774 | 0.787 | 0.803 |
| **MTL (best in-domain)** | **0.755** | **0.783** | **0.801** | 0.819 |
| MAE† (ImageNet) | 0.421 | 0.462 | 0.519 | 0.536 |
| BYOL† (ImageNet)| 0.440 | 0.531 | 0.533 | 0.552 |
| SimCLR† (ImageNet)| 0.432 | 0.474 | 0.496 | 0.530 |

(† = directly fine-tuned from ImageNet pretrained models with different
backbones.)

**Gain of the best in-domain method over ImageNet:**

- 80 subjects: +0.073
- 200 subjects: +0.046
- 400 subjects: +0.002
- 800 subjects: **−0.005**

So the honest published number is **positive only in the low-data
regime**, and it **vanishes or reverses** once the downstream labelled set
reaches ~800 subjects.  Compared to training from scratch, in-domain SSL
is a large win (+0.257 at 800 subjects), but that is not the right
baseline — ImageNet/DINOv2 is.

## 2. Which objective, and is the answer sensitive to it?

**Yes, very sensitive.**  Generic SSL objectives do poorly on this task:

- In-domain MAE: 0.624 at 800 subjects (worse than ImageNet).
- In-domain BYOL: 0.753.
- In-domain SimCLR: 0.800.
- Anatomy-aware MTL (relative slice orientation + relative slice
  location): **0.819**.

The winning objective exploits the **geometry of anatomy-oriented imaging
planes** — it is not a generic pixel-reconstruction or contrastive task.
This matters for scoping: a plain "pretrain MAE on fastMRI knee slices"
arm would likely underperform the ImageNet/DINOv2 baseline, based on
Zhang's numbers.

## 3. How much unlabelled data before the gain appears?

This is the part that most directly gates the decision.

Zhang et al. already used **~10,000 fastMRI knee exams** for pretraining.
The downstream gain over ImageNet was:

- Large at 80 labelled subjects (+0.073).
- Modest at 200 (+0.046).
- Tiny at 400 (+0.002).
- Negative at 800 (−0.005).

Our labelled corpus is **~4,400 studies** — more than five times Zhang's
largest labelled split.  The curve says that at that labelled scale the
in-domain gain over ImageNet is **approximately zero**.

A 2024 survey of self-supervised pretraining in radiology (VanBerlo et
al.) reaches the same aggregate conclusion: gains are "most prominent
when unlabelled examples greatly outnumber labelled examples."  Here the
ratio is inverted — our labelled set is large, and the unlabelled set is
only ~1,500 fastMRI volumes / ~231 studies.

## 4. Does sequence/contrast mismatch matter?

Zhang et al. took care to match sequence and contrast: they selected only
**sagittal T2W fat-suppressed** images from fastMRI for pretraining
because that is the sequence in the downstream Stanford/MRNet task.  They
do not run an ablation with mismatched sequences, but their deliberate
matching implies they expected mismatch to hurt.

Other evidence:

- **PRISM** (Qiu et al., *Large-scale Multi-sequence Pretraining for
  Generalizable MRI Analysis*, arXiv 2025) was explicitly motivated by
  the fact that MRI sequence heterogeneity breaks naive pretraining.
  They handle it with 336,476 volumes across 34 datasets — three orders
  of magnitude more than our 1,500 fastMRI volumes.
- **"Using knee-trained Deep Neural Networks for Brain MRIs"** (PTB
  technical report) shows that even within MRI, knee-trained
  reconstruction networks suffer domain shift when evaluated on brain
  data, confirming that "MRI" is not a single domain.

So if the RSNA corpus contains a different sequence mix than fastMRI knee
(proton-density vs T2, different fat-sat parameters, etc.), the effective
in-domain pretraining data is smaller than the raw count.

## Bottom line

- **Published gain over ImageNet/DINOv2 at our labelled scale: ~0.**
- **Best objective is anatomy-aware, not generic MAE/SimMIM/DINO.**
- **Data-scaling curve says our large labelled corpus already captures
  what ImageNet misses; adding 1,500 in-domain unlabelled volumes is
  unlikely to help.**
- **Sequence/contrast mismatch is a real risk and would further shrink
  the effective pretraining set.**

On the evidence, a card week spent pretraining on fastMRI/SKM-TEA for
this classification task is not well-supported.  The honest move is to
price what is already in the pool, as Claude suggested.
