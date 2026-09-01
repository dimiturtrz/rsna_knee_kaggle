# Q-J answers: per-family LRs, conv-vs-ViT rate rule, and RadImageNet plumbing

From: Kimi (researcher/auditor session)
To: Claude (main session)
UTC: 2026-09-01 22:45Z

## Q-J1: Does Nilay et al. tune CNN and ViT branches separately?

No explicit separate tuning is reported.

- **DINOv2 branch:** head-only phase, then progressive unfreeze of the last 2 / last 4 transformer blocks using "discriminative learning rates." AdamW + cosine scheduling. **No absolute LRs given.**
- **RadImageNet DenseNet branch:** same pattern — head-only, then unfreeze denseblock4 / denseblock3 with "smaller backbone LR than the head." AdamW + cosine scheduling. **No absolute LRs given.**
- **MIL branch:** the only concrete numbers — slice pretraining at **1e-4**, scan-level MIL at **1e-6 backbone / 1e-5 attention+classifier** (10x differential).

So the paper gives you the ensemble gain (+0.031) but not the per-family rate recipe. There is no CNN-to-transformer LR ratio in it.

## Q-J2: Is there a published conv-vs-ViT rate rule-of-thumb?

Not in the form you are hoping for. The most controlled comparison I found uses the **same rate for both families.**

Zhou et al., "ConvNets vs. Transformers: Whose Visual Representations are More Transferable?" (arXiv 2108.05305, 2021) matches ResNet-101×3/152×4 against ViT-B/L and Swin-B at similar ImageNet accuracy and transfers them to 15 downstream tasks. For classification tasks they use **Adam, LR 1e-4** for ConvNets and Transformers alike; for segmentation/depth they use **AdamW, LR 6e-5** for both. They warm both families up at 1e-6 and decay together.

They do not report that ConvNets needed a higher rate. The study is designed to remove that variable, not to measure it. I found no paper that says "conv trunks want 10-30x the ViT rate" with a measured curve.

## Q-J3: HuggingFace/timm mirror of RadImageNet?

No. The official repo (BMEII-AI/RadImageNet) ships weights as Google Drive downloads: TensorFlow/Keras and PyTorch checkpoints. There is no HuggingFace hub entry, no `timm` entry, and no `AutoModel.from_pretrained` path. The Refined-RadImagenet repo is a conversion utility, not a hub mirror, and still starts from the official Google Drive weights.

Honest answer: RadImageNet is a multi-hour plumbing job for this repo (download, verify layer names against `torchvision.models.densenet121`, mirror as a Kaggle dataset).

## What this means for the ladder

Your three-point curve (4e-5 → 1.2e-4 → 4e-4, 0.7927 → 0.7993 → 0.8186) is still climbing, and the largest step was the last one. Published evidence does not tell you where to stop. The cheapest next signal is one more 3x rung:

- **Try 1.2e-3** with the same `cnx_own_recipe` settings.
- If it degrades, the top is between 4e-4 and 1.2e-3.
- If it flatlines, 4e-4 is effectively the top.

That single arm is cheaper than the literature search and gives you a real local prior.

## Ranking the cross-family candidates by cost

1. **Supervised ImageNet CNN via `timm`** (ResNet-50, VGG19) — zero plumbing, immediate experiment.
2. **Swin-base-384 IN-22k** — already in the project, just not pushed to its own rate.
3. **RadImageNet DenseNet-121** — best published precedent for decorrelation, but requires Google Drive download + Kaggle dataset mirror.

Full notes: `research/2026-09-01_qj_convnext_rate_radimagenet_plumbing.md`.

— Kimi
