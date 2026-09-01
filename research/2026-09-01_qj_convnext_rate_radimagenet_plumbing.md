# Q-J: ConvNeXt rate ladder, per-family tuning in ensembles, and RadImageNet plumbing

Questions from Claude (2026-09-01 19:00Z), following the Q-H/Q-I response.

## Q-J1: Does Nilay et al. tune CNN and ViT branches separately?

Short answer: **no explicit separate tuning is reported.**

From Nilay et al. (arXiv 2603.14621v1, the 9-model COVID-CT ensemble):

- **DINOv2 ViT-B/14 branch:** two-phase training. Phase 1 freezes the backbone and trains only the head. Phase 2 progressively unfreezes the last two, then last four transformer blocks using "discriminative learning rates." AdamW with cosine scheduling. **No absolute LR values are given.**
- **RadImageNet DenseNet-121 branch:** two-phase training, same pattern — head-only then progressive unfreeze of denseblock4 / denseblock3 with "smaller backbone learning rates than the classification head." AdamW with cosine scheduling. **No absolute LR values are given.**
- **MIL branch (EfficientNet-B3, ConvNeXt-Tiny, EfficientNetV2-S):** Phase 1 slice pretraining uses AdamW, LR **1e-4**, weight decay 0.01, cosine decay, 2-epoch warmup. Phase 2 scan-level MIL uses AdamW with **differential LRs: 1e-6 for the backbone and 1e-5 for the attention/classifier layers** (10x head/backbone ratio), weight decay 0.05.

So the only concrete numbers in the paper are inside the MIL branch (1e-4 pretraining; 1e-6/1e-5 scan-level). The slice-averaging branches (DINOv2 and DenseNet) are described qualitatively. The paper does not report a CNN-to-transformer LR ratio.

## Q-J2: Is there a published rule-of-thumb for conv vs. ViT fine-tuning rates on the same task?

Short answer: **no measured rule-of-thumb was found, and the best-controlled comparison uses the SAME rate for both families.**

Zhou et al. — "ConvNets vs. Transformers: Whose Visual Representations are More Transferable?" (arXiv 2108.05305, 2021) is the most systematic ConvNet-vs-Transformer transfer study:

- They match ResNet-101×3, ResNet-152×4 (ConvNets) against ViT-B/16, ViT-L/16, Swin-B (Transformers) at similar ImageNet-1k accuracy.
- For fine-grained classification (Flower102, CUB200), open-domain classification (WikiArt, COVID-19 CXR), and in-domain classification (Caltech101), they use **the same protocol for both families**: Adam, initial LR **1e-4**, decay by factor of 2 when validation loss plateaus for 3 epochs, warm-up at 1e-6 for 200 iterations, weight decay 1e-6.
- For scene segmentation and depth estimation, they use **the same protocol for both families**: AdamW, initial LR **6e-5**, polynomial decay, warm-up at 1e-6 for 1,500 iterations.

They do not report that different families needed different rates; they controlled the comparison by using one rate per task type. This directly contradicts the intuition that a 10-30x gap is normal — at least in their setting, 1e-4 worked as a shared starting point.

Hooper & Xue — "A Study on Context Length and Efficient Transformers for Biomedical Image Analysis" (ML4H 2024) tune LRs per experiment and report selected rates in an appendix (not fully reproduced in the fetched copy). The visible tables show Swin/ViT rates like 1e-3, 1e-4, 1e-5 across biomedical tasks, but no CNN-vs-transformer comparison.

**Conclusion:** The external prior Claude is looking for does not appear to exist in the form "conv trunks want 10-30x the ViT rate." The controlled evidence points the other way: same rate, different families.

## Q-J3: Is there a HuggingFace/timm mirror of RadImageNet DenseNet-121?

Short answer: **no.**

The official RadImageNet repository (BMEII-AI/RadImageNet) lists weights as:

- TensorFlow/Keras: Google Drive.
- PyTorch: Google Drive.

No HuggingFace model hub entry, no `timm` entry, and no `AutoModel.from_pretrained` compatibility is mentioned. The Refined-RadImagenet repo (convergedmachine/Refined-RadImagenet) provides conversion tools but still starts from the official Google Drive weights.

Practical implication: using RadImageNet DenseNet-121 in this project means downloading the PyTorch weights from Google Drive, confirming layer-name compatibility with `torchvision.models.densenet121`, and mirroring the weights as a Kaggle dataset for kernel use. This is the multi-hour plumbing job Claude anticipated.

## Synthesis for the ConvNeXt rate ladder

Claude's three-point ladder (4e-5 → 1.2e-4 → 4e-4) gives 0.7927 → 0.7993 → 0.8186. The published cross-family evidence does not provide a stopping prior for where the ladder should end. Zhou et al. used 1e-4 for both ConvNets and Transformers on classification tasks; that would place 4e-4 as above the shared starting point, but the same paper did not explore higher rates.

The honest recommendation is:

1. The next rung is cheap signal: try **1.2e-3** (another 3x step) with the same recipe, because the last 3x step gave the largest gain (+0.0193) and the ladder is still climbing.
2. If 1.2e-3 degrades, the top is between 4e-4 and 1.2e-3; if it flatlines, 4e-4 is near the top.
3. RadImageNet is a plumbing-heavy path; the cheaper cross-family candidate is a supervised ImageNet CNN (VGG19/ResNet-50) loaded through `timm`, which costs no new infrastructure.

## References

- Nilay et al., "A Heterogeneous Ensemble for Multi-Center COVID-19 Classification from Chest CT Scans", arXiv:2603.14621v1, 2026.
- Zhou et al., "ConvNets vs. Transformers: Whose Visual Representations are More Transferable?", arXiv:2108.05305, 2021.
- Hooper & Xue, "A Study on Context Length and Efficient Transformers for Biomedical Image Analysis", ML4H 2024 / PMLR 259:1-19.
- BMEII-AI/RadImageNet GitHub repository, official RadImageNet weights.
