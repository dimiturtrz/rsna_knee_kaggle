# Diffusion feature extraction for knee-MRI classification — how to smoke-test OrthoDiffusion

2026-09-02, Kimi. This note collects the practical recipe for turning a pretrained 3D diffusion U-Net into a classifier backbone, applied to OrthoDiffusion for the RSNA 12-label task.

## Why this matters

OrthoDiffusion ships as a generative diffusion model, not a classifier. To use it for our task we must:
1. Load the three orientation-specific 3D U-Net denoisers.
2. Run a noisy volume forward and extract intermediate features.
3. Pool / fuse / classify into our 12 labels.

The literature on diffusion discriminative features is now strong enough to give us a starting recipe and avoid a blind hyperparameter sweep.

## What the OrthoDiffusion paper does

From the README and paper:
- Stage I trains a **single-layer linear classifier** on pooled per-plane features, freezing the diffusion backbone (`linear_classifier.py`, `--classes 8`).
- Stage II fuses the three planes with `linear_fusion.py`:
  ```bash
  --timestep 200,150,50
  --blockname mid_0,mid_0,mid_2
  --fusion simple_concat
  ```
- Pooling is **SAP (spatial average pooling)** by default.

The paper reports macro AUROC 0.908 on 8 knee abnormalities, so this recipe is already calibrated for knee MRI.

## General evidence from 3D MRI diffusion feature extraction

Li et al. (2025), *Are Diffusion Models Effective Good Feature Extractors...* ([PMC12772265](https://pmc.ncbi.nlm.nih.gov/articles/PMC12772265/)), pretrain a DDPM on 59,830 UK Biobank brain T1WIs and test on BraTS2020 and ADNI1 classification. Their findings are the best published guidance for how to extract discriminative features from a 3D diffusion U-Net:

- **Best blocks are middle U-Net blocks**, not the deepest bottleneck or the early encoder. In their architecture, blocks b={19,20,21} gave the strongest linear-probe performance. This matches the 2D diffusion literature and suggests OrthoDiffusion's `mid_*` blocks are the right place to read.
- **Timestep matters and is task-dependent.** Performance peaks at intermediate noise levels. In their experiments the peak was t≈40 for BraTS2020 and t≈110 for ADNI1, while very low timesteps (clean images) were worse. OrthoDiffusion's fusion recipe uses {200,150,50}; for classification we should also try lower timesteps (e.g., {50,100,150,200}) and verify on a validation split.
- **A small learned fusion module helps.** They propose CATS (Channel Attention along Timestep + Channel Attention along Space), with only 1.7M trainable parameters, and it beats both frozen-feature linear probes and from-scratch 3D CNNs on ADNI1. The practical message: after extracting multi-block/multi-timestep features, a lightweight attention-based aggregator is worth the small parameter cost.
- **Spatial pooling:** log-sum-exp (LSE) pooling over spatial dimensions outperformed mean and max pooling in their ablation.
- **Fine-tune the diffusion backbone on target data** when possible. Their DDPM was first pretrained on UK Biobank and then fine-tuned on the smaller target dataset (BraTS2020 / ADNI1) before feature extraction. For OrthoDiffusion this translates to at least unfreezing the later blocks / adapters during our 12-label training (LP-FT), not keeping the whole U-Net frozen.

## Practical smoke-test recipe for RSNA

### 1. Load the weights
- Source: HuggingFace `lanstat0123/orthodiffusion` (preferred; code is MIT; Kaggle mirror lists "unknown" license).
- Expected structure: three 3D U-Net denoisers, one per plane (sagittal, coronal, axial).

### 2. Preprocess RSNA studies to OrthoDiffusion input
- Paper uses 16 central slices at 256×256 per plane.
- Our pipeline is slice-oriented; we need to stack the selected slices into a 3D tensor per plane. Resample / crop / pad to 256×256×16.
- Intensity normalization should match the model's training (likely z-score or percentile window; inspect training code).

### 3. Extract features
For each plane:
- Add noise at timestep `t` (try a small grid: {50, 100, 150, 200}).
- Run the noisy volume through the 3D U-Net.
- Extract features from block(s) `mid_0`, `mid_2`, and possibly `mid_1`.
- Pool spatially (start with SAP, then try LSE).
- Optional: extract from multiple blocks/timesteps and fuse with a small MLP/SE-style module.

### 4. Fuse planes and classify
- Concatenate or cross-attend the three plane features.
- Train a 12-label linear or small MLP head with BCE.
- Use masked loss only for labels present in each batch.

### 5. Training strategy
- **LP-FT:** freeze U-Net, train head for a few epochs, then unfreeze adapters/last blocks with 10-100× lower LR.
- **Anti-forgetting:** L2-SP or LoRA to stay near the pretrained weights.
- **Validation:** hold out a small RSNA validation set and an external-only split if we add MRNet/fastMRI+.

## What could go wrong

- **Memory:** 3D U-Net forward at 256×256×16 is heavy. We may need to downsample to 128×128×16 or use gradient checkpointing for the smoke test.
- **Plane mismatch:** RSNA may not have all three planes for every study. The fusion head needs to handle missing planes (drop plane, use zero feature, or train plane-dropout).
- **Label mismatch:** only 5 of OrthoDiffusion's 8 labels map to RSNA. The head must be replaced; do not try to reuse the released 8-class head.
- **Timestep sensitivity:** the best timestep for our 12 labels may differ from the paper's 8. Budget a small validation sweep.

## Bottom line

Use OrthoDiffusion as a **knee-specific 3D backbone**, extract middle-block features at intermediate timesteps, pool with SAP/LSE, fuse planes, and add our own 12-label head. The published recipe (timesteps {200,150,50}, blocks `mid_0`/`mid_2`, SAP, simple concat) is the starting point; the brain-MRI diffusion literature says lower timesteps, middle blocks, and a learned fusion module are likely to improve it.

## Sources

- Lan et al., 2026, *OrthoDiffusion* ([arXiv:2602.20752](https://arxiv.org/abs/2602.20752), [GitHub](https://github.com/lt-0123/OrthoDiffusion)).
- Li et al., 2025, *Are Diffusion Models Effective Good Feature Extractors for Global Discriminative Tasks in 3D MRI?* ([PMC12772265](https://pmc.ncbi.nlm.nih.gov/articles/PMC12772265/)).
