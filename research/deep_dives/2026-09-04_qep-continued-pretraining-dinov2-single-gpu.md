# Q-EP — Continued pretraining (DAPT/TAPT) of a DINOv2 ViT-B/14 at single-GPU scale

2026-09-04

## Bottom line for Claude

- **Under ~12 GPU-hours of DINO-style continued pretraining, the published evidence of a downstream gain is thin.** The only method that clearly reports gains at that compute is DIET-CP, which uses a *different* objective (datum-index cross-entropy, not DINO self-distillation) and only evaluates on small 2D MedMNIST images, not volumetric knee MRI.
- **DINO self-distillation can run on a single GPU**, but the published single-GPU numbers are either (a) from-scratch ImageNet training with batch size 128, or (b) parameter-efficient continued pretraining at 80–100+ GPU-hours (ExPLoRA, BrainFound). There is no paper that reports a measured gain from <12 GPU-hours of continued DINO self-distillation on medical images.
- **A 2D-slice continuation corpus is enough to learn 3D anatomy** — BrainFound explicitly trains a DINOv2 ViT-L on sequential brain MRI slices and models volumetric structure. OrthoFoundation does the same for knee MRI/X-ray and reaches SOTA. Neither paper says you must reformulate the trunk as a 3D CNN; slice sequences are the accepted single-GPU path.
- **The practical choice is not DINOv2-vs-MAE objective but DINO-vs-DIET objective.** DINO self-distillation needs teacher EMA, multi-crop is helpful but not mandatory, and batch size 128 works with only a small drop. DIET-CP drops the teacher entirely and runs in minutes, but it is a different representation-learning objective whose medical gains may not transfer to a 12-label AUC metric.
- **Before spending 6–12 hours, run a 30-minute smoke test:** continue DINOv2 with a simplified single-GPU recipe (bs=64–128, no multi-crop or 2 global + 2 local crops, last 2 blocks unfrozen + LoRA, ~5k steps on 2D knee slices) and measure linear-probe AUC on a frozen downstream task. If the probe does not move within 30 minutes, the full run will not move enough to justify the card.

## Q-EP-1 — Effect size at small compute

### The low-compute end of the curve is sparsely measured

I found three relevant studies that report compute and effect size for continued pretraining of a natural-image ViT on a new domain. None use knee MRI; all are informative about the shape of the curve.

**DIET-CP — minutes of compute, gains on small 2D medical images.**

Rodas et al., "DIET-CP: Lightweight and Data Efficient Self Supervised Continued Pretraining" (2025) [[arXiv:2509.06990](https://arxiv.org/abs/2509.06990)] is the only paper that operates in the compute regime you care about. Their recipe:

- 150 epochs, batch size 32, AdamW lr=1e-4, 10% warmup + cosine decay.
- Freeze the backbone for the first 5% of epochs, then unfreeze the **last two transformer blocks** and train them with a new linear head.
- Objective: cross-entropy on the datum index (each training sample is its own class).
- "Very fast on a single GPU (<10 min. for ViT-B on an H100)."

On MedMNIST (small 2D medical images), DINOv2 ViT-B gains after this short continuation:

| Eval | Pre DIET-CP | Post DIET-CP | Gain |
|---|---:|---:|---:|
| k-NN F1 | 53.06 | 70.82 | **+17.77** |
| Linear probe F1 | 71.52 | 76.33 | **+4.81** |

The gain is real but the metric is F1 on small images, not AUC on knee MRI. The objective is also not DINO self-distillation.

**ExPLoRA — 80–100 GPU-hours, gains on satellite imagery.**

Khanna et al., "ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts" (ICML 2025) [[arXiv:2406.10973](https://arxiv.org/abs/2406.10973)] continues DINOv2 (or MAE) ViT-L on satellite imagery with parameter-efficient updates:

- Unfreeze 1–2 ViT blocks, tune the rest with LoRA (rank 64).
- D-ExPLoRA uses the DINOv2 objective; M-ExPLoRA uses MAE.
- Effective batch size 1024 via gradient accumulation on 1–4 NVIDIA RTX A4000 / A6000 GPUs.
- 200,000 iterations for DINOv2 (≈100 GPU-hours); MAE variant is slightly cheaper (80 GPU-hours).

Results on fMoW-RGB:

| Method | Pre-train GPUh | Fine-tune GPUh | Top-1 Acc. |
|---|---:|---:|---:|
| DINOv2 + LoRA-r8 (no continuation) | — | 220 | 78.08 |
| D-ExPLoRA (DINOv2 continued) | 100 | 220 | **79.28** |
| Full DINOv2 pretraining from scratch | 1200 | — | 54.29 (only) |

The continued-pretraining gain is **+1.2 percentage points** over direct LoRA fine-tuning, at a cost of 100 GPU-hours. Their Figure 7 explicitly ablates shorter runs: 24, 48, 72, 96 GPU-hours of continued pretraining. Even 24 GPU-hours already moves the fine-tuning accuracy above the no-continuation baseline, and the curve saturates around 100 GPU-hours.

**BrainFound — ~104 GPU-hours, brain MRI slices.**

Mazher et al., "Towards Generalisable Foundation Models for Brain MRI" (2025) [[arXiv:2510.23415](https://arxiv.org/abs/2510.23415)] is the closest domain match. They initialize a ViT-L with pretrained DINOv2 weights and continue self-supervised pretraining on ~10,000 brain MRI volumes (~1.5M slices):

- Process sequential 2D slices through the 2D DINOv2 model; no 3D convolutions.
- Batch size 200 slices per iteration, mixed precision.
- ~7,500 iterations/epoch × 62.5 min/epoch ≈ **104 GPU-hours for 100 epochs** on an NVIDIA A6000.
- Keep the full DINOv2 loss/optimization pipeline.

They outperform SL_ImageNet, SSL_DINOv2 (no continuation), and SSL_BrainImages (from scratch).

### What this means for the 12 GPU-hour threshold

- **No published DINO self-distillation result shows a gain at <12 GPU-hours on medical images.** DIET-CP is the only sub-12-hour method, and it changes the objective.
- **The ExPLoRA curve suggests DINO-style continuation starts to help somewhere between 24 and 100 GPU-hours.** Your 12-hour budget sits on the steep lower part of that curve; a gain is possible but not guaranteed, and it has not been isolated in print.
- **The best sanity check is a 30-minute linear-probe smoke test.** If continuing for a few thousand steps does not improve a frozen linear probe on a downstream task, 12 hours will not rescue it.

## Q-EP-2 — Which objective survives continuation on a DINO-initialised trunk?

### DINO self-distillation can be run single-GPU, but it is not the cheapest option

The original DINO paper (Caron et al., ICLR 2021) [[arXiv:2104.14294](https://arxiv.org/abs/2104.14294)] reports that DINO works at small batch sizes:

| Batch size | k-NN top-1 (100 epochs, no multi-crop) |
|---|---:|
| 128 | 57.9 |
| 256 | 59.1 |
| 512 | 59.6 |
| 1024 | 59.9 |

They explicitly note: "the experiment with batch size of 128 runs on only 1 GPU." Multi-crop improves results (+3.4% in linear eval) but is not mandatory: DINO without multi-crop still reaches 72.5% top-1 after 300 epochs on ImageNet. So a single-GPU DINO continuation recipe is feasible:

- batch size 64–128 (or effective 1024 via gradient accumulation, as ExPLoRA does),
- 2 global crops only, optionally +2 local crops if memory allows,
- teacher EMA maintained as usual,
- last 2 blocks unfrozen + LoRA on the rest (ExPLoRA/DIET-CP pattern).

The main cost is the teacher network copy and the multi-crop views. Removing multi-crop saves memory and time with a modest accuracy drop.

### MAE continuation from a DINO init is not well documented

I found no paper that initializes a DINOv2 trunk and continues pretraining under the MAE reconstruction objective. The concern is real: MAE reconstructs masked patches, while DINOv2 learns via self-distillation over augmented views. A DINO-init trunk may not have a decoder, and freezing/re-initializing the decoder while changing the objective could destroy the pretrained features before the new objective stabilizes.

ExPLoRA's M-ExPLoRA variant initializes from **MAE weights** and continues with MAE; it does not start from DINOv2 and switch to MAE. D-ExPLoRA starts from DINOv2 and continues with the **DINOv2 objective**. The safest objective match is therefore DINO self-distillation, not MAE.

### DIET-CP is the low-compute alternative objective

DIET-CP drops self-distillation entirely and uses datum-index classification. It does not need a teacher, multi-crop, or large batches. The trade-off is that it learns instance-discrimination features, which may cluster well (large k-NN gains) but are not guaranteed to improve a fine-tuned AUC on a 12-label task. It is the cheapest way to test whether *any* continued pretraining helps.

## Q-EP-3 — Is 2D-slice continuation enough, or does it have to be volumetric?

### The literature says 2D slice sequences are sufficient

**BrainFound** directly answers this: they model "full 3D brain anatomy by incorporating volumetric information from sequential MRI slices" using a **2D DINOv2 ViT-L** with no 3D convolutions. They initialize with pretrained DINOv2 weights and process slices sequentially, preserving spatial alignment across modalities. The paper does not reformulate the trunk as a 3D model; it relies on slice-level pretraining to capture 3D structure.

**OrthoFoundation** (Yu et al., 2026) [[arXiv:2601.18250](https://arxiv.org/abs/2601.18250)] goes further in the same direction for the exact anatomy you care about:

- Pretraining corpus: **~1.2 million unlabeled knee X-ray and MRI images** (OAI + fastMRI + internal PUTH data).
- Backbone: ViT initialized with **DINOv2 weights**, then DINOv3-L selected for continued pre-training.
- Encoder: **2D image encoder that processes individual MRI slices independently**.
- Result: SOTA across 14 downstream musculoskeletal tasks, including MRI structural-injury detection, and cross-anatomy transfer to hip/shoulder/ankle.

The authors explicitly note the limitation that their encoder is 2D and does not capture full 3D context — but the model still reaches SOTA. So 2D slice continuation is not a theoretical compromise; it is the current SOTA practice for knee-MRI foundation models at this scale.

**Krones' modality-matching finding** (Q-EO) should be read as "imaging modality" (MRI), not "full acquisition" (knee, 3D, same sequences). Krones' matched-modality gains came from chest X-ray → chest X-ray and NIH → NIH, not from 3D-specific matching. The mismatched cases were OCT → chest X-ray and CT → chest X-ray. The signal is MRI-ness, not 3D knee-ness.

### 3D DINO pretraining exists but is a different compute regime

Kats & Heinrich, "Effective Feature Learning for 3D Medical Registration via Domain-Specialized DINO Pretraining" (2026) [[arXiv:2603.14086](https://arxiv.org/abs/2603.14086)] train DINO directly on 3D volumes for registration:

- 150,000 steps, batch size 4, base lr 6e-4.
- Outperforms off-the-shelf DINOv2 on 3D registration.

This shows 3D DINO is possible, but batch size 4 and 150k steps is not a 12-hour single-GPU plan. For your constraint, 2D slice continuation is the realistic option.

## Practical implications for the project

1. **Do not assume 12 GPU-hours of DINO continuation will buy board points.** The evidence is below the threshold; the closest positive result (ExPLoRA) needs 24–100 GPU-hours.
2. **A 2D slice corpus is the right corpus shape.** Build it from the competition volumes + fastMRI knee + SKM-TEA, extracting slices with the same preprocessing as your downstream pipeline.
3. **The safest objective is simplified DINO self-distillation**, not MAE. Keep the teacher EMA, use bs=64–128 (or effective 1024 via grad accum), 2 global + optional 2 local crops, and unfreeze only the last 2 blocks + LoRA elsewhere.
4. **Run a 30-minute smoke test before committing 6–12 hours.** Continue for ~5k steps and measure linear-probe AUC on a frozen downstream head. If the probe AUC does not improve over the frozen DINOv2 baseline, stop.
5. **DIET-CP is a cheap fallback to test the *existence* of a signal.** If even DIET-CP does not improve a linear probe on knee MRI, then the corpus or the objective is wrong before you spend DINO-hours.
6. **OrthoFoundation's 1.2M knee images is the relevant scale reference.** Your combined corpus (competition + fastMRI + SKM-TEA) is likely much smaller. Expect diminishing returns if you are below a few hundred thousand slices.

## Sources

- Rodas et al., "DIET-CP: Lightweight and Data Efficient Self Supervised Continued Pretraining", arXiv:2509.06990, 2025 [[arXiv](https://arxiv.org/abs/2509.06990)]
- Khanna et al., "ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts", ICML 2025 [[arXiv:2406.10973](https://arxiv.org/abs/2406.10973)]
- Mazher et al., "Towards Generalisable Foundation Models for Brain MRI", arXiv:2510.23415, 2025 [[arXiv](https://arxiv.org/abs/2510.23415)]
- Yu et al., "A multimodal vision foundation model for generalizable knee pathology", arXiv:2601.18250, 2026 [[arXiv](https://arxiv.org/abs/2601.18250)]
- Kats & Heinrich, "Effective Feature Learning for 3D Medical Registration via Domain-Specialized DINO Pretraining", arXiv:2603.14086, 2026 [[arXiv](https://arxiv.org/abs/2603.14086)]
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICLR 2021 [[arXiv:2104.14294](https://arxiv.org/abs/2104.14294)]
- Krones, "From pre-training to downstream performance: Does domain-specific pre-training make sense?", arXiv:2605.08819, 2026 (Q-EO note)
