# Q-BM — gentle fine-tuning for a small supervised CNN trunk under weak supervision

## Sources

- Davila et al., *Comparison of fine-tuning strategies for transfer learning in medical image classification*, arXiv:2406.10050 (2024).
- Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, arXiv:2202.10054 (2022).
- Mei et al., *RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning*, PMC9530758 (2022).

## The collapse is almost certainly LR, not trunk capacity

RadImageNet's own transfer experiments on knee MRI (ACL/meniscus) used a DenseNet-121 trunk and report AUC 0.96–0.97 with full unfreeze at LR **1e-4**. The frozen ridge probe on our data already reads **0.8197**, above the incumbent. A `--backbone-lr 4e-4` is 4× higher than the LR RadImageNet itself used on a clean, small knee task, and our labels are report-derived and noisier. The 0.6716 score has the signature of catastrophic feature distortion (Kumar et al.) rather than an empty trunk.

## Backbone LR rule for supervised CNN trunks

Davila et al. sweep LRs {1e-3, 1e-4, 1e-5} for FT/LP-FT/unfreeze on ResNet-50/DenseNet-121/VGG-19 across medical modalities. For DenseNet-121 the best LR is often at the low end; under weak/noisy supervision the safe starting point is **1e-5**, with **1e-6** as a conservative floor. The head can stay at the usual **1e-3–1e-4**, giving a **1:10 to 1:100 backbone:head ratio**. The 4e-4 ratio is inherited from DINOv2-style self-supervised ViTs and is not appropriate for a supervised RadImageNet CNN.

## Mechanisms: partial unfreeze, LLRD, LP-FT

- **Partial unfreeze / gradual unfreeze**: Davila et al. find gradual unfreeze helps, but the gain over LP-FT is smaller and less consistent. For DenseNet-121, unfreezing the last dense block (or last two) is a low-risk baseline.
- **Layer-wise LR decay (LLRD)**: Not directly tested in the above medical papers; evidence is thinner. Use as a later refinement, not the first experiment.
- **LP-FT (linear probe then fine-tune)**: Kumar et al. show LP-FT changes pretrained features ~20× less than direct FT and outperforms both LP and FT in-distribution and out-of-distribution. Davila et al. replicate this on medical trunks, with LP-FT improving over FT in >50% of scenarios and giving the most consistent gains for DenseNet-121.

## Recommended next arm

Run **LP-FT**:

1. Use the existing ridge head weights as the initialized head (the "LP" is already done).
2. Unfreeze the whole DenseNet-121 trunk.
3. Backbone LR **1e-5** (or **1e-6**), head LR **1e-4**, with early stopping (patience 3–5 epochs).
4. Keep the same geometry that gave the frozen probe its best reading (336px, 12 slices, band 0.1–0.9, pool `cls_mean_focal`, slice read `max`).

This is the cheapest experiment that directly tests whether the trunk can be nudged without destroying the features that already score 0.8197.

## Diagnostic for feature distortion

Without running a sweep, monitor:

- **Feature drift**: cosine distance or CKA between trunk outputs at epoch 0 and the current epoch. Rapid growth means the LR is too high.
- **Probe-on-current-features**: re-evaluate the ridge probe on features at epochs 1, 2, 3. If probe AUC drops while fine-tuned AUC rises, the trunk is forgetting useful representations.
- **Training loss**: if loss explodes or validation AUC drops in the first 2–3 epochs, stop and lower LR.

## Limitations / what is thin

- Published medical LR sweeps rarely go below 1e-5; the 1e-6 floor is extrapolated from noisy-label settings.
- LLRD evidence in this specific literature is sparse.
- The exact DenseNet-121 block count to unfreeze is not settled; "last dense block" is a conventional heuristic.
