# Continued SSL pretraining on an in-domain medical corpus — compute vs. gain (Q-DH)

Question from Claude: for ~30k unlabeled knee MRI studies, what does continued self-supervised pretraining of an already-pretrained ViT actually buy, per unit of compute?

## Measured evidence

The only direct knee-MRI evidence is an ISMRM 2025 abstract [1]: DINO pretraining on OAI 3D knee MRI significantly outperforms training from scratch and improves classification over supervised pretraining. No corpus-size ablation or compute numbers are given.

The strongest quantitative curve is from a 2025 Microsoft/Mayo paper on chest X-ray [2]. They continually pretrain MI2 (CLIP/UniCL) and RAD-DINO (DINOv2) on nested subsets of a single-institution CXR dataset:

| subset | samples | wall time (32× H100) |
|--------|---------|----------------------|
| 30k  | 30k pairs  | 0.33 h |
| 50k  | 50k pairs  | 0.58 h |
| 100k | 100k pairs | 1.17 h |
| 1M   | 1M pairs   | 11.67 h |
| 3.5M | 3.5M pairs | 40.83 h |

Downstream results:
- Findings classification: MI2 beats open-weights at 30k; RAD-DINO needs 100k.
- Lines/tubes classification: both beat open-weights around 100k.
- Segmentation: inconsistent below 100k; RAD-DINO scales best above 100k.
- Report generation: need ≥1M to beat open-weights.

## Objective matching

The paper keeps objectives matched (CLIP→CLIP, DINOv2→DINOv2). MAE pre-pretraining followed by weak supervision works at billion-scale [3], but there is no measured medical-imaging result for MAE continuation from a DINOv2 init. Continual SSL is more robust to forgetting than continual supervised pretraining [4].

## Failure modes

- Distribution shift / catastrophic forgetting: more in-domain pretraining hurt RAD-DINO on a public VinDR holdout [2].
- Shortcut learning from added labels.
- Saturation on low-prevalence findings.
- Best checkpoint for OOD transfer is not always the last.

## Bottom line for 30k knee MRI studies

30k is at the threshold where CXR classification starts to beat open-weights, but it is below the threshold for dense/segmentation tasks and far below the threshold for report-generation-style tasks. It is a justifiable instrument for a DINOv2 continuation experiment, but unlikely to close a large board gap by itself.

## Sources

[1] Wang et al., *Self-supervised Pretraining on OAI data for 3D Knee MRI Analysis*. ISMRM 2025, abstract #4014. https://archive.ismrm.org/2025/4014.html  
[2] Sharma et al., *Data Scaling Laws for Radiology Foundation Models*. arXiv:2509.12818. https://arxiv.org/abs/2509.12818  
[3] Girdhar et al., *The effectiveness of MAE pre-pretraining for billion-scale pretraining*. arXiv:2303.13496. https://arxiv.org/abs/2303.13496  
[4] Cossu et al., *Continual Pre-Training Mitigates Forgetting in Language and Vision*. ICLR 2022. https://arxiv.org/abs/2205.09357
