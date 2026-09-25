# SAM / MedSAM neck output vs. raw ViT tokens for classification transfer

Date: 2026-09-04
Question: Should a classification trunk read SAM's 256-channel neck output or the 768-dimensional raw patch tokens?

## Conclusion

The open literature does **not** contain a direct comparison of the two read-out points for SAM / MedSAM classification transfer. The indirect evidence points toward reading the raw 768-d tokens:

- The neck output is weak on semantics (SAMantics linear-probe result on ImageNet).
- A medical report-generation model explicitly drops the MedSAM neck to use patch embeddings (FODA-PG).
- The neck was designed for the mask decoder and compresses the representation 3×.

For a project whose ViT-B incumbents already hand the head 768-d tokens, reading raw tokens removes a width confound and aligns with existing adapters.

## Papers reviewed

### SAMantics — Yang et al., 2024, arXiv:2411.15288

Title: *There is no SAMantics! Exploring SAM as a Backbone for Visual Understanding Tasks*

- Linear-probes the frozen SAM image encoder on ImageNet-1K.
- Attaches the linear head to the encoder output, i.e. the **neck output** for SAM.
- Results: SAM 11.06% top-1 accuracy vs. DINOv2 77.43%.
- Interpretation: the neck output is semantically poor for classification.
- Limitation: does not compare raw pre-neck tokens.

### FODA-PG — Shu et al., 2024, arXiv:2409.03947

Title: *FODA-PG for Enhanced Medical Imaging Narrative Generation: Adaptive Differentiation of Normal and Abnormal Attributes*

- Uses the MedSAM ViT encoder as an image encoder for medical report generation.
- Explicitly **omits the MLP neck** "to focus on extracting patch embeddings".
- Direct precedent for discarding the neck in a medical imaging task.
- Limitation: generation task, not classification.

### Raptor — An et al., 2025, arXiv:2507.08254

Title: *Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models*

- Uses frozen 2D foundation models as slice feature extractors for 3D volume classification.
- Compares SAM vs. MedSAM: AUC 0.850 vs. 0.872 on 3D MedMNIST (Appendix A.1, Table A.1).
- Refers to "the encoder of SAM / MedSAM" but does **not** state which read-out point is used.
- The small SAM→MedSAM gain suggests the medical fine-tuning adds only modest classification signal.
- Limitation: no neck-vs-token ablation.

### MedSAM — Ma et al., 2024, Nature Communications

Title: *Segment anything in medical images*

- Fine-tuned SAM on 1M+ medical image-mask pairs.
- Updated **all trainable parameters in the image encoder and mask decoder**; prompt encoder frozen.
- Therefore the neck was fine-tuned, but no layer-wise or parameter-delta analysis was found.

## Architecture reminder

SAM's vision encoder (HuggingFace `SamVisionEncoder`):

1. Patch embedding + absolute position embedding.
2. ViT-B transformer blocks → 768-d patch tokens.
3. Convolutional neck (1×1 conv → LayerNorm → 3×3 conv → LayerNorm) → 256-channel feature map.

`kneemri/models/sam_trunk.py` currently loads `SamModel.vision_encoder`, which returns the **neck output** as a feature map.

## Recommendation

Read the raw 768-d tokens for the MedSAM classification arm. This:

- matches the read-out width of the project's other ViT-B trunks,
- avoids a task-specific segmentation bottleneck,
- is supported by the only medical-task precedent that explicitly chose a read-out point.

The remaining uncertainty can only be resolved by an internal ablation; the literature does not provide a settled answer.

## References

- [SAMantics](https://arxiv.org/html/2411.15288v1)
- [FODA-PG](https://arxiv.org/html/2409.03947v1)
- [Raptor](https://arxiv.org/html/2507.08254v1)
- [MedSAM](https://www.nature.com/articles/s41467-024-44824-z)
- [SAM encoder architecture / issue #588](https://github.com/facebookresearch/segment-anything/issues/588)
