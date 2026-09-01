# 3D knee MRI classification: what the literature actually says

2026-09-01, Kimi. Quick survey triggered by the MRNet page, focused on whether a "3D path" is likely to beat the slice-aggregation the project already ships.

## MRNet (2018): the canonical baseline is still 2D-per-slice + max over slices

[MRNet](https://stanfordmlgroup.github.io/projects/mrnet/) (Bien et al., PLOS Medicine 2018) is the paper usually cited for knee-MRI deep learning. Its architecture is:

- input: a series of 2D slices (`s × 3 × 256 × 256`, channel-replicated to 3);
- per-slice feature extractor: pretrained AlexNet;
- global average pooling per slice → `s × 256`;
- **max pooling across slices** → 256-d vector;
- logistic regression to combine the three series (sagittal T2, coronal T1, axial PD).

On their internal validation set it reached AUC 0.937 for abnormality, 0.965 for ACL, and 0.847 for meniscal tears. On an external sagittal-T1 set from Rijeka, ACL AUC was 0.824 without retraining and 0.911 after training on the rest of the external data.

This is **exactly the operation the project already settled on**: whole-stack per-slice features followed by elementwise max. MRNet confirms it is the field's starting point, not a new idea.

## What 3D convnets have added

A 2024 systematic review of 54 knee-MRI deep-learning studies (Mead et al., *Eur Radiol* 2025) reports that **2D models averaged higher metrics than 3D models**, but **combined 2D+3D data averaged the highest AUC-ROC (0.934 vs. 0.931 for 2D alone and 0.871 for 3D alone)**. The authors caution that the 3D numbers may reflect smaller datasets and less mature pipelines rather than a true capacity gap. Only 8 of 54 studies used 3D MRI, and 3 of those used both 2D and 3D.

Key 3D-specific findings:

- [Pedoia et al., JMRI 2019](https://doi.org/10.1002/jmri.26246): 3D CNN for meniscus and PFJ cartilage degenerative changes, using OAI data. The value was severity staging across a 3D volume, not a solo AUC win over 2D.
- [Li et al., Eur J Med Res 2022](https://doi.org/10.1186/s40001-022-00883-w): 3D Mask R-CNN for meniscus segmentation + tear detection. Segmentation-first, not end-to-end classification.
- [Müller-Franzes et al., *Sci Rep* 2025 / arXiv:2411.15802](https://arxiv.org/abs/2411.15802): **Medical Slice Transformer (MST)** — DINOv2 per slice + a Slice Transformer attending across slices. On a 1,199-patient knee-MRI set for meniscus tears, MST reached AUC 0.85 ± 0.04 vs. a 3D ResNet at 0.69 ± 0.05. The win comes from the 2D self-supervised backbone plus inter-slice attention, not from a volumetric convolution.
- [Zhang et al., *Radiology: AI* 2025 / arXiv:2405.02784](https://arxiv.org/abs/2405.02784): **MR-Transformer** adapts ImageNet-pretrained DeiT-Ti to 3D knee MRI for total-knee-replacement prediction by replicating 2D position embeddings across slices. On OAI/MOST it matched MRNet on three of four contrasts and significantly outperformed 3D CNN baselines (TSE, 3DMeT), but its gain over MRNet was limited to one contrast (COR STIR AUC 0.82 vs. MRNet 0.78). This is another instance where a 3D-aware transformer with natural-image pretraining reaches parity with MRNet but does not clearly surpass the 2D-per-slice recipe.

## Implications for this project

| path | what it means here | status in repo |
|---|---|---|
| **Whole-stack max over per-slice features** | MRNet's operation; already shipped | `slice_pooling=max`, `whole_stack=True` |
| **Attention over the slice sequence** | MST's move; replaces max with learned aggregation | `slice_pooling=attend` tested, peaked at 0.8476 vs max 0.8457 — under the floor (`attend_true`, README:302) |
| **True 3D convolution / tubelet trunk** | V-JEPA2 is the only tubelet trunk tried; underperforms DINOv2 base by −0.017 | `vjepa2l` at 0.8343 (Q6, `interpretations/kneemri/2026-09-01_q6_untried_encoder_families.md`) |
| **Multi-plane fusion** | MRNet combines sagittal/coronal/axial; the project uses sagittal+coronal slots | `extended_slots` added one coronal slot and read +0.0002 — axis saturated |
| **Combined 2D + 3D data** | literature's best AUC shape, but requires 3D acquisitions the competition may not provide | not applicable unless external 3D corpus is ingested |

## Verdict

A pure 3D convolutional path has **not** beaten the 2D-per-slice + aggregation recipe in the published knee-MRI literature. The one recent method that does beat a 3D CNN (MST) does so by keeping the 2D DINOv2 backbone and adding a slice-level transformer — which this project has already measured as flat under its own operating point. The V-JEPA2 tubelet experiment is the project's own test of a true 3D trunk, and it also trails DINOv2.

So the **3D path is not obviously promising**, and the cheaper next steps are still:

1. improve the **supervision** (label quality, calibration, reader blending);
2. improve **slice aggregation** only if a new attention mechanism is paired with a stronger inductive bias than the current `attend` probe showed;
3. ingest **external 3D data** only after the SSL objective on the existing cache is validated.
