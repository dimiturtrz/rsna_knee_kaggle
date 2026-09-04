From: kimi

## A-DU — Foreign-architecture census of public knee-MRI / RSNA-knee checkpoints

Status: read-only audit. I inspected Kaggle datasets, Kaggle Models, and HF-flavoured artefacts via the Kaggle API and by unpacking checkpoints. Licences are quoted from the artefact metadata, not inferred.

### What we already have seated

The champion pins one foreign seat: `raptor_v4_swa` from `dreaddevelopment/raptor-knee-widedense` (CC0-1.0), a CoAtNet_RM_2 at 384 px with a per-finding attention head over three-slice windows, rebuilt by `kneemri.inference.foreign_member`. This note looks for *additional* distinct families.

### Census, ranked by distance from our `facebook/dinov2-small` per-slice max-pool reader

| rank | slug / model | family | stack consumption | licence (quoted) | notes |
|------|--------------|--------|-------------------|------------------|-------|
| 1 | `fabotelli/pillar0-knee-finetune` (Kaggle Model, `atlas-336-2ep/1`) | Pillar-0 BreastMRI 3D foundation model | true 3D volume: uint8 `[D,256,256]` → z-score → trilinear to `[192,384,384]` → channel-replicate ×3 → visual encoder → Linear(1152→12) | Educational Community License 2.0 (artefact `LICENSE`) | Only true 3D/volumetric checkpoint found. Fine-tuned on RSNA knee; gold-58 macro-AUC 0.797. Cross-anatomy transfer, not a DINOv2 derivative. |
| 2 | `dreaddevelopment/raptor-knee-widedense` | CoAtNet (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`) | three-slice RGB windows across 5 slots, per-finding attention pool | CC0: Public Domain (Kaggle API `licenseName`) | Already seated. Single model 0.924 public LB. |
| 2b | `dreaddevelopment/raptor-knee-arms` | mixed: CoAtNet 384, ConvNeXt-v2 Base 336 (`cnv2b336`), EfficientNetV2-L 480 (`effv2l480`) | same three-slice windowing as widedense | CC0: Public Domain | Same producer, same decode; three different backbones in one dataset. |
| 2c | `dreaddevelopment/raptor-knee-finespacing` | CoAtNet v9 (`coatnet_v9`) | same windowing | CC0: Public Domain | Another CoAtNet variant from same producer. |
| 3 | `ishan1989/rsna-knee-5fold-weights` | EfficientNetV2-S | 5-fold ensemble; key names confirm `features.X.block` MBConv/fused-MBConv | CC0: Public Domain | Pure CNN family, no transformer. |
| 3b | `tim9510019/rsna-knee-v33-15models` | ResNet-50 (`track_5a_biomedclip`, `track_5b_geometry`) + EfficientNet-B0 (`v32_effnet`) | per-plane ResNet/EfficientNet branches | CC0: Public Domain | BiomedCLIP-flavoured ResNet-50 weights; geometry variant is the same ResNet-50 architecture. |
| 3c | `tim9510019/rsna-knee-v34-master-models` | EfficientNet-B0 (`v42_best_fold_*`), ResNet50/DNA-Helix (`v42_best_dna_helix_384`), self-CNN | multi-branch per-plane | CC0: Public Domain | DNA-Helix is a hybrid: 2D conv branches + positional embeddings + view embeddings. |
| 3d | `sahil00001/rsna-knee-phase14-weights` | ResNet (`shared_encoder.conv1/layer1...`) | unknown decode | CC0: Public Domain | Single 70 MB checkpoint; ResNet family confirmed by state-dict keys. |
| 3e | `ruicompany/rsna-knee-a5-rad` (Kaggle Model) | ResNet-50 RadImageNet | 3 planes × 8 slices × 224² | MIT (Kaggle Model `licenseName`) | Packaged alongside a DINOv3 ensemble; the ResNet-50 arm is the distinct piece. |
| 3f | `vidish0987/.../convnext-tiny-5fold-ema` (Kaggle Model) | ConvNeXt-Tiny | 5-fold EMA | Apache 2.0 | 2.7 GB, 5 folds; metadata names ConvNeXt-Tiny explicitly. |
| 4 | `mattiaangeli/knee-mri-fold-weights` | DINOv3 ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m`) | 6 slots × 16 slices × 336², slot-token conditioning, `xcodex` pool | CC0: Public Domain | Same generation as our DINOv2-small but newer pretraining. |
| 4b | `tim9510019/rsna-knee-v34-master-models` | DINOv3 ViT-S/14 (`best_dinov3_finetuned_384`) | multi-view ViT specialist | CC0: Public Domain | DINOv3 variant. |
| 4c | `starkhushi/rsna-knee-v6f2-weights` | DINOv2 ViT-S/14 | 5 slots with slot embedding, 518² input (pos-embed `[1,1370,384]` = 1 cls + 37×37 patches) | CC0: Public Domain | Very close to our reader: same backbone family, only decode differs. |
| 4d | `zaidaliiq1000/rsna-knee-ens55-weights` | DINOv2 ViT-S/14 | target-attention head with GRU slot aggregation | Other (specified in description) | 55 tiny 3.2 MB heads over a shared DINOv2 feature extractor. Licence is not a standard permissive tag. |
| 4e | `pilkwang/rsna-knee-weights` | DINOv2-small (`facebook/dinov2-small`) | 12 slices, cls_mean pool, 130 mm crop | CC0: Public Domain | 20 DINOv2-small clones; functionally a sibling of our own monoculture. |
| 4f | `phoenixdroid/rsna-knee-models` | DINOv2-small | 16 slices, 224² | CC0: Public Domain | Another DINOv2-small clone, lower resolution. |
| 4g | `girishbose/dinov2-vits14-rsna-knee` | DINOv2 ViT-S/14 pretrained | — | Apache 2.0 | Not fine-tuned on knee; just the public pretrained backbone. |
| 4h | `girishbose/dinov2-vitb14-rsna-knee` | DINOv2 ViT-B/14 pretrained | — | Apache 2.0 | Same, base size. |
| — | `royceplayz/rsna-segformer-weights` | SegFormer-B5 (ADE20k) | semantic segmentation head | Apache 2.0 | Not a classification checkpoint; standard HF ADE20k model repackaged. Not usable as-is. |
| — | `cjinny/mrnet-v1` | — | dataset only | Unknown | MRNet v1.0 data, no trained weights. |
| — | `pilkwang/rsna-knee-llm-labels` | — | labels only | CC0: Public Domain | Report-derived soft labels, not weights. |
| NC | `riachk/rsna-knee-abnormality-detection-tool` etc. | unspecified | — | CC BY-NC 4.0 | Three models by Ria Chakraborty; NC licence = not shippable under competition winner obligations. |

### Deduplication / copy status

- **DINOv3 same lineage**: `mattiaangeli/knee-mri-fold-weights` and the `a5` half of `ruicompany/rsna-knee-a5-rad` carry *identical* configs (`vit_small_patch16_dinov3.lvd1689m`, 336 px, 16 slices, `xcodex` pool, slot-token conditioning). Treat as the same architectural lineage, not two distinct foreign families.
- **Dread Development trilogy**: `widedense`, `arms`, `finespacing` share the same producer and the same three-slice decode; `arms` adds ConvNeXt-v2 and EfficientNetV2-L variants, so it contributes two new backbone families beyond the already-used CoAtNet.
- **Tim9510019 v33 vs v34**: v33’s `v32_effnet` and v34’s `v42_best_fold_*` both look like EfficientNet-B0; likely related but not byte-identical. v34 adds DNA-Helix and the DINOv3 specialist.
- **Pilkwang / phoenixdroid / zaidaliiq1000 / girishbose** are all DINOv2-family and do not add architectural distance.

### Shippability summary

- **Permissive and usable**: all CC0 and Apache/MIT/ECL entries above.
- **Not shippable**: Ria Chakraborty’s CC BY-NC 4.0 models.
- **Unclear / needs legal check**: `zaidaliiq1000/rsna-knee-ens55-weights` reports licence "Other (specified in description)" but the description is empty in the API; avoid unless clarified.
- **Not relevant as classification weights**: `royceplayz/rsna-segformer-weights` (segmentation ADE20k), `cjinny/mrnet-v1` (dataset), `pilkwang/rsna-knee-llm-labels` (labels).

### Bottom line

The landscape is **not** four distinct checkpoints. There are at least **7 structurally different families** publicly available and permissively licensed:

1. Pillar-0 (true 3D BreastMRI foundation model) — most distant.
2. CoAtNet / ConvNeXt-v2 / EfficientNetV2-L (Dread Development raptor series).
3. EfficientNetV2-S (`ishan1989`).
4. ResNet-50 (`tim9510019` v33 biomedclip/geometry, `ruicompany` a5-rad rad arm, `sahil00001`, `yunusgmsoy`).
5. EfficientNet-B0 (`tim9510019` v32/v42).
6. DNA-Helix hybrid (`tim9510019` v34).
7. ConvNeXt-Tiny (`vidish0987`).

Everything else is a DINOv2/DINOv3 variant or a copy/repack. The biggest gaps relative to our monoculture are **Pillar-0 (3D)** and the **Dread Development CNN family (CoAtNet/ConvNeXt/EfficientNet with windowed attention)**. The former needs a full 3D feed; the latter can slot into the existing `foreign_member` decode with minor feed adjustments.
