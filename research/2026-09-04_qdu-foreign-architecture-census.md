# Foreign-architecture census of public knee-MRI / RSNA-knee checkpoints

Date: 2026-09-04
Prompt: Claude Q-DU — enumerate publicly downloadable knee-MRI/RSNA-knee checkpoints and notebooks, classify by architecture family, quote licences from artefacts, deduplicate copies, flag gated/NC, rank by distance from the project's `facebook/dinov2-small` per-slice reader.

## Method

- Read `comp/champion.json` and `comp/members.roster` to identify the already-seated foreign member.
- Used Kaggle API (`/api/v1/datasets/view/{slug}`, `/api/v1/datasets/list/{slug}`, `/api/v1/models/list`, model download endpoints) with the stored `~/.kaggle/kaggle.json` credentials.
- Unpacked checkpoint archives and inspected state-dict keys / embedded configs with `torch.load` to confirm architecture families rather than relying on names.
- Quoted licences from artefact metadata (`licenseName`) or the packaged `LICENSE` file.

## Already known / seated

- `dreaddevelopment/raptor-knee-widedense` — CoAtNet_RM_2 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`) at 384 px, three-slice windows, per-finding attention head. CC0-1.0. Seated as `raptor_v4_swa` in `raptorweight_v1` via `kneemri.inference.foreign_member`.

## Census by architecture family

### True 3D / volumetric

- `fabotelli/pillar0-knee-finetune` (Kaggle Model `PyTorch/atlas-336-2ep/1`) — Pillar-0 BreastMRI foundation model fine-tuned on RSNA knee MRI. Licence: **Educational Community License 2.0** (packaged `LICENSE`). Input is a uint8 volume `[D,256,256]` that is trilinearly resized to `[192,384,384]`, channel-replicated to 3 channels, fed through the Pillar-0 visual encoder, and pooled to 12 outputs via a linear head (`Linear(1152→12)`). This is the only public true-3D checkpoint identified. Gold-58 macro-AUC 0.797.

### CNN families (non-DINOv2)

- `dreaddevelopment/raptor-knee-arms` — same producer/decode as widedense but contains three backbones: CoAtNet 384 (`raptor_ft_coatnet384.pt`), ConvNeXt-v2 Base 336 (`raptor_ft_cnv2b336.pt`), EfficientNetV2-L 480 (`raptor_ft_effv2l480.pt`). CC0-1.0.
- `dreaddevelopment/raptor-knee-finespacing` — CoAtNet v9 (`raptor_ft_coatnet_v9_full.pt`). CC0-1.0.
- `ishan1989/rsna-knee-5fold-weights` — five `best_model_fold*.pt` files. State-dict keys (`features.X.block`, fused-MBConv / MBConv) confirm **EfficientNetV2-S**. CC0-1.0.
- `tim9510019/rsna-knee-v33-15models` — `track_5a_biomedclip_fold*.pth` and `track_5b_geometry_fold*.pth` are ResNet-50 (`branch_sag.0` conv1, `branch_sag.4` layer1, bottleneck blocks); `v32_effnet_fold*.pth` is EfficientNet-B0. CC0-1.0.
- `tim9510019/rsna-knee-v34-master-models` — includes EfficientNet-B0 (`v42_best_fold_*_384.pth`), ResNet-50/EfficientNet-based DNA-Helix hybrid (`v42_best_dna_helix_384.pth`, has `pos_embed_x`, `pos_embed_y`, `view_embed`, conv branches), and self-CNN (`v42_best_self_cnn_384.pth`). CC0-1.0.
- `sahil00001/rsna-knee-phase14-weights` — single `best_model_phase14_recB.pth`; state-dict keys (`shared_encoder.conv1`, `layer1...`) confirm **ResNet**. CC0-1.0.
- `ruicompany/rsna-knee-a5-rad` (Kaggle Model, MIT) — package contains a 5-fold DINOv3 ensemble plus a `rad/ResNet50.pt` RadImageNet ResNet-50 checkpoint. The ResNet-50 arm is the distinct CNN family member.
- `vidish0987/httpskaggle-commodelsvidish0987` / `convnext-tiny-5fold-ema` (Kaggle Model, Apache 2.0) — five `model_convnext_tiny_fold*_ema.pt` files; state-dict keys (`slice_encoder.stem`, `stages.X.blocks.Y.conv_dw`, `mlp.fc1`) confirm **ConvNeXt-Tiny**.
- `yunusgmsoy/rsna-knee-radhead-resnet50` (Kaggle Model, CC0-1.0) — package advertises ResNet50 + RadHead, but the downloaded artefact (`v52_rad_heads_yunus.pt`) contains only metadata and out-of-fold predictions, not trainable weights. Not a usable checkpoint.

### DINOv2 / DINOv3 family (closest to our monoculture)

- `mattiaangeli/knee-mri-fold-weights` — five folds of **DINOv3 ViT-S/16** (`vit_small_patch16_dinov3.lvd1689m`), 336 px, 16 slices, slot-token conditioning, `xcodex` pool. CC0-1.0.
- `tim9510019/rsna-knee-v34-master-models` — `best_dinov3_finetuned_384.pth` is DINOv3 ViT-S/14. CC0-1.0.
- `starkhushi/rsna-knee-v6f2-weights` — `model_v6f2_best.pt` is **DINOv2 ViT-S/14** at 518 px (position embedding `[1,1370,384]` = 1 cls + 37×37 patches), with a 5-slot embedding. CC0-1.0.
- `zaidaliiq1000/rsna-knee-ens55-weights` — 55 small heads over a shared DINOv2-vits14 (`arch: dinov2_vits14`, `feat_dim: 384`); head uses slot embedding, label query, GRU. Licence: **Other (specified in description)**; description is empty in API metadata.
- `pilkwang/rsna-knee-weights` — 20 models all `facebook/dinov2-small`, 336 px, 12 slices, `cls_mean` pool, 130 mm crop. CC0-1.0.
- `phoenixdroid/rsna-knee-models` — single `fold_1_best.pt`; config says `backbone: dinov2_vits14`, 224 px, 16 slices. CC0-1.0.
- `girishbose/dinov2-vits14-rsna-knee` and `dinov2-vitb14-rsna-knee` — public pretrained DINOv2 weights, not knee-fine-tuned. Apache 2.0.

### Not classification weights / not relevant

- `royceplayz/rsna-segformer-weights` — SegFormer-B5 fine-tuned on ADE20k (`SegformerForSemanticSegmentation`, 150 ADE labels). Apache 2.0. Semantic segmentation, not study-level classification; not usable as-is.
- `cjinny/mrnet-v1` — MRNet v1.0 dataset only; licence Unknown.
- `pilkwang/rsna-knee-llm-labels` — report-derived soft labels, not weights; CC0-1.0.

### Non-shippable (NC)

- `riachk/rsna-knee-abnormality-detection-tool`, `riachk/rsna-knee-abnormality-smart-detection`, `riachk/smart-rsna-mri-detection-tool` — Kaggle Models with licence **CC BY-NC 4.0**. Non-commercial, so not shippable under the competition winner obligations (which require public dataset + code, effectively permissive).

## Deduplication

- **DINOv3 same lineage**: `mattiaangeli/knee-mri-fold-weights` and the `a5` component of `ruicompany/rsna-knee-a5-rad` share identical configs (`vit_small_patch16_dinov3.lvd1689m`, 336 px, 16 slices, `xcodex` pool, slot-token conditioning). Count as one lineage, not two distinct families.
- **Dread Development raptor series**: `widedense`, `arms`, `finespacing` share producer and decode; `arms` adds ConvNeXt-v2 and EfficientNetV2-L as genuinely new backbone families beyond the already-used CoAtNet.
- **Tim9510019 v33/v34**: share EfficientNet-B0 lineage across versions; v34 adds DNA-Helix and DINOv3 specialist.

## Ranking by distance from `facebook/dinov2-small` per-slice reader

1. **Pillar-0** — true 3D volumetric foundation model (cross-anatomy transfer); largest architectural distance.
2. **Dread Development raptor series** — CoAtNet/ConvNeXt-v2/EfficientNetV2-L with three-slice windows and per-finding attention; different backbone families and different stack consumption.
3. **CNN-only families** — EfficientNetV2-S (`ishan1989`), ResNet-50 (multiple sources), EfficientNet-B0 (`tim9510019`), ConvNeXt-Tiny (`vidish0987`).
4. **Hybrid attention-conv** — DNA-Helix (`tim9510019` v34).
5. **DINOv2/DINOv3 variants** — newer pretraining or different resolution/decode, but same architectural family as our reader.

## Takeaway

There are at least **seven structurally distinct, permissively licensed families** available beyond the already-seated CoAtNet. The two most interesting axes are:

- **Pillar-0**: the only true 3D option, but requires building a full volumetric feed and the published gold-58 AUC (0.797) is well below our seated foreign member (0.923 board).
- **Dread Development `raptor-knee-arms`**: adds ConvNeXt-v2 Base and EfficientNetV2-L to the existing CoAtNet decode; lowest-friction addition because the feed infrastructure already exists.

The remaining CNN families (EfficientNetV2-S, ResNet-50, ConvNeXt-Tiny) are structurally different but require a new feed implementation each. The DINOv2/DINOv3 clones are the most available but add the least architectural distance and therefore are unlikely to break the monoculture plateau.


## Addendum: Pillar-0 licence and overlap (Q-DV follow-up)

The only genuinely new entry in the census was `fabotelli/pillar0-knee-finetune`. The follow-up asked for the licence text quoted at length and for any declared overlap.

### Licence

The packaged `LICENSE` file is the full Educational Community License 2.0 text. It grants a "perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work". It contains no NC clause and no competition-specific restriction. The only material deviation from Apache-2.0 is the narrowed patent grant in section 3.

Upstream `YalaLab/Pillar0-BreastMRI` on Hugging Face also tags `license:ecl-2.0` (`cardData.license` and `license:ecl-2.0` tag). The HF repo is click-through gated (`extra_gated_fields` requires Name / Organization), but the licence tag remains ECL-2.0.

Kaggle Model `licenseName` displays "Other (specified in description)" — this is a Kaggle UI limitation, not a licence conflict; both the packaged file and upstream agree on ECL-2.0.

### Overlap

The Kaggle Model `README.md` states:

> "The 58 holdout studies were excluded from training."

The checkpoint (`pillar_ft_a.final.pt`) contains only `model`, `head`, and `ep` keys — no fold list, no split file, no UIDs. The exclusion claim is prose-only and not machine-verifiable from the artefact. If accepted, the published 0.797 macro-AUC is a held-out number; if not, the model should be treated as potentially in-sample.

### Shippability summary for Pillar-0

- Licence: clean for competition use (ECL-2.0, no NC, no competition restriction).
- Gating: the Kaggle derivative is public and ungated; upstream HF base model is click-through gated.
- Overlap: declared held-out but not independently verifiable.

This leaves the decision on whether to spend decode-card time on it to a judgement call on the README statement.


## Addendum: Pillar-0 per-label AUC table (Q-DW follow-up)

Quoted from the packaged `README.md` of `fabotelli/pillar0-knee-finetune`:

| Target | AUC |
|---|---|
| ACL | 0.773 |
| MCL | 0.630 |
| Medial Meniscus | 0.750 |
| Lateral Meniscus | 0.781 |
| Medial OA | 0.941 |
| Lateral OA | 0.774 |
| PF OA | 0.758 |
| Effusion | 0.965 |
| Synovitis | 0.821 |
| Baker's | 0.842 |
| Contusion | 0.756 |
| Fracture | 0.775 |
| **Macro** | **0.797** |

The README does not provide per-label prevalence, confusion tables, or counts on the 58 studies. The training recipe mentions "series capped at 4/study, seed 0"; inference is described as "mean sigmoid over all series of the study", with no explicit inference-time series selection rule.


## Addendum: Pillar-0 split artefacts and kernel cost (Q-DX follow-up)

### Training-split evidence

No public artefact names the studies `fabotelli/pillar0-knee-finetune` trained on:

- The Kaggle Model package contains only `LICENSE`, `README.md`, and `pillar_ft_a.final.pt`; the checkpoint keys are `model`, `head`, `ep`.
- The README states *"The 58 holdout studies were excluded from training"* and gives a training recipe, but no UID list, fold file, or training-study count.
- Upstream `YalaLab/pillar-finetune` (GitHub) documents NLST/chest-CT fine-tuning only; no knee split.
- Upstream `YalaLab/Pillar0-BreastMRI` (Hugging Face) is the breast-MRI base model, gated behind Name/Organization fields, and is not the knee derivative.
- The Pillar-0 paper (arXiv [2511.17803v1](https://arxiv.org/abs/2511.17803)) covers abdomen-pelvis CT, chest CT, head CT, and breast MRI pretraining; "knee" does not appear.

The held-out claim is therefore prose-only and not independently machine-verifiable.

### Inference cost

Published numbers exist for the Atlas backbone, not for the knee checkpoint on Kaggle hardware:

- Pillar-0 paper architecture ablation: Atlas takes **0.2 s/sample** for a `384×384×384` volume at inference, vs 38.8 s/sample for a matched ViT-S. Pillar-0 is reported as 32×/15×/55× faster than MedGemma/MedImageInsight/Lingshu.
- Knee fine-tune README gives only **training** cost: *"2 epochs on one NVIDIA B200 (~70 min)"*.
- Local rebuild measurement (this project, `kneemri.inference.pillar_member`): **~0.70 s/series, ~2.6 GiB VRAM** at `[192,384,384]`.

No published inference cost exists for the knee checkpoint itself or for T4 hardware.

### Kaggle kernel limits

- Wall-clock cap per submission: **9 hours**.
- GPU observed/used in this competition: **T4 ×2**.
- Internet disabled during scoring; only uploaded datasets/models are available.

### Kernel fit estimate

Inference rule is *mean sigmoid over all series of the study* with no stated series cap. For ~1,300 test studies, forward-only wall-clock is on the order of tens of minutes on fast local hardware, but the actual T4 cost will be higher and the DICOM → `[192,384,384]` preprocessing must also fit inside the 9-hour budget.

## Amendment 2026-09-06 — the DINOv3 licence chain overrides the CC0 labels

The CC0-1.0 readings above come from Kaggle dataset metadata. For DINOv3-derived weights that
metadata cannot be the operative licence: the **DINOv3 License** (Meta Platforms, 2025-08-19,
`facebookresearch/dinov3` `LICENSE.md`) grants derivative-works rights (§1a) but requires that any
redistribution of the DINO Materials OR derivative works be under that same Agreement with a copy
provided (§1b(i)), plus publication attribution (§1b(ii)); derivative works are owned by their maker
as between maker and Meta (§5a). **No non-commercial clause, no share-alike** — materially more
permissive than DINOv2's CC-BY-NC-4.0. Verified full text pulled 2026-09-06.

Consequences:
- `mattiaangeli/knee-mri-fold-weights` (DINOv3 ViT-S/16): shippable under rule 2.6, but the citable
  licence is the DINOv3 License, not CC0 — ship `LICENSE.md` with the mount and carry the attribution.
- `tim9510019/...best_dinov3_finetuned_384.pth` (DINOv3 ViT-S/14): same chain.
- `starkhushi/rsna-knee-v6f2-weights` is **DINOv2** (CC-BY-NC-4.0): different chain — non-commercial
  plus mandatory attribution; CC0 metadata would be flat wrong there.
