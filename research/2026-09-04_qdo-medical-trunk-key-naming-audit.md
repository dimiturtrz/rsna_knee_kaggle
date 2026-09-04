# Q-DO — Audit of published medical trunk key naming on Hugging Face

**Date:** 2026-09-04  
**Trigger:** Claude question Q-DO after `Snarcy/RadioDino-b16` was found to ship a `config.json` claiming `ViTModel` while its weights use timm key naming, causing a silent random trunk load.  
**Scope:** Inspect `config.json` / `open_clip_config.json` and weight-file key prefixes for candidate medical trunks. No code changes.

## Method

For each repository I read the config file from the HF Hub and inspected the weight keys:
- For `model.safetensors`, read the JSON header from the first bytes.
- For `.bin`/`.pth`/`.pt` files, downloaded the weight file only and loaded it with `torch.load(..., map_location="cpu", weights_only=False)`, then listed key prefixes.

All findings below are from files actually opened, not inferred.

## 1. RadioDINO family — all timm-keyed, all mislabelled as `ViTModel`

| Repository | `config.json` architectures | Weight prefixes | Verdict | timm string | Width |
|------------|-----------------------------|-----------------|---------|-------------|-------|
| `Snarcy/RadioDino-b16` | `["ViTModel"]` | `blocks`, `cls_token`, `norm`, `patch_embed`, `pos_embed` | TIMM-KEYED | `vit_base_patch16_224` | 768 |
| `Snarcy/RadioDino-s16` | `["ViTModel"]` | same prefixes | TIMM-KEYED | `vit_small_patch16_224` | 384 |
| `Snarcy/RadioDino-s8` | `["ViTModel"]` | same prefixes | TIMM-KEYED | `vit_small_patch8_224` | 384 |

No `l16` or additional sizes were found on Hugging Face. Sample key: `blocks.0.attn.qkv.weight`. All three checkpoints have the same packaging fault as the known-broken `b16`.

Sources:
- [Snarcy/RadioDino-b16](https://huggingface.co/Snarcy/RadioDino-b16)
- [Snarcy/RadioDino-s16](https://huggingface.co/Snarcy/RadioDino-s16)
- [Snarcy/RadioDino-s8](https://huggingface.co/Snarcy/RadioDino-s8)
- [Radio-DINO GitHub](https://github.com/Snarci/Radio-DINO)

## 2. RadImageNet — no clean HF mirror; weights are torchvision-style

Two HF mirrors exist, neither ships `config.json`, neither uses timm keys.

### `convergedmachine/RadImagenet`

- Files: `resnet50.pth`, `resnet18.pth`, `resnet10t.pth`
- Checkpoint is a dict with keys `args`, `epoch`, `lr_scheduler`, `model`, `optimizer`; state dict is `sd["model"]`.
- ResNet50/18 keys are prefixed with `_orig_mod.` (torch.compile wrapper) and underneath are torchvision ResNet keys: `conv1.weight`, `bn1.*`, `layer1.0.conv1.weight`, etc.
- ResNet10t uses a custom variant (`conv1.0.weight`, `conv1.1.weight`, ...).
- Verdict: **OTHER / torchvision-style with `_orig_mod.` wrapper**.

### `Lab-Rasool/RadImageNet`

- Files: `DenseNet121.pt`, `ResNet50.pt`, `InceptionV3.pt`
- All keys prefixed with `backbone.`. Stripping the prefix yields torchvision ResNet / DenseNet / InceptionV3 keys.
- Verdict: **OTHER / torchvision-style wrapped in `backbone.`**.

Neither mirror is suitable for a `TimmHubWeights` loader. A torchvision loader with prefix normalization is required.

Sources:
- [convergedmachine/RadImagenet](https://huggingface.co/convergedmachine/RadImagenet)
- [Lab-Rasool/RadImageNet](https://huggingface.co/Lab-Rasool/RadImageNet)
- [BMEII-AI/RadImageNet GitHub](https://github.com/BMEII-AI/RadImageNet)

## 3. `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

- No `config.json`; uses `open_clip_config.json`.
- `open_clip_config.json` specifies `model_cfg.vision_cfg.timm_model_name: "vit_base_patch16_224"`.
- Weight file: `open_clip_pytorch_model.bin`.
- Top-level prefixes: `logit_scale`, `text`, `visual`.
- Vision trunk keys are nested under `visual.trunk.`, e.g. `visual.trunk.blocks.0.attn.qkv.weight`.
- Verdict: **TIMM-KEYED but NESTED under `visual.trunk.`**.
- timm string: `vit_base_patch16_224`, width 768.

Source: [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

## 4. MRI-CORE

- No Hugging Face model repository was found.
- Official release is on GitHub (`mazurowski-lab/mri_foundation`) with a `.pth` checkpoint distributed via Google Drive.
- Model is SAM-based (`sam_model_registry['vit_b']`); key naming is SAM-style (`image_encoder.*`, `prompt_encoder.*`, `mask_decoder.*`).
- Verdict: **Not on HF; not part of this `config.json` audit.**

Source: [mazurowski-lab/mri_foundation](https://github.com/mazurowski-lab/mri_foundation)

## 5. MedSAM

- Correct repo name: `flaviagiammarino/medsam-vit-base` (not `medsam-vit-b`).
- `config.json` architectures: `["SamModel"]`.
- Weight prefixes: `mask_decoder`, `prompt_encoder`, `shared_image_embedding`, `vision_encoder`.
- Sample key: `vision_encoder.layers.0.attn.qkv.weight`.
- Verdict: **MATCHES transformers `SamModel` naming**.

Source: [flaviagiammarino/medsam-vit-base](https://huggingface.co/flaviagiammarino/medsam-vit-base)

## Registry implications

| Model / repo | Loader type needed |
|--------------|--------------------|
| RadioDino b16 / s16 / s8 | `TimmHubWeights` with plain timm string |
| BiomedCLIP vision trunk | `TimmHubWeights` with subtree extraction (`visual.trunk.*`) |
| RadImageNet (both mirrors) | Custom torchvision loader + prefix strip (`_orig_mod.` or `backbone.`) |
| MRI-CORE | SAM loader, not HF Hub |
| MedSAM | transformers `SamModel` / plain HF loader |
