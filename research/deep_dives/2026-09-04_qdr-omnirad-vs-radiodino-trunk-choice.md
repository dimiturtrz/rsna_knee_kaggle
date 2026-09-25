# Q-DR — OmniRad vs RadioDino: which RadImageNet trunk for the one arm?

**Date:** 2026-09-04  
**Ask:** With one ~40-minute arm slot, choose between OmniRad and RadioDino. Need exact architecture/objective/key naming, downstream numbers (especially MSK/MRI), whether OmniRad is a filtered successor, and a decisive recommendation.

## Summary

- **OmniRad** = modified DINOv2 (global crops only), ViT-S/14 (384-d) and ViT-B/14 (768-d), timm keys `vit_small_patch14_dinov2` / `vit_base_patch14_dinov2`.
- **RadioDino** = DINO, ViT-B/16, ViT-S/16, ViT-S/8, timm keys `vit_base_patch16_224`, `vit_small_patch16_224`, `vit_small_patch8_224`.
- **OmniRad beats RadioDino** on MedMNIST v2, MedSegBench, and image captioning, but **neither model reports knee-MRI or MSK-specific downstream numbers**.
- **Corpus is the same RadImageNet**; the 1.35M vs ~1.2M difference is total images vs training split, not a filter.
- **Recommendation: RadioDino.** OmniRad has a licence conflict: README says CC BY 4.0, but `config.json` says CC BY-NC 4.0. The NC field makes OmniRad potentially non-commercial and therefore risky for a Kaggle submission. RadioDino's licence is unambiguously CC BY 4.0.

## Architecture and objective

### OmniRad (`Snarcy/OmniRad-*`)

From `config.json`:

| Variant | timm architecture | Features | Patch | Objective |
|---------|-------------------|----------|-------|-----------|
| small | `vit_small_patch14_dinov2` | 384 | 14 | Modified DINOv2, global crops only |
| base | `vit_base_patch14_dinov2` | 768 | 14 | Modified DINOv2, global crops only |

- Paper §3.1: "We further modify the original DINOv2 formulation by retaining only global crops and completely removing local crops."
- Key naming is pure timm; no HF `ViTModel` dual-format.

### RadioDino (`Snarcy/RadioDino-*`)

From `config.json`:

| Variant | timm architecture | Features | Patch | Objective |
|---------|-------------------|----------|-------|-----------|
| b16 | `vit_base_patch16_224` | 768 | 16 | DINO |
| s16 | `vit_small_patch16_224` | 384 | 16 | DINO |
| s8 | `vit_small_patch8_224` | 384 | 8 | DINO |

- Config also contains an HF `architectures: ["ViTModel"]` block that Q-DO found does not match the actual timm state dict.

## Downstream numbers

OmniRad adds benchmarks that RadioDino does not report:

- **MedSegBench:** six radiology segmentation datasets; OmniRad-base mean Dice 92.95%.
- **Image captioning:** radiological image captioning with BART; OmniRad > CLIP, DINOv2, DINOv3, RadioDino.
- **MedMNIST v2:** OmniRad improves over RadioDino on OrganAMNIST, OrganCMNIST, OrganSMNIST.

**No knee-MRI or MSK-specific numbers** are reported for either model.

## Corpus relationship

- RadioDino: "pretrained on the large-scale RadImageNet dataset (1.35 million CT, MRI, and Ultrasound images)".
- OmniRad paper Table 2: RadImageNet = 1,350,000 total images (1,080,000 train / 135,000 val / 135,000 test). OmniRad README/abstract round to "~1.2M", i.e. the training split.
- **No published corpus filter.** Same dataset, different objective/architecture.

## Licence finding (decisive)

### RadioDino
- README frontmatter: `license: cc-by-4.0`
- No conflicting licence field in `config.json`
- **Result:** unambiguously permissive, shippable.

### OmniRad
- README frontmatter: `license: cc-by-4.0`
- README text: "model weights are licensed under ... CC BY 4.0"
- `config.json` `pretrained_cfg.license`: **`cc-by-nc-4.0`**
- **Result:** licence conflict. The NC field in the weight config creates a non-commercial risk, making OmniRad unsuitable for a Kaggle submission unless clarified by the authors.

## Conclusion

Spend the arm on **RadioDino** (b16 or s8). OmniRad is newer and stronger on published benchmarks, but it cannot be treated as safely shippable while one of its own files declares a non-commercial licence. RadioDino has a known packaging quirk (already resolved in Q-DO) and a clean CC BY 4.0 declaration.
