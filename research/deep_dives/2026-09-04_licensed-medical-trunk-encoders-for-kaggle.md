# Licensed, mountable medical-imaging trunk encoders for Kaggle datasets

Question from Q-DG (Claude): which medical-imaging trunk encoders can be mounted as Kaggle datasets under a permissive license (CC-BY-4.0 / Apache-2.0 / MIT / BSD), downloaded without human gating, with ≤~100M params, and a pretraining corpus that is not chest-X-ray only?

## Findings

Very few models satisfy all filters. The cleanest options are SAM-derived segmentation backbones; the only purely-classification medical trunk with a verified permissive license is RAD-DINO, and it is CXR-only.

| model | source | license | params | corpus | knee-MRI fit |
|-------|--------|---------|--------|--------|--------------|
| MedSAM ViT-B | `flaviagiammarino/medsam-vit-base` (live HF mirror), `bowang-lab/MedSAM` (code) | Apache-2.0 | ~89M | 1M+ images incl. MSK MRI | best available |
| SAM-Med2D | `OpenGVLab/SAM-Med2D` (GitHub; HF id 404) | Apache-2.0 | ~89M | 4.6M images, 10 modalities | good (no HF repo found) |
| RAD-DINO / RadioDino-b16 | `microsoft/rad-dino` (MIT source) · `Snarcy/RadioDino-b16` (cc-by-4.0, mountable) | MIT / CC-BY-4.0 | ~86M | ~840k chest X-rays | poor (wrong domain), but mountable via `RadioDino-b16` |
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | unverified | ~86M image enc. | PMC-15M (diverse) | exclude until license clarified |
| MedImageInsight | via `guy-singer/MedImageInsight` | MIT | 360M image enc. | 3.7M images, 14 domains | too large |
| RadImageNet | radimagenet.com | unclear / by request | ResNets | broad radiology | not mountable |

No released encoder pretrained specifically on knee or MSK MRI with a permissive license was found.

### Corrections from mounting attempt (Claude, A-DG)

- `wanglab/medsam-vit-b` **404s on HF API**; the live, ungated mirror is `flaviagiammarino/medsam-vit-base` (Apache-2.0, 375 MB).
- `OpenGVLab/SAM-Med2D` **404s as a HF repo id**; code/weights are on GitHub, not readily mountable as a Kaggle dataset.
- `Snarcy/RadioDino-b16` is a real, ungated, **cc-by-4.0** RAD-DINO checkpoint and appears to be the id the project's `radiodino_corpus` registry row actually uses, distinct from `microsoft/rad-dino`.
- MedSAM declares **identity normalization** (`image_mean=[0,0,0]`, `image_std=[1,1,1]`), not ImageNet. SAM's position embeddings are sized for 1024 px and must be re-gridded once at build time for smaller inputs.

## SSL vs supervised transfer for knee MRI

The literature (Raghu et al. 2019; Azizi et al. 2021; Zhao et al. 2024) suggests the choice depends on target data size, color/grayscale mismatch, and domain overlap:

- **Transfer learning** (supervised pretraining) tends to win on large, colorful, domain-matched datasets.
- **Self-supervised learning** tends to win on grayscale, small-labelled, domain-mismatched targets.

Knee MRI is grayscale, small-labelled, and far from natural images, so medical-domain SSL should theoretically beat natural-domain SSL. The fact that DINOv2-small currently leads the pool likely reflects the absence of a fair medical competitor (RAD-DINO is CXR-only; other medical trunks are unlicensed, too large, or not mountable) rather than a universal superiority of natural-image SSL.

## Next steps

1. Test MedSAM download and run a frozen-backbone linear probe vs. DINOv2-small.
2. Add RAD-DINO as a "medical SSL, wrong domain" control.
3. Clarify BiomedCLIP license before considering it for a Kaggle dataset.
4. Re-evaluate the size ceiling if MedImageInsight becomes relevant.

## Sources

- MedSAM: https://github.com/bowang-lab/MedSAM (Apache-2.0 LICENSE verified)
- SAM-Med2D: https://github.com/OpenGVLab/SAM-Med2D
- RAD-DINO: https://huggingface.co/microsoft/rad-dino
- BiomedCLIP: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- MedImageInsight paper: https://arxiv.org/abs/2410.06542
- Raghu et al. 2019: https://arxiv.org/abs/1902.07296
- Azizi et al. 2021: https://arxiv.org/abs/2101.05224
- Zhao et al. 2024: https://arxiv.org/abs/2407.05592
