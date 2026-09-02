# SOTA pretrained models for knee MRI (2025–2026 emphasis)

This note surveys recently released foundation and SOTA models that are relevant to our RSNA 12-label knee-MRI task, with emphasis on 2025–2026 work and on models whose weights are actually available. The goal is to identify whether we should replace, continue-pretrain, or ensemble our current DINOv2/DINOv3 trunk with a knee-specific or MRI-specific foundation model.

## Executive summary

| Model | Year | Anatomy focus | Weights available? | License | Relevance to RSNA task |
|-------|------|---------------|-------------------|---------|------------------------|
| **OrthoFoundation** | 2026 | Knee (X-ray + MRI) | ✅ Yes, `OrthoFoudation-L.pth` | MIT | **Highest** — trained on 1.25M knee images including OAI+fastMRI; evaluated on ACL/PCL/MCL/Meniscus/Cartilage |
| **MRI-CORE** | 2024 | General MRI | ✅ Yes, `MRI_CORE_vitb.pth` | Apache 2.0 | High — SAM-based, segmentation feature extractor, 2D slice compatible |
| **MedDINOv3** | 2025 | General medical segmentation | Probably | Unknown | Medium — DINOv3 medical adaptation, segmentation focus |
| **OmniMRI** | 2025 | General MRI VLM | ❌ Not found | — | Medium-high in principle, but no public weights yet |
| **MedImageInsight** | 2025–2026 | General medical imaging | ✅ Azure model catalog | Commercial/MS license | Medium — general embedding model, not knee-specific |
| **Pillar-0** | 2025 | General radiology | Unknown | Unknown | Medium — radiology FM, mostly CT/chest emphasis in public materials |
| **Triad** | 2025 | 3D medical (MRI incl.) | Unknown | Unknown | Medium — 3D VFM, no knee-specific eval found |
| **CARVE** | 2026 | 3D medical volumes | Unknown | Unknown | Low-medium — 3D volume understanding, not knee-specific |

## 1. OrthoFoundation (2026) — the strongest candidate

[OrthoFoundation](https://github.com/ytrsk/OrthoFoundation) is a large-scale multimodal vision foundation model for musculoskeletal diagnosis and prognosis. It is the most relevant published model for our task because it was trained almost exclusively on knee images.

- **Architecture:** Vision Transformer initialized with DINOv3-L weights, then continued with DINO-style self-supervised learning.
- **Pretraining data:** 1,251,655 knee images — 357,670 radiographs and 893,985 MRI slices — from OAI, fastMRI, and a private multicenter cohort (Peking University Third Hospital).
- **License:** MIT.
- **Weights:** Released as `OrthoFoudation-L.pth` in the GitHub repository.
- **Reported knee-MRI results:**

| Task | Metric | OrthoFoundation |
|------|--------|-----------------|
| ACL injury | AUROC | **98.49** |
| PCL injury | AUROC | **84.31** |
| MCL injury | AUROC | **93.34** |
| Meniscus injury | Accuracy | **84.74** |
| Cartilage lesion | AUROC | **93.38** |

([Yu et al., 2026, *OrthoFoundation: A multimodal vision foundation model for generalizable knee pathology*](https://arxiv.org/pdf/2601.18250))

**Why it matters for us.** OrthoFoundation is a drop-in replacement candidate for our 2D trunk. It is already initialized from DINOv3-L and then further pretrained on the exact same public corpora we have local (OAI + fastMRI slices). If its representations are better for knee pathology than generic DINOv3, we can load `OrthoFoudation-L.pth` as the trunk and keep the rest of our architecture (slice pooling, slot heads, ensemble) unchanged. It also gives us a strong baseline to compare against if we continue pretraining our own DINOv3 on the same data.

**Risks.** The model is slice-based (2D), like our champion, so it will not fix any 3D-context limitation. The reported numbers are on internal test sets and may not transfer directly to the RSNA label definitions (e.g., RSNA excludes low-grade ACL sprains). And the private PUTH cohort is not reproducible, though the public portion is large enough to matter.

## 2. MRI-CORE (2024) — general MRI feature extractor

[MRI-CORE](https://github.com/mazurowski-lab/mri_foundation) is a SAM-based foundation model trained for MRI. It is anatomy-agnostic but designed specifically for MRI contrast and geometry.

- **Architecture:** Segment Anything Model (SAM) ViT-B image encoder, adapted for MRI.
- **Weights:** `MRI_CORE_vitb.pth` available from the repository.
- **License:** Apache 2.0.
- **Use case:** Few-shot segmentation, linear probing, feature extraction.
- **Input:** 2D slices normalized to [0, 1], 1024×1024 by default.

([Dong et al., 2024, *MRI-CORE: A Foundation Model for Magnetic Resonance Imaging*](https://github.com/mazurowski-lab/mri_foundation))

**Why it matters for us.** MRI-CORE can provide a strong MRI-specific feature extractor. Unlike OrthoFoundation, it is not knee-specific, but it is MRI-specific and has a permissive license. It is most useful if we want a segmentation pretraining signal (e.g., for CSNet-style surface networks or for adding a segmentation auxiliary task). For pure 12-label classification, OrthoFoundation is likely better.

## 3. MedDINOv3 (2025) — DINOv3 medical adaptation

MedDINOv3 investigates how to adapt Meta's DINOv3 for medical image segmentation. The paper identifies challenges (e.g., domain gap, patch resolution, global-vs-local features) and proposes adaptation strategies.

- **Base:** DINOv3.
- **Focus:** Medical segmentation.
- **Availability:** Paper on arXiv; weights/checkpoints not confirmed in our search.

([MedDINOv3, 2025, *How to adapt vision foundation models for medical image segmentation?*](https://arxiv.org/html/2509.02379v1))

**Why it matters for us.** If DINOv3 is already our trunk, MedDINOv3's findings tell us how to adapt it: which layers to freeze, how to handle patch size, and how much medical pretraining is needed. It supports the strategy of continuing our own DINOv3 pretraining on OAI+fastMRI slices rather than replacing the trunk.

## 4. OmniMRI (2025) — generalist MRI VLM

OmniMRI is a unified vision-language foundation model trained on 60 public MRI datasets (220,000 volumes, 19 million slices). It can perform reconstruction, segmentation, detection, diagnosis, and report generation from a single model.

- **Architecture:** Swin Transformer vision encoder + Qwen2.5 autoregressive Transformer + diffusion image decoder + text decoder.
- **Data:** 60 public datasets, including knee (12% of anatomy mix).
- **Weights:** Not found in our search.

([He et al., 2025, *OmniMRI: A Unified Vision–Language Foundation Model for Generalist MRI Interpretation*](https://arxiv.org/pdf/2508.17524))

**Why it matters for us.** If weights become available, OmniMRI could be used as a feature extractor or even as a zero-shot/radiology-report generator for pseudo-labeling. Without weights, it is only a signal that the field is moving toward VLMs. Keep on watchlist.

## 5. MedImageInsight (2025–2026) — Microsoft general medical embedding model

MedImageInsight is a 360M-parameter image encoder + 252M-parameter language encoder released through the Azure AI model catalog. It is a general-domain medical imaging embedding model, not knee-specific.

- **Availability:** Azure model catalog.
- **License:** Microsoft commercial/research license.

([Azure — MedImageInsight](https://ai.azure.com/catalog/models/MedImageInsight))

**Why it matters for us.** It is an alternative general embedding model, but it is unlikely to beat OrthoFoundation on knee MRI because it is not trained on a knee-heavy corpus. Useful only if we want a general baseline or if we are already using Microsoft Azure tooling.

## 6. Pillar-0, Triad, CARVE — watchlist

- **Pillar-0** (2025): a radiology foundation model pretrained on diverse imaging datasets. Public materials emphasize CT and chest; no clear knee-MRI eval or weights found yet.
- **Triad** (2025): a 3D vision foundation model for MRI. Potentially relevant for a 3D branch, but no knee-specific pretrained checkpoint identified.
- **CARVE** (2026): cross-slice anisotropic reallocation for 3D medical volume understanding. Interesting for 3D aggregation, but no public weights yet.

These are worth monitoring but not actionable today.

## 7. Recommendation

Given our current champion is a DINOv2/DINOv3 slice ensemble and we have fastMRI + MRNet + OAI on disk, the priority order is:

1. **OrthoFoundation-L** as trunk replacement. Load the released checkpoint, freeze or LP-FT, and measure RSNA holdout AUC. This is the single highest-impact model change we can make with publicly available weights.
2. **Continue-pretrain our own DINOv3-L** on OAI + fastMRI slices (the same public corpora OrthoFoundation used) if OrthoFoundation's private PUTH data is a concern or if we want a reproducible pipeline. Use the MedDINOv3 paper's adaptation recipe.
3. **MRI-CORE** as a segmentation/feature auxiliary, especially if we add a 3D segmentation pretraining branch using SKM-TEA masks.
4. **Watch OmniMRI / Pillar-0 / Triad** for public weight releases; they are not usable today.

## 8. Concrete next step

Download `OrthoFoudation-L.pth` and run a smoke experiment: replace the DINOv3 trunk in one of our smaller 224 px arms with OrthoFoundation-L, freeze the trunk, train only the heads, and compare RSNA holdout AUC against the same arm with generic DINOv3-L. If the gain is positive, the model becomes the new default trunk and the external-data merge (Scheme B) is built on top of it rather than on generic DINOv3.
