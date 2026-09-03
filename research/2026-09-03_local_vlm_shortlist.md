# Local vision+text model shortlist for knee-MRI label audit

**Date:** 2026-09-03  
**Constraint:** RTX 5090 32 GB (Blackwell, NVFP4-capable).  
**Goal:** one local model that can read both radiology reports and knee-MRI images.

## The Blackwell angle changes the list

On a 5090 the best speed/quality trade-off is usually an **NVFP4** weight (Blackwell-native 4-bit FP, ~97–99 % of BF16 quality) served through vLLM/SGLang, not a GGUF/Q4 in llama.cpp.  The original shortlist was Qwen2.5-VL-era; on Blackwell the practical question becomes: *which model has an NVFP4 checkpoint and actually loads on your card?*

| Model | Params | Format / VRAM | Vision? | Medically tuned? | NVFP4 on HF? | Verdict | License |
|---|---:|---|:---:|:---:|:---:|---|---|
| **Qwen3.8-27B** | 27 B | NVFP4 ~20–22 GB ([unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)) | ✅ | general | ✅ | **Best single-model candidate** on this GPU; 128k context, already has FP4 kernels, handles text + images | Apache 2.0 |
| **Qwen3.8-27B** | 27 B | GGUF Q4 ~16–18 GB | ✅ | general | ❌ | Fallback if NVFP4 vLLM path has issues; slower but proven stack | Apache 2.0 |
| **MedGemma-1.5-4B-IT** | 4 B | PyTorch/GGUF ~2–3 GB ([google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)) | ✅ | ✅ (radiology/chest/derm/path) | ❌ | **Cheap medical sanity-check / candidate-findings model**; maintained, 1000× more usage than Lingshu | Gemma / HADF terms |
| **MedGemma-27B-IT** (multimodal) | 27 B | GGUF Q4 ~16–17 GB | ✅ | ✅ | ❌ | Stronger medical VLM, but no NVFP4 and heavy; only if 4B is too weak | Gemma / HADF terms |
| **MedGemma-27B-text-IT** | 27 B | GPTQ-Int4 / GGUF ~16 GB | ❌ | ✅ | ❌ | Purpose-built for medical text, but Qwen3.8-27B likely matches or beats it on extraction/consistency | Gemma / HADF terms |
| **Lingshu-7B / 32B** | 7 / 32 B | GGUF / MLX only (~5–6 GB / ~18–20 GB) | ✅ | ✅ | ❌ | Research artifact — only 698 HF downloads for 32B; skip for a production-ish pipeline on Blackwell | MIT |
| **Qwen2.5-VL-7B / 32B** | 7 / 32 B | GGUF Q4 (~5–6 GB / ~18–20 GB) | ✅ | general | ❌ | Good general VLMs, but superseded by Qwen3.8 on this hardware | Apache 2.0 |

Sources: Qwen3.8-27B NVFP4 ~20 GB and Blackwell-only from Unsloth docs [[1]](https://unsloth.ai/docs/models/qwen3.8); MedGemma-27B Q4 ~16.5 GB [[2]](https://modelscope.cn/models/bartowski/google_medgemma-27b-it-GGUF); MedGemma 1.5 4B release notes [[3]](https://modelscope.cn/models/google/medgemma-1.5-4b-it); Lingshu low adoption and no NVFP4 observed on HF search.

## Split the use cases

| Task | Right tool | Why |
|---|---|---|
| **Report text re-reading / label audit** | Qwen3.8-27B NVFP4 (text-only endpoint is enough) | 128k context, FP4 speed, already deployed; adding a separate text-only MedGemma engine is redundant |
| **Sanity-check labels against images, generate candidate findings text** | MedGemma-1.5-4B-IT first, then Qwen3.8-27B if quality is insufficient | 4B is cheap to try and medically tuned; Qwen3.8-27B is the fallback that also sees images |
| **Actual Kaggle score improvement from MRI features** | Task-specific 3D CNN/ViT trained on competition data | VLMs are not the medal path for MRI/CT; they can assist labeling, not replace a trunk |

## Recommendation

If you really want **one model**, use **Qwen3.8-27B NVFP4**.  It does text + vision, fits on the 5090, and has community NVFP4 weights.  Its weakness is that it is not medically fine-tuned; for pure label-audit text work that probably does not matter, but for image sanity checks you may prefer MedGemma-1.5-4B-IT as a cheap second opinion.

A pragmatic two-model setup (which still avoids the old Qwen3.8-27B text-only + separate medical VLM bloat):
1. **Qwen3.8-27B NVFP4** for report text and any bulk extraction.
2. **MedGemma-1.5-4B-IT** for spot-checking images / generating candidate findings strings.

## Open questions / next steps

- Verify the local vLLM NVFP4 container actually loads on SM120 with FP4 benefit (vLLM issue #47749 warns of silent Marlin W4A16 fallback).
- Run a small head-to-head on knee MRI slices: Qwen3.8-27B NVFP4 vs MedGemma-1.5-4B-IT vs (if available) MedGemma-27B-IT for label-image consistency.
- Continue separate research into 3D knee-MRI architectures that can actually move the Kaggle score (task-specific CNN/ViT, not VLMs).
