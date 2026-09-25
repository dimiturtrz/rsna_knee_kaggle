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
| **MedGemma-27B-IT** (multimodal) | 27 B | **FP8-Dynamic** ([ig1](https://huggingface.co/ig1/medgemma-27b-it-FP8-Dynamic), 1,927 dl), ~27 GB; GGUF / MLX / bnb-4bit ~16–17 GB | ✅ | ✅ | ❌ | **FP8 only** — no GPTQ, no AWQ, no W4A16, no NVFP4. FP8 is Blackwell-native so the FP8 row is a real vLLM path. This cell was wrong twice: first as "GGUF/MLX only, zero vLLM-loadable quants" (an `fp8` repo-id census read as covering the text variant only; two of its rows are the multimodal one), then briefly over-corrected. Cited above so the third reading is checkable | Gemma / HADF terms |
| **MedGemma-27B-text-IT** | 27 B | bitsandbytes-4bit only, ~16 GB | ❌ | ✅ | ❌ | **No GPTQ, no AWQ, no W4A16, no FP8, no NVFP4** — an HF sweep of 51 quantized repos found only GGUF, MLX and bnb. bnb loads in vLLM but on the slow path (no Marlin/Machete kernels), and the two repos that exist have 740 and 480 downloads | Gemma / HADF terms |
| **gemma-3-27b-it** | 27 B | GPTQ-4b-128g ~16 GB ([ISTA-DASLab](https://huggingface.co/ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g), 641,658 dl); FP8-Dynamic ([leon-se](https://huggingface.co/leon-se/gemma-3-27b-it-FP8-Dynamic)) | ✅ | general | ❌ | **The decorrelation control.** Same architecture family as MedGemma minus the medical tuning, first-class vLLM support, no self-quantizing. If it matches MedGemma-27B-text on the annotated studies the medical finetune bought nothing on report-reading; if it does not, that difference IS what medical tuning is worth here. Either sign is a reading | Gemma terms |
| **gemma-3-12b-it** | 12 B | int4-AWQ (233,873 dl) / W4A16 (111,507 dl) | ✅ | general | ❌ | Smaller tier if 27B cannot be co-scheduled. Note MedGemma has **no 12B** — 4B and 27B only, so there is no medical counterpart to control against at this size | Gemma terms |
| **MedGPT-oss-20B**, **Hulu-Med** | 20 B / — | GGUF only (max 383 dl) | — | ✅ | ❌ | Dead for vLLM. Shortlisted on benchmark strength, dropped on availability | — |
| **Lingshu-7B / 32B** | 7 / 32 B | GGUF / MLX only (~5–6 GB / ~18–20 GB) | ✅ | ✅ | ❌ | Research artifact — only 698 HF downloads for 32B; skip for a production-ish pipeline on Blackwell | MIT |
| **Qwen2.5-VL-7B / 32B** | 7 / 32 B | GGUF Q4 (~5–6 GB / ~18–20 GB) | ✅ | general | ❌ | Good general VLMs, but superseded by Qwen3.8 on this hardware | Apache 2.0 |

**THE WHOLE TABLE IS SCOPED TO vLLM, and that scope is doing more work than it looks.** Every "not
runnable" verdict above means *no vLLM-loadable quant exists*, not *the model does not run here*. Through
**transformers + bitsandbytes** the constraint lifts: `medgemma-4b-it` is bf16 at ~9 GB and never needed a
quant at all, and `medgemma-27b-it` multimodal loads in bnb-4bit at ~16-17 GB. The serving stack was
confused for the model's availability once already in this file (see the withdrawal note below), so the
distinction is stated here rather than left implicit: the vLLM column prices THROUGHPUT, and a one-off
audit over 58 studies does not need throughput.

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
2. ~~**MedGemma-1.5-4B-IT** for spot-checking images / generating candidate findings strings.~~
   **WITHDRAWN 2026-09-03 as a GENERATOR, and the withdrawal was then PARTLY RETRACTED the same day.**
   The argument below still holds for what it was about — free-text candidate generation — and it still
   prices any future VLM proposal: **as a candidate GENERATOR only, never as an adjudicator.** What it
   does NOT establish, and was briefly read as establishing, is that MedGemma is disqualified from the
   label audit. Two corrections, both raised by `claude-main` and accepted here:
   - The hallucination and verbosity figures are from **free-text report generation**. The audit task is
     constrained per-finding extraction against a fixed `YES`/`NO`/`UNK` grammar over twelve named
     targets. A narrative-generation metric does not price a structured-extraction use, and **nobody has
     measured whether MedGemma over-calls under constraint**. That is now a question for the run, not a
     settled fact.
   - The "no vLLM-loadable checkpoint" census in the table above kills the **27B only**. `medgemma-4b-it`
     is 4 B, bf16 is ~9 GB, and it never needed a quant — a 27B availability finding was allowed to bleed
     onto the 4B. Through transformers rather than vLLM the constraint lifts further: `medgemma-27b-it`
     multimodal loads in bnb-4bit at ~16-17 GB.
   The over-calling framing survives as what AIMS the run — errors should land where we can see them —
   not as what kills a model. A radiologist reader study
   (arXiv:2512.00271 / Eur Radiol Jun 2026; 478 ED patients, same-day CXR+CT with CT as ground truth,
   three blinded thoracic radiologists, no fine-tuning) scored `google/medgemma-4b-it` at 71.4% clinical
   acceptability against radiologists' 74.3% — close — but at a **5.4% hallucination rate against 0.1%**,
   and it reaches that acceptability partly by emitting 275-word reports against a 5-word human median.
   That is high recall by carpet-bombing. Taken at face value it injects false findings at ~5%, which
   means auditing labels with a hallucinating auditor. Every positive it emits gets vetoed by a second
   reader. Two caveats bind: the study is CHEST X-RAY, where the medical pretraining corpus is dense, so
   71.4% does not port across anatomies and its knee behaviour is unmeasured and likely worse; and our own
   label defect is OVER-CALLING (`reader_agreement`: 108 confident false positives, 15.8% of every
   annotated cell, against 32 misses of which only 8 are a confident NO), so a carpet-bombing reader is
   aimed at the end we already have too much of. Point it at the over-called cells (`knee-pnwar`), where
   its errors land in a direction we can see.

## Open questions / next steps

- ~~Verify the local vLLM NVFP4 container actually loads on SM120 with FP4 benefit (vLLM issue #47749
  warns of silent Marlin W4A16 fallback).~~ **RESOLVED 2026-09-03.** Engine logs on the deployed container
  show `[AutoTuner]: Tuning fp4_gemm` across six tuning passes, so real FP4 kernels are active and it is
  not the Marlin fallback. KV cache 60,800 tokens, max concurrency 1.57x at 131,072 ctx.
- Run a head-to-head on the 58 annotated studies: **gemma-3-27b-it-GPTQ against MedGemma**, same harness,
  same `chat_template_kwargs: {"enable_thinking": false}` discipline. **Do NOT baseline it against our
  Qwen3.8 candidate table** — `knee-8hyr6` is a P0 saying that run FAILED (687 of 696 cells UNK, 57 of 58
  studies parsed nothing), so a comparison against it measures the artefact and reads the difference as
  medical tuning.
- **The census pattern, worth stating on its own:** general models are quantized to death and medical
  finetunes are not. Every medical checkpoint here is GGUF/MLX-only with download counts in the hundreds,
  while their general siblings ship GPTQ/AWQ/FP8 with counts in the hundreds of thousands. Nobody is
  serving the medical models in production, which is a fact about tooling support rather than about model
  quality — but it prices every medical option's real cost far above its benchmark row.
- **Medical-model family diversity is far thinner than the model count suggests**, and it prices any
  medical ensemble either lane builds. `Baichuan-M2-32B` reports `Qwen2ForCausalLM` with vocab 152064 —
  it is Qwen 2.5 under another name, and it was on a roster as a "second family" until that was checked.
  Lingshu, Hulu-Med and II-Medical are Qwen too. Outside Gemma and Qwen the whole medical space is
  `Baichuan-M1-14B` (own arch), `HuatuoGPT-Vision` (Yi/LLaVA), `OpenBioLLM` (Llama-3) and `MedFound`
  (Bloom, 2022). So a five-model medical roster can easily be two families wearing five names — the same
  correlation trap `distinct14` measured on our own pool, one level up. Census by `architectures` in
  `config.json`, never by the model card.
- `ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g` reports `Gemma3ForConditionalGeneration` with a live
  `vision_config` and ships `preprocessor_config.json` — **the GPTQ control keeps its vision tower**, so
  it stays a valid control if the audit ever goes multimodal.
- Continue separate research into 3D knee-MRI architectures that can actually move the Kaggle score (task-specific CNN/ViT, not VLMs).
