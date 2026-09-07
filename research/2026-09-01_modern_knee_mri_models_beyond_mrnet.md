# Modern knee-MRI foundation models beyond MRNet

2026-09-01, Kimi. Follow-up to the owner's steer that MRNet is 8 years old and there should be newer/better public knee-MRI transfer sources.

## TL;DR

**OrthoFoundation-L is the standout.** It is a 2026 DINOv3-L-based model, MIT-licensed, with released weights, pretrained on **1.25 million knee images** (357k X-rays + 894k MRI slices) from OAI/fastMRI plus a private cohort. It reports strong knee-MRI downstream numbers (ACL AUROC 0.9849, MCL 0.9334, cartilage lesion 0.9338) and is exactly the "strong init + in-domain SSL" mechanism the project has been trying to price. It is **not yet referenced anywhere in this repo**.

The repo already tracks RAD-DINO (knee-vp8q), DINOv3 (knee-8hn), and the MRNet/fMRI/KMAR-50K corpora in `docs/EXTERNAL_CORPORA.md` and `research/2026-09-01_knee-mri-corpora-survey.md`, but those are data sources or general encoders. OrthoFoundation is a *knee-specific* foundation model.

## Candidate models

### OrthoFoundation (2026) — knee-specific, DINOv3-L, weights available

* Paper: [arXiv:2601.18250](https://arxiv.org/abs/2601.18250)
* Code/weights: [github.com/ytrsk/OrthoFoundation](https://github.com/ytrsk/OrthoFoundation), `OrthoFoudation-L.pth`
* Licence: **MIT**
* Architecture: ViT-L/14 initialized from DINOv3-L, continued with DINO-style student-teacher SSL.
* Pretraining corpus: 1,251,655 knee images — 357,670 radiographs + 893,985 MRI slices. Public portion is **~600k samples** from OAI (~100k X-rays, ~200k MRI slices) and fastMRI (~300k MRI slices); the rest is a private multicenter cohort.
* Reported knee-MRI downstream (full fine-tuning):
  * ACL injury — AUROC 98.49
  * PCL injury — AUROC 84.31
  * MCL injury — AUROC 93.34
  * Meniscus injury — Accuracy 84.74
  * Cartilage lesion — AUROC 93.38
* Why it matters here: it is the same DINOv3 family the project already mounts, so the engineering delta is "swap the checkpoint" rather than a new encoder pipeline. The public data overlap with fastMRI/OAI means the licence surface is similar to data already under review.

### OAI-DINO (ISMRM 2025 #4014) — no public numbers

* ISMRM 2025 abstract #4014. DINO SSL pretraining on OAI 3D knee MRI.
* Public abstract claims improvement over training-from-scratch and supervised pretraining on downstream tasks.
* No weights found, no quantitative results in the public abstract. Not actionable until a paper or checkpoint appears.

### OrthoDiffusion (2026) — diffusion-based, weights on Kaggle, licence VERIFIED MIT (2026-09-07)

* Paper: [arXiv:2602.20752](https://arxiv.org/abs/2602.20752)
* Code: [github.com/lt-0123/OrthoDiffusion](https://github.com/lt-0123/OrthoDiffusion)
* Weights: [Kaggle dataset](https://www.kaggle.com/datasets/prvsiyan/orthodiffusion-official-knee-mri-foundation) (not an official Kaggle model mirror)
* Pretraining corpus: 15,948 unlabeled knee MRI scans, three orientation-specific 3D diffusion models.
* Downstream: segmentation of 11 knee structures + 8 abnormality labels; cross-joint transfer to ankle/shoulder.
* **Licence: PASSES.** The authors' own Hugging Face release `lanstat0123/orthodiffusion` carries **MIT**, so the weights are usable and the Kaggle re-upload is not the licensing object — pull from HF, not from the community dataset. This corrects the "licence unclear" reading this section carried until 2026-09-07; it was never a blocker.
* The remaining caveat is architectural, not legal: it is a 3D diffusion pipeline, not a drop-in 2D encoder. A frozen OrthoDiffusion trunk carried 0.7537 in LP-FT stage one and was declined on that number (`comp/members.declines:289-293`) — re-opened by a different recipe, never by re-quoting the licence.

### MRI-CORE (2025) — general MRI, SAM-based, Apache 2.0

* Paper: [arXiv:2506.12186](https://arxiv.org/abs/2506.12186)
* Code/weights: [github.com/mazurowski-lab/mri_foundation](https://github.com/mazurowski-lab/mri_foundation), Apache 2.0
* Corpus: >110k MRI volumes / >6M slices across 18 body locations; knee is one of the included anatomies.
* Architecture: SAM ViT-B with adapter-tuned encoders; designed for few-shot/zero-shot segmentation.
* Caveat: it is a segmentation-first model, not a classification encoder. Integrating it would mean replacing the DINO/V-JEPA backbone with a SAM image encoder and likely rebuilding the head. The Apache licence is clean, but the payoff is far less direct than OrthoFoundation.

### PRISM (2025) — multi-sequence general MRI, strong results, release status unclear

* Paper: [arXiv:2508.07165](https://arxiv.org/abs/2508.07165)
* Corpus: 336,476 volumetric MRI scans from 34 datasets (8 public, 26 private), multi-organ/multi-sequence.
* Benchmark: first place on 39/44 downstream tasks.
* Caveat: a quick search did not surface a public GitHub repository or weights for this exact work. Treat as promising but not yet usable until a release is located.

## What the project already knows

* `docs/EXTERNAL_CORPORA.md` documents SKM-TEA, fastMRI, KMAR-50K, OAI/MOST, and segmentation-era datasets. It does not list foundation-model checkpoints.
* `research/2026-09-01_ssl_continuation_from_strong_init.md` shows that continuing DINO/DINOv2/DINOv3 on in-domain unlabeled data can help, but only at large scale (30k–3.5M images). OrthoFoundation's 1.25M knee images sits at the high end of that evidence.
* Beads:
  * `knee-ombx` — MRNet as external data for supervised encoder pretraining, blocked on the owner licence call.
  * `knee-vp8q` — RAD-DINO (chest radiography init) as a partial-domain-match control.
  * `knee-8hn` — DINOv3 access/licence question.
  * No bead exists for OrthoFoundation or any knee-specific foundation model.

## Recommended next steps

1. **Treat OrthoFoundation-L as the next encoder-init experiment.** It is a direct replacement for the DINOv3-L backbone, MIT-licensed, with public weights. The control is `dinov3_base_lora_s2026`/`dinov3_large` paths; the question is whether a knee-continued DINOv3-L buys more than the raw DINOv3-L checkpoint.
2. **Resolve the Kaggle-shippability question before training.** The weights are MIT, but the base is DINOv3-L (Meta licence). The same DINOv3 Kaggle-mount issue already tracked in `knee-8hn` applies.
3. **Skip OAI-DINO until a checkpoint or numbers appear.** It is too thin to act on.
4. **OrthoDiffusion's licence is verified MIT (HF `lanstat0123/orthodiffusion`) and is not the reason to defer it.** What defers it is the measurement: a frozen trunk reads 0.7537 in LP-FT stage one, and it is a 3D diffusion pipeline rather than a drop-in 2D encoder.
5. **Keep MRI-CORE/PRISM on watch.** MRI-CORE is clean but segmentation-first; PRISM is strong but weights are not yet located.
