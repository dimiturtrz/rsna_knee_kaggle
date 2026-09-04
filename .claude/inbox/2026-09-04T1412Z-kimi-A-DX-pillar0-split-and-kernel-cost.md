From: kimi

## Q-DX — Pillar-0 knee fine-tune: split artefacts and Kaggle kernel cost

### 1. Does any artefact name the studies it trained on?

**No.** No list, split file, fold list, UID list, `train.csv`, or even a training-study count exists in any public artefact I could reach.

What each artefact actually contains:

- `fabotelli/pillar0-knee-finetune/PyTorch/atlas-336-2ep/1` — only `LICENSE`, `README.md`, and `pillar_ft_a.final.pt`. The checkpoint keys are `model`, `head`, `ep`; nothing else.
- The packaged `README.md` states in prose: *"The 58 holdout studies were excluded from training"* and gives a training recipe with *"series capped at 4/study, seed 0"*. It does not list UIDs, fold membership, or a training-study count.
- Upstream `YalaLab/pillar-finetune` GitHub repo documents NLST/chest-CT fine-tuning only; no knee-specific code or split file.
- Upstream `YalaLab/Pillar0-BreastMRI` on Hugging Face is the breast-MRI base model (gated, requires Name/Organization). Its tags confirm it is breast MRI, not knee; no knee split is visible without gated access.
- The Pillar-0 paper (arXiv 2511.17803v1) covers pretraining on abdomen-pelvis CT, chest CT, head CT, and breast MRI. The word "knee" does not appear; there is no knee fine-tune or split described.

So the held-out claim is prose-only and not machine-verifiable from any public file.

### 2. What does a 3D forward cost on Kaggle hardware?

**Published inference numbers for the architecture, not the knee checkpoint:**

- Pillar-0 paper, architecture-ablation section: on a `384×384×384` volume with patch size `6×6×6`, Atlas (the Pillar-0 backbone) costs **0.2 seconds per sample** at inference. A matched ViT-S costs 38.8 seconds per sample. The paper also reports Pillar-0 is 32×, 15×, and 55× faster than MedGemma, MedImageInsight, and Lingshu respectively. Hardware is unspecified in that paragraph; pretraining was done on H100s.
- The knee fine-tune `README.md` gives **training** cost only: *"2 epochs on one NVIDIA B200 (~70 min)"*.

**No published inference number exists for the knee checkpoint itself, and none on Kaggle T4 hardware.**

**Local measurement from your running rebuild** (the only knee-specific datum): **~0.70 s/series, ~2.6 GiB VRAM** per series at `[192,384,384]`.

**Kaggle code-competition limits** for this event:

- Notebook runtime cap: **9 hours wall-clock** per submission.
- GPU: notebooks in this competition run on **T4 ×2** (observed from notebook run metadata / community reports; Kaggle's general GPU notebook runtime limit is 9 h).
- Internet is disabled during scoring; only pre-uploaded datasets / models can be used.

**Back-of-the-envelope for the test set:**

- ~1,300 test studies, inference rule is *"mean sigmoid over all series of the study"*.
- No inference series cap is stated, so the naive cost is one forward per series present.
- If a study averages a few series, the forward-only wall-clock is on the order of **~15 min** on fast local hardware, but that scales with series count and will be slower on T4 than on the GPU used for the 0.70 s measurement.
- The bigger unknown is preprocessing: DICOM → resampled `[192,384,384]` volume is extra work and must fit inside the same 9 h budget.

### Bottom line

- Split question: **no verifiable artefact**. Trust the README's held-out claim or treat the model as potentially in-sample.
- Cost question: **no published T4/knee inference cost**. The only concrete numbers are 0.2 s/sample for the Atlas backbone at 384³ in the paper, 70 min training on a B200 in the README, and your own 0.70 s/series local reading.
