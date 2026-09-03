# State of the label lane and the model axes — 2026-09-03

Research synthesis for the standing question: "Labels are supposedly not the problem — are they really closed, and what model axes are still alive?"

## Labels: the lane is mostly closed, but three small doors are still open

The training labels are **report-derived** (`processed/.../labels/report_labels_v2.csv` and blend tables). The 58 studies in `train.csv` with all twelve finding columns filled are **radiologist image-read gold annotations**, not training labels. Every label correction is judged against those 58 (now 57 after one missing-row defect).

### What has been measured

- **The report reader is the weaker half.** On gold, the model fitted on report labels beats the labels themselves as a predictor by **0.047 AUC** over 696 cells (`interpretations/kneemri/2026-09-03_the-report-reader-is-the-weaker-half.md`). The gap is concentrated in Synovitis (+0.1075), Fracture (+0.0715), Lateral Meniscus (+0.0522) and Baker's (+0.0462).
- **Only one label correction survives.** `blend_hybrid_fx.csv` applies `Fracture := monotone_rescale(Fracture + 0.25 · Contusion)`. It is worth **+0.0632 on the Fracture column** and **+0.0053 on the twelve-label mean** (`interpretations/kneemri/2026-09-02_what-label-correction-is-actually-worth.md`). Every other global/per-finding correction tried so far is null or artifact.
- **The rubric re-read is built but unpriced on gold.** `core.data.labels.table_correction` can apply the published rubric to 502 English reports with backed contradictions; only 5 of those cells fall inside the gold 57, so a gold reading cannot judge it. The same is true for the negation pass.

### What is still actionable

1. **The missing gold study.** `report_labels_v2.csv` has no row for `1.2.826.0.1.3680043.8.498.73527530686853911124431549317032662220`, which is gold-positive for ACL, PF OA, Effusion, Contusion and Fracture. That is a pipeline defect, not a parser disagreement.
2. **Effusion over-call.** The audit shows a **70% false-positive rate** on gold negatives: report phrases like "small effusion" / "trace fluid" are read as positive while the annotator wants moderate/large distension.
3. **Synovitis under-call.** **52% false-negative rate** on gold positives; the reports often describe it implicitly or not at all.
4. **Multilingual rubric pass.** The English-only rubric correction touches 502 cells; expanding the subthreshold vocabulary to es/tr/hr/el/de/bg/nl would reach roughly **1,250 cells** at zero GPU cost.

### Bottom line on labels

The big label-correction axis is closed: the model already exceeds the supervision on the thin findings, and the surviving Fracture fix is worth only ~0.005 on the mean. The remaining items are targeted parser fixes and a multilingual rule pass. They should not be the main GPU bet, but they are cheap enough to queue.

## Models: the monoculture is the live problem, and two public bets are still unpriced

### What is closed

- **Ensemble/seat axis.** Sixteen board entries spanning 6-12 seats and every mechanism the project owns fit inside **0.005** (`interpretations/kneemri/2026-08-31_the-board-names-its-own-price.md`, bead `knee-va22`).
- **External-corpus seat axis.** One, two and four external-corpus seats read the same board value; six reads worse (`interpretations/kneemri/2026-09-02_the-external-corpus-seat-axis-closes.md`). Adding external studies as a *seating* lever is closed.
- **OrthoFoundation-L as a 2D trunk.** `ortho_dense_s2026` read **0.8452** vs its DINOv2-base control **0.8502**; the frozen probe showed the knee-specific continuation **lost signal** vs. DINOv3-L itself (`research/2026-09-02_sota_pretrained_knee_mri_models_2025_2026.md`).

### What is still open

1. **OrthoDiffusion (3D diffusion foundation model).** The first public knee-specific 3D foundation model with multi-label diagnosis weights: 15,948 knee MRIs, macro AUROC 0.908 on 8 abnormalities, 5 of 8 map directly to RSNA labels. It has not been smoke-tested in this repo. This is the single highest-impact untried public model.
2. **Slice coverage.** The public frontier reads **64 slices** per study; this project's 137 arm scripts top out at **20 slices**. A 224 px probe that reads all slices is affordable (~24 GB VRAM) and is the one mechanism that cannot miss a thin finding between samples (`research/2026-08-31_public-frontier-sweep.md`).
3. **External data as trunk pretraining, not seating.** The seat-count route is closed, but a trunk *continued* on external knee-MRI images is a different mechanism. The literature supports partial-label multi-task pretraining on MRNet + fastMRI+ + Rijeka with masked loss, LP-FT, L2-SP and domain mixing (`research/2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score.md`). Expected mean-AUC gain **+0.005 to +0.012**.
4. **Local vision-language re-read of reports.** Untried. Could help the four findings where the report reader loses, but only where the report actually contains text (Synovitis and Fracture are mostly silent; Baker's and Contusion are the realistic targets).

### Recommended priority

1. Smoke-test **OrthoDiffusion** (download HuggingFace weights, extract per-plane features, train 12-label head).
2. Arm **slice-coverage probe at 224 px** reading the full sagittal stack.
3. Queue **partial-label multi-task pretraining (Scheme B)** as the external-data trunk experiment.
4. Fix the **missing gold-study row** and the **Effusion/Synovitis parser targets** as zero-card side work.
5. Keep **multilingual rubric pass** as a language-task, not a GPU task.

## Sources

- `interpretations/kneemri/2026-09-03_the-report-reader-is-the-weaker-half.md`
- `interpretations/kneemri/2026-09-02_what-label-correction-is-actually-worth.md`
- `interpretations/kneemri/2026-09-02_report-to-label_audit.md`
- `interpretations/kneemri/2026-09-02_the-external-corpus-seat-axis-closes.md`
- `interpretations/kneemri/2026-09-01_external-data-state.md`
- `research/2026-09-02_sota_pretrained_knee_mri_models_2025_2026.md`
- `research/2026-09-02_modern_3d_knee_mri_models_beyond_mrnet.md`
- `research/2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score.md`
- `research/2026-08-31_public-frontier-sweep.md`
