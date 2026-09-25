# External data and foundation-model status quo — 2026-09-02

Kimi, read-only auditor/researcher session. This note ties together the
measurements already on disk, the exact foreign-data inventory, and the
actionable foundation-model landscape so the next arm decisions are grounded.

## 1. What has already been tried and what it read

### 1.1 External-data corpora

| corpus | decoded / on disk | labels vs RSNA 12 | measured result |
|---|---|---|---|
| Rijeka (kneeMRI) | 736 studies decoded | ACL severity only | `merged_blend_s2026` +0.0063 mean AUC; Lateral Meniscus +0.0196*, ACL +0.0161 |
| MRNet | 1,250 exams, 3 planes | abnormal / ACL / meniscus | part of three-corpus cache (+1,250 to merged); no separate member score yet |
| fastMRI+ | 1,173 reviewed, 974 with boxes | graded ACL/MCL/effusion/meniscus-any/cartilage/bone-edema | used only as unlabelled images so far (`fastmri_continuation_arithmetic`) |
| KMAR-50K | 1,444 scans / 62k slices | none | SSL candidate only |

The merged-corpus thread (`experiments/README.md`, `interpretations/kneemri/2026-09-01_external-data-state.md`) already proves that foreign knee MRI helps, but the gain is small (+0.002 to +0.008) and concentrated in ACL/meniscus. The open question is whether a **unified multi-task partial-label trainer** (Scheme B) can turn the richer fastMRI+ boxes and MRNet labels into a bigger gain than the simple merged cache did.

### 1.2 Foundation-model trunks

| trunk | frozen probe mean AUC | fitted member mean AUC | control |
|---|---|---|---|
| `dinov2-large` | **0.7852** | — | — |
| `v3large` (DINOv3-L) | 0.7446 | 0.8258 (`v3large_dense`) | `base224` 0.8502 |
| `ortholarge` (OrthoFoundation-L) | 0.7387 | 0.8452 (`ortho_dense`) | `base224` 0.8502 |

Sources: `logs/ortho_probe.log`, `logs/after_orthochain.log`, `logs/after_v3large_dense.log`.

Key readings:

- **OrthoFoundation-L is declined as a 2D trunk replacement.** The fitted member lost −0.0050 vs. its DINOv2-base control, and the frozen probe shows the knee continuation actually **subtracted signal** from its own DINOv3-L init (0.7387 vs. 0.7446). This is the catastrophic-forgetting shape reported for RAD-DINO on out-of-domain chest data.
- **The size confound matters.** `v3large_dense` (DINOv3-L, no knee continuation) scored 0.8258, worse than `ortho_dense` 0.8452, so the low `ortho_dense` number is partly "ViT-L overfits 4,407 report-derived studies," not purely "knee continuation hurts." But neither beat the DINOv2-base control.
- **The readout may be the real story.** DINOv2-L beats DINOv3-L and OrthoFoundation-L under the project's default max-pooled ridge probe. Q-AL literature (`research/deep_dives/2026-09-02_trunk_vs_readout_quality_medical_volumes_qal.md`) shows that the same frozen trunk can go from chance to near-perfect when the readout is made local, and learned attention pooling over slices can beat max/mean by up to +0.11 AUC. Bead `knee-02vn` tracks the re-probe.

## 2. Exact external-data inventory (re-measured 2026-09-02)

### 2.1 MRNet

- Path: `D:/data/volumetric/mri/raw/mrnet/`
- 1,250 decoded exams, 3 planes each (`sagittal/`, `coronal/`, `axial/`)
- Labels: `train-abnormal.csv`, `train-acl.csv`, `train-meniscus.csv` (and valid variants)
- Label counts (train+valid): binary abnormal, ACL, meniscus tear
- Licence: Stanford Research Use Agreement, non-commercial, no redistribution; project has re-read Kaggle rule 2.6(a) as a disjunction and concluded MRNet is permitted for competition use.

### 2.2 Rijeka (kneeMRI)

- Path: `D:/data/volumetric/mri/raw/kneemri_rijeka/kneemridataset.zip`
- 3.1 GiB, 918 rows in `metadata.csv`, project decoded 736
- `aclDiagnosis`: 0 = 690, 1 = 172, 2 = 55
- `kneeLR`: 448 left, 469 right
- `seriesNo` is heavily concentrated on 5 (767 rows), with a long tail 3–20; this is almost certainly a single sagittal PDW FS series per exam
- Only ACL severity is labelled; no meniscus/OA/etc.

### 2.3 fastMRI+

- Path: `D:/data/volumetric/mri/raw/fastmri/labels/Annotations/`
- 1,173 reviewed knee files (`knee_file_list.csv`)
- 974 files have at least one bbox finding; 199 reviewed with no findings
- 16,167 bbox annotations in `knee.csv`
- Top labels by count:
  - Meniscus Tear: 5,658
  - Cartilage – Partial Thickness loss/defect: 2,985
  - Joint Effusion: 1,311
  - Bone-Fracture/Contusion/dislocation: 1,060
  - Bone-Subchondral edema: 986
  - Periarticular cysts: 864
  - Ligament – ACL Low Grade sprain: 765
  - Ligament – ACL High Grade Sprain: 678
  - Cartilage – Full Thickness loss/defect: 615
  - Ligament – MCL Low-Mod Grade Sprain: 285
- `study_level` is "No" for 16,154 annotations and "Yes" for only 13; the bbox is the useful signal.

Honest RSNA mappings:

- **ACL**: low-grade and high-grade sprain → binary ACL (or keep graded as soft target).
- **MCL**: low-mod grade sprain → binary MCL.
- **Effusion**: Joint Effusion → Effusion.
- **Meniscus**: fastMRI+ does **not** split medial/lateral; either drop or use only as a weak "any meniscus" auxiliary.
- **Contusion**: pooled with fracture/dislocation; drop unless box location can recover contusion specifically.
- **Cartilage/OA/Synovitis/Baker's/Fracture**: no honest external source.

### 2.4 KMAR-50K and SKM-TEA

- KMAR-50K: 1,444 scans / 62k slices, CC BY 4.0, no labels — SSL/pseudo-label only.
- SKM-TEA: 155 patients, ~25k slices, Stanford DUA non-commercial, rich 3D segmentations — segmentation auxiliary path only.

## 3. Foundation-model landscape (2025–2026), repriced

| model | year | weights | knee-specific | actionable? |
|---|---|---|---|---|
| **OrthoDiffusion** | 2026 | ✅ HuggingFace + Kaggle | ✅ 15,948 knee MRIs | **Yes — smoke test is the top open trunk bet** |
| **OrthoFoundation-L** | 2026 | ✅ GitHub | ✅ 1.25M knee images | **Declined** (`ortho_dense` 0.8452 < control 0.8502) |
| **MRI-CORE** | 2024 | ✅ Apache 2.0 | ❌ general MRI | Auxiliary segmentation/feature only |
| **MedDINOv3** | 2025 | paper only | ❌ general medical | No weights yet |
| **OmniMRI** | 2025 | ❌ not found | general MRI VLM | Watchlist |
| **Decipher-MR** | 2025–26 | ❌ proprietary | general 3D MRI VLM | **Not actionable** — GE Healthcare, data and weights not public |
| **MedImageInsight** | 2025–26 | ✅ Azure | ❌ general medical | Commercial/MS licence; low priority |
| **Radio DINO** | 2025 | ✅ CC BY 4.0 | ❌ general radiology | Fallback 2D trunk only |

**Decipher-MR is a dead end for us.** The Nature npj Digital Medicine paper's data-availability statement says the pretraining corpus and held-out validation sets are proprietary and cannot be shared publicly. No public GitHub or HuggingFace weights were found. It is a strong model on paper, but it cannot be mounted in a Kaggle kernel.

**OrthoDiffusion is now the only public, knee-specific, multi-label-diagnosis 3D foundation model with weights.** It reports macro AUROC 0.908 on 8 knee abnormalities; 5 of 8 map directly to our RSNA labels (ACL, Medial Meniscus, Lateral Meniscus, MCL, Effusion). Its value is the 3D diffusion backbone, not the released head.

## 4. Recommended priority order

1. **Run the readout re-probe (`knee-02vn`).** It costs no training and may overturn the "DINOv2 is best" reading. If attention/patch-level readout shows DINOv3-L or OrthoFoundation-L beating DINOv2-L, the trunk-closure verdicts must be rewritten.
2. **Smoke OrthoDiffusion (`knee-6hfi`).** It is the only untested public knee-specific foundation model. If it transfers, it becomes a member candidate; if it fails, the 3D diffusion lane closes.
3. **Implement Scheme B (`knee-n6tk` / `knee-mwuv`).** Start with MRNet + Rijeka (already decoded, partial labels), then add fastMRI+ boxes. This is the cheapest external-data path that uses honest labels and can be pre-registered against RSNA-only and single-corpus baselines.
4. **Only then consider Scheme A/C SSL continuation** if Scheme B is positive and OAI access is resolved.

## 5. What not to do

- Do not chase Decipher-MR — no public weights.
- Do not treat OrthoFoundation-L as a live 2D trunk bet; it is measured and declined.
- Do not treat fastMRI+ meniscus or contusion labels as direct RSNA supervision; the anatomy/semantics do not map cleanly.
- Do not run another 2D knee-specific DINOv3 continuation; OrthoFoundation-L is already a stronger instance of that bet and it lost.

## Sources

- `interpretations/kneemri/2026-09-01_external-data-state.md`
- `research/deep_dives/2026-09-02_using_external_knee_mri_data_to_improve_kaggle_score.md`
- `research/deep_dives/2026-09-02_external_mri_data_merging_schemes.md`
- `research/deep_dives/2026-09-02_sota_pretrained_knee_mri_models_2025_2026.md`
- `research/deep_dives/2026-09-02_trunk_vs_readout_quality_medical_volumes_qal.md`
- `logs/ortho_probe.log`
- `logs/after_orthochain.log`
- `logs/after_v3large_dense.log`
- Disk inventory: `D:/data/volumetric/mri/raw/mrnet/`, `D:/data/volumetric/mri/raw/kneemri_rijeka/`, `D:/data/volumetric/mri/raw/fastmri/labels/Annotations/`
- Yang et al., *Decipher-MR: A Vision-Language Foundation Model for 3D MRI Representations*, arXiv:2509.21249 / Nature npj Digital Medicine 2026.
- Lan et al., *OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI Interpretation*, arXiv:2602.20752.
