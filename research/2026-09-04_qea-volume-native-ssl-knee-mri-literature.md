# Volume-native SSL pretraining for knee MRI (Q-EA)

Date: 2026-09-04
Prompt: Claude Q-EA — is there any published result where 3D / volume-native SSL pretraining on ~10^5 knee-MRI volumes beats a 2D ImageNet/DINO init on multi-label knee-MRI classification? If yes, smallest corpus / GPU-days / resolution / depth. Does any volume-native knee release already contain fastMRI? Was Zhang et al. 2024 (arXiv:2403.16499) 2D-slice or volumetric SSL?

## Short answer

No published 3D/volume-native knee-MRI SSL result beats a strong 2D ImageNet/DINO init on multi-label classification. The only 3D knee-MRI SSL evidence is an ISMRM 2025 abstract with no numbers and no 2D baseline comparison. All quantitative results are either 2D-slice SSL or generative/diffusion models.

## Direct 3D knee-MRI SSL evidence

### Wang et al., ISMRM 2025, abstract #4014

- Title: *Self-supervised Pretraining on OAI data for 3D Knee MRI Analysis*
- Source: https://archive.ismrm.org/2025/4014.html
- Method: DINO self-supervised pipeline on **OAI 3D knee MRI**.
- Reported result: "significantly outperforms training from scratch" and "improves classification performance over the supervised pretrained model."
- What is missing: no numerical deltas, no comparison to ImageNet/DINO 2D init, no corpus-size ablation, no compute cost. Corpus is OAI, not ~10^5 volumes.
- Verdict: the only directly relevant 3D knee-MRI SSL paper, but it does not answer the question.

## 2D-slice SSL evidence on knee MRI

### Zhang et al. 2024 (arXiv:2403.16499, Medical Image Analysis)

- Method: **2D-slice SSL**, not volumetric. Two pretext tasks on anatomy-oriented imaging planes: (1) regress relative orientation between intersecting planes, (2) regress relative slice location within a parallel stack.
- Pretraining data: **fastMRI knee** (among others).
- Downstream task: binary classification of abnormality, ACL tear, and meniscal tear on sagittal T2W fat-saturated knee MRI from the **Stanford MRNet dataset**.
- Key numbers (Table 5, mean AUC):

| Pretraining method | 80 subjects (10%) | 200 (25%) | 400 (50%) | 800 (100%) |
|---|---|---|---|---|
| From scratch | 0.433 | 0.429 | 0.474 | 0.562 |
| ImageNet | 0.682 | 0.737 | 0.799 | **0.824** |
| SimCLR (in-domain) | 0.698 | 0.761 | 0.774 | 0.800 |
| MTL (proposed) | 0.755 | 0.783 | 0.801 | **0.819** |

- Interpretation: at low data, proposed MTL beats ImageNet; at full MRNet data, ImageNet init (0.824) still edges MTL (0.819). Because this is 2D-slice SSL, it does **not** bound the 3D volume-native case.

### OrthoFoundation

- 2D DINOv3-L continuation on 1.25M knee images (894k MRI slices, including ~300k fastMRI coronal PD/PD-FS slices).
- Already tried in this project as `ortho_dense_s2026` and declined: holdout AUC 0.8452 vs. DINOv2-base control 0.8502.

### OrthoDiffusion

- 3D diffusion backbone pretrained on 15,948 private clinical knee MRIs.
- Not SSL classification pretraining; measured seat value in this project is +0.0001, already declined.

## Volume-native knee checkpoints whose corpus contains fastMRI

None identified.

- Zhang et al. 2024 uses fastMRI for pretext tasks but releases code, not a foundation-model checkpoint.
- OrthoFoundation includes fastMRI slices but is 2D.
- OrthoDiffusion and OAI-DINO do not list fastMRI in their pretraining corpora.

## Nearest quantitative scaling curve (not knee MRI)

Sharma et al., *Data Scaling Laws for Radiology Foundation Models* (arXiv:2509.12818) — continued SSL on chest X-ray with 32×H100:

| samples | wall time |
|---|---|
| 30k | 0.33 h |
| 50k | 0.58 h |
| 100k | 1.17 h |
| 1M | 11.67 h |
| 3.5M | 40.83 h |

Classification starts to beat open-weights around 30k–100k samples. This is the strongest published compute/gain curve for radiology SSL, but it is 2D CXR, not 3D knee MRI.

## Conclusion

A 3D volume-native SSL pretrain on ~10^5 knee-MRI volumes that beats a 2D ImageNet/DINO init on multi-label knee-MRI classification is **not present in the published literature**. The only 3D knee-MRI SSL result (OAI-DINO) is an abstract without numbers or the required baseline comparison. Arming this axis would be an original experiment, not a replication of an established floor.
