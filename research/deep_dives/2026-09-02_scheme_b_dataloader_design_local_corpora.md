# Scheme B dataloader design for local external knee-MRI corpora

With the academic/non-profit registrations for fastMRI and MRNet confirmed, the blocker is no longer legal — it is engineering. This note designs a single dataloader that can merge the local external corpora into the RSNA 12-label training pipeline under Scheme B: shared 2D trunk, per-corpus heads, masked binary cross-entropy, honest labels only.

## 1. Local corpora format summary

| Corpus | Path | Samples | Native format | Native labels |
|--------|------|---------|---------------|---------------|
| MRNet | `D:/data/volumetric/mri/raw/mrnet/` | 1,250 exams × 3 planes | `uint8` `.npy` stacks, ~36-44 slices of 256×256 | `train/valid-{abnormal,acl,meniscus}.csv` |
| fastMRI knee | `D:/data/volumetric/mri/raw/fastmri/` | 974 scans | `singlecoil_*.h5` k-space/image volumes | `labels/Annotations/knee.csv`: 16,167 slice-level bounding boxes, 21 classes |
| Rijeka | `D:/data/volumetric/mri/raw/kneemri_rijeka/kneemridataset.zip` | 917 volumes | `.pck` files, 32×320×320 `uint16` | `metadata.csv`: `aclDiagnosis` (0/1) + ROI coords |
| SKM-TEA | `D:/data/volumetric/mri/raw/skmtea/` | 155 scans | DICOM + HDF5 + Nifti segmentations | COCO-style detection annotations + 6-class segmentation |
| KMAR-50K | `D:/data/volumetric/mri/raw/kmar50k/` | ~3,000 volumes | `.nii` 640×640×22 | None (artifact / ground-truth pairs) |
| knee_clinical_seq | `D:/data/volumetric/mri/raw/knee_clinical_seq/` | Unknown | DICOM series | Unknown |

## 2. Design principles

1. **2D-first, not 3D-first.** The champion pipeline operates on 2D slices. External volumes should be decoded into slice tensors and fed through the same trunk. Any 3D pretraining is a separate, higher-risk branch.
2. **Honest partial labels.** A missing label means "not annotated," not negative. The loss mask must exclude that column for that sample.
3. **Per-corpus heads are optional.** The primary head is the RSNA 12-label head. Per-corpus auxiliary heads can help when labels do not map cleanly, but they are not required for Scheme B.
4. **Oversample under-represented positive classes.** External corpora are skewed toward ACL/meniscus/effusion. The RSNA task is also skewed. Use weighted sampling that emphasizes positive slices and late epochs.
5. **Hold out an external-only validation split.** Do not validate on RSNA holdout alone; track AUC on a held-out external subset to detect domain shift.

## 3. Label mapping to RSNA 12 targets

The RSNA targets, in order, are: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.

### MRNet mapping

| MRNet label | RSNA targets | Notes |
|-------------|--------------|-------|
| abnormal | — | Study-level flag, too coarse to supervise the 12-label head. Can be used as an auxiliary task or ignored. |
| acl | `ACL` | Direct. MRNet reports any ACL tear; RSNA wants high-grade partial or full. Treat MRNet positives as RSNA positives; the model will learn from RSNA negatives during fine-tuning. |
| meniscus | `Medial Meniscus`, `Lateral Meniscus` | MRNet does not distinguish medial/lateral. Apply the same label to both columns. |

### fastMRI+ mapping

fastMRI+ boxes are slice-local and class-specific. A scan can contribute many boxes across many slices.

| fastMRI+ label | RSNA targets | Handling |
|----------------|--------------|----------|
| Meniscus Tear | `Medial Meniscus`, `Lateral Meniscus` | No compartment info → both columns get the same slice-level label. |
| Displaced Meniscal Tissue | `Medial Meniscus`, `Lateral Meniscus` | Same as above. |
| Ligament — ACL Low Grade sprain | `ACL` | RSNA criteria require >50% fiber disruption. Low-grade sprain is below the RSNA threshold. Use as **ambiguous / do not supervise** or soft label 0.5. |
| Ligament — ACL High Grade Sprain | `ACL` | Direct positive. |
| Ligament — MCL Low-Mod Grade Sprain | `MCL` | RSNA wants high-grade partial or complete. Use as ambiguous / soft 0.5, or positive if the annotation confidence is high. |
| Ligament — MCL High Grade sprain | `MCL` | Direct positive. |
| Ligament — PCL/LCL sprains | — | No RSNA target. Ignore or use as auxiliary negative examples for the ligament columns. |
| Cartilage — Partial Thickness loss/defect | `Medial OA`, `Lateral OA`, `PF OA` | No compartment → apply to all three OA columns. RSNA wants >50% thickness loss, so partial-thickness should be ambiguous / soft. |
| Cartilage — Full Thickness loss/defect | `Medial OA`, `Lateral OA`, `PF OA` | Strong positive for all three OA columns. |
| Bone — Subchondral edema | `Contusion` | Direct positive (marrow edema without fracture line). |
| Bone — Fracture/Contusion/dislocation | `Contusion`, `Fracture` | Positive for both. |
| Bone — Lesion | `Contusion` | Ambiguous; likely old contusion or tumor. Use soft label or exclude. |
| Joint Effusion | `Effusion` | Direct positive. |
| Periarticular cysts | `Baker's` | Popliteal/Baker's cyst is the most common periarticular cyst around the knee. Map with lower confidence. |
| Soft Tissue Lesion | — | Too vague; exclude. |
| Muscle Strain | — | No RSNA target; exclude. |
| Joint Bodies | — | No RSNA target; exclude. |
| artifact | — | Exclude. |

**Partial-label mask.** For each external slice, only the RSNA columns that receive a mapped label are supervised. All other columns are masked out. For example, a fastMRI slice with only an ACL box supervises `ACL` and masks the other 11 columns.

### Rijeka mapping

| Rijeka label | RSNA targets |
|--------------|--------------|
| aclDiagnosis = 1 | `ACL` |
| aclDiagnosis = 0 | `ACL` (negative) |

Rijeka provides a full study-level ACL label, so the `ACL` column can be supervised for every volume. No other RSNA targets are annotated.

### SKM-TEA mapping

SKM-TEA is small (155 scans) but has dense labels.

| SKM-TEA source | RSNA targets |
|----------------|--------------|
| Meniscus segmentation | `Medial Meniscus`, `Lateral Meniscus` if morphological tear signs are present; otherwise use as unlabeled/negative. |
| Cartilage segmentation + thickness | `Medial OA`, `Lateral OA`, `PF OA` if full-thickness loss. |
| Pathology bounding boxes | Map by category to ACL/Meniscus/Effusion/Contusion/Fracture as available. |

Because SKM-TEA annotations are not fully aligned with the RSNA criteria, treat it primarily as a **segmentation pretraining** corpus (e.g., pretrain a U-Net/decoder on cartilage/meniscus masks) rather than a direct 12-label supervision source.

## 4. Unified sample format

Every loader returns the same sample dict:

```python
{
    "image": Tensor[C, H, W],           # 2D slice, normalized
    "corpus": str,                       # "mrnet", "fastmri", "rijeka", "skmtea", "kmar50k"
    "labels": Tensor[12],                # 0/1/NaN for each RSNA target
    "mask": Tensor[12],                  # 1 = supervise this column, 0 = ignore
    "weight": float,                     # sampling weight
    "meta": {...},                       # original identifiers, slice index, spacing
}
```

- `labels` uses `NaN` for missing labels; the loss function masks these out.
- `mask` is derived from whether the corpus annotates that column for this sample.
- `weight` comes from class-balanced sampling and corpus oversampling.

## 5. Sampling strategy

1. **Corpus-level mixing.** Draw each batch with a fixed corpus distribution, e.g.:
   - 40% RSNA (the target task)
   - 25% fastMRI
   - 15% MRNet
   - 10% Rijeka
   - 10% SKM-TEA / KMAR-50K (unlabeled/SSL)
   Adjust based on label density. fastMRI has the richest labels, so it should dominate external samples.

2. **Positive slice oversampling.** Within each corpus, oversample slices that contain at least one positive mapped label by 3-5×. This prevents the dataloader from returning mostly negative slices.

3. **Late-epoch emphasis.** Increase the external-corpus probability from 0.1 in early epochs to 0.5 in later epochs, letting the model first learn the RSNA distribution and then absorb external patterns without catastrophic forgetting.

4. **KMAR-50K as unlabeled augmentation.** KMAR-50K has no pathology labels. Use it for:
   - Reconstruction SSL (auto-encoder / masked-image-modeling objective)
   - Motion-artifact augmentation (train on artifact images, target ground-truth)
   - Domain-randomization style intensity augmentation

## 6. Validation splits

- **RSNA holdout:** standard fold-based holdout for the final metric.
- **External holdout:** hold out 10-15% of each external corpus before training. Track per-corpus AUC/accuracy on this split to catch domain shift.
- **MRNet official valid set:** use the provided 120-exam validation set as-is; do not merge it into training.

## 7. Implementation sketch

The dataloader can be built as a `torch.utils.data.Dataset` wrapper around per-corpus loaders:

```
MergedKneeDataset
├── MRNetDataset      → .npy + CSV
├── FastMRIDataset    → .h5 + knee.csv boxes
├── RijekaDataset     → zip + metadata.csv
├── SKMTEADataset     → DICOM/HDF5 + annotations
├── KMAR50KDataset    → .nii pairs (unlabeled)
└── ConcatSampler     → corpus-mixed, weighted
```

Each per-corpus loader is responsible for:
- Decoding its native format.
- Mapping native labels to RSNA 12-label `labels` + `mask`.
- Returning 2D slices (not volumes).
- Applying corpus-specific intensity normalization.

The shared transforms (resize, random affine, intensity augmentation) sit above the corpus loaders.

## 8. First experiment to run

1. Build only the **fastMRI** branch first. It has the cleanest label-to-RSNA mapping and the largest annotation count.
2. Train a small 224 px arm with frozen DINOv2 trunk + fastMRI slices mixed into RSNA batches at 25%.
3. Measure:
   - RSNA holdout AUC change vs. RSNA-only baseline.
   - fastMRI external validation AUC on held-out scans.
   - Per-label gain on the classes fastMRI supervises (ACL, Meniscus, OA, Effusion, Contusion, Fracture).

If the fastMRI branch reads positive, add MRNet and Rijeka. If it reads negative, the bottleneck is domain shift or label mapping, not data volume.

## 9. Open questions

- Should ACL low-grade sprain in fastMRI be treated as positive, negative, or soft 0.5? The RSNA criteria explicitly exclude low-grade sprains, so soft 0.5 or exclude is safer.
- fastMRI boxes have no medial/lateral compartment for meniscus/OA. Is it better to supervise both columns equally or to down-weight those labels?
- Does KMAR-50K's artifact/ground-truth pairing help robustness enough to justify the I/O cost of 3,000 large `.nii` volumes?
- What is in `knee_clinical_seq` and can it be labeled with the existing RSNA label tables?

The next step is to implement the fastMRI branch first, since it is the highest-confidence path.
