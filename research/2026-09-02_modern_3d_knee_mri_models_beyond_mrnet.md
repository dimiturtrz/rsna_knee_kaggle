# Modern 3D knee-MRI models beyond MRNet

MRNet (Bien et al., 2018) proved that a 3D CNN on knee MRI can reach radiologist-level AUC on abnormal / ACL / meniscus, but it is now eight years old and was trained on roughly 1,370 exams. Since then the field has moved in three directions that are relevant to our 12-label RSNA task: (1) shallow 3D CNNs with extruded ImageNet pretraining, (2) graph / surface convolutions that respect cartilage geometry and anisotropic spacing, and (3) video / tubelet transformers that treat an MRI volume as a stack of frames. This note reviews the concrete numbers, the data volumes, and how each approach could or could not raise our Kaggle score.

## 1. Shallow 3D CNNs with extruded 2D pretraining

Yeoh et al. (2023) trained 13 3D CNN variants on 400 OAI sagittal DESS volumes (135 no-OA, 265 OA) for binary knee-OA detection. The key result is that transfer learning by *extruding* 2D ImageNet weights along the third axis beats training from scratch by a wide margin:

| Model | Transfer AUC | Scratch AUC |
|-------|--------------|-------------|
| 3D ResNet18 | **0.945** | 0.763 |
| 3D ResNet34 | 0.925 | 0.763 |
| 3D ResNet50 | 0.900 | 0.529 |
| 3D DenseNet121 | 0.914 | 0.798 |
| 3D VGG11 | 0.928 | 0.870 |

([Yeoh et al., 2023, *Transfer learning-assisted 3D deep learning models for knee osteoarthritis detection*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10136763/))

**Why it matters.** A 3D ResNet18 with ~33M parameters and extruded ImageNet weights reaches 0.945 AUC with only 400 volumes. That is the same scale as MRNet and suggests that a modern 3D path does not need massive 3D pretraining to outperform random initialization. The best models are *shallow*; deeper 3D ResNets and DenseNets overfit. For us this means: if we reconstruct even a few hundred volumetric knee exams from external corpora (OAI, Rijeka, MRNet), a 3D ResNet18 pretrain is a cheap probe.

**Catch for our competition.** RSNA 2024 knee-MRI data was released as 2D PNGs without a guaranteed, consistent 3D grid. Reconstructing true 3D volumes requires DICOM series metadata and resampling to a canonical anisotropic spacing. Our current pipeline is slice-centric; a 3D branch would need a separate dataloader and a decision about how to handle studies with missing slices or variable spacing.

## 2. Graph and surface convolutions for anisotropic knee MRI

Zhuang et al. (2022) proposed CSNet, which builds a graph on the cartilage surface and runs patch convolution + graph attention. It was evaluated on 1,205 clinical sagittal T2 fat-suppressed knee MRIs graded for cartilage defect (Grade 0 / 1 / 2):

| Method | ACC | REC | AUC |
|--------|-----|-----|-----|
| CSNet (proposed) | **83.1%** | **82.2%** | **92.9%** |
| 3D-ResNet | 71.8% | 70.5% | 82.3% |
| 3D-MedT | 68.7% | 67.2% | 80.9% |
| ResNet101 (slice) | 86.3% | - | - |
| U-Net encoder (patch) | 87.2% | - | - |

([Zhuang et al., 2022, *Knee Cartilage Defect Assessment by Graph Representation and Surface Convolution*](https://arxiv.org/pdf/2201.04318))

**Why it matters.** CSNet explicitly solves two problems that kill ordinary 3D CNNs on knee MRI: thin cartilage voxels are drowned by background, and inter-slice spacing is heterogeneous. The graph representation adapts to different spacing automatically. The ablation shows that the *within-slice continuity edge* (As) contributes most; cross-slice edges (Aa) add a smaller but real gain.

**Catch for our competition.** CSNet is a cartilage-defect classifier and needs bone segmentation as a preprocessing step. It is not a general 12-label trunk out of the box. Adapting it would require segmentation masks for femur/tibia/patella, which we do not have for the RSNA test set. It is therefore more relevant as a *pretraining objective* on external data with segmentation labels (e.g., OAI with MOAKS/WORMS, or SKM-TEA) than as a direct competition model.

## 3. Video / tubelet transformers on 3D MRI

Akan et al. (2025) treat a 3D brain MRI as a video: each slice is a frame, and the Video Vision Transformer (ViViT) embeds non-overlapping 3D tubelets. On ADNI (351 scans, CN/MCI/AD), ViTranZheimer reaches:

| Model | Accuracy |
|-------|----------|
| ViTranZheimer (ViViT) | **98.6%** |
| ViT-BiLSTM | 97.5% |
| CNN-BiLSTM | 96.5% |

([Akan et al., 2025, *Leveraging Video Vision Transformer for Alzheimer's Disease Diagnosis from 3D Brain MRI*](https://arxiv.org/pdf/2501.15733))

**Why it matters.** This is the cleanest modern analogue to "3D ViT for MRI." It shows that spatio-temporal attention over slices can beat CNN-LSTM hybrids. The model is small (466k trainable parameters) and trained from scratch on 351 scans, which is encouraging for data-scarce medical settings.

**Catch for our competition.** Brain MRI in ADNI is relatively isotropic and skull-stripped; knee MRI is anisotropic and has large background regions. A vanilla ViViT would waste attention on background voxels unless combined with a foreground crop or mask. There is no published knee-MRI ViViT result yet, so this would be a genuine experiment, not a plug-and-play upgrade.

## 4. General 3D medical transformers (Swin UNETR, nnFormer, UNETR)

A data-scalable transformer benchmark (arXiv:2203.00131) found that pure ViT encoders (UNETR) underperform on 3D medical segmentation because patch tokenization loses structural information. Hierarchical Swin-style transformers (nnFormer, Swin UNETR) do better, but still lag behind ConvNets when medical training data is limited. NVIDIA's Swin UNETR is the main large-scale 3D medical SSL attempt, yet the paper notes that transfer from natural images is weak because of the domain gap.

**Implication.** For our 12-label classification task, a full 3D transformer backbone is unlikely to beat a strong 2D slice ensemble unless it is either (a) pretrained on tens of thousands of 3D knee volumes, or (b) constrained to a foreground region so the attention is not diluted.

## 5. How 3D models could improve our Kaggle score

Our champion is a 2D slice ensemble (DINOv2/DINOv3 trunks + slice aggregation). A 3D path can help in three non-exclusive ways:

1. **Pretraining trunk on external 3D volumes.** Train a 3D ResNet18 or CSNet on OAI/MRNet/Rijeka, then distill its features into the 2D trunk or use them as auxiliary inputs. Yeoh's result (0.945 transfer vs 0.763 scratch) shows that even a few hundred volumes give a strong initialization. Expected contribution: small but cheap, mostly complementary to Scheme B.

2. **3D-aware aggregation.** Replace slice-wise max/mean pooling with a lightweight 3D module (3D conv or temporal attention over slice embeddings) that models spatial relationships between slices. This keeps the 2D trunk intact but lets the model see that a meniscal tear spans several slices. Expected contribution: +0.002-0.006 mean AUC if slice ordering is reliable.

3. **Full 3D classifier.** Reconstruct volumes from the RSNA DICOMs and train an end-to-end 3D model. This is the highest-risk path: it requires series metadata, resampling, and a large compute budget, and published 3D knee classifiers still underperform slice-based methods on some tasks (e.g., RSNA 2020 ACL study found 3D CNN 89% accuracy vs 2D CNN 92%).

## 6. Data-volume reality check

| Corpus | Modality | Approx. volumes | Labels | Usable for 3D? |
|--------|----------|-----------------|--------|----------------|
| MRNet | knee MRI, 3 planes | ~1,370 | abnormal, ACL, meniscus | yes, but license unclear |
| OAI | knee MRI, sagittal DESS | ~4,800 subjects, many timepoints | KL grade, MOAKS/WORMS | yes, public with registration |
| Rijeka | knee MRI | 736 | ACL severity | yes, already on disk |
| fastMRI+knee | knee MRI, raw k-space | ~1,000+ | limited pathology labels | yes, but mostly unlabeled |
| SKM-TEA | knee MRI | ~1,000 | segmentation + labels | yes, multi-task |
| KMAR-50K | knee MRI | ~1,444 | unlabeled | SSL only |

A 3D ResNet18 probe needs only a few hundred volumes to beat random init (Yeoh). A CSNet-style probe needs bone masks. A ViViT needs either isotropic volumes or a foreground crop. Therefore the *cheapest* 3D experiment is a 3D ResNet18 pretrained on Rijeka + OAI + MRNet, then fine-tuned or distilled into our 2D pipeline.

## 7. Recommendation

Do not pivot the competition pipeline to full 3D. The champion is 2D for good reasons: variable spacing, large backgrounds, and the RSNA labels are slice-level aggregates. Instead:

- **Short term:** Add a 3D pretraining branch to Scheme B. Use Rijeka + OAI + MRNet to pretrain a 3D ResNet18 with extruded ImageNet weights, then use its features as auxiliary input or to initialize the 2D trunk via knowledge distillation. This is the same data-merge strategy as the external-data note, but with a 3D encoder.

- **Medium term:** Experiment with 3D-aware slice aggregation. Treat slice embeddings as a temporal sequence and add a small transformer or 1D conv over slice index. This is much cheaper than full 3D and directly improves the existing architecture.

- **Deprioritize:** End-to-end 3D classifiers and CSNet-style surface convolutions unless segmentation masks become available. They are higher complexity and lower expected return for this specific competition.

The bottom line: modern 3D models are better than MRNet, but the gains are largest when they are used to *pretrain* or *aggregate* rather than replace the 2D slice pipeline that already scores 0.937-0.941.
