# External knee-MRI data access audit

This note is the practical companion to the external-data and 3D-model research notes. It lists the knee-MRI corpora we have mentioned, how to actually download them, what their licenses require, and whether they can legally feed a Kaggle submission. The goal is to turn the Scheme B recommendation into a concrete procurement checklist.

## Summary table

| Corpus | Size | Access route | License / restriction | Usable for Kaggle? |
|--------|------|--------------|----------------------|--------------------|
| MRNet | 1,370 exams | Stanford AIMI / competition page | Research-use agreement; various sources list CC-BY or public domain, but official page requires agreement | Likely yes, if agreement allows competition use |
| OAI | ~4,800 subjects, many timepoints | NDA via Login.gov | Public-domain NIH data | Yes |
| Rijeka / KneeMRI | 917 volumes | Zenodo / Kaggle | **CC BY-NC-ND 4.0** | **No** — NC and ND clauses are incompatible with Kaggle rule 2.6a |
| fastMRI knee | ~1,000+ scans | fastMRI.med.nyu.edu | Custom NYU agreement, research/education | Likely yes, verify competition use |
| SKM-TEA | 155 scans | Stanford AIMI | Custom Stanford AIMI agreement | Likely yes, verify competition use |
| KMAR-50K | 1,444 scans | Mendeley Data | CC-BY (open access) | Yes |

## 1. MRNet (Stanford)

- **What:** 1,370 knee MRI exams from Stanford University Medical Center (2001-2012), with labels for abnormal, ACL tear, and meniscal tear. Train/val/test splits are public; the official test set is hidden and evaluated on Codalab (competition now closed).
- **Where:** [Stanford ML Group competition page](https://stanfordmlgroup.github.io/competitions/mrnet/) and [Stanford AIMI catalog entry](https://aimi.stanford.edu/datasets/mrnet-knee-mris).
- **Access:** The official download requires signing a research-use agreement on the Stanford AIMI portal. Some third-party sources (e.g., TensorFlow Datasets issues) list the license as CC-BY or public domain, but the canonical path is the AIMI agreement.
- **Competition risk:** The Kaggle host rule 2.6a generally requires external data to be "freely and publicly available." MRNet is publicly available under an agreement, but we need to confirm the agreement does not restrict competitive/commercial use. The bead `knee-2sbk` already flags this uncertainty.
- **Action:** Download and read the AIMI Data Use Agreement before any model trained on MRNet is shipped to Kaggle.

## 2. Osteoarthritis Initiative (OAI)

- **What:** Multi-center longitudinal knee OA study with ~4,800 participants, MRI (sagittal DESS), X-rays, clinical data, and MOAKS/WORMS-derived structural labels.
- **Where:** [NDA NIH OAI portal](https://nda.nih.gov/oai/).
- **Access:** Register via [Login.gov](https://login.gov/) on the NIH NDA site, request data permissions under "My Account → Data Permissions," add studies to a filter cart, and create a data package. Approval is usually fast for public datasets.
- **License:** Public-domain NIH data. No commercial restriction.
- **Competition risk:** Low. OAI is explicitly public and unrestricted.
- **Action:** Create an NDA account and request the baseline sagittal DESS MRI package plus KL grades. This is the safest external corpus to start with.

## 3. Rijeka / KneeMRI dataset

- **What:** 917 sagittal PD-weighted knee MRI exams from Clinical Hospital Centre Rijeka (2006-2014), labeled for ACL injury.
- **Where:** [Zenodo record 14789903](https://zenodo.org/records/14789903) and [Kaggle mirror](https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset).
- **License:** **CC BY-NC-ND 4.0** — non-commercial and no-derivatives.
- **Competition risk:** **High.** The non-commercial (NC) and no-derivatives (ND) clauses almost certainly violate Kaggle rule 2.6a, which requires external data to be freely available and usable for the competition. Training a model on this data and submitting predictions would be a derivative commercial use.
- **Action:** Do not use Rijeka in any shippable model unless the host explicitly waives the restriction. It can be used for *internal method development only* if we keep the model out of submissions. Update `comp/members.declines` with this reasoning.

## 4. fastMRI knee

- **What:** Fully anonymized knee MRI raw k-space and DICOM images from NYU Langone Health, originally released for accelerated MRI reconstruction.
- **Where:** [fastMRI.med.nyu.edu](https://fastmri.med.nyu.edu/).
- **Access:** Request access through the NYU fastMRI portal; a custom agreement must be signed.
- **License:** Royalty-free license for internal research and educational purposes (custom NYU agreement).
- **Competition risk:** Medium. The agreement is research-oriented; we must verify it permits use in a public competition with prize implications.
- **Action:** Read the fastMRI Data Use Agreement. If it restricts competitive use, either decline or request a waiver.

## 5. SKM-TEA (Stanford)

- **What:** 155 clinical quantitative 3D knee MRI scans with dense segmentations (cartilage, meniscus) and pathology bounding boxes. Two tracks: raw k-space and DICOM images.
- **Where:** [Stanford AIMI dataset page](https://stanfordaimi.azurewebsites.net/datasets/4aaafb9-c6e6-4e3c-9188-3aaaf0e0a9e7) (link from SKM-TEA paper) and [GitHub](https://github.com/StanfordMIMI/skm-tea).
- **Access:** Create a Stanford AIMI account, log in, and download. Dataset is 900 GB compressed / 1.6 TB expanded.
- **License:** Custom Stanford AIMI agreement.
- **Competition risk:** Medium, same as MRNet and fastMRI — verify the agreement allows competition use.
- **Action:** The main value of SKM-TEA is not scale (only 155 scans) but *dense anatomy labels*. It is most useful for pretraining a segmentation-aware 3D model or CSNet-style surface network, not for supervised 12-label classification.

## 6. KMAR-50K

- **What:** 1,444 multi-parameter knee MRI scans from 1,190 patients, paired with motion-artifact and artifact-free images.
- **Where:** [Mendeley Data](https://data.mendeley.com/datasets/xw7mrg7ntg).
- **Access:** Direct download from Mendeley Data after registration.
- **License:** Open-access / CC-BY (verify exact version on download page).
- **Competition risk:** Low. CC-BY is the most permissive common research license.
- **Action:** Useful for SSL / reconstruction pretraining and for data augmentation (motion-artifact simulation). Not labeled for our 12 pathology classes, so treat it as unlabeled data for Scheme A.

## 7. Kaggle rule 2.6a reminder

Kaggle competitions typically require that any external data used in a submission be:
- freely and publicly available to everyone, and
- usable under its license for the competition purpose.

A dataset that requires a research-use agreement is not automatically disqualified, but it must not impose restrictions that make it less available than the competition data itself. NC and ND licenses are generally disqualifying. Custom agreements need a line-by-line check.

## 8. Recommended procurement order

1. **OAI** — lowest risk, largest volume, public domain. Start here for Scheme B.
2. **KMAR-50K** — low risk, large volume, good for SSL / augmentation.
3. **MRNet** — moderate risk, verify agreement; valuable labels overlap our task.
4. **fastMRI knee** — moderate risk, verify agreement; mostly unlabeled but high-quality volumes.
5. **SKM-TEA** — moderate risk, small volume, high label density; useful only if we pursue 3D segmentation pretraining.
6. **Rijeka** — high risk (CC BY-NC-ND); decline for submissions unless host waives.

## 9. Local inventory update

Most of the corpora above are already on disk under `D:/data/volumetric/mri/raw/`. This changes the practical next step from "download" to "inspect format and decide license risk."

| Corpus | Local path | Format on disk | Notes |
|--------|-----------|----------------|-------|
| MRNet | `D:/data/volumetric/mri/raw/mrnet/` | 1,250 exams × 3 planes as `uint8` `.npy` stacks (axial ~44×256×256, sagittal/coronal ~36×256×256) + `train/valid-{abnormal,acl,meniscus}.csv` | Already preprocessed and ready to load. Only license check blocks use. |
| Rijeka / KneeMRI | `D:/data/volumetric/mri/raw/kneemri_rijeka/kneemridataset.zip` | 917 volumes as `uint16` `.pck` files (32×320×320) + `metadata.csv` with `aclDiagnosis` and ROI coords | License remains CC BY-NC-ND; data is available for internal experiments only. |
| fastMRI knee | `D:/data/volumetric/mri/raw/fastmri/` | 973 train + val `singlecoil_*.h5` files; `labels/Annotations/knee.csv` has 16,167 bounding boxes across 974 files and 21 pathology labels | This is the strongest immediate lever — labels map directly to RSNA classes (ACL, meniscus, cartilage, bone lesion, effusion, etc.). |
| SKM-TEA | `D:/data/volumetric/mri/raw/skmtea/` | 155 DICOM studies + annotations + segmentation masks | Small but dense; useful for 3D segmentation pretraining. |
| KMAR-50K | `D:/data/volumetric/mri/raw/kmar50k/` | ~3,000 `.nii` volumes across `ArtifactData_*` and `GroundTruthData_*` | Unlabeled for pathology; use for SSL/reconstruction/motion augmentation. |
| knee_clinical_seq | `D:/data/volumetric/mri/raw/knee_clinical_seq/` | DICOM series (batch2) | Unknown labels; needs inventory. |
| RSNA label tables | `D:/data/volumetric/mri/raw/knee_label_tables/` | Multiple CSVs with LLM/consensus labels and calibrated soft targets for RSNA studies | Potential pseudo-label / teacher sources for Scheme C. |

### fastMRI knee labels are the big surprise

`fastmri/labels/Annotations/knee.csv` contains 16,167 bounding-box annotations over 974 knee scans, with labels such as:

- Meniscus Tear (5,658 boxes)
- Cartilage — Partial / Full Thickness loss/defect (2,985 + 615)
- Joint Effusion (1,311)
- Bone — Fracture/Contusion/dislocation (1,060)
- Bone — Subchondral edema (986)
- Ligament — ACL Low Grade sprain / High Grade Sprain (765 + 678)
- Ligament — MCL, PCL, LCL sprains
- Bone — Lesion, Soft Tissue Lesion, etc.

These map well onto the RSNA 12-label set. Because the boxes are slice-local, they can be used for partial-label supervised pretraining (Scheme B) without requiring full 3D reconstruction. The NYU license still needs verification, but the data is already local.

### Revised procurement order given local state

1. **fastMRI knee** — already local, rich labels, strongest immediate lever. Verify NYU license first.
2. **MRNet** — already local and preprocessed. Verify Stanford AIMI license.
3. **KMAR-50K** — already local, use for SSL/augmentation.
4. **SKM-TEA** — already local, use for 3D segmentation pretraining if pursued.
5. **OAI** — not yet local (to our knowledge), but public-domain and low-risk; worth downloading for volume.
6. **Rijeka** — already local, but license likely bars competition use.

## 10. What we still need to verify

- Exact text of the Stanford AIMI agreement for MRNet and SKM-TEA.
- Exact text of the NYU fastMRI agreement.
- Whether the RSNA 2024 host has published an explicit external-data whitelist or FAQ.
- Whether Rijeka can be used under a separate research-only agreement without the NC/ND clauses.
- Whether `knee_clinical_seq` has any labels and what its license is.

The next concrete step is no longer downloading — it is reading the fastMRI and MRNet data-use agreements, then building a dataloader that can consume the local `.npy` / `.h5` / `.pck` / `.nii` formats under Scheme B.
