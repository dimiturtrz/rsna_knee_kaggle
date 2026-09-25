# Public knee-MRI corpora survey (knee-mfd8)

2026-09-01, Kimi. Six parallel survey agents, every field cited to a primary source (dataset page,
paper, or official repository); blogs excluded. Ranking axis: **(licence permits Kaggle use) ×
(exams) × (findings overlapping our twelve)**. Licences are quoted, not summarized; ambiguous means
ambiguous. Our encoder-transfer result (Lateral Meniscus +0.0196 off a corpus that never labels it)
means a label-free corpus is still worth something — the table does not filter on labels.

Already held, for reference: **KneeMRI/Rijeka** (736 exams on disk of published 917; ACL expert
reads; obtained via a Kaggle mirror — the mirror's own licence terms were never recorded in this
repo, verify before any redistribution) and **MRNet** (1,250 exams decoded, Stanford research-use
agreement; label tables now downloaded, fittability path in progress per knee-mfd8).

## The ranking

| tier | corpus | licence for Kaggle use | exams | findings overlap |
|---|---|---|---|---|
| 1 | **KMAR-50K** (Sichuan, 2025) | **CC BY 4.0 — yes, attribution only** | 1,444 paired scans / 62,506 slices, 1,190 patients | none (motion-artifact pairs) — encoder transfer only |
| 1 | **fastMRI knee + fastMRI+** (NYU/Meta/Microsoft) | images: NYU DUA "internal research or educational purposes only… may not be redistributed in any way" — usable, never redistributable; annotations: **MIT** | 1,594 k-space + **10,012 DICOM exams** | meniscus (no med/lat split), ACL graded, MCL, effusion, contusion (pooled w/ fracture), cartilage loss; NO synovitis, NO OA grade |
| 1 | **ACL-PCL** (Mendeley, 2025) | **CC BY 4.0 — yes** | ~970 images POST-augmentation (unique count unverified, small) | ACL/PCL torn/normal only; augmentation leakage risk unless deduped; provenance unverified |
| 1 (marginal) | Open Knee(s) (Cleveland Clinic) | CC BY 4.0 — yes | ~8 cadaveric knees | none (biomechanics/FE) |
| 1 (marginal) | Denver Natural Knee | custom non-commercial — yes w/ citation; co-authorship clause ambiguous | 7 cadaveric knees | none |
| 2 | **SKM-TEA** (Stanford, 2021) | Stanford DUA, **"restricted to non-commercial purposes"** (AIMI verbatim); prize money arguably commercial; DUA text gated behind login — UNVERIFIED | 155 patients, ~25k slices, qDESS 3T, 900 GB | meniscus tear types, ACL (full-thickness), effusion, cartilage lesion grades; bbox + confidence, not exam-level; NO MCL/synovitis/contusion/OA grade |
| 2 | **OAI** (NIH, via NDA) | NDA DUC: research use fine, **no redistribution (§3), 12-month deletion (§9), NDA Study per public disclosure (§11)**; status of trained weights genuinely ambiguous | 4,796 subjects, 3T bilateral longitudinal | MOAKS subset (~600 knees): meniscus damage (degenerative), effusion-synovitis composite, BML (not contusion), cartilage loss, Baker's; ACL "complete tear" only (partial = normal); OA grade via KL |
| 2 | **MOST** (NIA AgingResearchBiobank) | OHDTA: institutional PI + IRB + signing official, **12-month destruction, no redistribution**, purpose locked to approved request | 3,026 + 1,525 participants, bilateral, to 168 months | WORMS/MOAKS (1,854 knees): meniscus tear types, ACL, effusion-synovitis, cartilage, KL grade; OA cohort — no acute contusion, MCL weak |
| 2 | OAI-ZIB (2019) | masks "made publicly available" (no licence); images inherit OAI DUC | 507 volumes, 3T DESS | segmentation only; KL grade joinable from OAI clinical |
| 2 | Kneeview (Barcelona, 2025) | **CC BY-NC 4.0** + files restricted (access request) | 93 patients, T2-FS | segmentation only |
| 2 | OAIZIB-CM (Zenodo, 2025) | **CC BY-NC 4.0** — and sits in tension with OAI's no-redistribution DUC | 507 (re-release) | segmentation only |
| 2 | OAI-iMorphics | OAI DUC (no separate licence found — UNVERIFIED) | 88 subjects × 2 tp | segmentation only (cartilage) |
| 2 | IWOAI 2019 challenge set | OAI-derived, licence UNVERIFIED | 176 volumes | segmentation incl. **meniscus** (only one of the group) |
| 3 | SKI10 (2010) | **withdrawn 2019** — distributor lost redistribution rights; rules were segmentation-only use, no commercial | ~150 volumes | none (bone/cartilage masks) |
| 3 | Duke-110K | **not public** — "available… upon reasonable request", effectively unavailable | 6.1M slices (knee fraction unverified) | none released |
| excluded | Kaggle rehosts (kneemridataset = Rijeka duplicate; OAI-MRI-3DDESS = OAI-derived), PCIR "Knee MRI" (is CT), Roboflow KNEEMRI (trivial, unverified) | — | — | — |

## What the ranking says

1. **The clean-licence gap is real.** The two biggest prizes (OAI, MOST) are DUC/OHDTA-gated with
   no-redistribution and 12-month deletion — usable as local training data with institutional
   paperwork, never shippable, and the trained-weights question is unwritten in both agreements.
2. **fastMRI is the best licence×size×labels compromise**: 10,012 DICOM exams under a research DUA
   (no NC clause; the MLCommons footnote is a precedent that benchmark submission counts as
   non-commercial), plus fastMRI+'s MIT-licensed study-level/bbox labels covering six of our twelve
   findings' neighborhoods. Each user needs their own approval; data can never ride in a kernel.
3. **KMAR-50K is the only large, genuinely free, NEW population** (Chinese, 1.5T+3T Siemens, 62k
   slices) — zero finding labels, so it is an encoder-transfer asset exactly in the shape the merged
   work already monetized.
4. **SKM-TEA is the label-richest modern set** (meniscus tear type, ACL grade, effusion, cartilage)
   but its Stanford DUA is NC-restricted and the verbatim redistribution/derivative clause is gated
   behind the AIMI login — someone must register and read it before it is usable.
5. The segmentation-era group (SKI10/OAI-ZIB/iMorphics/Kneeview) is cartilage/bone masks only —
   no finding supervision; value is transfer population and OA-grade joins via OAI clinical tables.

## Per-corpus detail

### KMAR-50K — Sichuan Academy of Medical Sciences & Sichuan People's Hospital, 2025
Citation: Xi Y et al., "A Multi-view Open-access Dataset of Paired Knee MRI for Motion Artifact
Removal," Scientific Data 12:1173 (2025), doi:10.1038/s41597-025-05439-1. Data DOIs
10.17632/xw7mrg7ntg, 10.17632/95w9f5tzz8. Licence CC BY 4.0, verbatim: "You can share, copy and
modify this dataset so long as you give appropriate credit, provide a link to the CC BY license, and
indicate if changes were made, but you may not do so in a way that suggests the rights holder has
endorsed you or your use of the dataset." (The PAPER is CC BY-NC-ND; the DATA is CC BY 4.0.)
1,444 paired scans (artifact + clean rescan), 1,190 patients, 62,506 slices; cor/sag/tra; PD-TSE,
PD-FSE, T1-TSE, T1-FSE, T2-TSE, T2-FSE; Siemens Aera 1.5T + 3.0T; NIfTI, N4-corrected. No finding
labels. Direct download, no registration: data.mendeley.com/datasets/xw7mrg7ntg + /95w9f5tzz8,
≈20.7 GB total.

### fastMRI knee (NYU/FAIR, 2018; v2 2019) + fastMRI+ (Microsoft/NYU, 2022)
Citations: Zbontar J et al., arXiv:1811.08839 / Knoll F et al., Radiology: AI 2020;2(1):e190007;
Zhao R et al., Sci Data 2022;9:152 (fastMRI+). Images under the NYU fastMRI Data Sharing Agreement.
NYU page verbatim: "NYU fastMRI data may be used for internal research or educational purposes only
as described in the data use agreement and may not be redistributed in any way without prior
permission." Footnote verbatim: "Use of the fastMRI dataset for creating and submitting benchmark
results for publication on an MLCommons™ benchmark… is considered non-commercial use of fastMRI."
Paper verbatim: "while we do not allow any commercial use of the dataset itself, we do not
explicitly discourage the development or testing of software, algorithms, or other intellectual
property with the dataset." fastMRI+ annotations: MIT licence inside github.com/microsoft/fastmri-plus
("…to deal in the Software without restriction…"), but the labels key onto DUA-bound images.
K-space: 1,594 scans, coronal PD + PD-FS only, 320×320, 3 mm; 1.35 TB. DICOM: 10,012 exams / 9,290
patients, five sequences (cor PD, cor PD-FS, ax T2-FS, sag PD, sag T2-FS), 164 GB. fastMRI+ knee:
1,172 val studies, 16,154 bounding boxes, 22 categories — Meniscus Tear (663 studies), ACL high/low
grade (101/153), MCL (121 low-mod, 4 high-grade), Bone-Fracture/Contusion/Dislocation (119, pooled),
Cartilage partial/full-thickness (588/122), Joint Effusion (142), Periarticular Cysts (161); single
reader, no inter-rater stats; no synovitis, no OA grade. Obtain: fastmri.med.nyu.edu application
(institutional affiliation + proposed use); annotations git-clone, no registration.

### ACL-PCL — Mendeley Data, 2025
Md Shakil Hossain, East West University (Dhaka). DOI 10.17632/fjns2xzhrr.1; NO associated paper;
provenance UNVERIFIED (record claims manual expert annotation, names no hospital). CC BY 4.0
(verbatim clause identical to KMAR-50K's above). ~970 images post-augmentation over
axial/coronal/sagittal T2 + sag PD-FS, 640×640, YOLO format, 110 MB. Labels: Normal/Torn × ACL/PCL.
Train/val leakage risk from augmented copies; dedupe before use. Direct download.

### SKM-TEA — Stanford, 2021
Desai AD et al., NeurIPS 2021 Datasets & Benchmarks, arXiv:2203.06823. Licence: Stanford University
Dataset Research Use Agreement — AIMI verbatim: "The datasets have been available for free and are
shared broadly under a license that is restricted to non-commercial purposes." Paper verbatim (App.
A.5): "All public data is distributed under the Stanford University School of Medicine license and
the terms listed for the Lower Extremity Radiographs dataset." Full DUA text only post-login —
UNVERIFIED; obtain before use. 155 patients, ~25,000 slices, sagittal 3D qDESS two-echo 3T GE MR750,
512×512×160 at 0.3125×0.3125×0.8 mm, raw k-space + SENSE + DICOMs; 900 GB compressed / 1.6 TB
expanded. Labels: 16 detection categories as 3D bounding boxes with 0–5 confidence — meniscal tear
by type (no med/lat split), ligament tear by grade (ACL/PCL only, NO MCL/LCL), cartilage lesion
grades 1–3, effusion; plus 6-class cartilage/meniscus segmentations. NO synovitis, contusion,
Baker's, OA grade. Obtain: stanfordaimi.azurewebsites.net (free account = DUA acceptance);
code/loaders github.com/StanfordMIMI/skm-tea (MIT code).

### OAI — NIH/NDA
MRI protocol citation: Peterfy CG et al., Osteoarthritis Cartilage 2008;16(12):1433-41. Access: NDA
Data Use Certification (nda.nih.gov/oai). Verbatim DUC clauses (March 2021 version): permitted use
"scientific investigation, scholarship or teaching, or other form of research and research
development" (§2 — no NC clause); "Recipients agree to retain control over data and to not
distribute, sell, or move data, with or without charge, in any form, to any other individual,
entity, or third-party system" (§3); no publishing "derived data that could aid in the
re-identification of any of the study participants… including structural MRI images" (§5 — whether
trained weights count is AMBIGUOUS); downloaded data "will be permanently deleted… when research is
completed or this DUC is expired" (§9, one-year term, renewable §19); an NDA Study per "publication,
computational pipeline, or other public disclosure of results" (§11). 4,796 subjects, 3T Siemens
Trio, bilateral, longitudinal ~8 years. Labels: MOAKS semi-quantitative reads on ~600 knees (FNIH
Project 22) — meniscal damage by subregion (degenerative, not tear), effusion-synovitis composite,
BMLs (chronic, not contusion), cartilage loss, popliteal cyst; ACL scored normal/complete-tear only.
KL grades from radiographs standard. Download size UNVERIFIED.

### MOST — NIA AgingResearchBiobank
Cohort citation: Segal NA et al., PM R 2013;5(8) (PMC3867287). Access: NIA Outgoing Human Data
Transfer Agreement. Verbatim: "You may not share the data (including images) with any other
researcher not named in the Outgoing Human Data Transfer Agreement and will be required to either
destroy the data (including images) or apply for an extension at the end of 12 months"; "the
agreement you will sign explicitly applies to the research that you describe in your request";
commercial organizations MAY apply. Requires PI, IRB approval/waiver, institutional signing
official; multi-week review. 3,026 + 1,525 participants; bilateral knee MRI to 168 months (1.0T
OrthOne early, 1.5T GE third cycle). Labels: WORMS + MOAKS central reads (MOAKS project: 1,854 knees
paired 144m→168m) — meniscal tear TYPE + root + extrusion, ACL/PCL tear/repair, Hoffa-synovitis,
effusion-synovitis ("can include both synovitis and effusion" — non-contrast caveat), cartilage
14-subregion, BMLs; KL 0–4 from radiographs. OA cohort: no acute contusion; MCL only via WORMS.

### The segmentation-era group
- **SKI10** (Heimann et al., MICCAI 2010 workshop): withdrawn 2019 ("we are no longer allowed to
  share the SKI10 dataset"); original rules: use "exclusively for the purpose of developing
  segmentation algorithms", no commercial use, no redistribution. ~150 volumes multi-vendor 1.5T/3T;
  bone+cartilage masks only. Unobtainable legitimately.
- **OAI-ZIB** (Ambellan et al., MedIA 52:109-118, 2019): 507 OAI baseline DESS volumes,
  384×384×160 @ 0.36×0.36×0.7 mm; femur/tibia+cartilage masks; KL distribution
  60/77/61/151/158 joinable from OAI clinical. Masks 86 MB direct from zib.de ("made publicly
  available", no licence); images inherit OAI DUC.
- **OAIZIB-CM** (Zenodo 2025, CartiMorph): 12.7 GB re-release incl. images, **CC BY-NC 4.0** — in
  tension with the OAI DUC's no-redistribution clause; flag before relying on it.
- **OAI-iMorphics**: 88 subjects × 2 timepoints, cartilage masks, moderate-to-severe OA; via OAI.
- **IWOAI 2019**: 176 DESS volumes, the only one with MENISCUS masks; OAI-derived; licence
  UNVERIFIED.
- **Kneeview** (UPF Barcelona, Zenodo 2025): 93 patients T2-FS, radiologist-verified bone geometry;
  CC BY-NC 4.0, verbatim "NonCommercial means not primarily intended for or directed towards
  commercial advantage or monetary compensation"; files restricted, access request required.

### Tiny/obtuse
- **Open Knee(s)** (Cleveland Clinic/SimTK): ~8 cadaveric knees, CC BY 4.0 verbatim "licensed under
  a Creative Commons Attribution 4.0 International License"; MRI + mechanics + FE models; no
  clinical labels.
- **Denver Natural Knee Data**: 7 cadaveric knees, custom licence verbatim "…without restriction to
  use, copy, merge, and publish the Data for non-commercial purposes…", citation required, possible
  co-authorship condition for publications; no pathology labels.
- **Duke-110K**: not public ("available… upon reasonable request"); only model weights released
  (github.com/mazurowski-lab/mri_foundation).

## Recommended order of action

1. **fastMRI DUA application** (biggest clean-ish pool; fastMRI+ labels are MIT and already public).
2. **KMAR-50K download** (free, CC BY, new population; pure encoder transfer).
3. **SKM-TEA AIMI registration** to read the DUA verbatim — the NC clause decides it.
4. OAI/MOST only if institutional sponsorship is real for this project; both are multi-week,
   renewable-yearly, delete-on-expiry commitments.
