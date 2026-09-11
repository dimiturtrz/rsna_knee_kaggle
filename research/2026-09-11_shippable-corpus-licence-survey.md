# Shippable-corpus licence survey (researcher, for M314)

Question: does a knee-MRI (or comparable joint/MSK MRI) corpus exist whose licence permits shipping
trained model weights in a competition kernel? Every licence below was read as primary text (operative
clause quoted, URL cited), not by licence name from a summary page.

## Bottom line

**No corpus carrying labels overlapping our 12 findings permits shipping a trained model.** Every
labelled knee-MRI dataset (SKM-TEA, OAI, Rijeka, K2S, MRNet, fastMRI) is locked by a research-use DUA
or an NC/ND licence. The only MSK-MRI collections whose licence permits redistributing derivatives —
including commercially — are TCIA's CC-BY ones, and the only one still live with real joint MRI is
Soft-tissue-Sarcoma (51 subjects, no finding labels). Everything else permissive is non-joint MRI
(M4Raw brain k-space).

## Tier 1 — licence permits shipping weights

### TCIA Soft-tissue-Sarcoma — CC BY 3.0 (the only shippable MSK-MRI corpus)
- Collection page states "License: CC BY 3.0"; CC BY 3.0 legalcode §3 grants "(b) to create and
  Reproduce Adaptations … (d) to Distribute and Publicly Perform Adaptations … in all media and
  formats whether now known or hereafter devised." No NC/ND.
- Trained models: licence SILENT (its own terms cannot block distribution either way).
  Commercial/competition: yes. TCIA data-usage policy confirms commercial use under CC BY 3.0/4.0.
- Contents: 51 subjects (McGill + outside), T1/T2-FS/STIR MRI of extremities (much of it peri-knee),
  FDG-PET/CT, tumour contours, clinical follow-up. Labels: tumour segmentation + metastasis outcome —
  ZERO overlap with our 12 findings. Self-supervision target only.
- Also CC BY on TCIA: TCGA-SARC (261 subjects, some Leg MR, CC BY 4.0 — per-collection badge worth one
  more check), QIN-SARCOMA. **RIDER Knee MRI is GONE** from TCIA/IDC collection lists (NBIA API
  queried) — do not build on it.

### M4Raw — CC BY 4.0 (genuinely shippable, non-joint)
- Zenodo 8056074: CC BY 4.0; §2(a)(1) "produce, reproduce, and Share Adapted Material", no NC/ND.
- 183 subjects, 0.3 T, multi-channel BRAIN k-space, T1w/T2w/FLAIR, no labels. Self-supervision only.

### Mendeley "Knee and sacroiliac joints MRI scans" — licence unverified
- 6 knee 3D MRI (RA, pre/post contrast) + 9 SI-joint volumes; page displays no licence (house default
  CC BY 4.0, but silence is a finding, not a permission). Negligible size regardless.

## Tier 2 — blocked (primary text quoted)

### OAI — DUA, research/education only, no redistribution
OAI Data Use Agreement (Wayback copy of the UCSF PDF): purpose limited to "Human subjects research /
Educational activities"; §4(c) requires downstream users to obtain data directly from the OAI
Coordinating Center (no redistribution). Silent on trained models, but the purpose limitation confines
use to research/education — a Kaggle kernel is neither. ~4,796 subjects, 3T MRI (sag DESS-we + TSE),
MOAKS labels on a subset (OA, effusion/synovitis, meniscus, BML) — the best label overlap of any
candidate, which is exactly why this hurts. Access now via NDA, likewise research-gated.
Corollary: everything OAI-derived inherits this — SKI10 (MICCAI 2010), KNOAP2020, Kaggle
`mohamedberrimi/oaimri3ddess` (licence field literally "unknown"), and `kgaooo/oai-tissue-segmentations`
(uploader-tagged CC0, but a Kaggle tag cannot launder the OAI DUA; masks only, no images).

### SKM-TEA (Stanford, 155 qDESS knees) — explicit "no derivative works"
Paper Appendix A.5: distributed under Stanford SoM license "and the terms listed for the Lower
Extremity Radiographs dataset". Those terms, verbatim: "1. … personal, non-commercial research
purposes only … 4. YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY … 6. You must not modify,
reverse engineer, decompile, or **create derivative works** from the … Dataset." The strongest
anti-shipping clause in the survey (stronger than fastMRI's). 155 patients, 3T GE, qDESS 2-echo,
k-space + DICOM + T2 maps, cartilage/meniscus segmentation + bounding-box pathology labels for 16
categories — best label overlap after OAI, fully unusable.

### K2S (MICCAI 2023, 300 knee k-space + segmentation)
No public licence; access "from the coauthors upon reasonable request"; registry lists CC BY-NC-ND.
Blocked.

### Rijeka kneeMRI — confirmed identical on Zenodo
Zenodo 14789903: CC BY-NC-ND 4.0. The Kaggle re-upload (`sohaibanwaar1203/kneemridataset`, licence
"unknown") is the same data. No new finding.

### Kaggle sweep (licences read from each dataset's own metadata via API)
Nothing usable: `abbymorgan/pcir-knee-mri` is CC0 but a single-patient CT (toy); all other knee-MRI
hits are this competition's own data (competition rules govern those, not an external licence).

## Legal note — are trained weights a "derivative work"?

Genuinely unsettled. The U.S. Copyright Office's May 2025 Part 2 report holds that weights implicate
reproduction/derivative rights of training works only to the extent the model has MEMORISED
protectable expression it can regurgitate — a property a discriminative 2D/3D classification encoder
essentially does not have (the best available argument that our members' weights are not adaptations
of any scan). Creative Commons' own AI-training guidance frames training as copying but stops short of
declaring weights derivative works; doctrinal commentary (Kluwer Copyright Blog) notes
weights-as-numbers resist classification as protected expression; Data Provenance Initiative (Nature
MI 2024) surveys the open questions. Practical consequence: under CC BY/CC0 the question never needs
answering (the licence grants derivative creation and distribution regardless); it matters only
against ND licences and "no derivative works" DUAs, where the conservative reading controls.
