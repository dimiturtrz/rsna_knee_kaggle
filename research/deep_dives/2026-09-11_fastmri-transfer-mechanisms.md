# fastMRI / fastMRI+ as a transfer source — mechanisms (researcher, for M313-Q2)

Context: fastMRI/fastMRI+ can only be used OFFLINE (licence bars shipping trained models — see
`2026-09-11_shippable-corpus-licence-survey.md`), so the question is which offline use the literature
supports. All arXiv IDs verified.

## Q1 — the domain gap, measured

Every measurement runs the same direction: cross-site/cross-protocol transfer costs AUC, and small
amounts of target-domain training recover most of it. Protocol mismatch taxes the gain.

- **MRNet (Bien et al. 2018, PLoS Med, PMC6258509)**: Stanford sagittal T2-FS (n=1,370) evaluated
  zero-shot on Rijeka sagittal T1 (n=183): ACL AUC 0.824 vs 0.965 internal — a −0.14
  internal→external gap, shrinking to −0.05 vs a same-site model. Cross-site AND cross-sequence at
  once.
- **Tran et al. 2022 (Eur Radiol, Incepto)** — closest analogue to our setting (NLP-mined labels,
  multi-plane FS-PD/FS-T2, multi-scanner, n=19,765): internal 0.939; external zero-shot **0.941 on
  MRNet** and **0.860 on KneeMRI/Štajduhar** (the public corpus most like fastMRI knee); after light
  retraining (LR/10) 0.962 / 0.922. The authors credit multi-plane, multi-protocol, 12-centre
  training diversity for the small zero-shot gap.
- **Germann et al. 2020 (Invest Radiol, PubMed 32168039)**, surgical ground truth, n=512 from 59
  institutions: 0.967 in-house vs **0.898 outside-institution** — a −0.07 heterogeneity penalty —
  while 1.5T vs 3T made NO significant difference. Institution/protocol, not field strength, is what
  costs.
- **Lu et al. 2024 (arXiv:2409.04368)**, systematic: 7 datasets, 14 scanner pairs, 10 seeds. Mean
  cross-manufacturer AUC drop: **MRI −0.097**, X-ray −0.067, CT −0.02. MRI is the most vulnerable
  modality; mixing target-scanner data helps specifically for MRI, asymmetrically.

NULL: nobody has published fastMRI-knee → MRNet/KneeMRI (or reverse) classifier transfer. The closest
evidence says multi-plane/multi-protocol TRAINING diversity is what protects external performance —
which cuts against expecting much from a sagittal-PD-only SUPERVISED transfer, and for using fastMRI
as an SSL pool rather than a label source.

## Q2 — slice-level boxes, unreliable absence: the published handling

- **CheXpert (Irvin et al. 2019, arXiv:1901.07031)**: the canonical uncertain-label policies.
  U-Ignore (loss masking) is the standard baseline — competitive, never dominant: "different
  uncertainty approaches are useful for different pathologies" (U-Ones best for Atelectasis,
  U-MultiClass for Cardiomegaly). Measured on 200-study 3-radiologist consensus val.
- **PU learning measured in medical imaging**: Zhao/Pang/Liu/Ye (MICCAI 2021w / MELBA 2022,
  arXiv:2106.15918) — incomplete box annotations as positive-unlabelled, nnPU risk estimator
  (Kiryo 2017): mAP 0.530 vs 0.501 for treat-absence-as-negative (paired t, p<0.05). Demonstrated
  caveat: porting classification PU naively fails when unlabelled and positive samples share images —
  the fastMRI+ slice-to-study setting has exactly that structure.
- fastMRI+ itself (Zhao et al., Sci Data 2022, arXiv:2109.03812): single annotator per study, no
  inter-rater reliability, annotation not exhaustive — and the authors benchmark NO supervised
  classifier or label-handling on the knee data.

NULL: no head-to-head of PU-loss / U-Ignore / U-Zeros on fastMRI+ knee study-level labels exists.
Published fastMRI+ knee usage is anomaly detection, not supervised multilabel classification.

## Q3 — organ-specific continuation vs generic-medical breadth: the distinction is REAL

- Gururangan et al. (arXiv:2004.10964, "Don't Stop Pretraining"): DAPT gains on all 8 tasks; TAPT adds
  ON TOP of DAPT — the order is sequential general→domain→task, each narrower phase adds.
- Azizi et al. 2021 (arXiv:2101.05224): ImageNet SSL THEN continued SSL on in-domain medical images
  beat supervised ImageNet pretraining (+6.7% top-1 dermatology, +1.1% mean AUC CXR). Exactly the
  "continued on the organ" recipe.
- RETFound (Zhou et al., Nature 2023, PMC10550819): MAE continuation from SSL-ImageNet onto 1.6M
  retinal images beat SL-ImageNet, SSL-ImageNet and from-scratch-retina SSL, internal and external,
  5 seeds with CIs.
- MoCo-CXR (Sowrirajan et al. 2021, arXiv:2010.05352): in-domain CXR contrastive pretraining improved
  linear probes, strongest at small data.
- The generic-medical side is genuinely mixed, matching our five-trunk finding: "Revisiting Hidden
  Representations" (arXiv:2302.08272) found ImageNet-ResNet50 BEAT RadImageNet-pretrained on most of
  7 medical targets. RadImageNet's own claims (Mei et al. 2022, PMC9530758) are contested. The
  ORGAN-SPECIFIC continuation claim is the one with consistent replications. One measured caution:
  arXiv:2511.22001 (retinal SSL) found domain-specific pretraining helped only the HARDEST downstream
  task — the benefit is task-selective; verify per label, don't assume.
- Co-train vs pretrain order: NULL — no published head-to-head of supervised co-training with masked
  losses vs SSL-pretrain-then-fine-tune on an auxiliary noisy corpus, in knee MRI or anywhere in
  medical imaging. All the precedents (Azizi, RETFound, MoCo-CXR, Gururangan) are
  pretrain-then-fine-tune.

## Bottom line

The literature favours continued SSL on the fastMRI images (organ-specific DAPT) over supervised
co-training on fastMRI+ labels — the latter additionally inherits an unmeasured PU problem and a
single-sequence/single-plane gap the cross-institution literature prices at −0.07 to −0.14 AUC before
adaptation. All of this is OFFLINE-only use (licence), so its value is as an instrument: a
continuation arm on fastMRI images would be the cheapest test of whether ortho_dense's +0.0194
organ-continuation effect replicates on a second source.
