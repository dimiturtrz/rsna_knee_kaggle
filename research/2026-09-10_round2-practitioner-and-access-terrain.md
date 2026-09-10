# 2026-09-10 — Round two: practitioner terrain and data-access reality

Owner-directed discovery, round two: what winning teams actually do, what the public corpora
cost to use, and 2026-era models not yet priced. Cross-checked against the priced set; "covered"
means the project has already measured the axis. Provenance: letters D307/D308; companion files
2026-09-10_unseen-ceiling-breakers-discovery.md (round one) and
2026-09-10_unseen-candidates-mapped-onto-our-measurements.md (the mapping). Not committed.

## Genuinely new to us, ranked

1. **arXiv 2606.11606 (Muthyala et al., Jun 2026) [CLF] — the strongest round-two find.**
   Frozen embeddings of RAD-DINO, DINOv2-B/14, DINOv3-7B, BiomedCLIP, MedSigLIP probed on
   492,724 CXR + ChestX-Det10: **CLS/global pooling sits at chance (AUC 0.500–0.524) on
   small-footprint signal while patch-local pooling from the same forward pass recovers
   ≈1.0**; real-lesion small-vs-large stratum gap up to +0.243 AUC, recovered to ≥0.899 by
   region-restricted pooling. Generic DINOv3 ≈ medical FMs on global tasks (echoes Demir's
   negative). In our context: our members pool cls-mean, and our own deficit sits on FOCAL
   findings while reader ambiguity sits on diffuse ones
   (interpretations/kneemri/2026-08-26_label-ambiguity-and-our-deficit-do-not-overlap.md) —
   this paper is a measured, mechanism-level argument that a global-pooling trunk silently
   discards exactly the signal we are missing. It also lands on the cross-decode axis as a
   candidate toll mechanism (see cross-decode-toll file).

2. **Winner pattern: hand-corrected labels + canonical SHAPE alignment + external coordinate
   pretraining** (RSNA 2024 lumbar-spine medalists: 1st place self-describes
   "Data Cleaning + FE + External Data + Ensemble"; 4th place keypoint detection +
   surrounding crops; hengck23's repo carries hand-corrected annotation CSVs and mean-shape
   registration to a canonical reference; a shared Lumbar Coordinate Pretraining Dataset
   existed). The geometry-alignment half is distinct from our priced INTENSITY
   harmonisation. Our knee competition has no point annotations, so the direct analog is
   limited — but "winners hand-fix labels and hand-align anatomy" is the consistent
   practitioner differentiator.

3. **Winner pattern: segmentation-as-classification** — MIC-DKFZ took 7th of 1,147 in the
   2025 intracranial-aneurysm competition with pure nnU-Net segmentation and
   segmentation-derived scores. Countervailing: Demir CVPRW-2026's priced negative that a
   0.98-Dice task-specific U-Net produced the WEAKEST downstream features.

4. **Healthy-persona DDPM counterfactual features — Chen et al., MICCAI 2025 [CLF]**,
   MRNet N=1,130: inpaint a patient-specific HEALTHY knee from a DDPM trained on 218 healthy
   studies, extract radiomics from both, learned per-patient feature selection → 0.90
   abnormal / 0.87 ACL / 0.87 meniscus accuracy; ablations significant (p=0.023/0.018). New
   mechanism (counterfactual difference features), modest scale, accuracy not AUC.

5. **Practitioner pooled-corpus constant ~+0.02 CV** — RSNA breast-screening 20th place:
   VinDr + CBIS pooled, "boosted my CV by ~0.02" (secondary source, flagged). First measured
   pooled-corpus effect outside pairwise transfer — note it is CXR/mammo, and our merged3
   measurement says the knee version of this reads null (mapping file, row 1).

6. **AnyMC3D (arXiv 2512.12887v3, Siemens)** [CLF]: frozen 2D FM + ~1M-param/task plugins
   beat 3D architectures across 12 3D-classification tasks and explicitly beats the
   RSNA-Kaggle 3D winning solution as baseline. Citable external proof point for the
   project's 2D-slice architecture.

## Competition-public facts

- The 58-of-4,407 gold-label reality is now PUBLIC (Kaggle discussion 740162).
- A public Kaggle dataset of LLM-extracted report labels exists
  (stevenleehans/rsna-knee-llm-report-labels); label sharpening is live discussion.
- Public slice-geometry experiment: center-9 adjacent beats 9 equal-spaced across 24 by
  +0.018 on fold 0 (single fold, discussion 737597).
- Kaggle discussion pages are cookie-walled to fetchers — full thread text needs a
  credentialed `kaggle` CLI pass (main's lane; this lane spends no credentials).

## Data-access reality for joint pooling

- MRNet is mirrored on Kaggle (cjinny/mrnet-v1) — zero-friction in-kernel; licence on the
  mirror is "Unknown" and the CC-BY claim traces to a TFDS GitHub issue, not a primary
  licence document. Flag.
- **fastMRI licence CONFLICT, unresolved:** the paper says "we do not allow any commercial
  use", the AWS Open Data registry lists MIT. A prize competition is the grey zone — a
  human must read the NYU DUA before pooled weights ship.
- Still nothing published pooling ≥2 public knee corpora for classification — and per the
  mapping file, our merged3 is the only measured instance anywhere, and it reads null.

## Silent

MedGemma/MedSigLIP on MSK MRI (nothing; nearest is MedGemma-on-MURA radiographs, ASSERTED,
no numbers) · any MSK foundation model newer than OrthoFoundation/OrthoDiffusion (the
latter is already vendored in external/) · diffusion-generated knee MRI augmentation with a
classification delta (all evidence is X-ray or brain, +~3% acc) · MICCAI 2026 knee/MSK
multi-label grading papers · any 2026-H2 paper moving a multi-label MRI grading ceiling at
~4k studies.

## Flagged unverifiable

MRNet licence provenance · fastMRI paper-vs-registry conflict · the +0.02 VinDr/CBIS figure
(secondary mirror) · MedGemma/MURA [ASSERTED] · Kaggle discussion evidence is snippet-level
until a credentialed pass.
