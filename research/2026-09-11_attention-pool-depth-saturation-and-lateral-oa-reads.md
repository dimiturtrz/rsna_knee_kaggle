# Class-specific attention pool depth saturation + lateral tibiofemoral OA reads — literature synthesis

Researcher lane, 2026-09-11. Sources for letter D323 (main's M329). All citations verified against
abstract/full text at fetch time. Declared nulls are listed explicitly at the end.

## Q1 — does a class-specific attention pool's depth gain saturate, and where?

### Measured curves (the only two on-regime ones that exist)

1. **Ilse, Tomczak, Welling — ABMIL, ICML 2018, arXiv:1802.04712.** The ONLY verified paper sweeping
   instances-per-bag as the independent variable with pooling type as a factor. MNIST-bags, mean bag
   size {10, 50, 100}: attention AUC 0.996 vs instance+max 0.553 at bag size 100 / 50 training bags;
   at bag size 10, attention ~0.80–0.86 vs max ~0.55–0.75. Stated findings: attention's advantage is
   LARGEST in the small-sample regime; with enough data all pooling types converge; "the more instances
   in a bag the easier the MIL task becomes". Caveat: single binary attention query, not per-class
   branches. Bag sizes 10–100 are our regime (12–32).
2. **Li et al., MICCAI 2025, LNCS 15960:312–321, doi:10.1007/978-3-032-04927-8_30** — MIL over CT
   slices per lymph node, slice-count sweep: 9 slices AUC 81.12±2.78 → 18 slices 82.54±2.10 → 27
   slices 81.77±2.60. **Saturates by ~18, slightly degrades at 27** — a bag regime nearly identical to
   ours. Single-source-flagged (numbers from indexed text + mirror PDF).

### Mechanism in print for why per-class attention out-pays max at depth

- CLAM (Lu et al., Nat Biomed Eng 2021, PMC8711640), verified text: max-pooling "uses the gradient
  signal from only a single instance in each slide" — max's training signal is one slice deep no matter
  the bag depth, so extra slices past the best one add nothing. CLAM-MB runs "N parallel attention
  branches ... N unique slide-level representations ... each determined from a different set of highly
  attended regions ... strong positive evidence for one of N classes". Main's hypothesized mechanism
  (each class query surfaces candidates a global argmax discards) is this, verbatim in the primary
  paper; softmax-weighted sums keep extracting signal from added informative instances where argmax
  cannot.
- Ilse et al.: "attention serves as a gradient update filter during backpropagation".
- Counter-direction on record: FocusMIL (arXiv:2408.09449) argues max is more robust to domain shift
  (attention latches spurious correlations) — about shift, not depth; warns the attention>max ordering
  is not universal.
- Multi-branch > single-branch at fixed content, no depth sweep: CKD WSI study (J Pathol Transl Med
  2025): CLAM-MB 0.942 vs CLAM-SB 0.927 vs classic MIL 0.866; HMIL (arXiv:2411.07660) same direction.

### Regime-transfer caution

WSI bags are 10^3–10^5 patches (TransMIL, arXiv:2106.00908: ~8.8k Camelyon, ~14.6k TCGA) and no WSI
paper publishes an instance-count saturation/degradation curve for attention MIL. The two on-regime
curves above are the whole of the measured evidence. Both are consistent with our reading (max flat by
12–20, per-class attention still gaining at 20) and with a knee somewhere in the 18–32 window — but the
MICCAI curve is lymph-node CT, not knee MRI, and its 27-slice regression is within noise (±2.6).

**NULL declared:** no published accuracy-vs-instances curve separated by class-specific vs shared/max
aggregation exists. Our 12/16/20/24 measurement under both heads would be the first.

## Q2 — what reads LATERAL tibiofemoral OA

### Sequence/plane

- Cornerstone sequence for cartilage/OA morphology: 2D FSE intermediate-weighted FAT-SUPPRESSED, TE
  30–40 ms — Omoumi et al., Skeletal Radiol 2023;52:2185–2198, PMID 37154871. FS-IW also carries the
  BML read.
- BMLs specifically need fat suppression: Del Grande et al. 2014 (PMC4126144) — STIR/fs-T2 "most
  sensitive for bone marrow lesions"; Mirowitz et al., AJR 1994;162:215–221, PMID 8273669 — measured
  BML conspicuousness fs-T2/STIR ≫ T1. **Our non-FS extra acquisitions are blind to BMLs by physics.**
- No plane is compartment-specific: cartilage grading uses sagittal + coronal FS-IW together (MOAKS
  delineates femur/tibia subregions on the sagittal projection — PMC4058435).

### Is lateral harder / rarer? Yes on both, measured.

- Rarer: Stoddart et al., OAC 2021;29:445–455, PMID 33253887 (meta-analysis, 16 studies / 3,786
  knees): isolated medial TF OA 27%, isolated PF 18%, medial+PF 23%; ALL single/bicompartmental
  patterns involving lateral TF sum to 15% (95% CI 8.5–18.7). A rarer class at the same absolute error
  reads as a weaker AUC.
- Harder for radiologists AND CNNs (meniscus analogue, surgery-referenced): Fritz et al., Skeletal
  Radiol 2020;49:1207–1217, PMID 32170334: sensitivity medial 93–96% vs lateral 67–71%; inter-reader κ
  medial 0.876 vs lateral 0.741; DCNN AUC medial 0.882 vs lateral 0.781. Same compartmental asymmetry,
  same anatomy, hard gold standard.
- Lateral cartilage under-read FROM REPORTS: Krakowski et al., Appl Sci 2021;11:1552,
  doi:10.3390/app11041552 — 190 arthroscopy-correlated knees read from clinical reports: medial
  accuracy significantly higher than lateral; lateral tibial condyle AUC 0.578 (lowest of five
  locations); report sensitivity for any chondral lesion 24–42%. A report-derived-label study — the
  exact channel our ruler is mined from.

### Geometry: could a 0.1–0.9 coronal depth band under-sample the lateral compartment?

- Measured asymmetry that makes it plausible: Bloecker et al., JMRI 2013, PMC3696373 (OAI, 3D
  morphometry): lateral meniscus covers 59±6.8% of the lateral tibial plateau vs medial meniscus
  50±5.5% of medial (p<0.001) — proportionally more lateral articular surface is hidden under meniscus
  on every coronal slice.
- Supporting anatomy (textbook-grade, flagged as non-measurement): lateral tibial plateau is convex in
  the coronal plane (medial concave) and smaller; a convex plateau presents a thinner cartilage arc per
  coronal slice and tails off faster toward stack edges — a proportional band samples fewer INFORMATIVE
  lateral slices even at equal slice counts. Geometric derivation, not a published measurement.
- **NULL:** nobody has counted coronal slices over which lateral vs medial cartilage/joint line is
  visible, and no work exists on slice-sampling bias between compartments for banded MIL.

### Do MOAKS/WORMS/BLOKS treat the compartments differently?

- Subregion granularity is SYMMETRIC (MOAKS Hunter et al. OAC 2011, PMC4058435; WORMS Peterfy et al.
  OAC 2004; BLOKS Hunter et al. ARD 2008, PMID 17472995 — PMID single-source-flagged).
- What surfaces in a label table is RELIABILITY, not granularity: MOAKS's own reliability study has its
  weakest kappas exactly on cartilage/tibial features — inter-rater tibial cartilage area κ=0.36,
  tibial osteophytes κ=0.49 — vs 0.61–1.0 for most other features. The noisiest channel in the
  reference scoring system is the cartilage channel itself; report-based cartilage calls are noisier
  still (Krakowski).
- **NULL:** no quantification of how report text lumps/omits PF or lateral TF vs SQ scoring.

## Synthesis (as sent in D323)

Lateral TF OA's weakness is OVER-DETERMINED in the literature: rarer (15% of patterns), harder for
radiologists (κ 0.741 vs 0.876), hardest to read from reports (AUC 0.578 at LTC), noisiest feature
family in the reference scoring system (κ 0.36), and anatomically less exposed per coronal slice (59%
vs 50% meniscal coverage). None of it requires a modelling failure. The one fixable-on-our-side
candidate is the geometry term: a proportional depth band cannot adapt to the compartments' different
slice-extent profiles, and the lateral plateau's convex shape plausibly concentrates its informative
slices more narrowly. Unmeasured anywhere, including by us.

## Declared nulls

1. No per-class-vs-shared/max accuracy-vs-instances curve anywhere (WSI or medical MIL).
2. No WSI instance-count saturation/downsampling experiment for attention MIL.
3. No coronal slice-coverage asymmetry measurement (lateral vs medial).
4. No report-lumping quantification for PF/lateral TF compartments.
