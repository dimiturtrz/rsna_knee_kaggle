# Broad macro-AUC interventions at high baseline + concordance-adjudicated rulers — literature synthesis

Researcher lane, 2026-09-11. Sources for letter D325 (main's M332). All IDs verified at fetch time.

## PART 1 — what moves the MACRO mean broadly (M332 Q1), and the high-baseline regime (Q2)

### Family ranking by measured broad macro delta (all gains measured from ≤0.90 baselines)

1. **Input coverage (multi-plane/sequence)** — best-documented broad family in knee MRI. Belton et al.
   MIUA 2021 (arXiv:2108.08136, MRNet): single-plane → multi-plane fusion: ACL 0.953→0.972 (+0.019),
   meniscus 0.862→0.916 (+0.054), abnormal 0.952→0.957. Gain size inversely tracks baseline AUC.
   Confounded: plane-count tripling tripled parameters. Yang et al. QIMS 2025 (PMC12209677): planes
   contribute unequally per label (axial unexpectedly important).
2. **In-domain SSL pretraining** — Azizi et al. ICCV 2021 (arXiv:2101.05224): ImageNet-SimCLR + MICLe
   on unlabeled CheXpert → +0.011 mean AUC from a ~0.88 baseline, per-label table in supplement. The
   canonical BROAD pretraining delta, and it is +0.011. REMEDIS (arXiv:2205.09723): big headline is
   data-efficiency/OOD, not full-data AUC.
3. **Resolution** — plateau documented. Sabottke & Spieler, Radiology:AI 2020 (PMC8017385):
   ChestX-ray14, 9 resolutions 32²→600²: steep to 224, plateaus 256–448; only small findings keep
   gaining. Wollek et al. 2023 (arXiv:2306.06051): 256²→1024² = +0.015 mean, needs 4× resolution.
   **Our 336→392 null (−0.0002) is exactly what the published curve predicts.**
4. **Ensembling** — Pham et al. (arXiv:2005.12734): CheXpert 0.930 test / 0.940 val; the
   single→ensemble step crossed 0.93→0.94. +0.005–0.01 at the top.
5. **Pretraining corpus swap** — Transfusion (arXiv:1902.07208): measured NULL at full-data scale.
   Matsoukas CVPR 2022 (arXiv:2203.01825 — NOTE: 2204.02825 is a wrong ID for this paper): deltas in
   the ±0.01 band, shrink with target data size. RadImageNet (ryai.210315) contested: "Revisiting
   Hidden Representations" (arXiv:2302.08272) found ImageNet ≥ RadImageNet on 6/7 downstream sets.
6. **Trunk class** — wash at matched data: Matsoukas ICCVW 2021 (arXiv:2108.09038): CNN>ViT from
   scratch, ≈ with ImageNet pretraining, ViT>CNN only with SSL. MRNet backbone sweep (J Imaging 2022):
   ±0.01–0.02 per label, no winner. "When is 3D Worth It?" (arXiv:2606.06950): 2.5D beat 2D AND 3D on
   NLST. NULL: no knee-MRI video-trunk macro comparison.
7. **AUC-surrogate loss** — LibAUC/DAM (Yuan et al. ICCV 2021) won Stanford CheXpert 2020 — measured
   win on top of a ~0.92 leaderboard, but margin ~+0.001–0.005, leaderboard-scale not +0.01.
8. **Schedule/EMA/self-distillation/augmentation strength** — NULL: no measured broad macro deltas in
   multi-label radiology.

### Q2 — the 0.93–0.99 regime: published gains collapse toward noise

**No published demonstration of a +0.01–0.03 broad macro gain from a ≥0.93 baseline exists.** All
large deltas originate from 0.75–0.90 baselines. Evidence:

- MRNet lineage (domain-matched): baseline abnormal/ACL/meniscus = 0.937/0.965/0.847 (Bien et al. PLoS
  Med 2018, PMID 30481176). ELNet (Tsai et al. MIDL 2020, arXiv:2005.02706): the entire +0.057 landed
  on the sub-0.90 label; the two ≥0.93 labels moved +0.004 and −0.005. TransMed: +0.5% on 0.96-baseline
  ACL vs +5.7% on weaker meniscus. Belton multi-plane: +0.019 at 0.953 vs +0.054 at 0.862. **Three
  independent intervention families, same dataset, same law: delta ∝ (1 − baseline).**
- CheXpert SOTA progression: 0.88–0.89 (Irvin 2019) → ~0.93 (LibAUC 2020) → 0.930 test (Pham ensemble)
  — each step smaller, more apparatus.
- TorchXRayVision (Cohen et al. MIDL 2022, arXiv:2111.00595): seed/split AUC variance near the top of
  the range is the same order as late-stage intervention deltas.
- Mahmood et al. CVPR 2022 (arXiv:2207.01725): performance-vs-data is a power law — tail gains cost
  exponentially more data. NULL: no AUC-specific scaling-law fit for multi-label radiology.

**Bottom line for 0.93–0.99:** published-style broad single-model interventions read as ±0.005 noise at
that altitude; the only families with measured gains there are coverage additions that fix a specific
weak label, ensembling, and more data on a power law.

## PART 2 — concordance-adjudicated rulers (M332 Q3)

### Independence condition and how it's checked

Requirement: **conditional independence of ERRORS given the true label** (Hui–Walter). Measured
consequences of violating it:

- Vacek, Biometrics 1985;41:959-968 (PMID 3830260): conditional dependence → error rates of BOTH tests
  substantially underestimated — a correlated-pipeline concordance rule is believed more accurate than
  it is.
- Torrance-Rynard & Walter, Stat Med 1997;16:2157-2175 (PMID 9330426): bias more serious at LOW
  prevalence — rare labels are exactly the exposed regime.
- **Albert & Dodd, Biometrics 2004;60:427-435 (PMID 15180668) — the decisive negative result:** with
  few tests, likelihood comparisons and diagnostics CANNOT distinguish dependence structures. You
  cannot verify your 5 tables' independence from their own agreement pattern. The check must be
  external (gold audit) or structural (different information source).
- Sorin's design-level argument: image-based CV detector vs text-based LLMs = MODALITY independence —
  an argument 5 same-report, same-LLM-family tables cannot make.
- Voting theory: Berend & Paroush, Soc Choice Welfare 1998;15:481-488 — with correlated jurors,
  unanimity error does NOT vanish with K; the error floor is the correlated component.

### The object produced and the certified fraction (Sorin et al., JAMIA Open 2025, PMC12343119, verified)

- Corpus 11,999 CTA reports. Components: CV PE detector + 3 Gemini LLMs (cross-modality).
- Object: a HARD binary ruler over the full corpus — concordant cases (96.3%) accepted; any
  disagreement (448 = 3.7%) manually adjudicated by one radiologist.
- Audit: 400 balanced random concordant cases radiologist-reviewed → 100% matched → residual error
  ≲0.75% (rule-of-three).
- Discrepancy rate sets the REVIEW budget; only the audit bounds RESIDUAL error.
- Comparators: Geijer & Geijer, Insights Imaging 2018 (PMC5990995): human double-reading discrepancy
  0.4–22% across 46 studies. Funk et al., J Biomed Semantics 2015 (PMC4301805): ≥2-of-4 biomedical NLP
  systems' agreement → best F1 33%, precision ≤18% vs gold — the agreeing systems shared lexical
  machinery. **Shared-method agreement is weak evidence; measured.**

### Resolution: sampling error vs residual label error

- Hanley–McNeil (PMID 7063747), computed: AUC 0.9 at 27 pos/31 neg → SE 0.043, 95% half-width
  **±0.085** (D324's ±0.19–0.24 was WRONG — correction sent). At 27% prevalence, AUC 0.9: ±0.025 needs
  N≈1,100; N=4,407 → ±0.012. Paired ΔAUC=0.04 at 80% power: 225–919 per group (Emir et al., BMC Med
  Res Methodol 2026, PMC12924612). A 4,407-study concordance ruler resolves 0.05 on SAMPLING error by
  a wide margin.
- The binding constraint is residual ruler error: Northcutt et al., NeurIPS D&B 2021
  (arXiv:2103.14749) — ≥3.3% average test-set label error across 10 benchmarks; rankings of close
  models FLIP once mislabel prevalence rises ~6% (ResNet-18 beats ResNet-50 on corrected labels).
  Beyer et al. 2020 (arXiv:2006.07159, "Are we done with ImageNet?"): original consensus labels no
  longer best predictors of re-annotation; near-ceiling differences were partly label noise.
- Gold-58 bounds a ruler's error only coarsely (0 errors in 58 → ≤5% upper, rule-of-three). A
  STRATIFIED audit of concordant cells (Sorin's step 2) is what tightens: n≈400 audited cells with k
  errors → 95% bound (k+3)/400.
- Modern combination machinery: prediction-powered inference (Angelopoulos et al., Science 2023,
  arXiv:2301.09633) — valid CIs combining 4,407 machine labels + 58 gold; PPI-for-AUC estimator = NULL
  (framework covers means/quantiles/GLMs).
- Estimation over the 5-table array: Dawid & Skene 1979 (doi:10.2307/2346806), Raykar JMLR 2010
  (D320), Paun et al. TACL 2018 (Q18-1040) — with Albert & Dodd's identifiability warning applying.

### Pricing unanimous-cells-as-ruler (93.9% of 52,872 cells)

Published anchors for unanimous-but-wrong under correlation: Funk 2015 (precision 18%,
same-method NLP systems) at the bad end; Sorin (<0.75% residual, cross-modality) at the good end;
Northcutt (≥6% in consensus ImageNet labels) in the middle. Five same-report, same-LLM-family tables
sit nearer Funk than Sorin on the independence axis. If per-table cell error is ε with pairwise error
correlation ρ, unanimous-wrong is ~ρ·ε-scale, not ε⁴ (Berend–Paroush) — and the correlation plausibly
concentrates on exactly the hard cells (multilingual hedging, negation) that discriminate models. The
decider is an audit of unanimous cells against gold, over-sampling hard labels.
