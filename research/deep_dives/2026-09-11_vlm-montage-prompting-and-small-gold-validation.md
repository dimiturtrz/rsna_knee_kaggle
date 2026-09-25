# Generated-gold via VLM: montage geometry, prompt structure, small-gold validation — literature synthesis

Researcher lane, 2026-09-11. Sources for letter D324 (main's M330, plus M327-Q3 precedent).
All IDs verified by fetching abstract/PMC/Crossref/ACL pages at fetch time. Declared nulls at the end.

## Q0 (M327-Q3) — precedent for per-class leaderboard probing by constant substitution

**NULL for the specific design.** Nobody has published recovering per-class AUCs of a column-mean
multi-label metric by replacing one class's predictions with a constant. The oracle-attack family it
belongs to:

- Whitehill, "Exploiting an Oracle That Reports AUC Scores in Machine Learning Contests", AAAI 2016,
  arXiv:1506.01339 — attacks using leaderboard AUC feedback; proof that binary labelings consistent
  with an observed AUC grow exponentially in n.
- Whitehill, "Climbing the Kaggle Leaderboard by Exploiting the Log-Loss Oracle", 2017,
  arXiv:1707.01825 — full test-label recovery in batches on a real Kaggle competition (Intel &
  MobileODT), rank #4/848 at log-loss 0.00000 without training a classifier.
- Dwork et al., "The reusable holdout", Science 2015;349:636-8, PMID 26250683 (full version NeurIPS
  2015, arXiv:1506.02629) — prices adaptive holdout reuse generally (Thresholdout).
- Blum & Hardt, "The Ladder", ICML 2015, arXiv:1502.04585 — leaderboard mechanism robust to adaptive
  querying. Zheng, arXiv:1510.03349 — simplification, sample complexity cubic in leaderboard precision.
- "Reconstructing Test Labels from Noisy Loss Scores" (Amazon Science extended abstract) — reduced
  confidence, existence and PDF verified only.

Mechanism note: constant substitution works because column-mean AUC is linear in each column and a
constant column scores exactly 0.5 — one probe per class recovers that class's test AUC from the
board's own number. The literature says such oracles leak (Whitehill ×2) and that adaptive consumption
degrades holdout validity (Dwork, Blum–Hardt); a per-class probe would be a novel, citable instance.

## Q1 — montage geometry for a volumetric study read by a VLM through one image

Overall signal: **fewer tiles at higher per-tile resolution, or separate images, beat dense montages**;
tiling perturbs the 2D statistics VLMs are pretrained on.

### (a) Tile count / layout — measured

- SpatialMed (Ding et al. 2026, arXiv:2603.13800), Table 5 / §A.8: 3 orthogonal center slices vs a
  20-axial-slice tiled grid, three models — montage loses on average for all three (GPT-5.4: 39.23 vs
  35.90; InternVL3.5-30B: 52.73 vs 51.37; HealthGPT-XL32: 62.44 vs 41.69, −20.75). Gap LARGEST for the
  medical-specialized model: "tiling perturbs the visual statistics it relies on most". Montage won
  isolated sub-tasks (directional reasoning 57.14 vs 42.48) — coverage helps when resolution loss is
  tolerable.
- UCSF-PDGM-VQA (Ghosh et al. 2026, arXiv:2605.17140): brain-MRI VQA, 2,387 QA pairs; montage of best
  slices HARMED vs separate slices; MedGemma-1.5 multi-slice 63.57% vs single-slice 55.37% (+8 from
  separate multi-slice input).
- LoCoVQA (Sharma et al., EMNLP 2024 Findings, ACL 2024.findings-emnlp.312): VLM accuracy decays ~log
  in number of in-context images.
- Visual Haystacks (Wu et al., ICLR 2025, arXiv:2407.13766): degradation + placement bias as image
  count grows.
- ChatGPT-4V chest CT (Jpn J Radiol 2024, doi:10.1007/s11604-024-01606-3): practical ceiling ~4 CT
  slices per conversation.

**NULL:** no accuracy-vs-grid-size curve for a medical finding on montaged slices. The two 2026 papers
compare montage to fewer/better inputs, not montage sizes against each other.

### (b) Multi-plane tiles in one montage vs single plane

- SpatialMed IS this comparison in one direction: 3-orthogonal-plane at native-ish resolution beat
  single-plane 20-slice montage on average → literature favors multi-plane FEW slices over single-plane
  MANY tiles.
- UCSF-PDGM-VQA: multiple montages (axial+sagittal+coronal as separate images) still underperformed
  separate slices.
- Med3D-R1 (Lai et al. 2026, arXiv:2602.01200): multi-slice montage + orthogonal key slices as input
  design for a TRAINED 3D VLM (SOTA 41.92% CT-RATE) — precedent, not ablation.

### (c) Slice index / sequence name as text in tile

**NULL.** Adjacent: MedBLINK (ICCV-W 2025, arXiv:2508.02951) — models detect visual marks at 99–100%
but tasks REQUIRING use of the mark score barely above chance (41.3% vs 33.3% random); BLINK (ECCV
2024, arXiv:2404.12390) — GPT-4V 51.26% vs humans 95.70% on visual-prompt reference. Marks are seen,
not reasoned about. In-tile text indices are unproven, not established help.

### (d) Native resolution per tile vs downscale-to-fit

- LLaVA-UHD (ECCV 2024, arXiv:2403.11703): native-resolution adaptive slicing beats fixed low-res
  (+6.4 TextVQA, +3.2 POPE at equal compute); GPT-4V shows a PHASE CHANGE with input resolution —
  downscaling is not a smooth knob.
- MedBLINK resolution ablation (GPT-4o): negligible on its two most resolution-sensitive tasks.
- SpatialMed mechanism: montage cost = "substantial per-slice down-scaling"; its most
  resolution-sensitive sub-task is where montage lost most.
- Synthesis: resolution-sensitive findings punish dense grids; coverage-sensitive reasoning tolerates
  them. Synovitis is the former. No synovitis-like small-finding vs per-tile-resolution curve exists.

## Q2 — prompt structure for an under-called, low-prevalence finding

### (a) Grade vs binary — the one clean comparison points the OTHER way

- Chetla et al., Clin Ophthalmol 2025, PMC12411675 — 2,500 fundus photos, 8 prompt designs: graded
  5-level prompts produced a strong bias toward "absent" (sensitivity for mild DR 0.028–0.040);
  grading UNDER-CALLS — collapses adjacent grades into "no disease". No over-call penalty because the
  error mass went the other direction.
- Guo et al., MELBA 2025:004 — LLaVA-Med on MIMIC-CXR/CheXpert: pathology-EXPLANATION prompts (definition
  + visual findings) raised Recall substantially at minimal Precision cost (F1 +0.115 to +0.194);
  weak-learner-augmented prompt then cut Edema FPs by 78.5%. Closest published evidence that prompt
  engineering moves the FP:FN operating point in a CONTROLLED direction: explanation prompts recover
  FNs, reference-classifier prompts suppress FPs.
- Sorin et al., JAMIA Open 2025, PMC12343119 — 11,999 PE reports, 22 prompt variants: explicit
  "prioritize minimizing false negatives" moved recall to 0.997; FP counts ranged 3–56, FN 3–104 across
  variants; few-shot improved recall; chain-of-thought DEGRADED performance. Prompt framing alone spans
  a wide FP:FN range — a generated table does NOT inherit a fixed error shape, it inherits whatever the
  prompt sets.

**NULL:** no study measures, for an under-called imaging finding, grade-vs-binary sensitivity recovery
at a measured FP cost.

### (b) One finding per call vs all findings in one call

**NULL for a measured head-to-head in medical VLM/LLM labeling.** Adjacent: MADE benchmark
(arXiv:2604.15203) uses single joint prompt (community default) and finds self-verbalized confidence
unreliable; ImpressionGPT (arXiv:2304.08448, TMLR 2024) does per-category iterative prompting for
GENERATION; Sorin et al. run each prompt–report combination in a separate call by convention.

### (c) Self-consistency — measured in medical extraction

- Liu et al. 2026, PMC13060412 — 3 open-weight LLMs, 4 binary clinical extraction tasks, 100 repeats:
  majority-vote self-consistency over 10 samples raised Fleiss' κ substantially and flattened the
  κ-vs-temperature curve for every model/task; F1 preserved or modestly improved. Prompt PARAPHRASING
  degrades agreement more than temperature — self-consistency over paraphrases prices a real variance
  source.
- Wang et al., Self-Consistency, ICLR 2023, arXiv:2203.11171 — the method; reasoning benchmarks.
- Sorin et al.: repeat runs at temperature 0 still vary across cloud regions (Gemini-Pro sensitivity
  0.90–0.998) — nondeterminism exists even at temp 0; repeat-and-vote prices it.

## Q3 — validating a generated label table against the gold you tuned on

Named protocols, in order of relevance:

1. **Split-the-gold — the CheXpert precedent** (Irvin et al., AAAI 2019, arXiv:1901.07031): labeler
   developed/tuned on a 200-study radiologist set, final comparison on a SEPARATE 500-study consensus
   test set. The canonical radiology answer: split, never touch the second half until frozen. At n=58:
   ~29/29, with CI implications below.
2. **Snorkel practice** (Ratner et al. NeurIPS 2016, arXiv:1605.07723; VLDB 2020, PMC5951191; DryBell
   SIGMOD 2019, arXiv:1812.00417): small hand-labeled dev set for LF iteration, generative model
   estimates LF accuracies WITHOUT gold; unbiased evaluation requires gold the LFs never saw.
3. **Snuba** (Varma & Ré, VLDB 2018, PMC6879381): synthesize heuristics FROM the small labeled set with
   statistical guards (low-cardinality, high-accuracy-on-labeled-subset only) — treats the 58 as
   fitting data with overfit controls, not as a ruler.
4. **Partial identification** (Maia Polo et al., NeurIPS 2023/2024, arXiv:2312.04601): bounds on
   accuracy/precision/recall/F1 of a weakly supervised model WITHOUT ground truth, from LF marginals.
   The only published protocol whose explicit purpose is validating a programmatic label table when
   gold is too small to spend.
5. **Confident Learning** (Northcutt et al., JAIR 2021, arXiv:1911.00068): estimate the joint of
   given-vs-true labels from out-of-sample predicted probabilities — audit the GENERATED table's error
   rate without consuming gold.
6. **Cross-fitting** (Chernozhukov et al., Econometrics J 2018, arXiv:1608.00060): the generic
   machinery. NULL: nobody has operationalized cross-fitting of a labeling FUNCTION against n≈50 gold.
7. **WRENCH** (Zhang et al., NeurIPS 2021 D&B, arXiv:2109.11377): WS evaluation protocols are not
   comparable across papers — the field's own warning.
8. **Concordance-adjudicated ruler** (Sorin et al. 2025): ground truth by concordance between
   independent pipelines (vision model + LLM runs), manually reviewing only discrepancies (448) plus a
   400-case audit. If the pixel-VLM column and the report-extractor column are near-independent, this
   is a published way to get a LARGER ruler than 58 without re-annotation.

**CI widths at n≈50 — a null plus arithmetic.** No WS paper reports measured bootstrap CI widths at
n≈50. Hanley & McNeil (Radiology 1982;143:29–36, doi:10.1148/radiology.143.1.7063747) closed-form SE,
computed here (flagged as derived): at 27 pos / 31 neg, SE(AUC) ≈ 0.12 at true AUC 0.80, ≈ 0.096 at
0.90 → 95% CI half-width ≈ ±0.19–0.24 on the FULL 58, ≈ ±0.27+ on a 29/29 split. A split-the-gold
protocol at n=58 can certify "better than chance" or "catastrophic vs not" — it cannot resolve "the
VLM column beats the report column by 0.05".

## Declared nulls (searches that returned nothing)

- Per-class leaderboard constant-substitution probing (the M327 design itself).
- Accuracy-vs-grid-size curves for medical montages.
- Text-overlay (slice index / sequence) effect on montage reading.
- Grade-prompt sensitivity recovery with FP penalty for under-called findings.
- One-finding-per-call vs all-findings measured comparison in medical VLM labeling.
- Cross-fit labeling-function validation.
- Reported bootstrap CI widths at n≈50 in weak-supervision practice.
