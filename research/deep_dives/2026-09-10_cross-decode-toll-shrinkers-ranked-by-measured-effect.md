# 2026-09-10 — What shrinks a cross-acquisition decode toll on a classifier: ranked by measured effect

Field synthesis for the cross-decode toll question (our measured toll: −0.0336 mean AUC,
sd 0.0038, n=11 members, 909-study holdout, midpoint cut — the residual AFTER physical-mm
normalisation). Ranking is by measured classifier effect, proximity of the cell to ours.
[CLF] = classifier endpoint, [SEG] = segmentation, [ASSERTED] = claimed, no ablation.

## The ranked tier list

### Tier (a) — inference-time, model fixed, ranked

1. **CARVE +0.036 internal / +0.034 external** [CLF, multi-label, CT] — arXiv:2607.15556.
   The only published result in our exact cell (model fixed, multi-label, physical
   normalisation already paid — both cohorts HU-clipped and mm-resampled). Denominators:
   internal CT-RATE CT-CLIP zero-shot 0.713→0.749 = +0.036; external RAD-ChestCT
   0.499→0.533 = +0.034 **from a near-chance base** (an operating-point repair, not a
   toll recovery on a healthy base); fVLM external 0.612→0.642 = +0.030. Our setting
   (ρ=0.8818 ruler transfer) is the paper's *internal* regime, so +0.036 is the honest
   comparator — and the paper warns the base-model gap (+0.11 between backbones) exceeds
   any adapter gain. Precondition per §3: the base must already separate present/absent.
   **CARVE recipe, verified from full text:** k̂ = clip(round(Σⱼ p̄ⱼ), 1, 8), estimated
   label-free from Bernoulli prompt-pair probabilities (no caption retrieval); top-k̂
   objective L = −(1/|𝒫|)Σ log p_j − λ_neg (1/|𝒩|)Σ log(1−p_j), λ_neg=0.8. Weak views:
   additive Gaussian intensity jitter (σ=0.02, ≈20 HU) + integer translation up to ±3%
   per axis, circular shift — **no crops, flips, rotations, or slice subsampling**; the
   multi-view step "serves reliability filtering rather than data augmentation"
   (stronger views raise cross-view disagreement without improving the update
   direction). Retention: per-scan rank — the K = round(8 × 0.25) = 2 lowest-entropy
   views of that scan's own 8; p̄ (retained-view average) feeds ONLY the cardinality
   (Eq. 8); the top-k̂ loss (Eq. 9) runs per retained view, averaged (Eq. 10).
   Constants fixed once on internal validation, reused everywhere: V=8, ρ=0.25, Adam
   lr 1e-5, 2 steps, norm-affine only, episodic reset per scan; optional dispersion
   gate τ=0.091 is abstention-only and near-random as a viability detector (their
   Fig. 6) — do not use it as one. No lr/step retune at V=8 exists. **No V=1 ablation
   is quotable**: Fig. 2 sweeps V but every per-value number is figure-locked; the text
   says only "more views help until saturation". TENT on their internal leg for
   contrast: accuracy 0.633→0.605 (WORSE), recall 0.698→0.712 — the operating-point
   shift CARVE's cardinality term is designed to avoid.
2. **CoWA** (arXiv:2607.03715) [CLF, multi-label CXR, 6 cross-institution shifts]:
   co-occurrence-weighted entropy, BN-affine on DenseNet121. Verifiable: norm-ablation
   AUROC 74.7±8.0 (ℓ_F) vs 71.1/69.8/70.6 (ℓ1/ℓ2/ℓ∞). **Table 1 absolutes are
   image-locked in HTML and PDF, and the GitHub repo (github.com/woojin716/CoWA) is a
   verified ABSENCE: no results/, no committed logs, no eval output, no releases/tags,
   no wandb links** — the sweep scripts write tta_summary.csv locally and nothing was
   committed. CoWA stays UNPRICED, a stated absence to plan around. Claimed: only
   method never below unadapted baseline across shifts [ASSERTED, figure-only].
3. **DISARM++ +0.32 AUC** (arXiv:2505.03715) [CLF, MRI, MCI-vs-AD] — but n=41,
   logistic regression on PCA'd features, baseline was **raw/unprocessed**: the delta
   buys back normalisation we already do. Not comparable to our cell. **CORRECTION to
   earlier notes:** the "Caldera/DISARM++ 0.857→0.899" figure does not appear in the
   paper; the verifiable number is 0.6284→0.9459. Withdraw the misquote.
4. **Saponaro +0.09 AUC** (ABIDE ASD, ~1.1k) — **CORRECTION: this is ComBat on
   FreeSurfer morphometric FEATURES with a Random Forest, not histogram matching and
   not pixels.** Mechanistically distant; the number stands in its own tabular cell.
5. **LUMINA** (CVPR 2026, arXiv:2603.14644) [CLF, multi-vendor mammo, 1,824 imgs]:
   foreground-only histogram matching, text claims consistent improvement, **all deltas
   figure-only**; protocol caveat: both sides harmonized, not a held-out toll-recovery
   design.
6. **Albert 2023** (rectal MRI, 201 pts, 6 centers) [CLF+SEG]: the most honest
   normalisation comparison found. All arms N4-corrected; z-score significantly worse
   than percentile+HM (matches our Lu-2024 null); normaliser choice buys ~0.01–0.03
   AUC; **no normaliser repairs the cross-center classifier toll** (Dworak collapses
   to ~0.52–0.58 AUC regardless). The literature's own controlled study independently
   confirms the toll is normalisation-invariant — our flat −0.0336 seen from the
   literature's side.
7. **Kim 2026 TB-SFDA** [CLF, CXR, 6 sites]: SHOT-FT +0.011 AUROC average — small, and
   its own text says AUROC is "harder to increase" than the F1 operating-point repair.
8. **Warning — "I Can't Believe TTA Is Not Better"** (arXiv:2604.09697) [CLF, MedMNIST,
   12 combos]: standard augmentation-averaging TTA degrades accuracy in 11/12, up to
   −31.6 pts, mechanism = BN-stat mismatch. Our backbones are ViT/LN so the named
   mechanism may not transfer, but the direction is the evidence: shift-axis TTA is a
   hypothesis, not a free lunch. Gate per-member, compare against single-pass.

### Tier (b) — train-time, no new data: thin, partly refuted by our own pool

- The medical train-time shift-robustness literature is overwhelmingly [SEG] (BigAug,
  LDDG, RAS4DG, Karani, SITTA, InTEnt, On-the-Fly TTA, DG-TTA). **No paper prices a
  classifier toll against consistency-across-decodes or shift-invariance losses.**
- LDDG (NeurIPS 2020): BigAug **failed** to generalise for skin-lesion [CLF] —
  consistent with our n=11 flatness on the decode axis.
- VORTEX (k-space physics augmentation): robustness endpoint is MRI **reconstruction**,
  zero classifier numbers.

### Tier (c) — multi-label TTA / SFDA 2024–2026 with numbers

CARVE and CoWA are the entire medical multi-label TTA literature. ML-TTA/Bound Entropy
(ICLR 2025) is natural-images only. HEAL, Asymmetric SFDA: no verified numbers.

## The BN-only finding (pre-register both suspects)

Backbone audit of every medical-classifier TTA experiment found: **the tier is 100%
BatchNorm** — CoWA DenseNet121, HyDA (MICCAI 2025) DenseNet121, DiffCTA (MICCAI 2025)
ResNet50, Valanarasu MIDL 2024 U-Net BNs. The method papers (TENT, EATA, SAR, AdaBN)
have no medical classification experiments at all; CoTTA's LayerNorm backbone appears
only in driving segmentation. Closest LN/ViT medical datum: MedUnA (test-time PROMPT
tuning on CLIP, norm statistics untouched). Consequence: CARVE's +0.036 is a
BN-literature prior; a NULL on our LN/ViT members has TWO live suspects — the
k̂-counting line and the LN-vs-BN line — and any ledger line should name both. Our
single-view CARVE pass (−0.0307) sits in an unpublished cell on BOTH the view-count
axis and the backbone axis; it cannot be attributed to the counting rule alone, and
no published V=1 row exists to license or convict it. An 8-view pass would be, as far
as can be established, the first numeric V-sweep on an LN medical multi-label
classifier — pre-register both suspects so the reading cannot be claimed as a CARVE
replication either way.

## Dutt et al. (arXiv:2305.08252, full v4 PDF + appendices) — what is NOT in the paper

1. **LoRA configuration NOT STATED** — no rank, alpha, dropout, or target-module list
   in 20 pages; Appendix E's code listings cover BN/bias/attention tuning, TSA and
   diffusion methods, no LoRA. Only fingerprint: App-D Table 6 trainable-param counts
   (0.6M ViT-B / 1.5M ViT-L / 2.6M ViT-H). **No rank sweep exists; there is no
   published rank to get wrong.** Stated: batch 512, AdamW, per-strategy Optuna HPO
   (PEFT LRs ~10× full-FT), epochs never stated.
2. **The crossover is F1-only** — grep of the entire PDF for AUC/ROC returns ZERO
   hits; HAM10000 treated as single-label 7-class. "The crossover holds on AUC-style
   endpoints" is unproven by the source; our ortho_lora matched cell (−0.0103) is the
   only AUC multi-label datapoint at this capacity and contradicts the transfer.
3. Crossover location: ViT-B full FT optimal at 100% data, crossover at **~50% data
   ≈ 3.8k images — exactly our scale**; below it LoRA beats full FT by up to ~6% F1,
   gain growing with trunk size. ViT-L: LoRA overtakes at 80%, BitFit below 20%.
4. **No rank-vs-corpus rule exists anywhere** — Dutt varies METHOD choice with corpus
   size, never rank. If this axis resumes, rank is unlicensed by the literature and
   must be pre-registered on our own arithmetic (e.g. from the App-D param counts).

## Silent cells (nothing exists — do not spend a slot)

BN/LN statistic re-estimation on an MRI acquisition-shift classifier; test-time-only
histogram matching (train native, match test to reference); re-sampling TTA over the
decode axis; physics-parameter randomisation on a classifier; N4 as a classifier
intervention (zero classifier evidence — Albert applies it uniformly to all arms).

## One-sentence ranking

Of everything published, only CARVE (+0.036, post-normalisation, model-fixed,
multi-label, base-must-already-discriminate) sits in our cell; CoWA is the one
unpriced neighbour (a verified absence, plan around it); every larger number
(DISARM++ +0.32, Saponaro +0.09) is measured against a raw baseline or a feature
endpoint and buys back normalisation already paid; and Albert 2023 independently
confirms the toll is normalisation-invariant.

## Flagged unverifiable

CoWA Table 1 absolutes (image-locked, absent from repo); CoWA "never below baseline"
claim (figure-only); LUMINA deltas (figure-only); CARVE Fig. 2 V-sweep values
(figure-locked; manual PDF read would take one page); HEAL and Asymmetric SFDA (no
verified numbers at all).

Source letters: D270 (ranking + CARVE Eq. 8–9 + denominators), D274 (CARVE weak views,
retention, budget), D276 (CoWA absence, BN-only tier, Dutt config), D278 (no V=1 row,
No-CARVE denominator table). Corrections to D266 (DISARM++ misquote, Saponaro
mechanism) are folded in above.
