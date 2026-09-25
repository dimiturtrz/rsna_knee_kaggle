# Training methods that consume an externally-measured per-label asymmetric noise rate — literature synthesis

Researcher lane, 2026-09-11. Sources for letter D331 (main's M337). All citations verified at fetch
time.

## (a) Loss correction with a known transition matrix

**The named method exists and has a guarantee: Natarajan et al., NeurIPS 2013** ("Learning with Noisy
Labels") — the binary origin: an unbiased loss estimator that CONSUMES known asymmetric flip rates
ρ₊₁, ρ₋₁ — exactly a per-label 2×2 T as input, no estimation machinery. **Patrini et al., CVPR 2017
(arXiv:1609.03683)** generalises to multi-class forward/backward correction ("at most a matrix
inversion and multiplication, provided that we know the probability of each class being corrupted").

(i) **Multi-label survival:** Patrini's Theorem 1 is single-label multi-class, explicitly a
generalisation of Natarajan's binary theorem. Patrini never states a multi-label result — but the
binary root is the relevant one: twelve columns with per-column 2×2 T's ARE twelve binary problems.
Unbiasedness is linear in the loss vector, multi-label BCE is a sum of binary risks, and
class-conditional noise factorises per label — so per-column forward correction is unbiased
per-column and label correlation does NOT break the expectation argument. **This composition is
unpublished — "follows from linearity", stated nowhere and tested nowhere.**

(ii) **Multi-label/medical extensions:** Li et al. 2023 (arXiv:2309.12706) proves identifiability of
class-dependent per-label transition matrices in multi-label learning without anchor points — the
strongest formal statement that per-label 2×2 matrices are the right object (it estimates T
internally, but validates the framework an external T plugs into). Ghiassi et al. ACML 2023 (TLCM,
PMLR v189) — corruption matrix estimated on TRUSTED data, the closest published "externally measured
T as input" in multi-label; denominators MS-COCO/NUS-WIDE at 30–60% injected corruption, mAP not AUC.
Hendrycks et al. NeurIPS 2018 GLC (arXiv:1802.05300) — trusted subset → T, single-label. Medical
multi-label with report-extracted labels: BoMD (ICCV 2023, arXiv:2203.01937); Yang et al. 2023 (PMID
35616775, CXR14 n=112,120, final AUC 0.827).

(iii) **Soft targets:** Patrini's theorem is stated for one-hot targets only. Mechanically the
correction is defined for soft targets (the linear map on the loss vector), and unbiasedness for a
soft target equal to the expected noisy label follows by linearity of expectation — **but no paper
publishes or tests that statement.** Closest: Lukasik et al. ICML 2020 (PMLR v119) — label smoothing
≡ injecting symmetric noise, competitive with loss correction under noise; PENCIL (Yi & Wu, CVPR
2019) — soft label updates with a transition model, no external T.

**Sensitivity to a mis-specified T — measured:** Patrini's own known-vs-estimated-T comparison:
estimated T̂ loses a MEDIAN ~10 accuracy points (range 0 on binary IMDB to 27 on MNIST at high
noise); "small estimation errors have a mild, not catastrophic effect", but at CIFAR-100 + high noise
bad estimation "destroys most of the gain". Xia et al. 2019 T-Revision (arXiv:1906.00189): without
anchor points T is poorly learned and consistent classifiers "significantly degenerate". No
controlled "T wrong by ±2pp" study exists.

## (b) The asymmetric case

- Natarajan 2013 IS the asymmetric method (known ρ₊₁ ≠ ρ₋₁ as input).
- **Menon et al., ICML 2015 (PMLR v37)** — corrupted binary labels and PU learning in one framework
  ("mutually contaminated distributions"): (1) **AUC can be optimised WITHOUT knowing the corruption
  parameters at all** — FN-dominant noise is a monotone distortion of class probability, so ranking is
  preserved; (2) given the parameters, general risks are minimisable; (3) parameters are estimable
  from corrupted data. **The PU mapping is published here: pure-FN noise with FP=0 IS PU learning.**
  Practical upshot: since our metric is per-label AUC and our noise is FN-dominant, part of what
  T-correction buys is recoverable parameter-free; T's residual value is calibration and per-label
  risk-exactness, not ranking.
- Instance/part-dependent T estimation (Xia et al. 2020, arXiv:2006.07836) — unnecessary with an
  external T.
- **NULL: no controlled "correct asymmetric T vs symmetric assumption" ablation exists.** Patrini
  Table 2 is closest: under asymmetric injected noise, corrections with known T are "overall the best
  performing... forward usually the best".

## (c) No-T alternatives, priced on the one axis that matters

The single published denominator in our regime: **BoMD's NIHxPDC benchmark (arXiv:2203.01937, Table
4)** — multi-label CXR, 13 labels, noise-injected report-derived training labels, clean OpenI test,
mean AUC:

| injected ps/pl | BCE | NVUM (ELR-family) | BoMD (re-label/graph) |
|---|---|---|---|
| 0/0 (inherent NLP noise) | 85.65 | 87.89 | 89.76 |
| 20/20 | 83.99 | 85.34 | 88.00 |
| 20/40 | 81.42 | 82.83 | 86.26 |
| 60/60 | 72.39 | 76.91 | 74.11 |

So gains ARE measured above 0.85 baseline on multi-label medical: +1.4 (ELR-family) to +4.0
(re-labelling/graph) from 0.84–0.86 baselines. Nothing above 0.90 baseline anywhere.

**The warning that matters for us: on BoMD's NIH-Google pneumothorax benchmark (Table 3), naive
per-label forward correction scored 80.8 vs plain BCE 87.0 — F-correction UNDERPERFORMED BCE in the
only multi-label medical head-to-head that exists.** And on NIH→OpenI/PadChest, noise-cleaning
families HURT (DivideMix 72.8/75.5, FINE 63.7/70.9 vs BCE ~83).

Canonical no-T papers' denominators do not transfer (all accuracy, single-label, CIFAR-class):
Co-teaching arXiv:1804.06872; ELR arXiv:2007.00151; symmetric losses arXiv:1712.09482; GCE
arXiv:1805.07836.

## (d) The null, priced

- Patrini's known-vs-estimated comparison: external T buys a median ~10 points on hard multi-class —
  but on the BINARY asymmetric task (IMDB, 5% one-directional), estimated T matched known T. Twelve
  binary columns sit on the EASY side of that evidence.
- Menon 2015: under FN-dominant noise, AUC-optimality needs no noise parameters — free-lunch ranking.
- **The exact null main asked for — "given-T vs internal-estimate at 5–15% asymmetric noise on
  multi-label medical at ≥0.85 baseline" — is unpublished.** That absence is the answer; and the one
  adjacent measurement (BoMD pneumothorax) points against naive per-label F-correction.
