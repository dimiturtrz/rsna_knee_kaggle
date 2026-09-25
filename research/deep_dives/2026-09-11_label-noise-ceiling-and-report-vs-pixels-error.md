# Annotator disagreement vs attainable AUC; report-vs-pixels error for knee MRI — literature synthesis

Researcher lane, 2026-09-11. Sources for letter D330 (main's M336). All citations verified at fetch
time.

## (a) Studies with BOTH an agreement statistic AND a model AUC on the same labels

1. **MRNet (Bien et al., PLoS Med 2018, PMC6258509)** — the cleanest knee read-off: reference =
   majority of 3 MSK radiologists; Fleiss κ = 0.508 (abnormality) / 0.800 (ACL) / 0.745 (meniscus);
   model AUCs on the same labels 0.937 / 0.965 / 0.847. The LOWEST-agreement task still yields 0.937 —
   κ and attainable AUC are not tightly coupled.
2. **Fritz et al., Skeletal Radiol 2020, PMID 32170334** — knee meniscus vs surgery: inter-reader κ
   0.876 (medial) / 0.741 (lateral); radiologist sensitivity MM 93–96%, LM 67–71%; DCNN AUC 0.882 (MM)
   / 0.781 (LM). Agreement, model AUC, and reader-vs-truth error in one study.
3. **CheXpert (Irvin et al., arXiv:1901.07031)** — labeler vs 2-radiologist consensus: macro-F1
   mention 0.948 / negation 0.899 / uncertainty 0.770; models on 5-radiologist majority-vote test:
   AUC 0.85–0.97; the report-label curves (LabelL/LabelU) sit below the model curve. Pham et al.
   (arXiv:1911.06475): 0.940 mean on the hidden test.
4. **Majkowska et al., Radiology 2020, PMID 31793848** — 4 CXR findings, radiologist-adjudicated
   image-review reference, population-adjusted AUCs 0.72–0.95.
5. **McKinney et al., Nature 2020, PMID 31894144** — mammography double-reading + arbitration; AI AUC
   exceeded average radiologist by +0.115 absolute.
6. **GreenRFM (arXiv:2603.06467, 2026 preprint)** — the closest pipeline analog: LLM-extracted report
   labels (Merlin CT). Expert re-read audit (n=565): κ=0.848, sens 0.903, spec 0.978, hard FP 0.53%;
   silver-trained encoder AUC 87.41% vs curated-label 85.78%. Audit-informed prompt fixes raised
   binary F1 0.943 → 0.985 — residual extractor error is REDUCIBLE.

Null: nobody regresses model AUC against measured reader disagreement as an explicit axis; the above
are co-located numbers, not fitted ceilings.

## (b) Functional forms f(e) — the eval-noise forms exist and are exact; the train-noise ceiling does not

- **Menon, van Rooyen, Ong, Williamson (ICML 2015, PMLR v37)** + restatement in Fogliato et al.
  (arXiv:2003.13808, Prop B.1): for a FIXED scorer under EVALUATION-label noise:
  - one-sided positive-label noise rate ρ: **AUC_obs = (1−ρ)·AUC* + ρ/2**
  - symmetric flip rate ρ: **AUC_obs = (1−2ρ)·AUC* + ρ**
  Read-off at AUC*=0.95: symmetric ρ=3% → 0.923 (−0.027, ~3× our whole 0.010 gap); one-sided FN at
  ρ=10% → 0.905. A few % of eval-label noise alone could mask the 0.944→0.954 gap — but ONLY noise in
  the evaluation labels does this.
- **Train-label noise is theoretically different:** under class-conditional noise with ρ₊+ρ₋<1 the
  Bayes-optimal RANKING is unchanged (Menon 2015) — training noise need not cap achievable ranking
  given enough data; practical degradation is memorization, not an information-theoretic ceiling.
- **Northcutt et al. (arXiv:2103.14749):** ranking flips at ~5–6% test-label error — benchmark
  validity, not a ceiling (already the D327 decision-band anchor).
- **GreenRFM Table S9/Fig 3e–h — the only measured noise-injection curve on report-derived labels:**
  symmetric flips are catastrophic (40% noise → macro-AUC 0.529 from 0.856 baseline); but
  UNCERTAINTY-TYPE corruption (uncertain→negative/positive/random — the realistic extractor error
  shape) costs only 0.03–0.09 AUC. Structured report noise ≪ arbitrary flips. (2026 preprint, not
  peer-reviewed — flagged.)
- Surveys: Frénay & Verleysen (IEEE TNNLS 2014, PMID 24808033) — separates train vs eval noise;
  Karimi et al. (Med Image Anal 2020, PMID 32623277) — medical label-noise survey, observer
  variability the dominant source.

Null: no validated universal f(e) fitted on real radiology label noise; no second measured
AUC-vs-injected-noise curve beyond GreenRFM.

## (c) Report-vs-pixels error, knee MRI (the term our audit cannot see)

Report-level measurements (routine clinical reports vs truth — the right population):

- **Krakowski et al., Appl Sci 2021 (doi:10.3390/app11041552)** — 190 patients, reports by 18
  radiologists/10 centers vs arthroscopy, 453 chondral lesions: only ~30% of reports adequate on
  cartilage in all locations; report sensitivity 24–42% by location; grade I 4.8%, grade IV 39%;
  per-location AUC 0.578–0.737. Cartilage/OA report-vs-truth error is LARGE and one-sided (false
  negatives).
- **Boutany et al., SICOT-J 2025 (doi:10.1051/sicotj/2025046)** — 31 knee dislocations, reports vs
  operative findings: only 35.5% of reports fully matched surgery; report sensitivity ACL 71%, PCL
  61%, MCL 93.5%, LCL 64.5%, PLC 51.6%. Cites image re-review studies at much higher accuracy — the
  reading-vs-reporting gap, measured.
- **Fritz 2020** — lateral meniscus under-read ~30% by MSK radiologists vs surgery.

Meta-analyses (measure READING under research conditions — optimistic bounds on report error):
Phelan 2016 (21 studies, n=1,339): ACL sens 87/spec 93; MM 89/88; LM 78/95. Li 2017 (PMC5548790):
ACL 87/93. Crawford 2007 (PMID 17785279): MM 91.4/81.1, LM 76/93.3. Zhang 2013 (PMID 22906758,
40-study cartilage): sens 75/spec 94 for lesion presence — grading far worse.

Scoring reliability (reader-vs-reader) for the soft channels: MOAKS (PMC4058435) — tibial cartilage
κ=0.36, tibial osteophytes κ=0.49, Hoffa-synovitis intra-rater κ=0.42; effusion-synovitis inter-rater
≈0.90 (Gilles 2023). Bone bruise MRI detection sens 83–96% (Uçar 2012, PMC3497884).

**Nulls:** no report-text-vs-image-re-review disagreement study for routine knee MRI; NO published
report-vs-truth miss rates for effusion, synovitis, Baker's, contusion, or fracture on knee MRI —
five of our twelve columns' report-level error is unmeasured anywhere.

## Bottom line

- Eval-label noise has an exact linear price: ~2–3% residual test-set noise ≈ our entire 0.010 gap.
  Our audit bounds extractor error only.
- Train-label noise has NO theoretical ceiling on ranking (Menon) and structured report-type noise is
  measured cheap (0.03–0.09 AUC worst case, GreenRFM) where random flips are catastrophic.
- Report-vs-truth error is real, asymmetric (FN-dominant), and concentrated exactly in our taxonomy's
  soft channels: low-grade cartilage (sens 5–39% by grade, report level), lateral meniscus (~30% FN),
  complex-trauma ligaments (29–48% FN). ACL/effusion are clean (sens 87–94%, κ~0.9).
- Expected profile: a PER-LABEL ceiling gradient, not a uniform ceiling — and no published f(e)
  exists to price it without measuring report-vs-truth error on our own corpus (which the 58 can do
  only coarsely).
