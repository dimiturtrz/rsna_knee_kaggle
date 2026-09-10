# 2026-09-10 — The discovery candidates mapped onto our own measurements

Cross-reference of the discovery harvest (`2026-09-10_unseen-ceiling-breakers-discovery.md`)
against the project's measured state (`interpretations/kneemri/`, champion pin, ledger).
Purpose: prevent re-asking what we have already priced, and name the cells our context
does NOT answer. Status codes: CLOSED (measured by us) / PARTIALLY-PRICED (bound by adjacent
measurements) / OPEN. Provenance: research-lane mapping 2026-09-10; letters D303–D307.
Not committed by this lane.

## The three highest-value rows (contradictions of the harvest)

**1. Joint multi-corpus pooling — the harvest's "biggest unbuilt cell" is BUILT here and null.**
`merged3_corpus` (6,393 studies = 4,407 + Rijeka 736 + MRNet plane fills, domain-tagged,
partial label overlap) is the only measured instance ANYWHERE of domain-tagged
partial-overlap multi-corpus knee training. Result (interpretations/kneemri/
2026-09-03_the-corpus-axis-saturates-at-two-sites.md): **−0.0024**, and the direct
supervision column (ACL, 1,248 expert image reads) moved **−0.0013** — "KIND does not
compose." External-seat board dose curve bends down: 0.944/0.944/0.943/0.940. The
four-corpus variant (with fastMRI-knee) is licence-blocked for Kaggle use (NYU DUA), so the
remaining untested variant is moot for the competition. → CLOSED, by us.

**2. Automatic label cleaning — our audit IS the CleanPatrick negative, at our scale.**
2026-09-01_q14_label_cleaning_go_no_go.md: corpus-wide auto-classifier flags 876
missed-negation cells but ~1.5% of TRUE positives trip the same rules; net fixable mass
≈700 cells = **1.66% of the label matrix**, against a measured correction ceiling of
+0.0053 (the Fracture+Contusion rule). Soft negation pass read −0.0024 on gold. → CLOSED.

**3. Report-supervision family — the corpus is 60% non-English.**
2026-08-17_the-corpus-is-not-english.md: English = 40.1% of the corpus; 70.0% of classified
non-English reports grade findings in their own vocabulary vs 20.2% visible to English
patterns; Croatian converts at 0.0%. Every report-text method silently assumes English.
Yue & Johansson's predicted negative (label derivable from report → privileged-text
distillation gains nothing) is independently confirmed here three ways: our model beats its
own labels on 9/12 findings; ambiguity sits on diffuse findings while our deficit is focal;
`unread_dense` (OOF scores into 27.3% UNK cells) read +0.0020. → CLOSED.

## The rows our context SUPPORTS (open, with in-repo positive evidence)

**1. Frozen in-domain FM features (Demir)** — the only candidate with a positive controlled
in-repo measurement: ortholarge vs v3large (same arch, same read, differing only in
OrthoFoundation's knee continuation): **+0.0202/+0.0203/+0.0165/+0.0167 across four
readouts** (2026-09-03_the-frozen-probe-re-read.md). Open cell: the recipe — ortholarge
"probes best, serves worst" (frozen probes 0.762–0.767 vs fitted 0.85+; served
0.8452–0.8464 vs ~0.855 DINOv2 seats), undiagnosed; `ortho_fresh_lowlr` is the named unrun
control. OAI/KL-grading endpoint unmeasured here (our OA labels are report-derived).

**2. Curriculum as schedule** — `certainty_ramp` +0.0077 (above the 0.0062 floor — one of
the few above-floor supervision gains owned); external convergence (Zhou +0.0245, same
shape). Static reweighting is null (−0.0036); prevalence-balancing refuted twice. The
deciding empty cell is pre-registered: `disagreed_ramp` (disagreement under a ramp) —
refutation branch closes the whole difficulty axis.

**3. Prioritised relabel, ruler side (Bernhardt)** — the noisy-ruler premise is independently
confirmed: proxy vs annotation ruler differ by +0.045 on identical weights
(2026-08-25_our-model-beats-its-own-labels.md); local gains double on the board
(2026-08-13_what-the-labels-actually-contain.md). Fixable populations enumerated (502 backed
rubric cells → ~1,250 corpus-wide). The blocker is a RULER-SIZE problem, not mechanism:
15 of 876 and 5 of 502 flagged cells fall inside the 58-study gold ruler — invisible to it.
Ledger tail: "the cheapest thing that would settle it is more gold, not a cleverer statistic."
Unpriced question: whether targeted correction of the named cells moves TRAINING (not gold-58).

## Partially priced (bound, not killed)

- **Prior-initialised plane fusion (Kunitomi +0.058, baseline-inflated)**: everything around
  it is measured flat (second decode +0.0008; learned combiner +0.0015–0.0021 under the
  floor; per-label weighting oracle +0.0002 over average). The specific mechanism — per-label
  plane weights INITIALISED FROM CLINICAL PRIORS, trained end-to-end in one model — never
  isolated.
- **Xie AOLM attention localisation (+0.01–0.04 per label)**: all our localisation-shaped
  interventions changed WHAT PIXELS EXIST, not where the trunk looks within them (crop80
  gate: Medial Meniscus −0.012 wrong sign; pooling moves Lateral Meniscus but cannot create
  it). Within-slice spatial attention localisation has never been isolated — genuinely
  unrun, though three adjacent nulls bound its plausible size.
- **Coreset/pruning at ~4k**: the dose curve is measured (half_data −0.0234; concave, each
  doubling buys ~0.48 of the last; next doubling ≈ +0.009) — removing studies moves DOWN the
  curve; a LEARNED coreset vs uniform subsampling at fixed N is unmeasured but capped by
  that curve.

## OPEN (exists here, built, unrun — cheapest candidate on the list)

- **Spectral/frequency-hierarchical training signal**: `kneemri/training/spectral_mae.py` +
  pretrain.py exist (masked spectral MAE aimed at focal detail); the validation run is
  priced at 3–4 h card (2026-09-01_q7_external_data_for_ssl_pretraining.md) and per that
  note "should run before any external decode". Whether the objective changes frozen-block
  representations at all is unmeasured (`encoder_drift` gate). Nothing else on the list is
  simultaneously this untested and this cheap.
