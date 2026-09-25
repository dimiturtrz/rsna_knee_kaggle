# Multi-source label merging and agreement-only source ranking (researcher, for M323-Q2 / M326)

Durable version of letter D320. Regime: merging report-text labels with bounding-box annotator
labels (threshold mismatch); ranking five label tables (93.9% unanimous, 4 of 5 LLM extractions of
the same reports) with no clean ruler beyond gold-58.

All IDs verified (one bait ID discarded by the sweep; FlyingSquid is 2002.11955).

## M323-Q2 — sources differing in THRESHOLD: the standard estimator is named and published

- **Raykar et al., "Learning From Crowds", JMLR 2010** (jmlr.org/papers/v11/raykar10a) — per-annotator
  sensitivity α^j and specificity β^j fitted jointly with the classifier by EM. This is your regime,
  and the asymmetry you want falls out of the E-step for free: P(y=1 | y^j=0) ∝ (1−α^j)·p /
  [(1−α^j)·p + β^j·(1−p)] — a NEGATIVE from a strict (low-sensitivity) source is automatically
  down-weighted against a negative from a lenient source, with the class prior p playing exactly the
  role of your hand-built prevalence ratio. **Your asymmetric-confidence encoding is a one-parameter
  hand approximation to this Bayes posterior** — the estimator gives you the fitted version instead of
  a ratio picked by hand.
- **The explicit per-source cut point on a shared latent severity** — the exact model you described —
  is Raykar, "Annotation models for crowdsourced ordinal data" (NeurIPS 2011 CSS workshop): an
  ordered-logit model where each annotator carries his own ordered thresholds over a latent severity.
  Same machinery as the IRT graded-response model (Samejima). So "fit the shift, don't pick the ratio"
  has a published home.
- GLAD (Whitehill 2009) is the WRONG shape — symmetric ability cannot represent "strict on positives,
  accurate on negatives"; Paun et al. (TACL 2018, DOI 10.1162/tacl_a_00040) measured Dawid-Skene-style
  bias variants beating GLAD-type models on annotator characterisation.
- On the secondary question ("is no-box safe to read as negative at all"): the detection literature
  treats non-exhaustive annotation as MISSING, not negative — Soft Sampling (arXiv:1806.06986)
  measured silently dropping 30% of annotations costing ~5% mAP with re-weighting recovering most of
  it; Background Recalibration Loss (arXiv:2002.05274) the same for one-stage detectors; and Castro et
  al. (Nat Commun 2020) gives the missingness frame: the box annotator's "not boxed" is close to MNAR
  (non-boxing depends on the finding's salience), which is exactly the case where absence-as-negative
  is biased. NULL on a classification-specific treatment — it routes through PU learning, which D318
  already covered.

## M326-Q1 — agreement-only ranking at K=5, 93.9% unanimous, four sources sharing a generator:
below the floor, and the width of the bound is the answer

Two literatures converge on the same bad news, with a published diagnostic:

- **The latent-class diagnostic-test literature** (the original home of "no gold, correlated raters"):
  Hui & Walter (Biometrics 1980) set the floor — three+ conditionally INDEPENDENT tests are needed to
  identify sens/spec/prevalence at all. Vacek (Biometrics 1985) and Torrance-Rynard & Walter (Stat Med
  1997) measured the direction and size of the bias under conditional dependence. Albert & Dodd
  (Biometrics 2004, "A Cautionary Note on Robustness…") is your regime named: estimates are NOT robust
  to violated independence, and high-concordance patterns are precisely where the prior dominates the
  data. Dendukuri & Joseph (Biometrics 2001) is the correlated-variant fix — pairwise dependence
  parameters, identifiable only with informative priors or more tests.
- **The weak-supervision line**: Snorkel structure learning (Bach et al., arXiv:1703.00854) can DETECT
  the shared-generator edges among your four LLM tables — but its own identifiability condition needs
  a triple of not-all-pairwise-dependent sources, and four correlated extractions collapse your
  effective independent views toward 2 (one non-LLM table + one LLM cluster), below the K≥3 floor.
  FlyingSquid (arXiv:2002.11955) gives closed-form triplet estimation with bounds that survive model
  misspecification; Mazzetto et al. (AISTATS 2021, PMLR v130) aggregates dependent sources with
  guarantees.
- **At 93.9% unanimity specifically**: only the 6.1% discordant cells carry information — ~3,225 cells
  over 12 findings × 10 confusion entries, tens of observations per parameter on the rare findings.
  The honest published tool is PARTIAL IDENTIFICATION: Maia Polo et al. (NeurIPS 2024,
  arXiv:2312.04601) derive Fréchet bounds on accuracy/precision/recall from weak labels alone with no
  ground truth. In your regime the bounds will be wide, and **that width is the diagnostic**: no
  agreement-only estimator resolves the ±0.012 you need, regardless of sophistication. CROWDLAB
  (arXiv:2210.06812) sidesteps pure agreement by adding a trained classifier's features as tie-breaker
  — it beats Dawid-Skene on consensus estimation but inherits whatever the classifier knows.

So the clean form of the null you offered: **at 93.9% unanimity with a shared generator across four of
five sources, agreement-only ranking is not identifiable at the resolution you need — gold or the
board.** That licenses the slot.

## M326-Q2 — evaluation on a ruler built from the old supervision: partially named, no canonical term,
and the field's remedy is a modality swap

- Lampert, Stumpf & Gançarski (IEEE TIP 2016, DOI 10.1109/TIP.2016.2544703) measured exactly this
  structure: when ground truth is ESTIMATED rather than independent, evaluating against the estimate
  inflates apparent performance — empirically, on segmentation benchmarks. That is your confound one
  level up.
- Northcutt et al. (arXiv:2103.14749) quantify how little test-label corruption reverses rankings:
  ≥3.3% average test-set errors across 10 benchmarks, and corrected labels flip ResNet-18/50 and
  VGG-11/19 orderings at your margins. (Independent noise, not shared-generator — your case is worse.)
- The medical-imaging instance of your exact confound is the field's own practice: VisualCheXbert
  (arXiv:2104.00793) trains on report-labeler outputs but EVALUATES on radiologist image annotations,
  and CheXpert reserves a radiologist-read test set — the de facto acknowledgment that a
  report-derived ruler measures agreement with the labeler. Rajpurkar's thesis documents radiologists
  labeling reports disagreeing with the same radiologists labeling the images.
- The mechanism's closest named cousin: "LLM Evaluators Recognize and Favor Their Own Generations"
  (arXiv:2404.13076) — a shared generator between evaluatee and evaluator produces measurable upward
  bias, causally implicated. Yours is the label-table version.
- Remedies in print beyond "get clean labels": (i) partial-identification bounds instead of point
  metrics (Maia Polo); (ii) **swap the ruler's MODALITY, not its instance** — evaluate against
  image-annotated labels to break the shared-generator link (the CheXpert pattern; for us that is what
  gold-58 already is, which re-prices its scarcity); (iii) latent-class-correct the eval table against
  the other sources first — attenuates, does not remove. NULL on a closed-form bias magnitude for your
  exact case.

M324/M325 (multi-view as samples, cross-view consistency, test-time adaptation, MIL bag-size
dilution) are in the second sweep; letter follows.

— researcher
