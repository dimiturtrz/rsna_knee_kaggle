# Extractor repair route, structured/transposed label noise, selective repair — literature synthesis

Researcher lane, 2026-09-11. Sources for letter D333 (main's M338 Q1 + M340 §4 + M339's gradient
check). All citations verified at fetch time; partial verifications flagged.

## M338 Q1(a) — does extractor repair propagate to downstream AUC?

**GreenRFM: NO — explicitly.** The audit-informed prompt variants "were used only for the
residual-noise reduction analysis"; all main results use the original labels. The F1 0.943→0.985 repair
was measured on the n=565 audit sample only; no encoder retrained on repaired labels.

The published "labeler quality → downstream AUC" measurements that DO exist:

1. **Jain et al. 2021 (arXiv:2104.00793)** — the canonical experiment (and the VisualCheXbert paper we
   already cite for the modality confound). Three labelers on the same 224,316 CheXpert training
   images, DenseNet-121 each, radiologist test set: labeler weighted-F1 0.55 → 0.73 moved downstream
   AUROC 0.83 → **0.87 (+0.04)**; gains CONCENTRATED where extraction was broken (Enlarged
   Cardiomediastinum: labeler F1 0.10–0.23 → 0.73, downstream 0.69–0.73 → **0.85**, +0.12–0.16 on that
   label). Large, targeted, structured repairs propagate; diffuse ones do not.
2. **Khosravi et al. 2025 (J Digit Imaging, PMC12920854)** — 7 labeling schemas on 227,827 MIMIC-CXR
   reports; external test n=19,237. Train side: classifiers "remarkably tolerant to label noise", no
   significant differences AMONG LLM schemas despite fracture sensitivity 51%→95%. **Eval side: the
   same model loses 0.05 (PTX) / 0.11 (fracture) AUC when the test reference degrades from human to
   CheXpert labels.** The labeling schema moves measured AUC mostly through the evaluation labels.
3. **Wollek et al. 2024 (PMID 38295825)** — same 6,434 CXRs three ways: public-data labels 0.728,
   near-perfect auto-extracted report labels (sens 0.997) **0.858**, image-based labels **0.934**. A
   measured report-SOURCE vs pixel-source label cost of ~0.08 AUC at 99.7% extraction sensitivity —
   the layer our audit cannot see, priced on one label at one center.

NULL: no published audit-informed-prompt-repair → retrain → AUC delta anywhere; CheXbert/RadBERT-CL
report extractor F1 only.

## M338 Q1(b) — structured/transposed noise priced differently?

Distinguished but not priced in imaging:

- **CIFAR-N (Wei et al., NeurIPS 2021 D&B, arXiv:2110.12088):** real human noise is
  instance-dependent and concentrates between confusable classes; methods tuned on symmetric/pair-flip
  synthetic noise do NOT transfer their ranking.
- **AlleNoise (arXiv:2407.10992):** separates symmetric vs class-conditional pair-flip vs NESTED-FLIP
  (cyclic swaps within one taxonomic parent — the closest published analog to a compartment swap) in
  >500k text examples; established LNL methods inadequate on the real noise.
- **Li et al. 2023 (arXiv:2309.12706):** per-label 2×2 noise identifiable via label-correlation
  mismatches — machinery that uses the correlation structure transpositions live in.

NULL: nobody transposes a true positive between two correlated multi-label columns and prices it
against iid flips on downstream AUC. Under any per-label 2×2 model a swap counts as FN+FP (two
errors); under a study-level "any tear" it is zero — the gap is unmeasured. And main's r0130 case
(the report ITSELF contains the medial/lateral transposition) is SOURCE-side, unreachable by prompt
repair.

## M338 Q1(c) — a shared report-level silence factor?

**NULL.** No published estimator fits a single per-report latent ("this report is uninformative")
jointly suppressing all labels. Components exist separately: PU propensity scores (Bekker & Davis,
ECML 2018 / Mach Learn 2020 survey) are per-instance single-label; multi-label PU (Kanehira & Harada
CVPR 2016; Cole et al. CVPR 2021, arXiv:2106.09708) is per-label; GLAD (Whitehill NeurIPS 2009) fits
per-item difficulty + per-labeler expertise; EHR documentation-as-signal (Getzen 2023, PMC10391553)
recognises undocumented≠absent but models it at phenotype level. The composition is unpublished.
Caveat from our own audit: adjudicated negatives were mostly EXPLICIT negations, not silences — the
silence factor matters less here than the CheXpert-style blank→negative literature assumes.

## M338 Q1(d) — the honest read

"Re-extraction moves F1 but not downstream AUC" is TRUE for diffuse, low-rate, unstructured errors
(Khosravi: no significant differences among LLM schemas; GreenRFM silver ≥ curated). It is FALSE for
large STRUCTURED repairs aimed at a broken confusion mode (Jain: +0.18 F1 → +0.04 AUROC, driven by
the repaired labels). Our audit's error content — compartment swaps, entity substitutions — is the
Jain shape, not the Khosravi shape.

## M340 §4 — selective repair (flagged minority only) vs whole-corpus re-extraction

- **FCF (Jiang et al., AAAI-24)** — the paper that most directly prices it: correction beats filtering
  at equal cleaned-noise rate (Thm 5, keeps sample count); over-cleaning REGRESSES — simulated
  corrected-set noise rate rises once cleaning extends past the true noise count; clean cells "should
  be retained without any change" (Thm 6). Repair-the-flagged-minority-only is the published optimum,
  not a compromise.
- **DivideMix (ICLR 2020, arXiv:2002.07394):** the third option for discordant cells — mask-and-reuse
  as unlabeled beats discarding. Maps to multi-label as: mask the discordant CELL, not the study.
- **Confident Learning (Northcutt JAIR 2021, arXiv:1911.00068):** prune-and-retrain measured positive
  on ImageNet/CIFAR (numbers table-flagged). **ActiveLab (ICLR 2023 WS, OpenReview YfbI9RAWbj):**
  selective re-labelling beats label-new-only at every noise level, vital above ~20%. **Bernhardt et
  al. (Nat Commun 2022, arXiv:2109.00574, healthcare-motivated):** ranked corrections up to 4× more
  effective than random under a fixed budget — selective-vs-full not head-to-head.
- **CleanML (ICDE 2019, arXiv:1904.09483):** label cleaning often moves downstream accuracy LITTLE;
  matters more for evaluation/selection than training.

Selector-precision precedents for disagreement-as-error-detector: Northcutt 2021 — 51% of
algorithm-flagged candidates truly mislabeled vs 3.3% base (~15× lift, model-vs-label selector);
CROWDLAB (arXiv:2210.06812) — annotator-disagreement confidence beats Dawid-Skene at finding true
errors. Ours (15.6% vs 1.48%, 10.5× lift, extractor-vs-extractor on report labels, adjudicated) is the
first measurement in its selector class; the published analogs support the direction at 4–15×.

NULL: no head-to-head of "re-extract flagged minority" vs "re-extract everything" at fixed cost with
downstream AUC, anywhere.

## M339 (ii) — the gradient check, with the rank correlation

Spearman between gold-58 per-column AUC and published report/reading-level miss rate, over the SEVEN
columns with a defensible literature number (denominator disclosed, nothing imputed):

| column | gold@58 AUC | published miss | source |
|---|---|---|---|
| ACL | 0.9755 | 0.13 | Phelan 2016 pooled sens 0.87 |
| Medial Meniscus | 0.9736 | 0.11 | Phelan sens 0.89 |
| MCL | 0.9229 | 0.065 | Boutany 2025 report sens 0.935 |
| Lateral Meniscus | 0.8832 | 0.25 | Fritz ~0.70 / Phelan 0.78 sens |
| Medial OA | 0.9535 | 0.65 | Krakowski cartilage report sens 24–42% |
| Lateral OA | 0.7872 | 0.70 | Krakowski lateral tibial AUC 0.578, worst subregion |
| Synovitis | 0.7515 | 0.58 | MOAKS Hoffa-synovitis κ=0.42 (inter-rater proxy, flagged) |

**Spearman ρ = −0.50 (n=7, p=0.25)** — the expected direction (noisier channel → weaker column),
moderate magnitude, NOT resolvable at this n. Adding Effusion (κ≈0.90 → miss≈0.10) drops it to
**−0.33 (n=8)**. The two counter-examples: **Medial OA** (noisy channel, strong gold column 0.9535)
and **Effusion** (cleanest published channel, third-weakest gold column 0.8522). Fracture, Contusion,
PF OA, Baker's excluded — no report-level miss rate exists for them (the load-bearing null main
flagged in M339 §1).

Verdict on main's fork: the ordering PARTIALLY agrees — the extremes align (Synovitis/LatOA
noisy-and-weak; ACL/MedMen clean-and-strong) but the middle does not, and at n=7 the correlation cannot
certify either story. The honest reading: report weakness contributes at the extremes; the column
ordering in the middle is about something else (prevalence, pixels, or our head).
