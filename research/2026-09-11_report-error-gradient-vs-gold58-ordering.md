# Published report-level error rates vs our gold-58 column ordering — a consistency check

Researcher lane, 2026-09-11. Background item offered in D332 (ii). Places the per-finding
report-vs-truth error rates from `research/2026-09-11_label-noise-ceiling-and-report-vs-pixels-error.md`
next to our two ruler orderings (ledger block README:14541: our per-column AUC on ledger896 vs gold58).

## Our two rulers (statistic: per-column ROC AUC; populations ledger896 / gold58; cut midpoint)

| column | ours@ledger896 | gold@58 | n+ (gold) | published report/reading error |
|---|---|---|---|---|
| Synovitis | 0.8546 | **0.7515 (rank 1)** | 27 | MOAKS Hoffa-synovitis κ=0.42; reports 84% UNK (D20) |
| Lateral OA | 0.8178 | **0.7872 (rank 2)** | 11 | cartilage report sens 24–42%, lateral tibial AUC 0.578 (Krakowski); MOAKS tibial cartilage κ=0.36 |
| Effusion | 0.8542 | **0.8522 (rank 3)** | 35 | MOAKS effusion-synovitis κ≈0.90 — expected CLEAN |
| Lateral Meniscus | 0.8460 | 0.8832 (rank 4) | 23 | ~30% FN vs surgery (Fritz); pooled sens 78% (Phelan) |
| Fracture | 0.8066 | 0.8931 (rank 5) | 18 | unmeasured at report level |
| Contusion | 0.8685 | 0.8961 (rank 6) | 19 | MRI-vs-truth sens 83–96% (Uçar), report level unmeasured |
| PF OA | 0.8392 | 0.9035 (rank 7) | 21 | cartilage channel, compartment unmeasured separately |
| MCL | 0.7908 | 0.9229 (rank 8) | 9 | report sens 93.5% (Boutany) — expected clean |
| Medial OA | 0.8763 | 0.9535 (rank 9) | 15 | cartilage channel but medial read better than lateral (Krakowski) |
| Baker's | 0.9076 | 0.9547 (rank 10) | 12 | unmeasured |
| Medial Meniscus | 0.8983 | 0.9736 (rank 11) | 26 | sens 89–91% |
| ACL | 0.8706 | 0.9755 (rank 12) | 24 | sens 87–94% |

## What is consistent

- **Synovitis weakest on gold** ↔ the noisiest channel in the reference system (κ=0.42) AND 84% report
  silence. Over-determined.
- **Lateral OA second-weakest** ↔ cartilage is the worst report channel and lateral is its worst
  subregion. Over-determined.
- **ACL and Medial Meniscus strongest on gold** ↔ cleanest published channels (87–94%, 89–91%).
- **MCL is mid on gold (0.9229) while worst on our ruler (0.7908)** ↔ reports are clean (93.5% sens) —
  consistent with M328's reading that MCL's offline weakness is a RULER artefact (2:1 over-call), not a
  reading difficulty.

## The one mismatch: Effusion

Effusion is our third-weakest gold column (0.8522) against the cleanest published channel (κ≈0.90,
routinely commented). Hypotheses, none priced: (i) the gold label's threshold differs from the reports'
(any trace effusion vs clinically significant) — a definitional gap, not an error rate; (ii) at 35/58
positives the AUC is computed on 23 negatives — the coarsest neg count in the table, so the estimate is
noisy (Hanley-McNeil SE at 35/23, AUC 0.85 ≈ 0.055); (iii) genuine model gap. Not resolved here.

## Caveat

The gold ordering confounds model capacity with label quality (the model trained on report labels).
This table is a CONSISTENCY check, not an attribution: the pattern "published-noisy channel = weak gold
column" holds for the extremes and breaks only for Effusion.
