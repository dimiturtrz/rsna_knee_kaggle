# The label-error selector is v2-vs-the-field: lone dissent is 75% wrong, measured twice

2026-09-11 · researcher lane · spins off the 402-cell blind audit (D327/D334, main's M340/M342)

## The chain

1. The audit (402 blind-adjudicated cells, pre-registered bounds) measured: five-way unanimous cells
   err 1.48% (bound 2.37%); five-way discordant cells err 15.6% (bound 20.3%). B = 0.0787 → REFUTED
   as-is (bands: <3% licensed, 3–6% conditional, >6% refuted).
2. Voter-set sensitivity (D334): `llm_labels_full` is a calibration outlier (corpus positive rate
   0.556 vs ~0.27; Synovitis 0.97; lone-dissents 25.7% of cells). Four-way without it: unanimity
   95.05%. Main's re-tabulation on existing verdicts: B′ = 0.0714 → REFUTED again; and the
   decomposition: 52 steven_full-only dissent cells err 1.9% (indistinguishable from unanimous);
   12 four-way-discordant cells err 75%.
3. This note: the corpus-wide structure behind that decomposition.

## Corpus structure under the four real tables (binarised ≥0.5, 4,406 studies × 12 = 52,872 cells)

| tier | cells | share |
|---|---|---|
| 0-of-4 positive | 36,430 | 68.90% |
| 4-of-4 positive | 13,823 | 26.14% |
| lone dissent (3-1 / 1-3) | 2,526 | 4.78% |
| 2-2 split | **93** | 0.18% |

"≥2-of-4 dissent" — the intuitive purity-preserving selector — is 93 cells corpus-wide. Unusable.

## The discovered asymmetry: WHICH table dissents is everything

- flight and steven_blend lone-dissent on **0.0%** of cells; lixin 0.4%. The three foreign tables vote
  as one cluster (shared machinery suspected — steven_blend is a blend that may contain the others).
- Lone dissent ≈ **v2 lone-dissent**: 2,315 cells (4.38% of corpus), 90% of them v2-says-1 while the
  cluster says 0 (v2 over-calls the field).
- Per column: Medial Meniscus 567, Contusion 556, Lateral Meniscus 431, ACL 298, Lateral OA 141,
  Medial OA 97, MCL 83, PF OA 49, Effusion 36, Synovitis 31, Baker's 26, Fracture 0.

## Purity, measured twice independently

- Pack adjudication (main, blind): 9/12 four-way-discordant cells wrong = **75%**; 11 of the 12 are
  v2-lone-dissent cells.
- Gold-58 cross-check (57 studies, 684 cells; no adjudication spent): on the 33 lone-dissent gold
  cells, v2 is the dissenter in 32 and errs on **75%** (24/32). When v2 votes with the majority
  (n=1), no error.
- Baselines for contrast: steven_full-only dissents 1.9%; unanimous-cell error vs gold 17.8% (that
  figure is the report-vs-pixels + prevalence layer of D330, separable from extractor error).

Two instruments, one number: **the v2-lone-dissent selector flags 2,315 cells of which ~75% are wrong
— ≈1,700 probable label errors, pre-identified, concentrated in MedMen/Contusion/LatMen/ACL.**

## Caveats

- The cluster votes as ONE reader: effective voter count is two (v2, the field), so "four-way
  unanimity" is not four independent opinions; conditional independence is more violated than the
  five-way framing suggested.
- n=12 and n=32 behind the two 75% readings — the intervals are wide, but they overlap and come from
  independent references (adjudicated report reading vs radiologist gold).
- The audit's pre-registered verdict is unaffected: B = 0.0787 (five-way) / 0.0714 (four-way), both
  REFUTED for the corpus as-is; this note is about the selector, not the verdict.

## What it prices

The repair target is not the 13.94% blended-discordant set (circular voter set) nor the 30.7%
five-way set (diluted by steven_full's lone noise). It is the **2,315-cell v2-lone-dissent set**:
4.4% of the corpus, ~75% error, direction known (over-call), columns known. `confident_span`
(masking 27.3% UNK mass) COST 0.0116 — subtraction by confidence fails; this selector is subtraction
by measured error, which the audit has now priced.
