# The holdout is not the defect — balance is already there and a gold holdout cannot resolve us — 2026-09-18T08:43Z

## Why this file exists

"Should we choose another 900?" has now been asked four times, and each time it has been answered from
scratch out of prose. The three stated requirements were: the holdout should be **balanced**, it should
contain the **58 quality (radiologist-annotated) studies**, and it should be **fresh** rather than the
draw ~450 members were selected against. This file measures all three so the next asking is a read.

Card cost of everything below: **zero.** It is label arithmetic and a hash.

## 1. Balance — already satisfied, nothing to buy

Fold 0 (n=909) against the whole corpus (n=4407), positive rate per finding, `report_labels_v2`:

| finding | corpus | fold 0 | delta | rel | binomial sd | sigma |
|---|---|---|---|---|---|---|
| ACL | 0.2782 | 0.2739 | −0.0043 | −1.5% | 0.0149 | 0.3 |
| MCL | 0.1740 | 0.1881 | +0.0141 | **+8.1%** | 0.0126 | **1.1** |
| Medial Meniscus | 0.5326 | 0.5325 | −0.0001 | −0.0% | 0.0165 | 0.0 |
| Lateral Meniscus | 0.2544 | 0.2508 | −0.0035 | −1.4% | 0.0144 | 0.2 |
| Medial OA | 0.3674 | 0.3674 | +0.0001 | +0.0% | 0.0160 | 0.0 |
| Lateral OA | 0.2467 | 0.2596 | +0.0130 | +5.3% | 0.0143 | 0.9 |
| PF OA | 0.4518 | 0.4499 | −0.0018 | −0.4% | 0.0165 | 0.1 |
| Effusion | 0.6049 | 0.6161 | +0.0111 | +1.8% | 0.0162 | 0.7 |
| Synovitis | 0.1316 | 0.1320 | +0.0004 | +0.3% | 0.0112 | 0.0 |
| Baker's | 0.2528 | 0.2398 | −0.0130 | −5.1% | 0.0144 | 0.9 |
| Contusion | 0.2943 | 0.3003 | +0.0060 | +2.1% | 0.0151 | 0.4 |
| Fracture | 0.0713 | 0.0770 | +0.0058 | **+8.1%** | 0.0085 | 0.7 |

**Every finding is inside 1.1 sigma of a random draw of 909.** The two largest *relative* gaps (MCL,
Fracture, both +8.1%) are the two rarest findings, which is where a percentage is loudest and a sigma is
not. There is no imbalance to correct, so the balance requirement licenses no work.

This is a property of the mechanism, not luck: `ReportFolds` assigns by `sha1(seed:report_text) % folds`,
which is a uniform draw over report groups. A re-draw would produce a different table of the same shape.

## 2. The 58 gold studies — a seed cannot do it, and the ruler could not resolve us if it could

They are scattered: **{fold 0: 13, 1: 8, 2: 16, 3: 11, 4: 10}**. Assignment is a hash of report text, so
searching seeds for a draw that puts all 58 in one fold is a `(1/5)^58` event. The requirement is not a
re-draw at all — it is an **explicit study list**, a change to `ReportFolds`'s source of truth.

Before paying for that, what the resulting ruler could measure. Hanley–McNeil sd at AUC 0.85 on the full
58, per finding (positives out of 58): ACL 24 → 0.055 · MCL 9 → 0.083 · Medial Meniscus 26 → 0.053 ·
Lateral Meniscus 23 → 0.056 · Medial OA 15 → 0.066 · Lateral OA 11 → 0.076 · PF OA 21 → 0.058 · Effusion
35 → 0.050 · Synovitis 27 → 0.053 · Baker's 12 → 0.073 · Contusion 19 → 0.060 · Fracture 18 → 0.061.

**Mean per-label sd 0.062. Macro sd 0.018 with labels independent, 0.046 at corr 0.5.** Paired — the form
we actually decide on — scales the 909-study floor by `sqrt(909/58) = 3.96`: **0.0062 → ~0.025.**

We decide at **0.0062**. A gold holdout at n=58 is **4x too coarse to rank anything this project runs**,
and it would cost ~2.6 h of card (a control refit under a new split and `--hold-annotation-studies`) plus
the explicit-list change. Going 13 → 58 is a genuine 2.1x improvement and both ends of it are unusable.

This agrees with a reading the repo already had and stated in a different denominator:
`2026-08-24_no_denominator_resolves_a_per_label_deficit.md:99` prices the annotation ruler at n=58 as
**"no, ±0.15"** per label. That is this same number seen per-finding instead of macro.

**The blocker is the count, not the split.** What would make a gold ruler usable is more annotation —
roughly 58 → 400 to bring the paired resolution under 0.01 — and that is an acquisition question, not a
holdout question. No seed, no list and no retrain substitutes for it.

## 3. Freshness — the only live part, and it is small

The real defect in fold 0 is that it is **spent as a selection instrument**: ~450 members have been ranked
against one fixed draw, so the top of that ranking carries a winner's-curse component. The board agrees —
three top-ranked members in a row landed low, `headlrsolo_v1` at −1.8 sd against the board fit.

A fresh draw removes it. The constraint is that **every existing member trained on folds 1-4**, so any new
holdout drawn from the full 4407 hands them training rows — the `raptor_solo` 98.6%-in-sample defect that
voided a board row. That leaves two shapes, and neither is free:

* **sub-carve the current 909** — honest for all ~450 historical records, zero card, but `n` halves so the
  paired floor grows by `sqrt(2)` to **0.0088**, and it inherits the same draw it is meant to escape.
* **a fresh forward-only split** — fixes the curse, but no existing member can be scored on it, so it costs
  a **~2.6 h control refit** before the first arm can be read, and severs comparability with the ledger.

Priced against what it returns — roughly **0.003 board** of recovered honesty in selection — a 2.6 h refit
that also strands ~450 historical rows is a bad ratio while the card has arms worth more than that queued.

## The verdict

**Do not rebuild the holdout.** Balance is already there (measured, 1.1 sigma worst case); the gold-study
requirement needs an explicit list and would still yield a ruler 4x too coarse to decide anything;
freshness is real but worth ~0.003 against a 2.6 h refit and a severed ledger.

**What re-opens it:** more annotations. At ~400 gold studies the gold ruler resolves under 0.01 and a
holdout built around them becomes the project's best instrument rather than its most expensive one.

## Evidence

* `core/data/corpus/folds.py` — `ReportFolds.assign`, `sha1(seed:report_text) % folds`; `TRAIN_ONLY`/`EXCLUDE_NONE`.
* `kneemri/evaluation/gold_rows.py:31` — `GoldRows`, the 58 all-or-nothing annotated studies.
* `kneemri/evaluation/annotated_readings.py:11-18` — `--hold-annotations` does NOT withhold the rows; 45 of
  58 train in fold 0, so a reading there is 78% in-sample (knee-x4muo). `--hold-annotation-studies` is the
  flag that does, and no member has been fitted with it.
* `interpretations/kneemri/2026-08-24_no_denominator_resolves_a_per_label_deficit.md:99` — the same verdict per label.
* `interpretations/kneemri/2026-08-26_the-label-table-axis-is-exhausted.md` — the label-table axis, already closed.
