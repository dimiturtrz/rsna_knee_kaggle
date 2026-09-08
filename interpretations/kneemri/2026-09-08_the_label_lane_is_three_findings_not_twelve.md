# The label lane is three findings, not twelve — and the pooled gap spans zero

`supervision_gap --labels report_labels_v2.csv`, 58 radiologist-annotated studies, 696 cells,
2026-09-08T16:30Z. REFERENCE = the report label used as a predictor of the radiologist. ACHIEVED =
what published members (out-of-fold) reach against the same radiologist.

## 1. The headline number does NOT say labels are the ceiling

```
labels - model: +0.0266  [-0.0082, +0.0610]  over 58 studies
```

The interval **spans zero**. Pooled over twelve findings, these 58 studies cannot say whether the reader
or the model is the weaker half. Anyone quoting "labels are the ceiling" as a global claim is quoting a
number whose own interval contains the opposite claim. Four of twelve per-finding bands exclude zero;
twelve reads, so that is a shape, not twelve findings.

## 2. The shape, and it splits both ways

| finding | REFERENCE | ACHIEVED | model − reader | reader FP | reader FN |
|---|---:|---:|---:|---:|---:|
| Effusion | 0.8149 | 0.9453 | **+0.1304** | **69.6%** | 2.9% |
| Contusion | 0.7530 | 0.8421 | **+0.0891** | **51.3%** | 15.8% |
| Synovitis | 0.6870 | 0.7378 | +0.0508 | 9.7% | **51.9%** |
| Medial OA | 0.9085 | 0.9504 | +0.0419 | 18.6% | 6.7% |
| Baker's | 0.9321 | 0.9547 | +0.0226 | 10.9% | 8.3% |
| Fracture | 0.8701 | 0.8861 | +0.0160 | 15.0% | 27.8% |
| Lateral OA | 0.7892 | 0.7099 | −0.0793 | 14.9% | 18.2% |
| PF OA | 0.8861 | 0.8005 | −0.0856 | 21.6% | 23.8% |
| Medial Meniscus | 0.9435 | 0.8498 | −0.0938 | 21.9% | 3.8% |
| MCL | 0.9762 | 0.8798 | −0.0964 | 18.4% | 0.0% |
| ACL | 0.9865 | 0.8640 | −0.1225 | 26.5% | 4.2% |
| Lateral Meniscus | 0.8422 | 0.6497 | −0.1925 | 28.6% | 13.0% |

## 3. The mechanism: the top of the table is the top of the reader-error table

The three findings where the model most beats its own labels are the three worst reader cells, and they
are the same three in the same order on the error that dominates each: **Effusion FP 69.6%** and
**Contusion FP 51.3%** are the two largest false-positive rates in the corpus, and **Synovitis FN 51.9%**
is the largest miss rate. Nothing else is close. That is not twelve independent draws landing in an
order — it is the over-call `noise_binding` already measured the members learning at tracking
**+0.538 to +0.700**, showing up on the axis that says which findings carry it.

So the label lane has a target, and the target is **three findings, not the corpus**. A better reader
is worth what it fixes on Effusion, Contusion and Synovitis and approximately nothing elsewhere.

## 4. The bottom of the table is NOT a label problem and no reader helps it

ACL: the report predicts the radiologist at **0.9865** with FP 26.5% / FN 4.2%, and the model reaches
0.8640. MCL 0.9762 → 0.8798 with a **0.0%** miss rate. Lateral Meniscus 0.8422 → 0.6497. On these the
signal is in the label and the model is failing to extract it from the pixels. Cleaning labels there
cannot move them; showing the model more of the image might. These are also three of the four bands that
exclude zero, so this half is the better-measured half.

## 5. What this does NOT license

The 58 studies are 1.3% of the corpus. Every per-finding cell here is a 58-study read with a base rate
between 9 and 35 positives, and the pooled interval spans zero. This ranks findings; it does not price a
reader. It says WHERE to look, and it pre-registers what a reader gain must look like to be real:
concentrated on Effusion / Contusion / Synovitis. A learned-reader gain spread evenly over twelve
findings is a null wearing a mean.

## 6. Standing counter-evidence, stated before the lever

`confident224` (README:326) cleared 14,445 UNK cells and **cost 0.0071**. A reader that resolves more
cells is not automatically better; a reader that is more SILENT is measurably worse. The candidate on
the table (`local_reader_annotated.csv`, 21.0% silent vs the lexicon's 25.9%) is the only one of the
three surveyed readers that is less silent than the incumbent, which is why it is the candidate and
Qwen3.8-27B (30.2% silent) is not.

## 7. CORRECTION, same day, 16:41Z — the per-finding map does not transfer, and the axis was already closed

Everything above is true of our holdout and false as a plan. Four board instruments, all spent
2026-09-04/05, close the per-column axis as a class, and I wrote §1-§6 without reading them:

- **`colnull_v1`** (ref 56020638) read **exactly 0.500** — the board metric is a plain unweighted mean of
  twelve per-column AUCs. A column gaining *d* moves the board by *d*/12.
- **`colmcl_v1`** (ref 56020054) → board MCL **0.980**, and **`colbaker_v1`** (ref 56023244) → board
  Baker's **0.992**. Those are the LOWEST (0.8104) and HIGHEST (0.9208) columns of our own offline
  ordering, and their board intervals **overlap**. `colbaker` explicitly retracts `colmcl`'s "the map
  inverts" reading — two points cannot show ordering at this resolution. What they establish is LEVEL:
  a 0.110-wide offline spread lands within noise of itself on the board.
- **`colhalf_v1`** (ref 56037952) confirms it by an independent route — the six columns our pool scores
  worst average **0.932** on the board against the strong six's **0.956**. A **0.024** gap against a
  **0.110** offline spread. Its own words: *"the board's 0.056 of headroom is distributed, not pooled…
  every one of the twelve columns has to move about 0.010 of AUC; there is no weak lane to attack and
  no per-column shortcut exists."* It cancels the ten remaining single-column ablations.

So the split in §2 ranks our SUPERVISION, not the board's difficulty, and §3's "the label lane is three
findings" is a statement about which of our labels are worst — not about where board points are. **No
per-finding plan may be priced off this table.** The `focal_dense` observation in the ledger keeps its
value as mechanism (it is still the only intervention that moves ACL and Lateral Meniscus at all) and
loses all of its value as a board projection.

What survives, and it is the only thing that does: an intervention has to move **all twelve columns
together**. That is what a supervision change does — it retrains every seat — and it is why the learned
reader (`knee-wko44`, M227) is the lane this table hands off to, rather than any per-finding work.
