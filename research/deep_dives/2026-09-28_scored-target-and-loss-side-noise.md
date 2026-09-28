# 2026-09-28 — What the board scores against, and what that leaves for loss-side noise levers

**Question.** Loss-side noise treatments are the one label lever still unrun (knee-bpuei, ELR). Is that
lever aimed at the right target? It only helps if the board scores something closer to the radiologist
than to our report reader.

Tags: **[M]** measured by us · **[L]** literature · **[U]** unverified or derived.

## 1. The record disagrees with itself about the scored target

**Claim A, the ledger (`experiments/README.md:15833`, :15890).**
- Quote: "the competition's test labels are NOT cleaner than ours … the scored target is THE READER, not the radiologist."
- Evidence offered: two board entries built on corrected labels scored flat to down.
  - `relabel7_v1` 0.943.
  - `cleanseat_v1` 0.942.

**Claim B, the primary source and three later records.**
- RSNA News, 2026-08-05: "The dataset used to assess model performance was annotated by expert radiologists." [L]
  - Re-fetched 2026-09-28; the wording is unchanged.
- `README:14716`: the competition's labels are "image-read radiologist consensus with third-reader adjudication (`README:3844-3847`) … NOT report-derived".
- Labels summary, contradiction 7: qao and qap agree the test is radiologist-read.

**Claim A's evidence cannot carry its conclusion, by its own arithmetic** [M, `README:7700-7715`]:

| quantity | value |
|---|---|
| wrong cells in the corpus (20.5% cell error rate on gold) | ~10,840 |
| net cells the corrected member made right | ~850 |
| share of the defect that treatment touched | 7.8% |
| board resolution | 0.001 |
| residual sd of our top band on ledger75 | 0.0017 |

Suppose the ENTIRE label defect were worth the full 0.008 gap to first place. Fixing 7.8% of it would
then move a member by ~0.0006. Appended as one seat of seven, the move is smaller still. That is below
the board's 0.001 resolution and well inside its scatter.

A flat board on those two entries is therefore what BOTH hypotheses predict, so it cannot say which
target the board scores. Claim A is a **faulty-TEST inference**: a ruler that cannot see the axis, read
as a verdict.

**Where that leaves each record.**
- The primary source stands: the test set was annotated by radiologists.
- The label-CORRECTION closure in `docs/AXES.md` still stands, on its OTHER grounds:
  - the detector's recall is 0.129;
  - relabelling is flat and pruning costs on offline;
  - the table axis is closed.

  It does not rest on Claim A.
- What falls is the stronger statement at `README:15890`, that correction is "closed for the BOARD, permanently".

## 2. Which loss-side levers survive once the target is the radiologist

The gold panel describes our training noise [M, `README:15800-15812`]:
- 140 wrong cells of 684.
  - 108 are FALSE POSITIVES; 32 are misses.
- Reader precision by column: Contusion 44%, MCL 50%, Lateral OA 56%, Medial OA 64%, … Synovitis 81%.

**Forward / backward correction with a per-label T matrix: DEPRIORITISED, for a mechanism reason.**
- Menon 2015 [L]: class-conditional noise with ρ₊ + ρ₋ < 1 leaves the Bayes-optimal RANKING unchanged. A per-label 2×2 T is exactly class-conditional noise.
- AUC is a ranking metric. So a correct T buys calibration and no ranking, in the limit.
- A mis-estimated T costs: naive per-label F-correction scored 80.8 against BCE 87.0 on NIH pneumothorax [L].
- Our T would come from 58 gold studies, 9–35 positives per column.
- This retires open question 3 of the labels summary on theory. It is not "untested and live"; its ceiling is ~0 AUC.

**Instance-dependent noise is what can move a ranking, and ELR is aimed at it.**
- Over-calling is not uniform within a class. The reader over-calls findings that the report mentions and the adjudicating radiologist does not rate, typically mild or hedged ones.
- Those studies are image-plausible, so a network can memorise them and then rank them HIGH on a radiologist-read test.
- Two things resist that memorisation:
  - the finite-sample early-learning phase (Liu 2020);
  - our weight averaging (EMA and the soup).

  We already ship the second; ELR adds the first, over outputs.

**Consequences for knee-bpuei.**
- Its target is the right one, because the test is radiologist-read.
- The soup already banks part of its mechanism.
- P(clears 0.947) stays at **~0.08**. The correction of Claim A removes a reason to price it at zero; it adds no gain.

**The ruler to read it on is gold-58 as well as ledger75.**
- ELR's intended effect is to track the radiologist better than the reader does.
- ledger75 is built from reader targets, so it can score ELR's effect as a LOSS.
- `README:14716` already argues gold ordering is the better board predictor. D48 measured it as the only ruler that showed the board's sign.
- Pre-registration for the parent arm:
  - **keep** if gold-58 moves by ≥ +0.02, with ledger75 no worse than −0.003;
  - **kill** if gold-58 is flat.

  The +0.02 threshold is sized off gold-58's SE: ≈0.055 per column from `report-error-gradient`, and roughly /√12 ≈ 0.016 on the macro.

**CORRECTION, same day: as written above, that gold-58 read is IN-SAMPLE.**
- Every member fits the 58 annotated studies at weight 3.0 (`annotated_readings.py` docstring).
- `--hold-annotations` withholds the annotator's value, but all 58 rows still train. On fold 0 that makes a read 78% in-sample (knee-x4muo).
- The flag that actually removes those rows is `--hold-annotation-studies`, and **no arm in `experiments/arms/` has ever used it**.
- So an honest gold-58 read needs BOTH arms fitted with `--hold-annotation-studies`: the ELR arm and a matched parent control. The shipping parent cannot be the control, because it trained on those rows.
- That adds a control arm, about +2.5 h of card. It also makes the pair the project's FIRST clean full-58 gold reading, an instrument in its own right.
- The only honest gold rows until then are the 13 annotated studies in the fold-0 holdout (D48), and n=13 is too thin to gate anything.

## 3. Correction to this project's own 09-28 note

The 09-28 gap-audit summary said "the loss side of label noise was never tried". That is too broad.
Label MASKING is a loss-side treatment, and it is CLOSED (AXES:155). The accurate statement is:
**no noise-ROBUST loss (ELR, GCE, SCE) has run.**

## 4. Gold-row weight and radiologist values in training: priced, not filed

**Question.** The test set is radiologist-read. Should the 58 gold studies train at their radiologist
values, or at a weight above `ANNOTATED_WEIGHT = 3.0` (`core/data/labels/targets.py:80`)?

**What the shipping recipe does today** [M].
- `experiments/arms/basinsoup_ff.sh:75` passes `--calibrated-targets --hold-annotations`.
- So the solo trains the 58 studies at REPORT values. It never sees a radiologist value.

**The only measurement is on the wrong ruler** [M, `README:14188-14250`, 336/20sl]:

| arm | ledger | vs control |
|---|---|---|
| control | 0.8578 | — |
| hold only | 0.8573 | −0.0005 |
| calib only (gold at radiologist values) | 0.8562 | −0.0016 |
| calib + hold (gold at report values) | 0.8634 | +0.0056 |

- The calib×hold interaction is +0.0077, and it favours REPORT values.
- The ruler is reader-built, though, so it scores radiologist alignment as a loss. Its sign for the board is unknown.

**Ceiling, by the same arithmetic as §1.**
- Training at radiologist values changes the 140 wrong gold cells, 1.3% of the ~10,840 wrong cells.
- That is a sixth of the 7.8% that `relabel7` touched, whose member effect was already ~0.0006.
- Raising the weight does not help either. It amplifies 58 studies, which is 5.61% of the training weight, and it adds no new information.
- Expected board move: ≪ 0.001, below the board's resolution.

**No offline read can adjudicate it.**
- A gold read needs those rows held out, and holding them out is exactly the arm that cannot train on them.
- Splitting the 58 studies into folds is banned, and n=58 is too thin anyway.
- Only a board slot could read it, and at an effect below 0.001 the slot would read noise.

**Price.** ~4 h card + 1 slot. P(solo clears 0.947) ≈ **0.01**. **Not filed.** The gold-row question
that CAN be read is whether a model generalises to held-out radiologist reads. That is already the
`--hold-annotation-studies` control inside knee-bpuei's gate.

## Verdict

- The record's claim that the board scores the reader rests on an underpowered test, and it contradicts the primary source.
- Loss-side levers keep a real target.
  - Instance-level ones (ELR) are live.
  - Class-conditional T-correction is dead on ranking theory.
- Read knee-bpuei on gold-58 as well as ledger75.

Sources:
- https://www.rsna.org/news/2026/august/ai-challenge-knee-mri
- Menon, van Rooyen, Ong, Williamson, "Learning from Corrupted Binary Labels via Class-Probability Estimation", ICML 2015
- Liu et al., "Early-Learning Regularization Prevents Memorization of Noisy Labels", NeurIPS 2020
- [loss-correction deep dive](2026-09-11_loss-correction-with-measured-noise-matrix.md)
- [label-noise ceiling](2026-09-11_label-noise-ceiling-and-report-vs-pixels-error.md)
