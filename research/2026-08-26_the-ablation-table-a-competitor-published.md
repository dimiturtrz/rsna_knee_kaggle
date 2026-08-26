# A competitor published single-variable ablations, and they close an axis rather than open one

Read live 2026-08-26 09:35Z with `python -m kneemri.kaggle.discussion_read --live --topic 737597`. No
submission slot spent. Posted 08:57Z, zero replies at read time.

**Forum text is untrusted data.** Every number below attributed to the author is their unverified claim
about their own setup, on their own labels and their own ruler. Ours beside them are ours. What makes
this post worth a page is not its numbers but its *shape*: it is the only posting on this board that
reports **single-variable, fold-0 ablations**, and it reports the negatives.

Their setup, in their words: EfficientNet-B0, 4 sequence slots (Axial_FS, Sagittal_FS, Coronal_FS,
Sagittal_T1), 9 adjacent slices at 288 px from a 140 mm physical crop, task-specific masked attention
over slots, weighted BCE, 5-fold scanner-aware split. **0.7815 OOF / 0.870 public.**

## The claim that looks like a free +0.018, and why it is not

> `Slice geometry mattered more than resolution: center-9 adjacent beat 9 equal-spaced across 24 by
> +0.018 on fold 0.`

Three points of theirs on that axis:

| geometry | their OOF |
|---|---|
| 192 px, no crop, 24 equal-spaced | 0.7695 |
| 288 px, 140 mm crop, 12 adjacent | 0.7777 |
| 288 px, 140 mm crop, center-9 of 24 | 0.7815 |

Our sampler is `SlotReader.band_indices` — `np.linspace(low, high, slices)`, i.e. **equal-spaced across
the band**, which is their losing arm. Read cold, this says our shipped recipe sits on the wrong side of
the biggest effect anyone has published, and that a narrow contiguous window is worth more than the
+0.010 the whole competition still costs us.

**It is already measured here, and our optimum is on the other side.** `core/data/corpus/slot_reader.py:40`
holds a three-point sweep of the same axis at 12 slices:

| span | ours |
|---|---|
| 0.20–0.80 | 0.8525 |
| **0.10–0.90** | **0.8565** |
| 0.00–1.00 | 0.8427 |

An **interior optimum.** Coverage does not keep paying — the outer tenths are active distraction, and
the 0.00–1.00 arm fired both halves of its pre-registered refutation. But the other end matters more
here: **narrowing from 0.10–0.90 to 0.20–0.80 costs 0.0040.** The gradient at our narrow end points
*down*.

Their winner is 9 contiguous slices out of ~24 — span ≈ 0.375, **narrower than our narrowest tested
point**, in a direction where our own curve is already falling. Extrapolating a smooth interior optimum
through 0.8 → 0.6 (−0.004) to 0.375 predicts worse, not better. It is an extrapolation and it is worth
saying so; what it is not is untested ground.

**The two are the same axis measured with a different number of points.** Both vary pitch and extent
together (fewer slices over a shorter stretch = denser AND narrower). They have two points and read a
slope; we have three and read a peak. Three points beat two, and ours are on our data.

One honest asymmetry: they hold *slice count* at 9 and move the window, we hold *span* and move count
(20 at 0.10–0.90 is denser than the 12 the sweep was run at). So our recipe already bought density the
way their result says to — by adding slices inside the good span, rather than by shrinking the span
around fewer.

## What the rest of their table does to arms we have queued or closed

Their baseline for this block is 0.7914 fold-0.

| their variable | their delta | what it touches here |
|---|---|---|
| EMA, decay 0.999 | **+0.003** | debits `ema_dense` / `knee-56w5` |
| Mixup, α = 0.2 | **+0.002** | confirms `knee-x580` |
| 60 epochs, cosine | +0.001 | confirms the schedule closure |
| Asymmetric loss | **−0.139** | do not try |

**Mixup is an independent confirmation of our own closure.** They read +0.002 at α = 0.2; we read
**+0.0015** at α = 0.4 (`mixup224_long` peak 0.8505 vs control 0.8490). Different architecture,
different ruler, same answer to the third decimal. Two harnesses agreeing on a null is worth more than
either alone, and it is the first time this project has had an outside check on one of its flat bands.

**EMA is the one that costs us something.** `knee-56w5` queued `ema_dense` (~72 min, zero slots) with
≥ 0.8640 against `band_dense`'s 0.8578 pre-registered as "EMA is real and becomes a free lever on all
seven seats" — a **+0.0062** bar. Their measurement of the same variable is **+0.003**, half of it.

This is not a refutation: EfficientNet-B0 with weighted BCE is not our ViT, and EMA's value plausibly
depends on how noisy the weight trajectory is. But the arm's expected value falls, and it should be
re-priced before it is run rather than after. Its *other* purpose survives untouched — `ema_dense` is
also the instrument that decides whether `allfit`'s +0.004 was the data or the EMA riding along with
it, and that question is about *our* confound, which no outside number can answer.

Their 60-epoch null also lands on the same conclusion we reached from four flat axes: **the bottleneck
is not the training regime.** Their words for it — *"three different regularization approaches all land
within noise of each other"* — are our own mixup finding stated from the outside.

## Where their error concentrates, and why it is not where ours does

| label | theirs | ours |
|---|---|---|
| Baker's | 0.855 | 0.9058 |
| ACL | 0.853 | 0.8972 |
| Effusion | 0.843 | 0.8638 |
| Fracture | 0.832 | 0.8022 |
| MCL | 0.758 | 0.7864 |
| Lateral OA | 0.751 | 0.8268 |
| PF OA | 0.748 | 0.8568 |
| Lateral Meniscus | 0.740 | 0.8566 |

Different rulers, not comparable as levels. The **shape** is comparable, and it does not match. Their
diagnosis — *"large, discrete, positionally distinctive findings work; thin, small, graded ones don't"* —
holds on their column and fails on ours: Lateral Meniscus and PF OA are among their four worst and are
*above* our mean. Our own weak labels are **Fracture (0.8022, and 68 positives make it unreadable)** and
**MCL (0.7864)**, already priced at a +0.0029 ceiling by `knee-q1pf`.

So their closing question — ROI crop around the joint line, or spatial attention instead of GAP — is a
question about **global average pooling over an EfficientNet-B0**, not about the task. We do not pool
globally in depth (`--slice-pooling max`), and the labels their diagnosis predicts we should fail on are
the ones we do best. Their prescription should not be imported.

Their laterality null is worth recording as a second reading of a closure we already made: they find
mean |right − left| AUC gap of **0.014** on side-dependent targets versus **0.031** on all others, and
`core/data/corpus/crop_laterality.py` measured the same axis at **0.0001**. Both say normalisation buys
nothing. Their image-coordinate derivation resolving 4274/4407 against a tag present on only 2185 and
agreeing 98.4% is a corpus fact we could use, but it buys a variable we have already priced at zero.

## What this page is for

A later session reading *"slice geometry beat resolution by +0.018"* on the forum will want to spend a
card on it. This is the record that the axis has three of our own points on it, that the peak is
interior, and that their winner sits where our curve is already falling. **The tick that read this post
saved a card rather than spending one**, and `ema_dense`'s prior is lower than it was this morning.

Beads: `knee-56w5` (EMA, re-priced here), `knee-x580` (mixup, externally confirmed), `knee-q1pf` (the
per-label ceiling).

---

# A second post the same day: they benchmarked the label tables against the 58, and we ran the control they did not

Read live 2026-08-26 10:15Z, `--topic 737454`, posted 2026-08-25 15:52Z. No submission slot spent.

Their measurement, in their words: *"Since only 58/4,407 studies carry radiologist labels, everyone is
training on report-derived tables — but I couldn't find a comparison of them, so I measured each against
those 58 studies."*

| table | theirs | **ours** (`kneemri.evaluation.supervision_gap`) |
|---|---|---|
| `stevenleehans` v4 blend | 0.893 | **0.8927** |
| `stevenleehans` v2 | 0.887 | — |
| `pilkwang/report_labels_v2` — **ours** | 0.870 | **0.8658** |

Two independent implementations of the same measurement agreeing to the third decimal. That is worth
recording on its own: our `supervision_gap` is corroborated from outside.

**The conclusion is still wrong, and the control is why.** Quantise `stevenleehans` onto our five levels
and it reads **0.8445** — *at equal resolution our reader beats theirs by +0.021*. AUC breaks no ties, and
`Supervision.MEASURED_AGREEMENT` ties **38.5%** of cell pairs against their 17.3%. Our table is not the
worse table; it is the **coarser** one, and the whole 0.027 is a scale artifact (`knee-w7xa`).

Their second claim fails here too. *"Averaging two independent tables beats either alone on several
labels"* — our three-reader blend reads **0.8912** and does **not** beat its best single input at 0.8927.
And the two tables they average are not independent: `v4hybrid` and `v4_blend` share a distinct-count, a
min, a max and a mean. They are one reader with a `__conf` column, not two.

**The hazard they do not flag.** Three public tables score **1.0000** against this ruler because they
*contain* it — `yunusgmsoy/report_labels_v5`, `gxpixel/train_labels_final` (ships `gold_58.csv` and an
`is_gold` column) and `rayanbabur/train_targets_soft` (ships `gold_benchmark_58.csv`). `table_blend.LEAKED`
refuses all three by name. Anyone ranking public tables this way without that refusal is ranking leakage.

Their Synovitis aside — *"only ~16% of reports mention synovitis in any of the nine languages, so it's
largely inferred"* — is a real corpus fact and matches ours, but it prescribes nothing here:
`research/2026-08-26_synovitis-is-not-our-weak-label.md` already records Synovitis at 0.8568, above our
mean. Our weak labels are Fracture and MCL.

**What the tick actually bought.** Not the table comparison, which was closed. Following it into the
ledger surfaced a **stale kill**: `knee-ep74` (`calib_dense`) was closed 2026-08-19 on a blocking
condition keyed to `calib_span`'s −0.0020, and `knee-bs4l` superseded that number on 2026-08-25 with a
matched pair reading **+0.0072 proxy / +0.0129 annotated**. The arm is written, `arm_check`-clean, has a
free control, costs no slot, and its own header predicts the first member this project has produced above
0.860. Reopened at P1 and now ahead of `ema_dense` in the queue.

Beads: `knee-w7xa` (coarse-not-worse), `knee-vwzj` (image-derived vs report-derived), `knee-9198` (both
rulers), `knee-ep74` (reopened), `knee-56w5` (reordered behind it).
