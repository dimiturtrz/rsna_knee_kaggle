# The project owns a 2.2x-finer ruler and has only ever pointed it at label changes — 2026-09-18T09:24Z

*No card. One tool run (`paired_floor`, CPU, 1000 resamples) plus an audit of what the existing five-fold
arms were read on.*

## The problem this is about

Our best member reads **0.9059 ledger75**; board 0.947 needs **0.9124**; the required gain is **+0.0065**
against a paired floor of **0.0062**. So the ruler we decide on can barely see the thing we need. And
`knee-dry3h` closed twenty-two defaulted leaves with the words *"not one alternative beats it above the
floor"* — a sentence about the FLOOR as much as about the leaves. `CLAUDE.md` licenses composing
sub-floor increments explicitly; what blocks composing them is that we cannot tell which are real.

## 1. The wide ruler's floor, measured rather than assumed

`python -m kneemri.evaluation.paired_floor --labels report_labels_v2.csv`, on the published out-of-fold
predictions where five folds partition all 4,407 studies:

| scale | mean per-label paired | mean unpaired | tighter |
|---|---|---|---|
| 881 studies (our ruler's size) | 0.0321 | 0.0683 | 2.2x |
| 4,407 studies (the corpus) | 0.0143 | 0.0308 | 2.2x |

`0.0321 / 0.0143 = 2.24`, against `sqrt(4407/881) = 2.24`. **The sqrt scaling holds to two decimals**, so
it may be applied to the macro floor rather than re-measured: **0.0062 -> ~0.0028.**

That is the whole quantitative claim. A wide-ruler read resolves at 0.0028 where ours resolves at 0.0062,
and the gain we need (+0.0065) sits at 1.05 floors on ours and 2.3 floors on the wide one.

## 2. The gap in the record: it has never been used for a recipe knob

Two arms have been run on the five-fold rotation, and both were **label** changes:

* `oof_reader_fold[0-4]` (v5reader) — read **PAIRED gold@58**, rankpooled −0.0112, sd 0.0122.
* `oof_band_folds` (v6band50) — read **PAIRED gold@58**, rankpooled −0.0066, sd 0.0062.

Their sds are 0.0062–0.0122, i.e. **no better than the fold-0 ruler they were meant to improve on**. That
is not a defect of the wide ruler and it is not bad luck. **A label change cannot be scored against the
labels it changes**, so both arms had to fall back to the 58 radiologist studies, and n=58 is the
resolution ceiling this project already wrote down (`2026-09-18_the_holdout_is_not_the_defect.md` §2:
paired gold resolves ~0.025; the `~0.0062` above is a pooled-across-folds estimator of the same 58).

**A recipe knob does not touch the labels.** `--backbone-lr`, `--slot-dropout`, `--head-lr`, depth,
`--slices`, pooling — every one of them can be read on the full 4,407 out-of-fold stitch against `v2`,
at 0.0028. **No arm has ever done this.** The instrument was built, used twice, and both times on the one
class of question that structurally cannot use it at full width.

## 3. What it costs, and the one assumption it rests on

The rotation is affordable only at the instrument recipe. Measured, not estimated:

* five folds at **224 px / 8 epochs** — `oof_band_folds` took **2.1 h** of card (README:13080, 13084).
* five folds at the **shipping recipe** — 5 x 3.5 h = **17.5 h**, refused on ratio (knee-700cm).

So the cheap wide ruler is the 224/8 one, and it can deliver the **SIGN** of a knob at 0.0028 — not its
magnitude at the shipping corner. That is the assumption the whole program rests on, and the evidence for
it is **n=1**: distillation reads **+0.0121** at 224/8 (gen-2 born-again, five folds, every fold up) and
**+0.0055/+0.0056** at the shipping corner. Same sign, magnitude compressed about 2x.

`CLAUDE.md` already states the rule in the form this needs — *only the SIGN transfers* — and a sign at
0.0028 is exactly what picks which of `knee-dry3h`'s twenty-two leaves are worth composing.

## The program, priced

1. **De-risk the transfer first, at zero card.** Find every knob with a reading at BOTH scales and check
   sign agreement. n=1 is not enough to spend 8 h on. This is the next step and it is arithmetic.
2. If sign transfers: **2.1 h per knob** buys a resolved sign on 3–4 of the twenty-two leaves (~8 h).
3. Compose the winners at the shipping recipe: **3.5 h**.

Total ~12 h for a plausible **+0.006 to +0.009** — three composed increments the current ruler cannot
distinguish from zero. Against 17.5–35 h for `knee-700cm`'s shipping-recipe rotation, which buys a finer
opinion about members we would still have to build, this is the better ratio by roughly 3x.

**What kills it:** step 1 coming back with knobs whose sign flips between scales. Then the 224/8 rotation
measures 224/8 and nothing else, and the twenty-two leaves stay closed at 0.0062.

## Evidence

* `kneemri/evaluation/paired_floor.py` — the tool; its own docstring frames the 2.2x as the unmeasured
  claim of knee-3vh. Run 2026-09-18T09:23Z, 1000 resamples, `report_labels_v2.csv`, output in §1.
* `experiments/README.md:13046-13082` — `oof_reader_fold[0-4]`, gold@58, 2.1 h, read negative.
* `experiments/README.md:13084-13099` — `oof_band_folds`, gold@58, the three estimators and their sds.
* `experiments/README.md:13954-13972` — gen-2 born-again, five folds, mean +0.0121, every fold up.
* `interpretations/kneemri/2026-09-17_no_teacher_is_above_the_student.md:14-26` — the +0.0056 shipping-corner
  distillation term, two independent reads agreeing to 0.0001.
* `experiments/arms/oof_folds.sh:24-29` — "THE RECIPE IS DELIBERATELY NOT THE BEST ONE", and why.
* `knee-dry3h` — the twenty-two leaves, none resolving above 0.0062.
* Denominators: ledger75, fold 0 / split_seed 2026, 909 studies, macro paired floor 0.0062. Wide ruler:
  five disjoint folds, union 4,407, `report_labels_v2`.
