# Their split is public after all, their "sites" are slots, and the six-point gap was never measured

2026-09-06. Three corrections to what this project believed about `mattiaangeli`'s DINOv3 folds at
12:30Z today. All three came out of reading their kernel source rather than their cfg, and each one
reverses a claim I had already written into an arm header and a letter.

## 1. THE SIX-POINT GAP DOES NOT EXIST. It was a label-table number wearing a member's clothes.

`foreign_recipe.sh` was launched against this sentence: *"their fold-0 solo reads 0.8874 on our 909
studies where our best DINOv3 member reads 0.8268."* **The 0.8874 is not a member score.** It is
`experiments/README.md:3684` — the **gold-58 macro AUC of a blended LABEL TABLE**, `ours + w*(foreign −
0.5)`, measured on 58 radiologist-annotated studies over ~696 cells, where "foreign" means a foreign
label table and not a foreign model.

So the comparison was: a label table's AUC on 58 studies against a member's holdout AUC on 909. Different
quantity, different instrument, different denominator. **Their checkpoints have never been priced on our
holdout at all** — nothing in the ledger, the interpretations or `comp/` carries a number for them.

The gap the arm was aimed at is not a small gap or a shaky gap. It is not a measurement.

## 2. `cond: token` IS SLOT CONDITIONING, NOT SITE CONDITIONING. `n_sites` is a dead key.

Their `Net.forward` (`bend_the_knee_to_dinov3_ensembled.py:2095`):

```python
f = self.enc.forward_features(im, slot) if self.cond == 'token' else self.enc.forward_features(im)
```

and at load time (`:2118`): `if cfg['cond'] == 'token': enc = ViTSlotToken(enc, N_SLOT_TYPES)`.

The token injected into the trunk is the **series slot type** — the same six-slot vocabulary
(`SAG_FLUID_FS`, `COR_T1`, …) this project already carries through its own pipeline. `Net.__init__` never
takes an `n_sites` argument; on the `token` path nothing consumes it. Kimi's
`research/2026-09-04_qdu-foreign-architecture-census.md:37` already called it "slot-token conditioning",
correctly, two days ago — I read `n_sites: 109` off the cfg and invented a site axis around it.

**What this costs and what it buys.** It costs the "site conditioning changes the objective" argument
entirely; there is no per-site calling threshold in their model. It buys a much cheaper port: a learned
embedding of a slot type we ALREADY have, injected as a trunk token, with no new metadata anywhere in
the pipeline and nothing to derive at test time.

**The one measurement worth keeping from the wrong hypothesis.** `canonical_metadata.csv` over 4,407
studies, grouped on `scanner + field_strength_t + pixel_spacing_mm + slice_spacing_mm + sequence`, yields
**122 acquisition signatures** (median group 4; the five largest 519/351/328/307/272; 93.4% of studies in
a group of ≥20; **45 singletons**). That is a real axis and it is header-derived, so it exists at test
time by construction — but it is not their 109, and no evidence now says a model should be conditioned on
it. It is recorded here so the next person who wonders does not re-measure it, not as a lever.
`vendor_ruler.py`'s standing reading is unchanged: `InstitutionName`, `StationName` and
`DeviceSerialNumber` are absent from the corpus, and the competition's own tables carry no site column.

## 3. THEIR SPLIT IS PUBLIC. It is an md5, which is why grepping for `KFold` found nothing.

At `:1510`:

```python
rep = train_df.set_index('StudyInstanceUID')['Report'].fillna('')
grp = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % 5 for s in st_tr])
va = np.array([i for i in keep if grp[i] == 0])
tr = np.array([i for i in keep if grp[i] != 0])
```

This morning I searched both their kernels for `KFold`, `GroupKFold`, `StratifiedGroupKFold`,
`train_test_split` and `random_state`, found zero of each, and closed the admissibility question as a
clean negative: *"their split is not public, `m_f*.pt` stay board-or-nothing."* **That conclusion was
wrong, and the method that produced it was wrong** — I searched for the vocabulary a split is usually
written in instead of reading the code that makes one. Their split is deterministic, keyed on report
text, reproducible by anyone with `train.csv`, and it is grouped by report so duplicate reports cannot
straddle it.

**Recomputed against our 909-study holdout, this instant:**

| | studies | share |
|---|---|---|
| in their **holdout** (`grp == 0`) — out-of-sample for their fold-0 | **209** | 23.0% |
| in their **train** (`grp != 0`) — **in-sample** for their fold-0 | **700** | 77.0% |

Group sizes over our holdout: 209 / 161 / 173 / 188 / 178.

**Any number their fold-0 checkpoint reads on our 909 is 77% in-sample and must not be quoted.** That is
the same defect `comp/board.jsonl` voided `raptor_solo` for on 2026-08-29 (98.6% in-sample), and it would
have been repeated here.

## What this unlocks, and it is the largest item

Because the split is computable, **their five folds become honestly priceable**: fold *f*'s checkpoint is
out-of-sample exactly on `grp == f`, so each of our 909 studies has a member that never saw it, and a
fold-matched reading covers the whole holdout with no contamination anywhere. The foreign family moves
from "board-or-nothing" to a measurable, admissible candidate — which was the question item 2 was
supposed to answer, answered backwards this morning.

## Status of the arm this reverses

`foreign_recipe_v3small_336_20sl` is on the card at epoch 4, and its stated motivation is void. It is
being allowed to finish anyway, for a reason that does not depend on the retracted number: **v3small at
336 px, 20 slices, whole-stack has never been priced here** — every DINOv3 number in the ledger is 224 px
— and `trajectory_check` puts its generous ceiling at 0.8435 against our best DINOv3 at 0.8268. Thirty
minutes of card for an unpriced geometry axis is worth spending; truncating it would buy a floor instead
of a number. What it is NOT is a replication of anything, and its header's pre-registered branches all
rest on the retracted 0.8874 and are void with it.
