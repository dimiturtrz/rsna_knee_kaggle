# Tail overlap closes the seat lane, and doubles as a leak detector

*2026-09-17, morning. CPU arithmetic over 391 `.holdout.npz` files already on disk. No GPU, no
submission. `python -m kneemri.evaluation.tail_screen --run labelwise_depth24_drop40_long_s2026`
joined against `python -m kneemri.evaluation.ledger75 --holdouts D:/models/rsna_knee_abnormality`.*

## The question, and why the three earlier closures did not answer it

knee-jy5yu, knee-lyzc1 and knee-5hbtz each closed the ensemble lane, and each priced candidates on
**solo score**. knee-i97cx then measured why that cannot work: the champion member's worst tenth is
SHARED — tail overlap 0.631 against a chance 0.100, 8 private studies of 90 — so the shelf fails the
same studies. A seat only pays if it fails DIFFERENT ones, which makes decorrelation the axis and solo
score a proxy blind to it. `StudyAttribution.shared` reports one aggregate over a sample of twelve, and
an aggregate cannot surface a candidate: a shelf whose mean is 0.631 may still hold one member at 0.25.

So the screen was run over the whole shelf, one row per member, ranked ascending.

## What it read

391 members share the reference's exact 909-study order; 51 do not and were skipped. Shelf mean 0.593,
floor 0.022, ceiling 1.000.

The ceiling is the arithmetic check that the screen works: `solobase_hold` reads 1.000, and its
`ledger75` is 0.9053 at epoch 18 over 909 rows — the reference member's own numbers, exactly. It is the
same member under a second name.

**The floor is three distinct classes, and none of them is a seat.**

```
0.022  0.4593  focus_latmen_s2026         WEAK  — below chance AUC
0.156  0.5259  focus_mcl_s2026            WEAK
0.167  0.6766  radnet_family_s2026        WEAK
0.178  0.6766  mae_matched_s2026          WEAK
0.256  0.7017  encoder_adapt_s2026        WEAK
0.267  0.6907  radiodino_corpus_s2026_run1 WEAK
0.278  0.7704  radnet_pinned_s2026        WEAK
--- the strong-and-decorrelated band, every row of it VOID ---
0.300  0.8804  foreign_dinov3_band0892    703/909 in-sample
0.300  0.8811  foreign_dinov3_band1684    703/909 in-sample
0.300  0.8905  foreign_dinov3_band1387    703/909 in-sample
0.300  0.9021  dinov3f0_solo              703/909 in-sample
0.311  0.8879  foreign_dinov3_band1090    703/909 in-sample
0.311  0.8964  foreign_dinov3_matched     703/909 in-sample
0.322  0.8698  foreign_dinov3_band208     703/909 in-sample
0.322  0.8882  foreign_dinov3_band1486    703/909 in-sample
0.322  0.8917  foreign_dinov3_tta3        703/909 in-sample
0.322  0.8941  foreign_dinov3_band1189    703/909 in-sample
0.333  0.8894  foreign_dinov3_tta4        703/909 in-sample
--- the honest strong floor starts here ---
0.433  0.9004  recal_holdannot_s2026
0.456  0.8479  swin_calib_dense_s2026
0.489  0.8509  swin_wholetop_s2026
0.511  0.8507  swin_distill_fit_s2026
0.522  0.8526  ortho_fresh_lowlr_s2026
```

Below 0.433, **every** member is either too weak to seat (`ledger75` 0.46 to 0.77, against the
reference's 0.9053) or void on an in-sample reading. The five `raptor_*` members that sit at 0.478 to
0.511 with `ledger75` 0.9005 to 0.9425 are void on the same count — `comp/members.declines:812` prices
that set at 98.6% in-sample.

## The closure

**The honest shelf's floor is 0.433 against a mean of 0.593 and a chance of 0.100.** That is not a
member that fails different studies; it is a member near the middle of a tight distribution. So the
ensemble lane closes on MECHANISM rather than on the solo-score arithmetic the three earlier beads
used, which is the stronger closure and is what this reading was spent to get.

And `recal_holdannot_s2026`, the honest floor, is already declined in writing as a seat
(`members.declines:84`: 0.012 under `calib_dense_hold` at the 224 corner, kept as an instrument). The
lowest-overlap honest member on the entire shelf is a member the roster had already refused for an
unrelated reason.

**What re-opens it.** Not another screen of this shelf — it has now been read on the axis that matters.
A member that does not yet exist, whose recipe changes what a study looks like rather than how it is
scored, and which is screened here BEFORE it is priced on a board slot. Re-opening this by re-quoting a
solo score is answering a different question again.

## The second reading, and it is an instrument

**Tail overlap separates the in-sample members from the honest ones with no errors.** All eleven
strong-and-void members occupy 0.300 to 0.333, a band no honest member enters; the honest strong floor
begins at 0.433, a clear gap of 0.100. The mechanism is not subtle — 703 of the 909 studies are those
members' TRAINING data, so they do not fail them, and their tail is drawn from the 206 that are left.
Low overlap at a high score IS the leak signature.

That is worth keeping because it is cheap where the existing check is not. `foreign_overlap` needs the
md5 split and the member's own fold assignment; this needs two `.holdout.npz` files and no provenance
at all, so it prices a member whose training set nobody can enumerate — which is the exact situation
every foreign member arrives in.

**Stated as a screen and not a verdict.** A member can be decorrelated for an honest reason, and this
reading cannot tell the two apart on its own; what it can do is say which members are worth asking
`foreign_overlap` about first. On this shelf the correlation is perfect, on a denominator of 11 void
members and 310 honest ones above the merit level.

## The denominator

391 aligned members of 442 with a holdout record; 51 skipped for a different study order, which is the
alignment rule `AttributionCorpus.members` already enforces and not a choice made here. Tail = the
worst 90 of 909 studies, each member cut at its own tenth. `ledger75` is the 0.75 cut, not the
midpoint — a number quoted from this page carries that cut.

## 2026-09-17, 11:00Z: the other half of the closure — every CAPACITY axis is measured too

The section above says a re-opening needs "a member that does not yet exist, whose recipe changes what a
study looks like rather than how it is scored". This is the census of whether such a recipe is available,
because if the input axes are also spent then the shelf is saturated on both sides and that is a
different kind of fact than a seating closure.

**The shelf is ONE recipe.** Read off `torch.load(..., weights_only=True)["options"]` for the ten
highest-`ledger75` honest members, every one is identical on `image_size 336`, `train_size 0`,
`compile_encoder True`, `backbone_lr 4e-05`, `pool cls_mean`, `slot_pooling attend`, `encoder_layer -1`,
`encoder_weights None`, `encoder_adapt_depth 0`. They differ on exactly two knobs — `slice_pooling`
(labelwise, except `oof_teacher_s2026` at max) and `stochastic_depth` (0.0, except
`labelwise_drop_drop40_long_s2026` and `fieldwide_s2026` at 0.05) — and they span 0.9007 to 0.9053
`ledger75` on the same 909 rows. A 0.005 spread across twenty members is what a saturated recipe looks
like.

**And every axis that would leave that recipe has already been run.**

```
axis                     member                 ledger75   delta vs its OWN control
ViT block 9 of 12        encoder_layer_9_s2026    0.8459    -0.0227  (3.7x floor; seat value -0.0034)
DINOv2-base vs small     base_dense_s2026         0.8812    -0.0057  (inside the 0.0062 floor)
392 px on a 448 cache    res392_ws_s2026          0.8817    +0.0066  (but see below)
80 mm joint crop         crop80_tissue_s2026      0.8597    negative
36 slices                stack224 / cover_labelwise  --     -0.0033 / -0.0014
reference                labelwise_depth24_drop40_long_s2026  0.9053
```

**The one positive is already spent.** `res392_ws` read +0.0066 over its control, and
`experiments/README.md:310-312` separates the cause: `rate_wholestack` moved the encoder rate 8e-6 to
4e-5 at an unchanged 336 px and read 0.8525 against res392's 0.8523. The gain was the RATE, not the
pixels — and the stamp above shows every current member already runs `backbone_lr 4e-05`. There is no
resolution lever left to pull; it was pulled and it is in the champion.

**The caveat that keeps this honest, and it is the thing that would re-open the lane.** Those arms peaked
at epochs 6 to 10 against controls from the 12-epoch era; the reference peaks at epoch 18 under
labelwise pooling, drop40 and a long schedule. So the deltas are priced on a recipe the champion no
longer is, and the standing rule is that **only the SIGN transfers across a recipe change**, never the
size. Four of the five signs are negative, which is a closure. The fifth, DINOv2-base, is a TIE inside
the floor — and a tie is the one outcome a longer budget could plausibly move, since a bigger trunk is
exactly the thing a 12-epoch budget under-trains.

So the single re-opening arm this census licenses is **DINOv2-base under the CURRENT recipe** — 336 px,
labelwise, drop40, 24 epochs — and it must be priced as an arm whose own pre-registration predicts a
tie, not sold as a lever. It is not armed here: a base trunk is roughly 3-4x small per epoch against a
reference arm that already costs hours, which is a poor ratio against a mechanism whose best prior
evidence is a null.

## 2026-09-17, 11:05Z: the closure's stated MECHANISM was wrong — it is the split, not the recipe

The sections above read shared tail failure as a fact about the recipe. It is a fact about the SPLIT, and
the difference matters because only one of the two is re-openable.

**345 of 364 members carrying both a checkpoint and a holdout record are fold 0 of `split_seed 2026`.**
Read off `options` across every `*.holdout.npz` on disk: fold 0 -> 345, fold 1 -> 5, fold 2 -> 5,
fold 3 -> 5, fold 4 -> 4, plus 79 whose checkpoint would not load. So the shelf the screen ranked is
almost entirely ONE training set. Members fitted on the same 3497 studies failing the same 90 is not
evidence that the recipe cannot decorrelate; it is what that construction guarantees.

**And the screen cannot see the class that could.** Alignment is its denominator rule —
`AttributionCorpus.members` keeps only members whose study order equals the reference's — and a fold-1
member holds out a DIFFERENT 909 by construction. The 51 members reported as skipped are therefore not
noise around the edge of the reading; they are the only members whose tails could have differed, removed
before the ranking began.

**The conclusion survives anyway, for a reason that is not the one given.** All 19 off-fold members are
the obsolete OOF recipe — `oof_{band,clean,gen2,reader,}_fold{1..4}_s2026`, every one `variant=small`,
12 slices, 8 epochs, against the reference's 24 slices and 18. No STRONG member exists off fold 0, so
there was in fact nothing for the screen to find. "The ensemble lane is closed" stands; "no honest member
fails different studies" does not, and may not be quoted.

**What this makes priceable, and the obstacle that stops it being cheap.** The only untested decorrelation
source is the current recipe fitted on another fold. It cannot be read on this split: fold-0 and fold-1
holdouts are DISJOINT, so no study is held out by both and no honest offline number for such a pair
exists. An honest reading needs OOF-complete predictions over the 4406-study corpus — each study scored
by the fold that did not train on it — which is what the `oof_*_fold*` family was built for and why it
sits at 12 slices and 8 epochs. So the arm is four arms plus a corpus reading, not one, and the
≤0.002 ensemble premium this project quotes against it was itself measured entirely WITHIN fold 0, under
the same confound. That is the honest case for pricing it and the honest reason it is not armed here.

**Also verified while reading the stamps, replacing an assertion made from bead prose.** Every one of the
champion's seven seats stamps `variant=small` and `crop_mm 130.0`, at 12 to 24 slices, fold 0 or -1.
The trunk claim in the census above is now read from the checkpoints.
