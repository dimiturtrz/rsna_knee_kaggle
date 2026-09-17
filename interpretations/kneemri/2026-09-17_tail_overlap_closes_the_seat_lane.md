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
