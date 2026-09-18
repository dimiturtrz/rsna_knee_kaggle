# Six knobs read at both image scales: every sign agrees, no magnitude does — 2026-09-18T09:40Z

*No card. Two `ledger75 --only` calls over 29 members already on disk, plus the arm headers that name each
control. This is the de-risking step `2026-09-18_the_wide_ruler_was_never_pointed_at_a_recipe_knob.md`
made a precondition for spending ~8 h on the five-fold rotation (knee-ozwxx).*

## The question

The wide five-fold ruler is affordable only at **224 px / 8 epochs** (2.1 h) and unaffordable at the
shipping corner (17.5 h). So it can only ever deliver a knob's reading *at the instrument scale*. That is
worth something if and only if the SIGN survives the change of scale. The evidence for it was **n=1**
(distillation, +0.0121 at 224/8 against +0.0055/+0.0056 at the shipping corner). n=1 does not license 8 h.

## The pairs

Every delta below is **ledger75**, population `eb3ba199`, 909 studies, fold 0 — one cut for all of them, so
no row is a prose number of unknown ruler. Each pair is one knob against the control its own arm header
names, at one scale, with the other side of the pair the same knob at the other scale.

| knob | 224 px | 336 px | sign | clears 0.0062? |
|---|---|---|---|---|
| distillation (`--distill-weight 0.7`) | **+0.0121** (224/8, five folds) | **+0.0056** | agree **+** | both |
| LLRD, 12 blocks at a decayed anchor | llrd224 0.8737 − native224 0.8800 = **−0.0063** | band_dense_llrd 0.8765 − band_dense 0.8877 = **−0.0112** | agree **−** | both |
| capacity ladder (`--unfreeze-last 12`) | band_deep224 0.8718 − band224 0.8801 = **−0.0083** | band_deep_matched 0.8781 − band_dense 0.8877 = **−0.0096** | agree **−** | both |
| `--slot-dropout 0.40` + 24 epochs | blendhold_drop40_long 0.8879 − blendhold 0.8829 = **+0.0050** | band_dense_drop40_long 0.8918 − band_dense 0.8877 = **+0.0041** | agree **+** | neither |
| 24 epochs alone | native224_long 0.8760 − native224 0.8800 = **−0.0040** | band_dense_long 0.8867 − band_dense 0.8877 = **−0.0010** | agree **−** | neither |
| band 0.20-0.80 -> 0.10-0.90 | band224 0.8801 − native224 0.8800 = **+0.0001** | band_span 0.8851 − lr4e-5_ws 0.8840 = **+0.0011** | agree **+** | neither |

**Six pairs, six sign agreements, zero flips.**

## What that does and does not license

**Three of the six clear the floor on both sides.** Those are real agreements. The other three agree while
both readings sit under 0.0062, where a sign is close to a coin flip; three coins landing the same way is
`p = 0.125`, which supports the rule and does not carry it alone.

So the honest statement is **n=3 strong agreements and 3 weak ones, against 0 flips at any strength.**
knee-ozwxx pre-registered "signs agree on n>=4 with no flip" as the branch that licenses the rotation. On
strong pairs alone that is n=3; counting all six it is n=6. I am taking the branch as cleared, and writing
down that the count depends on how the weak pairs are treated so the next reader can disagree with the
call rather than with a number.

**The magnitude does not transfer, and not even in a consistent direction:**

* distillation — 224 **overstates** by 2.2x (+0.0121 vs +0.0056)
* LLRD — 224 **understates** by 1.8x (−0.0063 vs −0.0112)
* the ladder and drop40+long — near-equal at both scales

The earlier page guessed at "magnitude compressed about 2x" from the distillation row alone. That is wrong
as a rule: the compression factor has both directions in it. **A 224 delta may be read for its sign and may
never be scaled into a shipping-corner prediction**, which is `CLAUDE.md`'s existing rule arrived at from a
second direction.

## The part that is actually the point

The three weak pairs are weak *because they were read at fold 0, where the floor is 0.0062*. On the
five-fold wide ruler the floor is **0.0028**, and those same three deltas become 1.8, 1.4 and 0.4 floors
instead of 0.8, 0.6 and 0.02. Two of the three cross from unreadable to readable **without a single extra
epoch of training** — they were measured at the wrong denominator, not at the wrong size.

That is the whole case for the rotation restated as an arithmetic fact about deltas already on disk: the
project owns several sub-floor readings whose sign it cannot trust, and the instrument that would make
them readable costs 2.1 h each.

## What is still not settled

Five of the six pairs vary **image size only** (224 vs 336, fold 0, same epochs, same ruler). The actual
instrument corner varies image size **and** epochs (8 vs 24) **and** the ruler (4,407 vs 909). Only the
distillation row tests that full change, and it is one row. So what is established here is that **image
scale does not flip a sign**; that **8 vs 24 epochs** does not is supported by one row plus the two
epoch-budget pairs in the table, which agree but are both sub-floor.

That residue is now mostly closed, by the lookup rather than a run. `oof_fold0`'s recipe is byte-identical
to `native224` but for `--epochs 8` against `12`, and it holds out the same fold, so with `native224_long`
the three make an epoch ladder on one ruler:

| epochs | ledger75 |
|---|---|
| 8 (`oof_fold0_s2026`, the instrument corner) | 0.8775 |
| 12 (`native224_s2026`) | 0.8800 |
| 24 (`native224_long_s2026`) | 0.8760 |

**Total spread 0.0040 across the whole 8-24 range — inside the 0.0062 floor.** So the epoch budget moves
the baseline by less than the ruler can see, and a knob read at 8 epochs is read against essentially the
same baseline it will be deployed against at 24.

**The one place this can still lie is named precisely by the interaction below.** That flatness holds in the
ABSENCE of slot dropout, which is the regime the instrument recipe sits in. With slot dropout present the
epoch axis is NOT flat — drop40 and 24 epochs interact by +0.0113. So a knob that interacts with
regularisation will be measured at 8 epochs in the flat regime and deployed at 24 epochs with drop40 in the
interacting regime. **The oracle's known failure mode is therefore exactly one class: regularisers.** For
those, a rotation sign is a hypothesis, not a verdict.

## An interaction, found on the way and worth its own line

`--slot-dropout 0.40` and 24 epochs are **superadditive at both scales**, and each is negative or flat
alone:

* 224: drop40 alone **−0.0023**, long alone **−0.0040**, together **+0.0050** — a **+0.0113** interaction.
* 336: long alone **−0.0010**, together **+0.0041**.

The shape reproduces across the scale change, so it is not an artefact of either corner. It is also a
standing warning about the leaf sweep the rotation is meant to serve: `knee-dry3h` closed twenty-two leaves
**one at a time**, and a knob that reads −0.0040 alone and +0.0050 in company cannot be closed by a
single-knob read. Some of those twenty-two may be closed wrongly for this reason rather than for the floor.

## Evidence

* `kneemri/evaluation/ledger75 --holdouts D:/models/rsna_knee_abnormality --only <29 stems>` — run
  2026-09-18T09:36Z, two calls, every number above. Population `eb3ba199`, 909 studies, fold 0.
* Arm headers naming each control: `llrd224.sh:91`, `band_dense_llrd.sh:51`, `band_deep224.sh:65-66`,
  `band_deep_matched.sh:69-70`, `blendhold_drop40_long.sh:34`, `band_dense_long.sh:20`, `band224.sh:44`,
  `band_span.sh:26`, `native224_long.sh:58`, `blendhold.sh:47`.
* `interpretations/kneemri/2026-09-17_no_teacher_is_above_the_student.md:14-26` — the +0.0056 shipping-corner
  distillation term; `experiments/README.md:13954-13972` — the +0.0121 five-fold 224/8 term.
* `interpretations/kneemri/2026-09-18_the_wide_ruler_was_never_pointed_at_a_recipe_knob.md` — the floor
  measurement (0.0062 -> 0.0028) and the program this gates.
* Denominators: every delta ledger75 at fold 0 / split_seed 2026 / 909 studies, paired floor 0.0062. The
  0.0028 is the same macro floor at the five-fold union of 4,407, by the sqrt scaling measured in the page
  above (2.24 against 2.24).
