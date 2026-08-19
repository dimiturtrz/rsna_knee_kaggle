# Submissions

Kernels are named for the METHOD, not for the attempt: `kneemri-<source>-<variation>`. A new version of
one slug is that same method retrained or re-tuned; a different source or aggregation is a different
slug. The slug is the claim, so a grafted score can never quietly become ours.

Budget: **5 submissions per day**. Kernel *pushes* are unlimited — probe there, submit only when the
question needs the leaderboard to answer it.

## The method tuple

A slug alone cannot say what was run: `kneemri-trained` has meant five windowed members with overlap
TTA and three whole-stack members without it. Every entry is therefore a tuple over four closed axes,
and the tuple — not the slug — is what the coverage grid below is indexed by.

| axis | values | what it decides |
|---|---|---|
| **source** | `ours` · `theirs` · `fusion` | whose weights produced the prediction; `fusion` means both, decoded once |
| **members** | `1` · `2` · `3` · `5` · `20` · `23` | how many members are rank-averaged |
| **read** | `windowed` · `whole` | one group of slices per pass, or the whole stack |
| **passes** | `1/study` · `n/study` · `overlap` | test-time cost, and whether windows overlap |
| **views** | `identity` · `rot3` | rigid test-time views averaged over, at that many times the passes |
| **vote** | `rank` · `logit` · `quality` | how the members' tables become one |

`ours×5×windowed×overlap` is a complete method. Any two entries differing in one coordinate are the
only pair that isolates that coordinate, which is what makes a spend decision arguable rather than a
preference.

**The last two axes were added on 2026-08-17 and every entry above them is `identity`/`rank`.** They
are the two things an entry can vary WITHOUT retraining anything, and until that date they were
reachable from `member_curve` and from nothing that writes a submission — so the tuple had four axes
because two of them could not be spent, not because they did not exist. Offline both are flat (vote:
0.8581 rank against 0.8583 logit on the pair behind the 0.919; views: +0.0007 on the same pair), which
is exactly why they are worth a slot and not a retrain: a flat offline axis is a pure transfer question.

**`passes` is a no-op for a whole-stack member and is not a bet.** `Member.window` gives a whole-stack
member ONE centred window of its full depth, so `overlap=True` and `overlap=False` produce the same
window and a byte-identical submission. A `kneemri-overlap` kernel was written on 2026-08-17 to ask the
board about it and deleted before it ran, because the answer is not "0.0000 offline, unknown on the
board" — it is that there is nothing to submit that differs.

## One member of the pool is unverifiable, and it is the strongest one

`witness_check --device cpu`, 2026-08-17: **6 ok, 1 unverified, 0 refused**. The unverified one is
`res392_ws_s2026` — our best single member at 0.8523, fitted on 2026-08-12 before the witness existed,
so there is nothing to re-derive. It is also the pool's ONLY resampling member (392 px off a 448 cache),
which is the exact class the witness was built to catch. The fingerprint still pins its weights, image
size and group, and both rebuilders read its recorded config rather than defaulting, so "rebuilt at the
wrong resolution" is closed by construction; what is open is anything the fingerprint cannot see.

**Say it in the submission note rather than discovering it afterwards**: any entry naming `res392` ships
five verifiable members and one that is trusted. The fix is one training slot, and it pays twice —
re-running res392 at the control's rate (knee-a8x) also disambiguates the +0.0066 this project currently
credits to resolution but which changed the encoder rate 5× at the same time.

## Coverage grid

Five slots expire unspent at 00:00 UTC daily; they cannot be banked. This grid exists so a slot is
never spent on a cell that already has a number.

| tuple | status | LB |
|---|---|---|
| `ours×1×whole×1/study` | scored | 0.868 |
| `ours×2×whole×2/study` | scored | 0.882 |
| `ours×3×whole×3/study` | scored | 0.882 |
| `ours×5×windowed×overlap` | scored | **0.888** |
| `theirs×20×windowed×overlap` | scored | **0.888** |
| `fusion×23×windowed×overlap` | scored | 0.894 |
| `fusion×24×windowed×overlap` | scored | 0.898 |
| `ours×2×whole×2/study` (post-LoRA pool) | scored | **0.919** — best |
| `ours×5×whole×5/study` (post-LoRA pool) | scored | **0.911** |
| `fusion×23×whole×n/study` | **null** — T4 OOM, sample table scored | 0.500 |
| `ours×2×whole×2/study` (res392 + calibrated) | **null** — stale kit, no `Member.precision` | 0.500 |
| `ours×5×whole×5/study` | submitted 2026-08-13, result not recorded here | — |
| `ours×2×whole×2/study` re-run on kit `5ba86ee` | submitted 2026-08-13, result not recorded here | — |
| `fusion×25×windowed×overlap` (our 5 + their 20) | 2026-08-17 | **0.894** |
| `ours×5×whole×5/study`, five DIFFERENT recipes | **untried** | — |
| `ours×1×whole` (post-LoRA pool) — `kneemri-solo` | 2026-08-17 | **0.914** |
| `ours×3×whole` (post-LoRA pool) — `kneemri-trio` | 2026-08-17 | **0.913** |
| `ours×2×whole×vote=logit` — `kneemri-vote` | 2026-08-17 | **0.919** — identical to the RankAverage pair |
| `ours×2×whole×views=rot3` — `kneemri-views` | 2026-08-17 | **no score** — COMPLETE, blank public score |
| `ours×4×whole`, rate member for sliceattn — `kneemri-rate` | 2026-08-18 | **0.924** — best, and the only clean member A/B we have |
| `ours×2×whole`, rate member for res392 — `kneemri-ratepair` | 2026-08-18 | **0.921** — the pair's own member swap |
| `ours×4×whole`, lr1e-4 for intensity392 — `kneemri-solotop` | 2026-08-18 | **0.925** — best; pre-registered 0.927, so the swap rule missed |
| `ours×2×whole×views=rot2` — `kneemri-rateviews` | 2026-08-18 | **no score** — the second blank, and the one that named the cause |
| `ours×4×whole×views=rot2` — `kneemri-topviews` | 2026-08-18 | **0.925** — the SAME number as `solotop`, whose four members are identical and whose views are not. Two rigid views bought nothing here; the board prices decode DIVERSITY, not decode COUNT |
| `ours×4×whole`, band_dense for res392 — `kneemri-band` | 2026-08-19 | **0.928** — our best, and the first entry past the 0.925 plateau. Pre-registered 0.929 |

**`kneemri-solo`'s submission description is wrong and Kaggle will not let it be edited.** It says v3
"silently ran overlap=True and would have varied decode alongside count". It did not: `Member.window`
gives a whole-stack member one centred window and never reads the flag, so v3 and v4 predict
identically and 0.914 is a clean count reading either way. The confound was read off a log line that
printed the REQUEST rather than the decode — since fixed, and the only real defect in that episode.

## The member-count curve is complete, and it is not a curve

| members | 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| board | 0.914 | **0.919** | 0.913 | 0.918 | 0.911 |

Four slots of 2026-08-17 bought this row, on ONE pool, ONE kit (`6f93554`) and `overlap=False` throughout,
so count is the only thing that varies across it. **It is not monotone in either direction**: 1 ≈ 3 sit
below 2 ≈ 4, and the whole span is 0.008 — about 1.3 of the 0.0062 measured noise floor. A count axis
would have to be ordered to be an axis at all, and this one is not.

**So member count is not the lever; membership is.** The two top cells (2 → 0.919, 4 → 0.918) are the
subsets built from the strongest members, and the 3 that scores 0.913 is not a smaller version of them —
it is a different set. That is the same conclusion the joint-holdout subset search reached offline
(knee-2bly: every subset larger than five falls monotonically to 0.8547), now confirmed on the board,
which is the ruler that matters. It also closes the question the four slots were spent on: adding or
removing a member is worth nothing predictable, and the entry that ships should be a NAMED subset.

## Membership is the lever, and 2026-08-18 measured how much (0.924)

| entry | members | LB |
|---|---|---|
| `kneemri-chosen` | res392 + intensity392 + calibrated + **sliceattn** | 0.918 |
| `kneemri-rate` | res392 + intensity392 + calibrated + **lr4e-5** | **0.924** |

Same count, same kit's 23 loaded files, same `overlap=False`, same `RankAverage`, one member swapped.
**+0.006 on the board** — above the 0.0062 noise floor's own width, and it is the FIRST clean single-member
A/B this competition has given us: every earlier board delta varied count or pool or both.

It also transfers the offline number almost exactly. `lr4e-5_ws_s2026` is the control recipe at encoder
rate 4e-5 instead of 8e-6 and scored +0.0068 solo on the fold-0 holdout (knee-wnkv); the board moved
+0.006 for putting it in. Two things follow. The rate lever is real on the ruler that pays, not an
artefact of one fold — and a member's SOLO holdout gain predicted its ensemble board gain here, which
the joint-subset search has never managed (it ranked this set +0.0013 over `chosen`, five times too
small). Membership, priced by solo strength, is the axis to keep pushing.

### A second board point, and it is the same rule at a tenth the size

| entry | members | LB |
|---|---|---|
| `kneemri-efficient` v4 | **res392** + calibrated | 0.919 |
| `kneemri-ratepair` | **lr4e-5** + calibrated | **0.921** |

The same swap at count two. `lr4e-5` beats `res392` by +0.0002 solo — a twentieth of what it beats
`sliceattn` by — and the board moved +0.002, versus +0.006 for the larger swap. Two points, both of them
the board delta landing within a thousandth of the member's own solo holdout gain:

| swap | solo delta | board delta |
|---|---|---|
| sliceattn → lr4e-5, in the four | +0.0052 | +0.006 |
| res392 → lr4e-5, in the pair | +0.0002 | +0.002 |

That is a usable selection rule and the first one this file has had: **price a candidate member by its
solo holdout gain over the member it displaces**, not by the joint-subset search, which underpriced the
first of these by 5×. Two points do not fix a slope — what they establish is that the sign and the order
of magnitude carry, which is all a slot decision needs.

### The third point was pre-registered and it broke the ordering

`kneemri-solotop` — the four with `lr1e-4` displacing `intensity392`, a +0.0031 solo gap — was written
with **0.927** on it before it ran, which is +0.003 over 0.924 by the rule above. It scored **0.925**.

| swap | solo delta | board delta |
|---|---|---|
| sliceattn → lr4e-5, in the four | +0.0052 | +0.006 |
| res392 → lr4e-5, in the pair | +0.0002 | +0.002 |
| intensity392 → lr1e-4, in the four | +0.0031 | **+0.001** |

The sign held for a third time; the ORDERING did not. The largest of the three solo gaps after the first
bought the SMALLEST board move — less than the swap worth a fiftieth of it. So the rule survives only in
the weak form it was stated in (a better solo member does not hurt the board) and fails in the form that
would actually rank two candidates against each other. Every one of these three deltas is also inside the
0.0053 noise floor on its own; the first is the only one that clears it, and a rule fitted to one point
above the floor and two below it is a rule fitted to noise. **Do not price a slot with it.**

What that leaves is a bar, not a mechanism: 0.925 is the best entry to date, and the three swaps that
produced 0.918 → 0.924 → 0.925 have between them moved the board +0.007 for four member substitutions —
a rate that says membership-shuffling inside the existing pool is close to exhausted. The next real move
has to come from a member that is DIFFERENT rather than better (knee-nas5's crop/band/centre axis, which
is the one that transfers at 2.4×), or from supervision (knee-mxj, knee-jk3).

That 0.925 entry is now pinned in `comp/champion.json` — its kernel, kit commit, four members, serving
config, and the six principles it is evidence for — and `python -m kneemri.kaggle.champion_check` asserts
it can still be rebuilt. The hidden set is the one number here that cannot be talked into existing, so the
path that produced it is guarded rather than remembered: a roster line dropped while adding a member, or a
`chosen` tuple edited in place instead of copied, leaves every other gate green and the entry unbuildable.

It also kills the count curve's peak at two. The pair built from the better member scores 0.921 and the
four holding that same member scores 0.924, so 2 > 4 was a statement about `sliceattn` being in the four,
not about count. And `res392` is not dead weight: removing it from the four is what the 0.921 pair does,
and it costs 0.003. The 448 px cache stays.

## The graft is a cost on the board too, and it is the day's only result above the bar

| all-ours ×2 | all-ours ×4 | fusion ×24 | fusion ×25 |
|---|---|---|---|
| **0.919** | 0.918 | 0.898 | **0.894** |

The fifth slot fired `fusion×25` on a NEGATIVE offline prior, to find out whether the board disagrees
with the holdout about grafted members. It does not: **−0.025 against the all-ours pair**, four times the
0.0062 noise floor and the only 2026-08-17 delta that clears the 1.5% bar in either direction. Adding
their twenty CC0 members to our five costs more than every method change this project has landed.

That settles the largest uncertainty this file named. The 2.4× transfer compression measured on
`onedecode` was the reason to doubt the offline verdict on grafted members — a holdout that under-prices
diversity might have been under-pricing them too. It was not: knee-k5kk's offline 0.013–0.021 penalty
lands on the board at 0.025, the same sign and a larger size. The holdout and the board agree, so the
CC0 pool is closed as a source of members and every remaining entry is all-ours.

**`views` returned no score at all.** The kernel ran clean on the visible three studies — valid 3×13 CSV,
2/2 members fingerprint-checked — and the submission reached COMPLETE with a blank public score. rot3 is
the only arm that triples decode, so a hidden-rerun timeout was the reading that fitted. **That reading was
wrong, and a second blank on 2026-08-18 (`rateviews`) is what refuted it** — the visible run costs 2.9 s a
study, so ~11,000 studies fit inside the 9 h ceiling and time was never the binding constraint.

The mechanism is memory, in the one place the retry cannot see it. `SpatialViews._warp` warped a whole
chunk in a single `grid_sample`, and it sits OUTSIDE `Ensemble.forwarded`'s row-halving
`torch.OutOfMemoryError` retry, which wraps the FORWARD only. A 48-study chunk is 6 slots × 12 slices =
3,456 images at 336 px: the float copy (1.56 GB), the `(n, h, w, 2)` sampling grid (3.12 GB) and the
result (1.56 GB) are live together — ~6.2 GB of transient on a 16 GB T4, on top of the model and the
decoded batch. The OOM kills the kernel, the hidden rerun writes no submission, and from outside that is
exactly a COMPLETE entry with a blank score. The visible set is 3 studies, 216 images, a fifteenth of the
memory — which is why every one of these runs looked clean in a log and cost a slot anyway.

Fixed by chunking the warp at `WARP_CHUNK = 512` images with one grid expanded across the batch, bounding
the transient at ~230 MB regardless of chunk size; `test_warp_is_indifferent_to_the_chunk_size` asserts the
result is bit-identical to the unchunked path, so the fix moved memory and not arithmetic.

**The fix held, and the TTA cell is now MEASURED rather than untested.** `kneemri-topviews`, the first
views entry from a kit carrying it, returned a real **0.925** — the identical number `solotop` scored on
the identical four members without views. Two rigid views buy **0.000**. So the blank never was a bad
score, as the caution insisted; it was no score, and the real one turns out to be flat. `rot3` — three
views rather than two — was never re-fired, and this reading does not cover it.

**The four queued cells are one day's five slots spent on questions the holdout cannot answer, not on a
predicted gain.** Nothing in the current member pool predicts a gain over 0.919 offline: count, subset,
per-label weighting and vote are all measured and flat or negative, and per-label weighting is dead
outright (oracle 0.8590 against flat 0.8588). Under this file's own spend policy that leaves only the
second clause — a question that cannot be answered offline at all. Two of the four complete the board's
member-count curve on ONE pool (1 and 3, against the 2/4/5 it already holds), and two open axes the
submission path could not reach until today. `kneemri-vote` doubles as the closest thing to a repeat
this competition allows — same members, same weights, same decode, same runtime, +0.0002 offline — so
whatever it scores against the 0.919 is the board's own noise, which every read of a 0.004 board
difference in this file has so far been made without.

A `null` is not a floor: neither of those two cells has ever been measured, and re-running one is a
legitimate spend. A cell with a number is not, unless the members underneath it changed.

**The fifth slot is decided by what the first four return, and the rule is written BEFORE they land** so
the pick is not the day's last result read as a trend. The board already holds the count curve at 2
(0.919), 4 (0.918) and 5 (0.911); `kneemri-solo` and `kneemri-trio` add its two missing points.

- **If `views` gains over 0.919** — TTA is the only axis that moved, and the fifth slot extends it
  (more views on the same pair). This is the one branch where a cell with a number is worth re-entering,
  because the members underneath it would not have changed but the READING rule would.
- **If `solo` scores at or near 0.919** — member count buys nothing anywhere on the curve from 1 to 4,
  and the whole ensembling axis is priced out for this pool. The fifth slot then belongs to the only
  untried cell that does not need a training run: `fusion×25×windowed×overlap`. Its offline prior is
  NEGATIVE (knee-k5kk: all-ours beats the 24-member fusion by 0.013–0.021), so it is spent to bound how
  far the board disagrees with the holdout on grafted members — the 2.4× transfer compression measured
  on `onedecode` says that disagreement is the largest uncertainty in this file.
- **If `solo` collapses and `trio` does not** — the curve has a knee below 2 rather than being flat, the
  ensembling axis is alive after all, and the fifth slot re-enters the best-scoring member set at its
  best count rather than opening a new axis.
- **If nothing separates** — every arm inside ±0.002 — then the board's own noise is wider than every
  offline delta this pool can produce, which is itself the day's result. The fifth slot is then NOT
  spent on another arm: it is spent re-entering the 0.919 pair unchanged, because a third identical
  entry is the only way to turn a two-point noise estimate into a spread.

Through 2026-08-15 every scored pair said the same thing — the best single model this project has
produced (holdout 0.8518) sat 0.020 BELOW five weaker ones, a third member of an identical recipe bought
nothing, and a different recipe bought +0.006 — so diversity was the axis with a number behind it.

**2026-08-16 bounds that reading rather than extending it.** Diversity earned its number while the
members were WEAK and while half of them were somebody else's. Against the post-LoRA all-ours pool, two
of our members score 0.919 and five score 0.911, while twenty-four members drawn from two packages score
0.898 — so above some member quality, adding members is a cost. The rule that survives both eras is
about the SPREAD, not the count: members worth adding are ones that fail differently and score nearly as
well, and the published CC0 members no longer clear the second half of that.

**2026-08-19 adds the other half of that rule, from `kneemri-band`'s 0.928.** Four different member sets
had all returned 0.925 and the reading forming around that plateau was that member quality does not
transfer — that only decode diversity does. It does. `band_dense` for `res392_ws` is one named swap on
the 0.925 four, holding count, decode groups and views fixed, so member quality is the only thing that
moves: +0.0055 solo and +0.0036 joint offline, +0.003 on the board. The plateau was not the board's
ceiling for this shape; it was four swaps that happened to trade roughly equal members. What made the
difference here is that `band_dense` comes from an axis the pool did not previously contain — the sampled
BAND rather than the learning rate or the read size — which is the same "fails differently" clause,
arriving as a better member rather than as an extra one. Pre-registered 0.929, landed 0.928.

## Ledger

The `local` and `LB` columns of the all-ours rows also live in `comp/board.jsonl`, in the form something
can read — with the one field this table never carried, the SCOPE each local number was measured on.
`python -m kneemri.evaluation.board_transfer` reads it and answers what the prose cannot: the holdout
has not mis-ordered a single pair it can resolve, and it converts at a median 4.7 board points per
holdout point. See `interpretations/kneemri/2026-08-16_board_transfer.md`. **A new scored entry gets a
line there as well as a row here.**

| date | kernel | v | method | local | LB | note |
|---|---|---|---|---|---|---|
| 2026-08-10 | `kneemri-probe` | 2 | passthrough `sample_submission` | — | not submitted | environment + corpus probe |
| 2026-08-10 | `kneemri-graft-verify` | 6 | fingerprint check only | 20/20 @ 4.65e-06 | not submitted | proves the architecture port |
| 2026-08-10 | `kneemri-graft` | 1 | **GRAFT** — 20 published CC0 members, our data layer, rank-averaged | their holdout 0.8398 | **0.888** | **their result, not ours** |
| 2026-08-10 | `kneemri-trained` | 1 | **OURS** — 5 folds fitted by `kneemri.training.train` on report-derived labels | holdout mean 0.8162 | **0.868** | trained with augmentation silently disabled |
| 2026-08-11 | `kneemri-trained` | 3 | **OURS v2** — augmentation actually applied + OneCycle + overlap TTA | holdout mean **0.8269** | **0.888** | **equals the 20-member graft with 5 of our own** |
| 2026-08-11 | `kneemri-trained` | 4 | **OURS** — 2 whole-stack members (fold 0 + fold 1) | 0.8481 | **0.882** | +0.021 per member on the holdout, −0.006 on the board |
| 2026-08-12 | `kneemri-trained` | 5 | **OURS** — 3 whole-stack members | 0.8493 | **0.882** | a third member of the same recipe bought NOTHING |
| 2026-08-12 | `kneemri-solo` | 1 | **OURS** — one model, no ensemble (our strongest member) | **0.8518** | **0.868** | the price of dropping ensembling: −0.020 |
| 2026-08-12 | `kneemri-fusion` | 1 | **FUSION** — our 3 whole-stack + their 20, one decode, rank-avg | mixed | **0.894** | best entry; 20 of 23 models are theirs |
| 2026-08-12 | `kneemri-fusion` | 2 | same 23, whole-stack members read WHOLE instead of windowed | mixed | **0.500** | T4 OOM → no submission written → the deleted `fallback` scored the sample table (knee-d16) |
| 2026-08-13 | `kneemri-efficient` | 1 | 2 strongest whole-stack members, 2 passes/study | 0.8509 | **0.500** | kit 54 commits stale, so the bf16 member ran fp16 and the image budget was missing |
| 2026-08-13 | `kneemri-trained` | 5* | **OURS** — strongest 5 whole-stack members, stamped kit | 0.8482 | *(no score returned)* | one of the three "Notebook Threw Exception" entries (knee-b6p) |
| 2026-08-13 | `kneemri-efficient` | 2 | the same 2 members on the fixed kit | 0.8509 | *(no score returned)* | second of the three exceptions |
| 2026-08-14 | `kneemri-fusion` | ~8 | 22 members, one decode, fp16 serving, a failing member drops itself | mixed | **0.895** | **best entry** — supersedes the 0.894 |
| 2026-08-15 | `kneemri-fusion` | 10 | 24 members (our 6 + the published 20 less the v3 and the nine-slot member) | mixed | **0.898** | superseded by the 0.919 efficient v4 — our own share doubled from 3 members to 6; +0.003 on the 22-member fusion, directional |
| 2026-08-16 | `kneemri-fusion` | 11 | 25 members — the expired-v3 exclusion fixed, so `lora_r8_v3small_224` serves | mixed | **0.898** | submitted 05:05 UTC. Restoring the twenty-fifth member changed NOTHING against the 24-member 0.898, while the two-member all-ours entry submitted the same hour scored 0.919. |
| 2026-08-16 | `kneemri-trained` | 8 | **OURS** — the strongest five all-ours whole-stack members, no published weights | 0.8610 | **0.911** | **THE ARM THAT SETTLES THE 0.919.** Ours-only, five members, +0.013 on the 24-member fusion — so the jump belongs to the MEMBER POOL, not to the two-member configuration: two different all-ours entries both clear the mixed fusion by more than the 1.5% bar. Published CC0 members are now a COST, not a contribution. It also puts the ruler and the board on opposite sides: local holdout ranks these five ABOVE the pair (0.8610 vs 0.8581, +0.003) and the board ranks them BELOW it (0.911 vs 0.919, −0.008). |
| 2026-08-16 | `kneemri-efficient` | 4 | **OURS** — `strongest(2)` whole-stack members, 2 passes/study, no overlap | — | **0.919** | **best entry, and the first that is entirely ours** — +0.021 on the 24-member fusion, so it clears the 1.5% bar as a RESULT, not a directional move. TWO of our own members beat twenty-four mixed ones at a twelfth of the forward passes, which inverts the diversity reading every earlier pair supported. Read it as a QUESTION before a finding: this kernel has never returned a valid score before (v1 scored 0.500 on a stale kit, v2 threw), so the 0.919 is the first measurement of the cell, and what changed underneath is the MEMBERS — `strongest(2)` now ranks over the post-LoRA pool, not the August-13 one. The comparison that settles it is the pending `kneemri-trained` v8 (all-ours, more members) against this. **SETTLED the same day: v8 scored 0.911.** The post-LoRA all-ours pool is what carries the jump — both all-ours entries clear the fusion — and the two-member configuration is worth a further −0.008 on top, which is directional, not a result. |

| 2026-08-16 | `kneemri-chosen` | 2 | **OURS** — the four members chosen by their JOINT holdout (res392 + intensity392 + calibrated + sliceattn), disjoint windows | **0.8612** | **0.918** | submitted 17:45 UTC, scored within the hour. Selection, not tuning: `strongest(4)` scores 0.8588 and the subset behind the 0.919 scores 0.8581, so naming the members is +0.0031 for free — same weights, same decode, same runtime. Directional (under the 0.0053 floor), kept because it costs nothing. **It scored 0.918 against the 0.919 of the two-member entry — a tie, on a board where the two entries differ by two members and 0.0031 of holdout.** The holdout ranked it FIRST of every all-ours configuration and the board cannot separate it from the pair, which is the third time the ruler and the board have disagreed by less than they agree; read it as the ruler being unable to price member COUNT, not as selection failing. |
| 2026-08-16 | `kneemri-onedecode` | 2 | **OURS** — three members of ONE pixel group (calibrated + sliceattn + extended_ws) | 0.8553 | **0.904** | submitted 17:45 UTC, aimed at the efficiency board. Half the decoding for 0.0059 of holdout; the holdout carries no clock, so only the board can price that trade. **The board priced it at −0.014 against the 0.918 of `chosen`, 2.4x what the holdout charged for the same change (−0.0059).** The pixel-group collapse is therefore a real cost and not a free efficiency win, and — more useful than the entry itself — the holdout UNDERSTATES a diversity loss by more than a factor of two, which is the same direction the `trained` v8 disagreement pointed. |

## What counts as ours

The graft row is the field's floor measured on our harness. It is **not a result**: the weights are
`pilkwang/rsna-knee-weights` (CC0), and the only thing of ours in it is the data layer and the ensemble.
It exists so that every later number has something to be compared against, and so the submission path
was proven before a model depended on it.

`kneemri-trained` is ours end to end — our slot routing, our ordering, our laterality, our crop, our
head, our fitting. It starts below the graft, which is the honest starting position for a replication
plus one day's work.

## The two numbers, side by side

| | members | epochs each | per-member holdout | public LB | rank of 1061 |
|---|---|---|---|---|---|
| graft (their weights) | 20 (5 folds x 4 seeds) | 60 | 0.8398 | 0.888 | 349 |
| ours (one day) | 5 (5 folds) | 16 | 0.8162 | 0.868 | 393 |

**0.020 apart, from an ensemble four times smaller trained for a quarter as long.** The public median
is 0.819, so both clear it; the leader is 0.942, so neither is close. Rank 349 with the field's own
weights is the tell that a graft scores where forks score — 86 teams sit within 0.005 of it.

The comparison is only this clean because both sides train on the SAME public label tables and run
through the SAME data layer. Members and epochs are the two axes we have not spent, and they are cheap:
27 s per epoch on the 5090 over the memory-mapped cache. Spending them is the obvious next move and
also the LEAST informative one — it buys rank without teaching anything. The labels are where the
un-shared edge is.

## What 0.888 bought

The graft is not only a floor — it **prices our data layer in leaderboard units**. The weights are the
same CC0 members the public forks run, so the difference between our 0.888 and what those weights score
in other hands is our pipeline's reading of the pixels.

**Correction to the first reading of this.** 0.899 was taken from a public notebook's title and called
"the plateau"; on the actual leaderboard 0.899 is about 100th place and the frontier is 0.942. So the
0.011 gap is to 100th place, not to the people running these weights best, and the real distance to the
top is 0.054 — far too large to be the slot tie-break. Something structural separates the leaders from
the public stack.

That agrees in direction and rough size with the 0.025 out-of-fold residual measured locally, which for
three days made the slot tie-break the lead: when two candidate series have equal slice counts, a stable
`sort_values` picks the first in FRAME order and our `max()` picks the first in LIST order.

**The ORDER is refuted (2026-08-14), by one CSV read** — and that is less than it first appeared.
`Study.from_directory` probes `sorted(directory.iterdir())` and the directory names ARE the series UIDs;
`train_series.csv`'s frame order is that same sorted-UID order for **4,407 of 4,407 studies**. So the two
pipelines see a slot's candidates in the same order, and order cannot be the source of a divergence.
`SlotTies.disagreeing_order` measures it and the CLI prints it first.

**That was written up as "the tie-break is refuted", which overstates it.** The step from "same order" to
"same choice" assumes their rule is *first maximal by slice count* — an inference recorded here, never
verified, because **we have their checkpoints and their config but not their router**. A rule that breaks
ties by anything else (the last maximal, SeriesNumber, acquisition time) diverges no matter how the
candidates are ordered.

And the rebuilt oracle's own covariates argued it does: over 24 studies the residual correlated **+0.664
with contested slots**, the strongest of four routing covariates, surviving control for series count
(**partial +0.496, p ≈ 0.009**). The population whose divergence was declared impossible was the one that
best predicted where the two pipelines disagree.

**Their rule is LAST-maximal; ours is first.** Decoding the same studies under each rule and scoring a
published member against its OWN recorded predictions:

| | 16 studies WITH a contested slot | 40 studies, unselected |
|---|---|---|
| `tie=first` (ours) | 0.1277 / 0.0236 | 0.1205 / 0.0094 |
| `tie=last` | **0.0252 / 0.0061** | **0.0671 / 0.0062** |

Replicated on **4 of 4 published members across 3 seeds**. Once the rule is corrected the routing
signature disappears entirely — `corr(residual, series_count)` falls from **+0.647 to −0.070** — so the
series-count correlation *was* the tie-break and nothing else.

**It is one contributor, not the cause.** On the unselected sample the mean falls by a third while 32 of
40 studies still disagree above 0.01, against the 1e-6 a faithful pipeline reaches.

**Ten axes are now ruled out and our decode matches theirs on every one**: slice order (their
`rules.order = 'normal'` is what we already do), percentile values and scope, DICOM windowing (they apply
none — VOI LUT is far worse), crop rounding, band sampling and its dedupe, the resize filter, and the
disjoint-window inference policy. Several of those were assumptions recorded in this file and never
tested.

The rule is plumbed as a per-member reading axis (`Routing(scheme, tie)`), because our members were
FITTED under first-maximal — a global switch would fix the graft and break everything of ours. The cost
of serving each side correctly is that `Ensemble.reads` groups by it, so a mixed entry decodes the test
set twice.

**And no declared reading axis differs either.** The published members' own checkpoints record
`crop_mm 130.0`, `band 0.20,0.80`, `img 336`, `slices 12`, `group 3`, `pool cls_mean`, the identical slot
ORDER, and byte-for-byte the `rules` dict `member_record.py` hardcodes. So the residual lives in the
IMPLEMENTATION of those rules — decode, percentile normalisation, resize — not in any recorded policy.
`kneemri.evaluation.oof_oracle` is rebuilt and pointed at exactly that, reporting per-study residuals against free
covariates so a defect that concentrates says so.

## What the six scored entries say, read together

| entry | members | per-member holdout | public LB |
|---|---|---|---|
| one model, no ensemble | 1 | **0.8518** (our best ever) | 0.868 |
| whole-stack | 2 | 0.8481 | 0.882 |
| whole-stack | 3 | 0.8493 | 0.882 |
| ours v2, old windowed recipe | 5 | 0.8269 | **0.888** |
| the published CC0 members | 20 | 0.8398 | **0.888** |
| **ours 3 + their 20** | **23** | mixed | **0.894** |

Built to isolate one variable, and they do. **The best single model this project has produced loses to
five much weaker ones by 0.020** — more than whole-stack (+0.0147) and more than every knob combined.
And a third member of our OWN recipe bought nothing (0.882 → 0.882) while a different recipe bought
+0.006, so it is DIVERSITY and not count. Members that fail the same way do not help each other.

That is also why the holdout cannot pick an entry alone: the 5-member set is +0.021 WORSE per member
than the 3 whole-stack set and scored +0.006 better.

## v2: our five members equal their twenty

| | members | public LB | rank of 1077 |
|---|---|---|---|
| graft (their weights) | 20 | 0.888 | 367 |
| ours v1 | 5 | 0.868 | 412 |
| **ours v2** | **5** | **0.888** | **367** |

**+0.020 on the leaderboard, 45 places, from three defects in our own code** — augmentation never
called, no learning-rate schedule, and the TTA default left off. No new idea, no bigger model, no
extra data.

The graft has since stopped being a floor and become a *control*: same data layer, same ensembling,
same submission path, differing only in whose weights and how many. Matching it with a quarter of the
members says our per-member training is now roughly on par with the published one — which is what the
holdout was already saying (0.8269 against their reported 0.8398, on different splits) and is now
confirmed on the hidden test, where neither of us chose the studies.

The remaining 0.054 to the leader is untouched by any of this, and nothing measured so far explains
it. It is not the schedule, the augmentation, the labels, the slice count or the TTA.

## What the ablations say (fold 0, matched; ruler held at pilkwang)

|  | pilkwang labels | v4_blend labels |
|---|---|---|
| geometric augmentation | **0.8202** | 0.8194 |
| + intensity augmentation | **0.8221** | 0.8203 |

Both imported ideas are null: intensity augmentation +0.0014 on average, better labels −0.0013, no
interaction. Set against what fixing our own code was worth — augmentation never called +0.0110, the
missing schedule +0.0061, the TTA default +0.0035 — **repairing what we had beat importing what others
had by an order of magnitude.**

That is not an argument against reading the field: the reading is what found the schedule, and the
`.903` notebook is what made label quality checkable at all. It is an argument against assuming a
published gain transfers. Every imported idea here either needed adapting or died on a mechanism
specific to this architecture — channel-stacked slices, side-defined targets, a ViT that is not
translation-equivariant, readers that are not independent.

Intensity augmentation stays behind `--intensity-aug` rather than being deleted: the holdout is drawn
from the same corpus as training, so it is structurally blind to the cross-scanner axis that transform
exists for. It is worth one submission slot when a slot is being spent anyway — a claim that is only
honest because it was written down before the result.

## The two oracles

Anything claiming to reproduce the published pipeline has to pass both, because they check different
things and neither implies the other:

- **Fingerprint** (`kneemri-graft-verify`) — each member's output on a seeded synthetic bag. Proves the
  ARCHITECTURE is the function those weights were fitted to. Blind to the pixels: it feeds noise.
- **Out-of-fold agreement** (`python -m kneemri.evaluation.oof_oracle`) — our pipeline against the member's own
  stored predictions on the studies it held out. Proves the PIXELS: routing, order, laterality, crop,
  percentile, resize. Measured at max|diff| 0.025 under our `tie=first` routing; **0.0671 max / 0.0062
  mean over 40 unselected studies once the tie rule is corrected to theirs** (see above).

The residual is an OPEN QUESTION, not a rounding error — a faithful pipeline would agree to about 1e-6,
as the fingerprint does. **The tie-break is a CONFIRMED contributor worth about a third of the mean**,
not the refutation an earlier draft of this section recorded: only the ORDER half was refuted, and
`tie=last` then beat `tie=first` on 4 of 4 published members across 3 seeds. Every DECLARED reading rule
is ruled out — the published checkpoints declare the same crop, band, slot order and rules we use, and
ten decode axes were swept without beating ours. What is left is the IMPLEMENTATION of those rules, and
32 of 40 studies still disagree above 0.01.

**This oracle was lost and rebuilt, which is why it is named as a module and not a script.** It lived in
`scratchpad/oof_oracle.py`, was never committed, and vanished with the weights it ran on — so for three
days two beads argued about a number nobody could recompute, and both were settled by facts sitting
inside those checkpoints the whole time. It reports per-study residuals against free covariates now,
because a max|diff| says a defect exists and nothing about where.

## Provenance warning on the live entry

The 0.888 submission's five members were trained and epoch-selected against a report-derived label
table that is **no longer reproducible** — it was not on disk anywhere and had no fetch step (see
`comp/README.md`). The members themselves are fine and the leaderboard score is real; what cannot be
reproduced is the holdout number that chose their epochs, or any comparison between them and an arm
trained today.

Anything retrained from here uses `pilkwang/rsna-knee-llm-labels` → `report_labels_v2.csv`, which is
fetched rather than remembered. Do not compare a new member's holdout against the old ones'.

## 2026-08-13: the all-ours weights finally reached Kaggle

Three nights running, the ledger recorded no submission at all — five slots expire unused at 00:00 UTC
every day, and they cannot be banked. The reason was a gate misapplied: an entry is supposed to be
justified offline by `member_curve` first, but that gate needs a GPU pass to produce predictions, and
"the GPU is busy" was treated as if it blocked SUBMITTING. It never did. The scored run happens on
Kaggle's T4.

| entry | members | passes/study | LB |
|---|---:|---:|---|
| `kneemri-efficient` | 2 (res392 0.8523 + calibrated_targets 0.8495) | 2 | **0.500 — null, see below** |
| `kneemri-trained` v6 | 5 all-ours whole-stack | 5 | pending |
| `kneemri-efficient` v2 | the same 2, RE-RUN on kit 5ba86ee | 2 | pending |

The members dataset had been stuck at 2026-08-11 — `res392`, `calibrated_targets`, `sliceattn`,
`intensity392` and `extended_ws` existed only on this machine. It now carries all six (498 MB), which is
also why `strongest(2)` selects a different, untested pair rather than repeating the 0.882 fold pair.

THREE SLOTS SPENT, TWO LEFT, and `kneemri-fusion` v4 was still running when the machine slept — its
output can be submitted whenever, so it becomes the next day's first entry rather than a loss. Which is
the operating fact worth keeping: a run is not tied to the day it started, only a SLOT is.

**That 0.500 is a NULL RESULT, not a floor, and the stale kit is the whole cause.** The OOM fix that
bounds the forward pass by IMAGES rather than studies landed at 2026-08-12 13:11 UTC (`9f31b35`); the
mounted kit was built at 09:46 UTC the same day, before it. So the run served whole-stack members
without the budget split, blew the T4's 16 GB exactly as the 08-12 entry did, and the fallback wrote a
constant table. The members were never measured. The same stale kit also cost `Member.precision`:

**Read that entry as a floor, not a score.** The mounted kit was built 2026-08-12 09:46 and HEAD was 54
commits ahead, so it lacked `Member.precision` — and `calibrated_targets` is a bf16 member, served under
fp16 on a T4 whose exponent range is narrower. Found mid-flight, too late to stop the run. The kit is
rebuilt and `build_kit` now stamps its commit into `kit.json`, so the drift is visible next time.

**Two kernels were one push away from spending a slot on a crash.** The package now contains a DINOv3
LoRA member, while `kneemri-trained` ran EVERY member and `kneemri-fusion` fused every member — both
mount DINOv2 only, with internet disabled. That is exactly how the 0.500 entry was spent. `trained` is
capped at `strongest(5)` and `fusion` now drops variants whose weights are not mounted.

## Gotchas paid for in full

- **Submission is notebook-only.** `kaggle competitions submit -f submission.csv` returns 400. The
  working form is `-k <owner>/<kernel> -v <version> -f submission.csv`.
- **Search for the artefact, not the container.** Three kernel runs died on mount-shape assumptions:
  the competition sits under `/kaggle/input/competitions/<slug>`, datasets under
  `/kaggle/input/datasets/<owner>/<name>`, and a `--dir-mode zip` upload is expanded on mount with its
  wrapping directory REMOVED.
- **Probe what you IMPORT, not only what you plan to use.** The runtime probe checked pydicom, timm and
  transformers (all present) and never checked `jaxtyping`, which our own modules import at module top.
- **A published checkpoint is a training artefact**, not a bare state dict: weights sit under `model`
  beside epoch, fold scores, config, out-of-fold predictions and its own fingerprint.
- **bfloat16 is EMULATED on the T4, and it costs about 5x.** `torch.cuda.is_bf16_supported()` returns
  True there, which reads as "fine" and is not: its own source returns True immediately for compute
  capability >= 8 and otherwise falls through to an emulation check. Turing has fp16 tensor cores and no
  bf16 ones, so bf16 matmuls leave the fast path. Measured on the card, a chunk of 48 studies at 336 px:
  **12 s in fp16 against 58 s in bf16**, same architecture, same resolution. Serve bf16 members in fp16
  on anything below sm_80 — `Ensemble.arithmetic` does, and every entry that has ever scored ran fp16.
- **A kernel that only LOADS is not a submission and costs no slot.** `kneemri-loadcheck` mounts the kit
  and the members, then loads, forwards and sends a 48-study chunk per member. Every runtime fact that
  cost a slot on 2026-08-13/14 — a class mismatch from a transformers version, an emulated dtype, a
  memory ceiling — is answerable there in twenty minutes for free. Run it after any kit rebuild.
- **Kaggle allows two concurrent GPU sessions ACROSS ALL PROJECTS.** On 2026-08-13 both were held by
  the sibling celltrack campaign, so kneemri runs serialised behind them: submissions are limited by
  SESSIONS, not by the five daily slots. A completed kernel version can be submitted at any later time,
  so a long run started before a machine sleeps is not wasted — it is submitted the next day.
- **READ THE ERROR BODY. The CLI throws it away, and a whole evening was spent on the theory it left
  behind.** `kaggle competitions submit` prints `400 Client Error ... CreateCodeSubmission` and nothing
  else. The body says:
  `{"error":{"code":400,"message":"This competition requires an output FileName for Notebook Submissions."}}`
  — the submit was missing `-f submission.csv`, which for a CODE submission names the notebook's OUTPUT
  file, not a local path to upload. Every 400 on 2026-08-15/16 was that one argument.
  I diagnosed it instead as "one code submission may be PENDING at a time, per competition" — the 400s
  did coincide with a pending entry, a second kernel got the same 400, and the story explained every
  observation. It was still wrong, and four of that day's five slots expired at 00:00 UTC (slots do not
  roll over) waiting for a queue that was never blocking. Thirty retries at 110 s apart tested the same
  wrong argument thirty times. **A retry loop is not a diagnosis** — if the first attempt fails without
  naming a cause, get the cause before automating the attempt:
  ```python
  api.competition_submit_code(None, "probe", "<comp>", "<owner>/<kernel>", <version>)  # raises with .response.text
  ```
  What IS measured about the queue: Kaggle reruns the notebook on the hidden test set before scoring, so
  a `PENDING` entry stays pending for HOURS (the 21:06 entry scored at about 05:00, ~8 h). And three
  entries submitted back to back at 05:05 were all accepted, so a pending entry blocks nothing.
- **A guard that reports its own alarm state is not a guard.** Every kernel run logged `kit UNSTAMPED
  (built before build_kit recorded its commit)` — including runs on kits that carried a perfectly good
  `kit.json`. `revision()` addressed the stamp with two `.parent` hops from `kneemri_src/kneemri/kaggle/`,
  which lands one directory short of `kneemri_src/kit.json`. The unit test built the module one level
  flatter than build_kit's real layout, so the wrong count was right in the test and nowhere else. Fixed
  by searching the ancestors for the stamp — and by giving the test the layout the kit actually has.
  The lesson generalises past this file: **a check whose failure branch is the one always taken looks
  identical to a check that works**, because nobody reads a warning that is always there.
- **`kaggle datasets version -p <path>` pastes the path into its upload-temp filename.** Any separator
  in it produces `[Errno 2] ... .kaggle/uploads\D_/data/.../kit_<file>.json` and every file fails. Push
  from the target's PARENT with a bare basename (`push.sh` does), which holds wherever the kit lives —
  the old fix, cd-ing into `comp/`, died the moment the kit moved out of the repo.
- **`machine_shape: NvidiaTeslaT4` gives two T4s and only four CPUs.** On a decode-bound pipeline the
  CPU count binds first: measured 12 ms per DICOM slice off the mount.

## The pool grew from 13 records to 36, and the best four are no longer the shipped four

Re-searched 2026-08-19 over every fold-0 holdout record on disk — 36 members, arithmetic only, no card,
while `band_full` trained. Greedy and exhaustive agree on every size, which is worth stating because
greedy is the only tractable reading past a dozen and this is the first time both were run:

| size | best subset | mix |
|---:|---|---|
| 1 | `band_dense` | 0.8578 |
| 2 | + `band_span` | **0.8640** |
| 3 | + `depth_regions_narrow` | 0.8661 |
| **4** | + `calibrated_targets` | **0.8666** |
| 5 | + `depth_regions` | 0.8671 |
| 8 | (flat) | 0.8674 |

The shipped champion roster reads **0.8648** under the same tool and the same vote, so the best four now
available is **+0.0018** over it at the same member count and the same serving cost. The count curve is
unchanged in shape — flat past four, peaking at eight by 0.0008 — so this is a MEMBERSHIP result, not an
argument for mounting more.

**It is a TWO-member swap, which is why it is filed rather than fired.** Both learning-rate members
(`lr1e-4_ws`, `lr4e-5_ws`) leave and `band_span` and `depth_regions_narrow` enter. Every board-transfer
reading in this file is on ONE swap at a time, and the third of those broke the ordering rule outright,
so the honest prediction here is a SIGN and not a number: 0.8666 > 0.8648 says the entry should not be
worse, and nothing in the record supports quoting a board figure for it.

It also does exactly what this file said the next move had to do. `band_span` is a member that is
DIFFERENT rather than better — it reads the 0.10–0.90 pixel group, which no shipped member reads — and
its second-pick jump (+0.0062, the noise floor to four decimals) is the largest single addition in the
curve. Membership-shuffling inside the OLD pool was close to exhausted; the pool was the thing that
changed.

Not submitted. `band_full` is still training and is on the same axis, so the roster question is worth
asking once, after it lands, rather than twice.

## Slot 1 of 2026-08-19: the searched best four, fired

Pre-registered BEFORE the run, because the whole value of this entry is whether an offline membership
search transfers at all.

`kneemri-best4` is `kneemri-band` with one line changed — `chosen` goes from (`band_dense`, `lr1e-4_ws`,
`calibrated_targets`, `lr4e-5_ws`) to (`band_dense`, `band_span`, `depth_regions_narrow`,
`calibrated_targets`). Same member count, same `overlap=False`, same `RankAverage`, same kit. Offline it
is 0.8666 against the champion's 0.8648 over the 909-study fold-0 holdout, greedy and exhaustive
agreeing at every size.

**THE PREDICTION IS A SIGN, NOT A NUMBER, and this file has earned that caution.** It is a TWO-member
swap; every board-transfer reading here is on one swap at a time, and the third of those broke the
ordering rule outright. So: **at or above 0.928**, and nothing in the record supports quoting a figure.
A result BELOW 0.928 does not refute the search — it refutes the transfer of a two-swap, which is a
different and more useful thing to learn, because it would say the holdout can rank single swaps and
cannot rank compound ones.

The dataset push that carries it mounts FOUR new members (`band_span`, `depth_regions_narrow`,
`band_full`, `calib_span`) while the kernel names two of them. Mounting is reversible and a slot is not;
the two unnamed ones exist so the next two roster questions today cost a kernel edit rather than a
dataset push and the twenty minutes it takes.

## Slot 2 of 2026-08-19: the first half of that swap, alone

Pre-registered BEFORE the run. Slot 1 is a TWO-member swap and cannot say which half carried, so this
slot takes the half that is cheapest to interpret and holds everything else at the champion.

`kneemri-oneswap` is `kneemri-band` with `chosen` going from (`band_dense`, `lr1e-4_ws`,
`calibrated_targets`, `lr4e-5_ws`) to (`band_dense`, `band_span`, `calibrated_targets`, `lr4e-5_ws`).
One member moves: `band_span` replaces `lr1e-4_ws`. `depth_regions_narrow` does not enter. Same count,
same `overlap=False`, same `RankAverage`, same kit, all four already mounted — the kernel edit is the
whole change.

Offline, over the same 909-study fold-0 holdout and by the same rank average the kernel ships:

| four | offline | vs champion |
|---|---|---|
| champion (`band_dense`, `lr1e-4_ws`, `calibrated_targets`, `lr4e-5_ws`) | 0.8648 | — |
| **this slot** (`lr1e-4_ws` → `band_span`) | **0.8661** | **+0.0013** |
| slot 1 (both swaps) | 0.8666 | +0.0018 |

So the single swap carries 0.0013 of the compound's 0.0018 offline — three quarters of it — and
`depth_regions_narrow` is the marginal quarter. Both numbers sit under the 0.0062 holdout noise floor,
which is exactly why the board is the ruler here and the offline table is only the ORDERING.

**PREDICTION, and it is again a sign.** At or above 0.928. The stronger claim this slot exists to test
is the ORDERING one: if slot 1 also lands at or above 0.928, then a compound swap transfers and the
holdout can rank more than one member at a time; if slot 1 lands below 0.928 and this one does not,
that is the cleanest evidence yet that only single swaps survive the trip to the board, and every
future roster question has to be asked one member at a time.

**REFUTATION.** This entry below 0.928 with slot 1 at or above it inverts the offline ordering and
would say the holdout cannot rank even a single swap on this pool — which retires offline membership
search for roster decisions outright, rather than leaving another subset to try.

These two numbers cost no card time. Both came from `record_mix`, which reads the per-study
probabilities each run already wrote beside its checkpoint; `member_curve` answers the same question by
re-deriving them through the pixel cache and would have needed the GPU a trainer is holding.

## Slot 3 of 2026-08-19: the same seat, the other candidate

Pre-registered BEFORE the run, and built as a MATCHED PAIR with slot 2 rather than as the leftover half
of slot 1.

`kneemri-swapdrn` puts `depth_regions_narrow` into the seat slot 2 gives to `band_span` — both replace
`lr1e-4_ws`, both leave `band_dense`, `calibrated_targets` and `lr4e-5_ws` untouched, both keep the
count, the kit, `overlap=False` and `RankAverage`. Two entries differing in ONE member each, from a
common champion, is the only shape from which a board ordering can be read.

| four | offline |
|---|---|
| champion | 0.8648 |
| slot 2 — `band_span` in that seat | 0.8661 |
| **slot 3 — `depth_regions_narrow` in that seat** | **0.8653** |
| slot 1 — both swaps at once | 0.8666 |

**THE QUESTION, stated as the pair.** The offline ruler orders these 0.8661 > 0.8653 > 0.8648, all
three inside the 0.0062 noise floor of each other. Slot 2 and slot 3 ask the board the same ordering.
If the board reproduces it, the holdout can rank single-member swaps even at sub-floor separations and
offline membership search is a usable instrument. If the board inverts it, the offline ruler is
measuring something that does not survive the trip, and roster decisions have to be bought one
submission at a time from here on.

**PREDICTION.** Both at or above 0.928, with slot 2 at or above slot 3. The refutation that matters is
not a low score — it is slot 3 ABOVE slot 2, which would invert the ordering while both still look like
wins, and is exactly the case a mean-only read would miss.

Read this pair together with slot 1: if the compound (0.8666) does not beat the better single, the
holdout is additive where the board is not.

## Slot 4 of 2026-08-19: the member that lost to its own control, in the same seat

Pre-registered BEFORE the run. Third entry into ONE seat — `lr1e-4_ws`'s — so all three read against
each other and against the champion with a single variable between them.

`kneemri-calibspan` puts `calib_span` there. That member is the one this ledger wrote up hours ago as a
FAILURE: 0.8545 alone against its matched control `band_span`'s 0.8565, the calibration coming back
negative at the shipped operating point after paying +0.0038 at the older one. It is also, by the
seat-value sweep over all 42 same-split records, the best fifth member the project owns (+0.0022), and
in this seat it is the best four the project has ever assembled offline:

| four (all replace `lr1e-4_ws`) | offline | member's own alone score |
|---|---|---|
| champion, unchanged | 0.8648 | `lr1e-4_ws` 0.8490 |
| slot 2 — `band_span` | 0.8661 | 0.8565 |
| slot 3 — `depth_regions_narrow` | 0.8653 | 0.8554 |
| **slot 4 — `calib_span`** | **0.8670** | **0.8545 — the worst of the three** |
| slot 1 — two swaps at once (`band_span` + `depth_regions_narrow`) | 0.8666 | — |

**WHAT THIS SLOT IS ACTUALLY TESTING, and it is not the mean.** The three candidates order one way
alone (`band_span` 0.8565 > `depth_regions_narrow` 0.8554 > `calib_span` 0.8545) and the exact opposite
in the seat (`calib_span` 0.8670 > `band_span` 0.8661 > `depth_regions_narrow` 0.8653). Member quality
and seat value are ANTI-correlated across these three. This project has recorded that shape before —
diversity beating member quality, six board entries built to isolate it — and has never tested it with
a member whose quality deficit was MEASURED against its own matched control rather than inferred.

**PREDICTION.** At or above 0.928, and at or above slot 2. The claim under test is that a member can be
worse than its control and still be the right occupant, because what it contributes is the part the
other three do not have.

**REFUTATION.** Slot 4 below slot 2 on the board, with the offline table ordering them the other way,
says the offline holdout systematically overprices decorrelation — that it rewards a member for being
different in ways the hidden test set does not care about. That would be the single most useful thing
to learn tonight, because every roster decision this project has made since the pool opened has been
made on this ruler.

Note what this entry is NOT: it is not a rehabilitation of the calibration lever. `calib_span` is still
worse than `band_span` as a member, and the ledger row that says so stands. A seat is a different
question from a score, and tonight is the first time the two have been asked separately.

### Slot 1 result: 0.927 — the compound swap does not transfer

Submission 55631509, `kneemri-best4`, returned **0.927** against the champion's 0.928. Pre-registered
condition was "at or above 0.928"; it is below, so the pre-registered reading stands as written: **the
transfer of a TWO-member swap is refuted, not the search.**

Read it at the right size. One board tick is the resolution this competition reports at, so 0.927 is
"not better", not "much worse" — the entry did not collapse, it failed to move. What died is the
inference that an offline four ranking 0.8666 over 0.8648 will rank the same way on the hidden test
when TWO members changed at once. Every board transfer this ledger has read remains a single swap, and
the one compound reading before tonight broke the ordering rule too. That is now twice.

What it does NOT say, and the distinction is the whole reason slots 2 and 3 were fired before this
score returned:

- It does not say `band_span` is a bad member — it is in this entry together with a second change.
- It does not say the offline ruler is broken. A +0.0018 offline separation is under a third of the
  0.0062 noise floor; a ruler that cannot resolve that separation is a ruler being asked the wrong
  question, not a broken one.
- It does not say the champion four is optimal. It says this particular pair of moves is not an
  improvement large enough to survive the trip.

Slots 2 and 3 put ONE member into ONE seat each and are already in the queue; slot 4 puts a third
candidate in the same seat. Those three answer the question this one cannot: whether the board
reproduces an offline ordering when only one thing moves.

### Slot 5 pre-registration: `kneemri-greedy4` — the mounted maximum, and the one-swap rule at a second point

`band_dense + calib_span + depth_regions_narrow + band_span`, offline **0.8679** against the champion
four's 0.8648 over the same 909-study fold-0 holdout. Same count, same `overlap=false`, same
`RankAverage`, one-line diff from `kneemri-best4`. Every member is already in `kneemri-members`, so no
dataset re-upload was needed and the entry costs a kernel push.

**HOW IT WAS FOUND, and why it is not the same search as slot 1's.** `record_mix --greedy` restricted to
the SIXTEEN MOUNTED members — the 36-record and 43-record searches that produced earlier entries ranked
members the kernel cannot load, which is how a candidate can look best offline and be unshippable. Slot
5 was originally going to be `attend_dense` in the `lr1e-4_ws` seat (seat value +0.0019, second of four
occupants) and was dropped for exactly that reason: it is not in the roster, its edge over `band_span`
is 0.0001, and a members re-upload to buy a separation a thirtieth of the noise floor is not a trade.

**THIS IS A THREE-MEMBER SWAP, WHICH IS THE POINT.** `lr1e-4_ws`, `calibrated_targets` and `lr4e-5_ws`
all leave; `calib_span`, `depth_regions_narrow` and `band_span` all enter. Every board transfer this
project has read is on ONE swap at a time, and the single compound reading it has — slot 1's two-member
swap, offline 0.8666 — came back **0.927 against 0.928**. So the prior here is negative and explicit.

**PREDICTION, written before the run.** The one-swap rule says this lands at or below 0.928 despite the
largest offline edge any entry has carried (+0.0031, half the 0.0062 floor). That is the outcome I
expect, and it is worth the slot because it confirms the rule at a SECOND point with a LARGER offline
edge, which is a much stronger statement than n=1: it would say the offline ruler's advantage does not
scale with how much of the roster moves, i.e. the holdout is pricing joint membership wrong rather than
pricing one member wrong.

**THE REFUTATION CONDITION, and it is the valuable half.** Above 0.928 refutes the one-swap rule
outright. Slot 1 would then read not as "compound swaps do not transfer" but as "that particular
compound swap did not transfer", and the offline greedy search becomes a tool this project can act on at
full roster width instead of one seat at a time — which changes how every remaining submission is
chosen, not just this one.

Read it against slots 2, 3 and 4, which are three single swaps into ONE seat. If the single swaps
transfer their ordering and this one does not, the failure is specifically in COMPOUNDING, and the two
readings together separate that from "the offline ruler is broken", which neither can do alone.

### Slots 2 and 3 result: 0.927 and 0.928 — the seat ordering does NOT reproduce, and the one-swap rule is not what separates them

| entry | swap into the `lr1e-4_ws` seat | offline (fold-0, 909 studies) | board |
|---|---|---|---|
| champion `band_v1` | — | 0.8648 | **0.928** |
| slot 1 `best4` | `band_span` + `depth_regions_narrow` (two at once) | 0.8666 | 0.927 |
| slot 2 `oneswap` (55631756) | `band_span` | 0.8661 | **0.927** |
| slot 3 `swapdrn` (55631811) | `depth_regions_narrow` | 0.8653 | **0.928** |
| slot 4 `calibspan` (55632193) | `calib_span` | 0.8670 | pending |

**BOTH PRE-REGISTERED CLAUSES FIRED, and they point the same way.** Slot 2's prediction was a sign — "at
or above 0.928" — and it came back **below**. Slot 3's clause named the interesting case explicitly:
"slot 3 above slot 2 inverts it while both still look like wins, which is the case a mean-only read would
miss." That is precisely the shape returned. The member the offline ruler ranked HIGHER in this seat
(`band_span`, 0.8661) scored LOWER on the board than the one it ranked lower (`depth_regions_narrow`,
0.8653). The offline seat ordering did not transfer.

**THE READING THAT SURVIVES ALL FOUR ENTRIES, and it is not the one slot 1 was written under.** Slot 1
was read as "compound swaps do not transfer, single swaps might." Slot 2 is a SINGLE swap and it scored
0.927 — identical to the compound. Slot 3 is a single swap and it scored 0.928 — identical to the
champion it replaced a member of. So the number of members moved is **not** what separates these
entries. What separates them is nothing: four entries, offline edges of +0.0005 to +0.0018 over the
champion, and a board that reports 0.927 or 0.928 with no relation to the offline order.

That is exactly what the noise floor predicts. **Every one of these offline edges is inside 0.0062** —
the largest is under a third of it. A ruler asked to resolve a separation a third of its own resolution
returns an ordering, because arithmetic always returns an ordering; it does not return a *finding*. The
board is not contradicting the holdout here. Both are saying the same thing, which is that these four
member sets are the same set as far as either can tell.

**WHAT THIS COSTS AND WHAT IT BUYS.** It costs the working assumption that offline seat value picks
submissions — within the floor it does not, and four slots were spent proving that at four points, which
is a real answer to a question that was genuinely open. It buys a hard rule for the remaining
submissions: **an entry whose offline edge over the champion is inside 0.0062 is not worth a slot.** The
member pool has been searched to its maximum (0.8679, slot 5) and that maximum is +0.0031 — still inside
the floor. Roster search is therefore exhausted as a source of board movement, and the next movement has
to come from a member that does not exist yet, not from a rearrangement of the sixteen that do.

**WHAT IT DOES NOT SAY.** It does not say the champion four is optimal — it says the alternatives are
indistinguishable from it. It does not say `band_span` is worse than `depth_regions_narrow`; one board
tick between two entries inside the floor is the same non-result in the other direction. And it does not
retire `record_mix` — a ruler that correctly reports "these are the same" is working. It retires the
practice of *acting* on its ordering below the floor.

#### Amendment to slot 5's pre-registration, written after slots 2 and 3 returned and before it fires

Slot 5 was pre-registered to test the ONE-SWAP RULE — "single swaps transfer, compound ones do not" —
at a second point. Slots 2 and 3 dissolved that rule before slot 5 fired: a single swap returned 0.927
and the compound returned 0.927, so member count was never the separator. **The hypothesis slot 5 was
written to test no longer exists**, and honesty requires saying so here rather than reading its result
against a clause its own evidence has retired.

What it can still answer is narrower and worth stating exactly. `greedy4` is the **offline maximum over
the mounted pool** (0.8679, +0.0031). Under the rule slots 2 and 3 just bought — an edge inside 0.0062
is not worth a slot — this entry is *itself* inside the floor and should be expected to return 0.927 or
0.928 like the other four. Firing it is therefore not a test of a live hypothesis; it is the cheapest
available check that the pool's maximum behaves like the pool's middle, which is the last thing that
could still show the offline ruler carrying signal below its own floor. If it too lands at 0.928, the
roster-search axis is closed by five independent points rather than four, and every subsequent slot goes
to a member that does not exist yet.

It fires only because the slot is free at the 00:00Z reset and no better-argued candidate is ready — a
NEW member is what the evidence now asks for, and both arms that could produce one (`focal_dense`,
`band_dense_long`) are still on the card. That is the honest reason, and it is a weaker one than the
pre-registration above claims.

### Slot 4 result: 0.929 — a new champion, and it refutes the rule slots 2 and 3 had just bought

Submission 55632193, `kneemri-calibspan`, returned **0.929**. That is the first entry this project has
put above 0.928, and `comp/champion.json` now pins it: `calibspan_v1`, kit `4bdf2a4`, members
`band_dense` + `calib_span` + `calibrated_targets` + `lr4e-5_ws`, one swap off the previous champion
(`lr1e-4_ws` out, `calib_span` in). `champion_check` confirms it rebuilds.

**THE CORRECTION, and it is to a section written an hour before this score returned.** "Slots 2 and 3
result" concluded that the board bears no relation to offline seat value inside the floor, and drew a
rule from it: *an entry whose offline edge over the champion is inside 0.0062 is not worth a slot.* Slot
4's edge was +0.0022 — a third of the floor — and it moved the board. **The rule is wrong as written and
is withdrawn.** It was drawn from three points at the moment the fourth was still pending, which is the
mistake, not the arithmetic.

Four occupants of ONE seat, which is what the four slots were spent to get:

| entry | member in the `lr1e-4_ws` seat | alone | offline seat | board |
|---|---|---|---|---|
| champion `band_v1` | `lr1e-4_ws` | — | 0.8648 | 0.928 |
| slot 2 `oneswap` | `band_span` | 0.8565 | 0.8661 | 0.927 |
| slot 3 `swapdrn` | `depth_regions_narrow` | 0.8554 | 0.8653 | 0.928 |
| **slot 4 `calibspan`** | **`calib_span`** | **0.8545** | **0.8670** | **0.929** |

**WHAT THE FOUR POINTS ACTUALLY SAY.** The offline ruler picks the WINNER and does not resolve the
FIELD. Highest offline seat value took the highest board score; second-highest offline (`band_span`)
came last on the board. So the ordering agrees at the top and inverts in the middle. That is a weaker
tool than a full ordering — it cannot rank two candidates a thousandth apart — and a far stronger one
than the "no relation" reading, which the very next data point contradicted.

**THE OTHER INVERSION IS THE ONE WORTH KEEPING.** Ranked ALONE, `calib_span` is the WORST of the three
challengers (0.8545 against 0.8565 and 0.8554), and this ledger wrote it up hours earlier as a failure
against its own matched control (−0.0020 at the settled operating point). The member that failed its own
control is the member that produced the best entry the project has. Member quality measured alone and
seat value in a four-member vote are different quantities, and here they are ordered oppositely. Pinned
into `champion.json` as a principle: never retire a member on its own control alone — score it in the
seat.

**WHAT IT DOES NOT SAY.** One board tick is this competition's reporting resolution, so 0.929 over 0.928
is "better" and not "much better", and a single hidden-set read cannot separate a real +0.001 from a
lucky one. The honest statement is that the entry is at worst equal to the previous champion and is the
only one of four seat occupants that did not land at or below it. It also does not resurrect the
one-swap rule: slot 2 was a single swap and returned 0.927, so single-vs-compound still is not the
separator — WHICH member goes in the seat is.

#### Second amendment to slot 5, after slot 4 returned 0.929

The first amendment said slot 5's hypothesis was dead and it would fire only because the slot was free.
Slot 4 changes that, and in the direction that makes the entry better argued rather than worse: the
offline ruler picked the winner among four seat occupants, so firing the offline MAXIMUM is now a
supported move rather than a spare-slot move. `greedy4` is exactly that maximum over the mounted pool,
and it contains `calib_span`, the member that just took the board.

Two things must be restated honestly against the NEW champion rather than the old one.

**Its edge shrank by two thirds.** `greedy4` is 0.8679; the champion is no longer 0.8648 but
**0.8670** (`calibspan_v1`). The pre-registration's "+0.0031, the largest offline edge any entry has
carried" is now **+0.0009** — a seventh of the noise floor, and smaller than any of the four seat
separations the board has already failed to resolve. On the reading slot 4 itself supports (the ruler
picks the top and not the field), a +0.0009 separation is inside the part the ruler cannot resolve.

**It is a three-member swap off the OLD champion but a two-member swap off the new one.** `calib_span`
and `band_dense` are now common to both; `calibrated_targets` and `lr4e-5_ws` leave, `depth_regions_narrow`
and `band_span` enter. Slot 1's two-member swap returned 0.927, and `band_span` — the entering member —
is the one that came LAST on the board among four seat occupants.

**REVISED PREDICTION.** At or below 0.929. The offline edge is too small for the ruler to be speaking,
and the entry compounds two swaps of which one brings the worst-transferring member of the four. The
useful outcome is the refutation: **above 0.929** would say the greedy search over the mounted pool
finds board-real improvements at separations the seat experiment says it cannot resolve, which would
mean the seat comparison, not the greedy search, is the blunt instrument.
