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
| `fusion×25×windowed×overlap` (our 5 + their 20), `fusion25` in `board.jsonl` | 2026-08-17 | **0.894** |
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
has not mis-ordered a single pair it can resolve, and it converts at a median **2.14** board points per
holdout point. See `interpretations/kneemri/2026-08-16_board_transfer.md`.  **A new scored entry gets a
line there as well as a row here.**

**THE COMPRESSION WAS 4.67 UNTIL 2026-08-24 AND EVERY NUMBER DERIVED FROM IT IS 2.2x TOO BIG.** The
older figure was fitted over six pairs spanning 0.0043 offline — inside our own 0.0062 holdout noise
floor — so it was a slope through noise and could not be falsified by the data that produced it. The
`weakseven` instrument (`knee-825t`) was built to sit outside that band and did: offline 0.8562 against
the champion's 0.8694, board 0.908 against 0.934, which is 0.0132 offline buying 0.026 board — a slope
of 1.97, and 2.21 against the chain floor. The tool's own refit over all twelve entries agrees at 2.14.
Prose written before that date quotes 4.67; the dated entries below are left as they were filed, because
a ledger that edits its own past readings stops being evidence. **Multiply by 2.14, not 4.67.**

The other half of that refit matters more than the constant. Inversions are 4 of 55 orderable pairs and
**0 of the 33 the holdout resolved at all** — every disagreement is under the floor. The holdout is not
a broken ruler that needs replacing; it is a directionally perfect ruler that was priced too steep. Its
SIGN can be trusted wherever it clears 0.0053, and its MAGNITUDE buys less than half of what this file
has been assuming since 2026-08-16.

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

#### Slot 5 `greedy4` landed 0.930 — the refutation this ledger wrote for itself, and it fired

**0.930. New champion, pinned as `greedy4_v1` (submission 55633771, kernel `kneemri-greedy4`, kit
`4bdf2a4` read from the kernel log's own line, members `band_dense` / `calib_span` /
`depth_regions_narrow` / `band_span` read from the kernel's `chosen` tuple).** The revised prediction
above was **at or below 0.929**, and the useful outcome it named was the one that happened: *above 0.929
would say the greedy search over the mounted pool finds board-real improvements at separations the seat
experiment says it cannot resolve.* That clause fires as written.

**EVERY ARGUMENT AGAINST FIRING IT WAS TRUE, AND IT WON ANYWAY.** The edge had shrunk from +0.0031 to
+0.0009 when the champion improved under it — a seventh of the noise floor. It compounded two swaps
where the one-swap entry had already returned 0.927. It brought in `band_span`, the member that came
LAST on the board among four occupants of one seat. Three separate reasons to expect a flat or negative
tick, all of them drawn from this ledger's own measurements, and the entry took the board.

**THE FIVE-POINT TABLE IS NOW THE THING TO READ, not any single pair.**

| entry | offline joint holdout | board |
|---|---|---|
| `band_v1` | 0.8648 | 0.928 |
| `swapdrn` | 0.8653 | 0.928 |
| `oneswap` | 0.8661 | **0.927** |
| `calibspan_v1` | 0.8670 | 0.929 |
| **`greedy4_v1`** | **0.8679** | **0.930** |

Four of five are monotone in the offline score across a 0.003 offline spread. The single inversion is
`oneswap` — and `band_span`, the member whose entry that was, is IN `greedy4_v1` and did not hurt it.
So the inversion is a property of THAT four, not of the member, which is the second time this ledger has
had to separate a member from the company it was measured in (the first was `calib_span`, written up as
a control failure and then the best entry we had).

**WHAT UPGRADES.** The standing principle — *the offline ruler picks the winner and does not resolve the
field* — is now too weak as written. Over five points it orders the field at 4/5. It is a ranking tool
with one known failure, and its maximum is worth firing rather than arguing down. Both statements are
pinned into `champion.json`.

**WHAT DOES NOT.** One board tick is this competition's reporting resolution, so 0.930 over 0.929 is
"better" and not "much better", and five points with a 0.003 spread cannot establish a slope. The honest
claim is a decision rule, not an effect size: **fire the offline maximum over the mounted pool and stop
talking the margin down.** Total movement against the original baseline is 0.925 → 0.930, +0.005 over
five entries, which is still a third of the 1.5% bar and a tenth of knee-jk3's 0.054 frontier gap. The
entries are compounding; they are not yet a result.

#### Slot 6 `longseat` landed 0.930 — the clause passes at its weakest possible value, and the ruler runs out of resolution

**0.930. A TIE, not a win.** Submission 55637658, kernel `kneemri-longseat`, `band_dense_long` in for
`band_span`, everything else held at `greedy4_v1`. Pre-registered at **0.930 or above** with the stated
refutation that anything below refutes `champion.json`'s newest principle — *fire the offline maximum* —
at the first chance it has had to fail. It did not fail. It also did not move.

**READ THE PASS HONESTLY: it is the weakest value in the passing range.** The clause was written as an
inequality because a tick is the board's reporting resolution and a one-tick prediction cannot be made
sharper than that. But the entry carried **+0.0012 of offline seat value**, the largest single move
available anywhere in the mounted pool — `python -m kneemri.kaggle.seat_value` over all 56 same-split
checkpoints returns exactly three positive seats and this is six times the second (`attend_dense`,
+0.0002). So the best move the pool contains bought **zero board ticks**. That is a fact about the ruler,
not a fault in the entry.

**THE SIX-POINT TABLE, which is what the ledger should now be read on.**

| entry | offline joint holdout | board |
|---|---|---|
| `band_v1` | 0.8648 | 0.928 |
| `swapdrn` | 0.8653 | 0.928 |
| `oneswap` | 0.8661 | **0.927** |
| `calibspan_v1` | 0.8670 | 0.929 |
| `greedy4_v1` | 0.8679 | **0.930** |
| `longseat` | **0.8691** | **0.930** |

Still 4/6 monotone with the one known `oneswap` inversion, and the new point is not an inversion — it is
a **plateau**. The chain rises across the first 0.003 of offline spread and then stops responding over the
next 0.0012. The simplest reading consistent with all six is that the offline ruler tracks the board while
the offline gaps are large relative to the board's tick, and saturates once they are not.

**WHAT THIS COSTS THE STANDING PRINCIPLE, precisely.** *Fire the offline maximum* survives — firing it has
now never lost a tick across two consecutive chances. What does not survive is any expectation that a
further offline maximum will KEEP paying: the pool's best remaining move was just spent and returned
nothing, and the second-best is +0.0002, a thirtieth of the noise floor. **The mounted pool is exhausted
as a source of board movement.** That is a stronger and more useful claim than the tie itself, and it is
the argument for spending the card on an axis the pool does not contain rather than on another occupant of
the same seat — which is what `v3base_band` (queued) is for.

**THE CHAMPION PIN DOES NOT MOVE.** `comp/champion.json` stays on `greedy4_v1`. A tie does not dethrone,
and re-pinning would trade a kit whose provenance is already verified by `champion_check` for one that
would have to earn the same verification to sit at the same score. `longseat` is recorded here as an equal,
not promoted.

**Total against the original baseline: 0.925 → 0.930, +0.005 over six entries.** Unchanged by this entry,
which is the point of writing it up as a tie.

#### Slots 7 and 8 `crossenc` and `maxdiv` — pre-registered, fired against our own ruler

**WRITTEN BEFORE THE SCORES.** Slot 6's conclusion was that *the mounted pool is exhausted as a source of
board movement*: the largest seat move the pool contained (+0.0012, six times the runner-up over all 56
same-split checkpoints) bought zero ticks. Every entry since `chosen_v2` has been the offline maximum over
that pool, and the offline maximum has stopped paying. These two entries stop optimising the ruler and
start testing it.

**THE ONE THING THE RULER IS KNOWN TO BE WRONG ABOUT.** Collapsing three members into a single pixel group
cost **0.0059** on the 909-study fold-0 holdout and **0.014** on the board — our own holdout under-prices
DIVERSITY by 2.4x (knee-nas5, knee-7d1). Every offline number below is produced by that same biased ruler,
so a small negative reading on a member whose value IS its diversity is not independent evidence against
it; it is the bias, quoted.

| entry | members | offline joint holdout | vs `longseat` 0.8691 | vs pin `greedy4_v1` 0.8679 |
|---|---|---|---|---|
| `crossenc` | `longseat` four + `dinov3_base_lora` | 0.8678 | −0.0013 | −0.0001 |
| `maxdiv` | that five + `crop80_tissue` | 0.8668 | −0.0023 | −0.0011 |

Both costs are inside the 0.0062 three-seed noise floor — a fifth and a third of it. `dinov3_base_lora` is
the pool's only DINOv3-**base** member and the first non-DINOv2 checkpoint any entry has ever named; all
seven scored entries are 100% DINOv2-small. `crop80_tissue` is the only member reading a different
physical crop. The entries are **nested**, so their difference isolates the crop axis and `crossenc`
isolates the encoder axis.

**COUNT IS NOT THE VARIABLE.** Greedy forward selection over the seventeen mounted members reads
0.8691 / 0.8692 / 0.8692 / 0.8691 at four / five / six / seven, flat to a seventh of the floor. Moving from
four to six costs nothing by itself, so all of the −0.0023 is attributable to WHICH members, not HOW MANY.
The FOUR MEMBERS principle in `champion.json` is untouched by these entries and is not what they test.

**PRE-REGISTERED, REFUTATION FIRST.**
- **≤0.929 on both** refutes the diversity-mispricing hypothesis at two doses and two axes at once, and is
  the outcome these entries exist to make available. It would promote slot 6's "exhausted" from a claim
  about MOVES to a claim about the POOL, close the DINOv3 line as a member source (four arms, never
  shipped), and make every future entry greedy-optimal by rule.
- **≥0.932 on either** says diversity is a board lever the offline ruler cannot see, and the roster's whole
  selection procedure has been fitted to the wrong objective. The next card-time then goes to a second
  encoder trained properly, not to another DINOv2 recipe.
- **0.930–0.931 on both** is the reporting resolution around a tie and still buys something: it says a WEAK
  member of a different family costs nothing, which is the argument for spending the card on a STRONG one.
- **A split result attributes by nesting**: `crossenc` at/above 0.930 with `maxdiv` below 0.929 puts the
  loss on `crop80_tissue` and leaves the encoder verdict intact; the reverse ordering isolates the encoder.

**THE CHAMPION PIN DOES NOT MOVE ON EITHER RESULT UNLESS ONE CLEARS 0.930.** A tie does not dethrone.

**THE RESULT — BOTH REFUTED, AND THE REFUTATION IS THE CLEANEST THIS LEDGER HOLDS.**

| entry | members | offline | board | vs `longseat` 0.930 |
|---|---|---|---|---|
| `longseat` | 4 | 0.8691 | **0.930** | — |
| `crossenc` | 5 (+`dinov3_base_lora`) | 0.8678 | **0.926** | −0.004 |
| `maxdiv` | 6 (+`crop80_tissue`) | 0.8668 | **0.925** | −0.005 |

The pre-registered ≤0.929 clause fired on both, at both doses, on both axes. **The diversity-mispricing
hypothesis is refuted.** The 2.4× under-pricing measured on the decode axis (knee-nas5, knee-7d1) does not
extend past it, and no recombination of the mounted pool is worth a further slot.

**THE SIGN IS THE PART WORTH KEEPING, AND IT IS THE OPPOSITE OF THE ARGUMENT THAT FIRED THESE.** The board
ordering is monotone in the offline ordering — 0.8691 / 0.8678 / 0.8668 maps to 0.930 / 0.926 / 0.925 — so
the holdout ranked these three entries correctly. What it got wrong is the MAGNITUDE, and in the direction
that makes it dangerous rather than conservative: −0.0013 offline became −0.004 on the board, −0.0023
became −0.005, a **~3× amplification of HARM**. The bias in this ruler is that it *understates what a weak
member costs*, not that it understates what a diverse one is worth. Every entry fired on "the offline
delta is inside the noise floor so the diversity is free" was fired on a misreading of the one measurement
that motivated it.

**THESE ARE THE FIRST TWO ENTRIES TO BREAK THE 0.927–0.930 BAND.** Eight scored entries had sat inside
0.003 of each other across wildly different member sets, and that band was being read as the price of
recombination — a floor. It is not a floor. It is the range spanned by recombinations of *members that are
individually strong*. `dinov3_base_lora` scores 0.8268 alone, 0.032 below the pool's best, and adding it
cost four ticks; adding `crop80_tissue` (0.8345) cost one more. **Member quality is priced on this board
and the offline holdout hides three quarters of the bill.**

**WHAT THIS CLOSES.** The DINOv3 line as a member source: four arms, never shipped, and now measured as
strictly negative on the board. Adding members to the champion four on a "free because inside the floor"
argument. The `crop80` crop axis at this quality level. **WHAT IT DOES NOT CLOSE**: whether a STRONG member
of a different encoder helps — both pre-registrations named that as the surviving question, and the
monotonicity here is what makes it answerable rather than idle, because it says the board reads member
quality faithfully enough to reward it.

**The champion pin does not move.** `greedy4_v1` at 0.930 stands.

#### Slot 9 `basedense` — a STRONG member of a different encoder, and the un-confounding of count from quality

**WRITTEN BEFORE THE SCORE, AND AFTER SLOTS 7 AND 8 HAD RETURNED.** `longseat`'s four plus
`base_dense_s2026` — DINOv2-**base**, 768 wide against small's 384 — as a fifth. Same RankAverage vote,
same `overlap=False`, same four incumbents, so this is a one-member delta off a known 0.930.

| entry | members | offline | board |
|---|---|---|---|
| `longseat` | 4 strong | 0.8691 | 0.930 |
| `crossenc` | 5 (weak fifth, 0.8268) | 0.8678 | 0.926 |
| `maxdiv` | 6 (weak fifth + weak sixth, 0.8345) | 0.8668 | 0.925 |
| **`basedense`** | **5 (strong fifth, 0.8512)** | **0.8690** | ? |

**THE CONFOUND THIS ENTRY EXISTS TO BREAK.** Every entry that has ever scored at/above 0.927 named FOUR
members. The only two entries with five and six members are the two that just lost four and five ticks —
and both of them added members that are individually WEAK. So *member count* and *member quality* are
perfectly confounded in everything this board has told us. This entry holds the count at five and moves
the quality: 0.8512 against 0.8268, taken **seventh** by greedy forward selection over eighteen members
where `dinov3_base_lora` is taken sixteenth, and sitting inside the flat top of the greedy curve
(0.8691 / 0.8692 / 0.8692 / 0.8692 at four through seven).

**THE PREDICTION IS A TIE AND IT IS STATED AS ONE.** Under the rule slots 7 and 8 just established — the
holdout ranks correctly and multiplies harm by about three — this entry's −0.0001 offline reading predicts
about −0.0003 on the board, which is under the resolution. Nothing in the mounted pool clears the +0.0015
the board actually resolves; the greedy ceiling over all eighteen members is 0.8692 against the incumbent
0.8691. **This is the last recombination that answers a question rather than re-shuffles**, and it is
fired for the answer, not for the tick.

- **0.930–0.931 (expected)** separates quality from count: a strong fifth holds the board where a weak
  fifth drops it four ticks. Adding members is then not intrinsically priced, and DINOv2-base at effective
  batch 12 becomes the next thing worth card-time.
- **0.932 or above** says encoder capacity is a board lever this holdout cannot see, and the roster has
  been selected against the wrong objective.
- **0.929 or below** says a fifth member is priced regardless of quality, which retires ADDING entirely
  and leaves only swaps and new recipes — the strongest possible closure of the recombination line.
- **THE CONFOUND, NAMED FIRST.** This member trained at `--batch 6` (336 px × 20 slices × 768 wide does
  not fit in 32 GB at 12). `batch_teardown` is on the card measuring `band_dense`'s recipe at batch 6 and
  at batch 6 with `--accumulate 2`. If batch 6 costs materially on its own, a 0.929-or-below reading here
  is a batch verdict, not an encoder verdict.

**The champion pin does not move unless this clears 0.930.**

##### Slot 9 result: **0.931** — a new champion, count is not what was priced, and the ruler misses UP for the first time

`basedense` (55654130) returned **0.931**. It clears the pin, so `comp/champion.json` moves to `basedense_v1`
(kernel `kneemri-basedense`, kit `e3c884e`, the five members read from the kernel's own `chosen` tuple).

**THE PRE-REGISTERED CLAUSE RESOLVES ON ITS UPPER EDGE.** The band written before the score was
*"0.930–0.931 (expected) separates quality from count"*, and 0.931 is the top of it. A **strong** fifth member
holds the board and takes a tick where a **weak** fifth dropped four. So:

- **The `FOUR MEMBERS` principle every champion since 2026-08-13 carried is REFUTED**, and it was never
  measured — it was read off a board where count and quality were perfectly confounded. `crossenc` and
  `maxdiv` did not lose four and five ticks for having five and six seats; they lost them for the members
  they put in those seats (0.8268 and 0.8345, taken 16th and 15th by greedy). The champion pin now records
  the refutation in place of the principle rather than dropping it.
- **`0.932 or above` did not fire**, so encoder capacity is not a board lever this holdout is blind to. One
  tick is this board's reporting resolution; 0.931 over 0.930 is *better*, not *much better*.
- **`0.929 or below` did not fire**, so adding is not retired. The recombination line stays open — but the
  greedy ceiling over all eighteen mounted members is 0.8692 against this entry's 0.8690, so what is open is
  a door onto an empty room. **New members, not new subsets.**

**THE NAMED CONFOUND WAS MEASURED, AND IT CLEARED.** This entry's own pre-registration warned that
`base_dense` trained at `--batch 6` for VRAM, so a low reading would be a batch verdict rather than an
encoder verdict. `batch_teardown` arm 1 (`small_batch6`) ran the control's recipe at batch 6 on the encoder
where 12 fits: peak **0.8598** at epoch 8 against `band_dense`'s **0.8578** at epoch 8, **+0.0020** — inside
the 0.0062 floor and on the *positive* side. The batch is not a lever at this operating point. `base_dense`'s
−0.0066 is an **encoder** reading, and 0.931 is not a batch artefact. (Full curve in `experiments/README.md`.)

**THE OFFLINE RULER MISSED, AND IN THE DIRECTION IT HAD NOT MISSED BEFORE.** Offline the five read 0.8690
against the four's 0.8691 and the prediction, stated plainly before the run, was a tie shading −0.0003. The
board paid **+0.001**. Seven pairs now exist:

| offline | board |
|---|---|
| 0.8648 | 0.928 |
| 0.8653 | 0.928 |
| 0.8661 | 0.927 |
| 0.8670 | 0.929 |
| 0.8679 | 0.930 |
| 0.8691 | 0.930 |
| **0.8690** | **0.931** |

The monotone reading holds across the whole spread and breaks only on the last pair, where a **0.0001**
offline separation inverts. That is not a failure of the ranking claim — it is the resolution limit of a
909-study holdout being reached. The operating rule tightens rather than reverses: **the holdout orders the
field, does not resolve under ~0.001, and is least trustworthy precisely where a member is offline-flat but
structurally different from the incumbents** — which is the only kind of member worth adding now.

**WHAT THIS BUYS THE NEXT ARM.** The strongest single fact in the entry is that a member which *failed its
own matched control outside the noise floor* took the board up a tick. `base_dense` is the only DINOv2-base
in a pool seven small members deep. Encoder *diversity* — not encoder *capacity*, which 0.932 would have
shown and did not — is the axis that paid. The follow-on is another member from an encoder family this pool
does not contain, judged on seat value and never retired on its own control.

#### 2026-08-21 — slots 1 and 2: `sixstrong` (55677573) and `sixfresh` (55677575)

Both are the champion `basedense` five plus ONE strong sixth, same RankAverage vote, same
`overlap=False`, nothing else moved. They differ in one property of the sixth: whether its recipe is
already seated. Fired in the same hour and read together, because neither alone separates "six is past
the useful count" from "the sixth was redundant".

| entry | sixth | solo | recipe distance | offline delta | **board** |
|---|---|---|---|---|---|
| `sixstrong` | `band_span_s2026` | 0.8565 | near-twin of incumbent `calib_span` | **+0.0002** | **0.930** |
| `sixfresh` | `lr4e-5_ws_s2026` | 0.8525 | shares no recipe with the five | **−0.0003** | **0.931** |

##### THEY SPLIT, AND THE SPLIT CLAUSE WAS PRE-REGISTERED

`sixfresh`'s own docstring, written before either score existed: *"If they SPLIT, the count is not the cap
and redundancy is: whichever sixth is the near-twin of an incumbent is the one that lost, and the next
entry is a seventh chosen for recipe distance rather than for solo score."* That is exactly what
happened, and in the predicted direction — the twin lost the tick, the stranger held the board maximum.

- **MEMBER COUNT IS NOT THE CAP.** Six members ties the champion five at 0.931 when the sixth is
  recipe-distant. `crossenc` (0.926) and `maxdiv` (0.925) did not lose four and five ticks for having
  five and six seats; combined with `basedense`'s refutation of the FOUR-MEMBERS principle, the count
  axis is now closed in both directions and no future entry should spend a slot on it.
- **REDUNDANCY IS THE CAP.** `band_span` is the pool's third-best solo checkpoint, 0.0040 ABOVE the
  sixth that tied — and it is the one that cost a tick, because `calib_span` (an incumbent) is its recipe
  with calibrated targets. A seat is priced by what it adds to the mix, not by what it scores alone.
- **NEITHER ENTRY BEAT 0.931**, so the champion pin does not move. The lane reopened is *how to choose*
  the next member, not *how many*.

##### THE OFFLINE RULER INVERTED FOR THE THIRD TIME, AND NOW THE MECHANISM IS NAMED

The holdout ranked `sixstrong` ABOVE `sixfresh` (+0.0002 against −0.0003) and the board ranked them the
other way. That is the third inversion at this margin — the champion itself scored 0.931 at offline
0.8690, below `longseat`'s 0.8691 which scored 0.930 — and the three together stop being scatter:

**RANK-AVERAGING A NEAR-TWIN BARELY MOVES A HOLDOUT RANKING, WHICH IS EXACTLY WHY THE HOLDOUT CANNOT SEE
REDUNDANCY.** Two members that agree on the ordering contribute one ordering twice; the rank average
shifts almost not at all, so the offline delta reads ~0 whether the sixth is a twin or a stranger. The
board, scored on different studies, pays for the disagreement the twin does not supply. The blindness is
not noise at the resolution limit — it is structural, and it points the wrong way *by construction* on
the one axis that now decides a seat.

Operating rule, tightened: **the holdout orders members by QUALITY and is blind to DISTANCE.** Use it to
reject a weak candidate; never to rank two candidates of comparable solo score against each other. That
is a ruler question, not a noise question, and no number of seeds fixes it.

##### WHAT THIS BUYS THE NEXT ARM

The pool holds seven small members, one base, and every one of them was fitted by this repo's own recipe
family on the same pixels with the same objective. Recipe distance inside that family is nearly
exhausted — `lr4e-5_ws` differs by a band, a rate and a pixel count, and that was enough for one tick.
The largest distance still available is not another recipe: it is a member whose ENCODER never saw this
family at all. That is `knee-jucq`'s volumetric trunk (V-JEPA 2, wired in `e291a4b`), which differs in
pretraining domain, architecture family and depth handling simultaneously — and it is now backed by a
measurement rather than by an argument about capacity, which is the argument `basedense`'s 0.932 clause
already refuted.

#### 2026-08-21 — slots 3 and 4: `sevengrade` (55678597) and `sixfresh` v2

Both submitted at about 23:20 UTC; a code submission stays `PENDING` for hours (the ledger's own measured
case is ~8 h), so neither scored the same day. Recorded here with the pre-registration so the readings
are resolved against what was written before them, not after.

| entry | seventh / change | what it asks | pre-registered reading |
|---|---|---|---|
| `kneemri-sevengrade` | `sixfresh`'s exact six + `grade_ladder_s2026` (solo 0.8565, offline **+0.0001**) | does DISTANCE keep paying past six, when the distance is in the OBJECTIVE? | **≥0.932** distance stacks, V-JEPA enters as an eighth · **0.931** ties, future members must DISPLACE · **≤0.930** seven is past the cap, count reopens |
| `kneemri-sixfresh` v2 | nothing — identical code, identical members, **refreshed kit** (`cf6a99f`, was `e3c884e`) | is the kit refresh transparent to serving? | **0.931** reproduced = transparent, `through_dense_s2026` and every V-JEPA member become shippable · anything else = the refresh moved serving, measured on a KNOWN entry |

`grade_ladder` is the seventh because the seated six vary the encoder, the band, the rate and the pixel
count and **not one of them varies the loss** — the severity ladder is the only axis with distance left
inside this recipe family. The offline +0.0001 is stated and set aside deliberately: the section above
this one establishes that the holdout is structurally blind to distance, so a near-zero offline delta is
uninformative about a seat, in either direction.

##### THE KIT-STALENESS TRAP, and why it cost a slot to close

The seventh was chosen on merit as `through_dense_s2026` and re-membered before shipping. `kit_check
--mounted` — the only gate that asks Kaggle what it will actually run — returned exit 1: the mounted
`dimiturnt/kneemri-kit` sat at `e3c884e`, **9 commits behind HEAD in the 30 files the kernel loads**, and
one of the nine (`ba76fd1 through_plane`) *implements the read that member was fitted with*. Mounting it
was a hard failure on the card, not a degraded score.

The fix is the trap. Refreshing the kit rewrites `model.py` (198 lines), `kernel_run`, `live_dataset` and
`study_cache` **under every member of the entry at once** — so the first entry shipped on a fresh kit
carries two variables, and a bad reading could not be attributed. Hence the split: `sevengrade` ships
against the SAME kit that produced the 0.931 (no severity or ladder commit is among the nine, so
`grade_ladder` needs none of them), and the kit refresh is validated ALONE, on an entry whose score is
already known. A slot spent on a control rather than on a candidate, because the control unblocks
`through_dense` and every V-JEPA member that follows it.

Named confound on `sevengrade`, recorded before the score: the ladder has never scored on the board, so a
low reading admits the reading "the objective does not transfer" as well as "seven is past the cap".
Those two separate only under the `through_dense` follow-up, which the kit refresh gates.

##### SLOT 5 WAS NOT SPENT

Every question ready to ship was gated on one of the two pending readings. The de-confounded
`through_dense` entry needs `sixfresh` v2 to read 0.931 first, and the required gates plus a ~9 min kernel
run did not fit the 26 minutes left before the slots expired. An expiring slot is worth zero; a slot spent
on an uninterpretable entry is worth less than zero, because it also spends the next day's comparison.

#### 2026-08-22 — a third ruler, and a prediction registered before its score arrived

`kneemri.evaluation.member_distance` exists because the 2026-08-21 split showed both existing rulers
pointing the wrong way on the axis that decided a seat. It reads NO labels: per-label rank correlation
(the same `Vote.ranks` the shipped vote applies) between a candidate and each seated member, summarised
as NEAREST (the twin detector — a seat is lost to one redundant neighbour, not to an average) and MEAN.

Against the pinned champion five, with the roster's own nearest-neighbour spread as the scale
(0.8801 - 0.8911):

| candidate | nearest | mean | seat_value | board |
|---|---|---|---|---|
| `through_dense_s2026` | 0.8922 | 0.8740 | +0.0008 (top of 24) | — |
| `lr4e-5_ws_s2026` | 0.9143 | 0.8816 | **-0.0003** | **0.931** |
| `band_span_s2026` | 0.9188 | 0.8925 | **+0.0002** | **0.930** |
| `grade_ladder_s2026` | 0.9248 | 0.8923 | +0.0001 | PENDING |

THE ORACLE IT HAD TO REPRODUCE, and did: `lr4e-5_ws` reads more distant than `band_span` on both
summaries, which is the ordering the board paid out and the OPPOSITE of the ordering `seat_value` and
solo score both gave. Reproducing the case a tool was designed from is not evidence, and this is stated
as a sanity check rather than a validation.

SO HERE IS THE PART THAT IS NOT CIRCULAR. `grade_ladder` reads the LEAST distant of all four — 0.9248
nearest, further from the roster's own spread than any other candidate — and `sevengrade` (55678597) is
still PENDING as this is written. If distance is what buys a seat, that entry should NOT exceed 0.931.
Registered now, before the score exists, because a prediction written after the fact costs nothing and
proves nothing. Note the confound already recorded above: `sevengrade` is a SEVENTH seat and also the
first entry carrying the severity objective, so a null there has two available explanations and only a
BEAT would cleanly refute this ruler.

AND THE POOL IS MEASURABLY EXHAUSTED FOR DISTANCE, which is the reading that matters for planning rather
than for one slot. Every candidate except `through_dense` sits CLOSER to some seated member (0.9143 -
0.9248) than the roster's own worst neighbour pair (0.8911). Recipe distance inside this family is spent:
band, rate, resolution, schedule and calibration all order the studies about the same way. That is the
independent argument for the V-JEPA 2 probe (knee-jucq) — a trunk whose pretraining never saw this
family is the only large distance left to buy, and this ruler now says so without appealing to the
board split at all.

WHAT IT IS NOT. It ranks; it does not price. Nothing measures how many board ticks a unit of distance
buys, one paired observation constrains it, and no slot should be spent on this number alone.

##### The full-pool sweep, and the rule it forces: distance is a TIE-BREAKER, not a ranking

All 119 same-split checkpoints run through both rulers at once (`logs/distance_sweep.log`; 8 smoke/other-fold
checkpoints were REFUSED by `RecordMix.shared` rather than silently scored on 11 shared studies, which is
the harness working).

THE FIRST THING IT SAYS IS THAT DISTANCE ALONE IS USELESS, and it says it loudly enough to be worth the
sweep on its own. The most distant candidates in the pool are the `encoder_adapt_*` family at nearest
0.5050-0.6248 — and their seat values are -0.0037 to -0.0044, the WORST in the pool. They are distant
because their orderings are close to broken, not because they carry new information. Across the extremes
the two rulers are strongly ANTI-correlated, so anything that ranked members by distance would have
proposed the four worst members available. Recorded because it is the failure mode this ruler will always
have: a coin flip is maximally distant from everybody.

SO THE USABLE RULE IS THE JOINT ONE, and the sweep is what makes it concrete. `seat_value` has a declared
blind band — below `BOARD_RESOLVABLE` = 0.0015 it "cannot tell the candidate from the incumbent", and 0
of 57 priced candidates clear it. Inside that band, where the quality ruler is by its own admission
silent, distance varies from 0.8428 to 0.9248. That is the whole content of the 2026-08-21 split:
`lr4e-5_ws` (-0.0003) and `band_span` (+0.0002) are both inside the blind band, and distance ordered them
the way the board paid. Use distance ONLY to break a tie the quality ruler has already declared, never to
rank across it.

THE SHORTLIST THAT FALLS OUT, non-negative seat value ordered by distance:

| candidate | seat value | nearest | mean |
|---|---|---|---|
| `deep24_s2026` | +0.0000 | 0.8428 | 0.8262 |
| `crop80_span_s2026` | +0.0005 | 0.8717 | 0.8508 |
| `attend_true_s2026` | +0.0004 | 0.8860 | 0.8583 |
| `aug_long_s2026` | +0.0005 | 0.8804 | 0.8667 |
| `through_dense_s2026` | +0.0008 | 0.8922 | 0.8740 |
| `grade_ladder_s2026` | +0.0001 | 0.9248 | 0.8923 |

`crop80_span` and `deep24` sit BELOW the roster's own best neighbour pair (0.8801), which is the scale
line that says a candidate brings an ordering the roster does not already hold twice. Neither has ever
been seated.

AND IT UNSETTLES THE GATED ENTRY, which is why it is written down before that slot is spent.
`kneemri-sevenplane` seats `through_dense` on the strength of its +0.0008 — the top seat value of
twenty-four. But +0.0008 is inside the blind band, so that ranking is one `seat_value` cannot support,
and on the axis that decided the last split `through_dense` is LESS distant than both `crop80_span` and
`deep24`. This does NOT retract the entry: `through_dense` still has the mechanism argument no other
candidate has (it is the only member whose input TENSOR differs), and the ruler proposing the change is
unvalidated. It is recorded as a live alternative to be settled when `sevengrade` reads, since that
reading is this ruler's own out-of-sample test.

##### THE READINGS, 2026-08-22 00:48 UTC — both entries resolved, both on their middle branch

| entry | pre-registered | scored | branch taken |
|---|---|---|---|
| `sixfresh` v2 (55678708) | 0.931 = the kit refresh is transparent | **0.931** | reproduced exactly |
| `sevengrade` (55678597) | 0.931 = seven ties, future members must DISPLACE | **0.931** | the middle one |

THE KIT REFRESH IS TRANSPARENT TO SERVING, measured on an entry whose score was already known. `cf6a99f`
rewrites `model.py` (198 lines), `kernel_run`, `live_dataset` and `study_cache` under every member at once,
and the same six members on the new kit return the same 0.931. That was the gate: `through_dense_s2026`
and every V-JEPA member are now shippable, and a future bad reading on a fresh-kit entry may be attributed
to the candidate rather than to the kit. This is the slot doing exactly what it was spent on.

AND THE SEAT COUNT IS NOW SATURATED FROM A THIRD DIRECTION. Five seats scored 0.925, six scored 0.931 and
0.930, seven scores 0.931. The seventh was not a weak pick — `grade_ladder` carries the ONLY axis no
seated member varies (the objective), which is the largest distance the family had left to offer — and it
bought nothing. Adding members is finished as a lever; the pre-registration's own words for the branch
are the operative rule from here: **future members must DISPLACE, not join.**

WHAT THE DISTANCE RULER ACTUALLY EARNED HERE, stated at the strength the evidence supports and no higher.
Its registered prediction was that the LEAST distant candidate of the four should NOT exceed 0.931, and it
did not. That is a survived prediction, not a validation, and the section above said so before the score
existed: a tie was available to two other explanations (a seventh seat is past the cap; the severity
objective does not transfer to the board), and only a BEAT would have cleanly refuted the ruler. So the
ruler is still unvalidated as a PREDICTOR and is still confined to breaking ties `seat_value` has already
declared. It has now made one out-of-sample call and not been wrong, which is the whole claim.

AND THE SEVERITY AXIS HAS FINALLY SCORED. `grade_ladder` is the first member carrying the severity
objective ever mounted on the board, and it ties. That does not kill the axis: it was seated as a SEVENTH
member on a count that is saturated, so the entry cannot separate "the objective adds nothing" from "no
seventh seat adds anything" — and the second is independently established by this very table. What it
does rule out is the cheap version of the hypothesis, that severity buys a board tick simply by being
mounted. A displacement entry is what would read it clean.

CONSEQUENCE FOR `kneemri-sevenplane`, which was gated on this reading. It is a SEVENTH SEAT, and the
reading above says a seventh seat is worth 0.000 regardless of which member fills it. Shipping it now
would spend a slot to re-measure the axis this entry just closed. It is therefore **withdrawn as an
addition and held as a displacement**: `through_dense` retains the mechanism argument no other candidate
has (it is the only member whose input TENSOR differs), so the question worth a slot is whether it
displaces a seated member at six, not whether it joins them at seven.

## `sevenplane` — 0.932, and it refutes the sentence directly above it (2026-08-22)

Submission 55681546. `sixfresh`'s exact six plus `through_dense_s2026`, same kit (`cf6a99f`), one
variable. **Public 0.932** — the first time any seventh seat has moved the board, and a new best.

**WHAT IT REFUTES, and it is this file's own words.** The `sevengrade` section above concluded that "a
seventh seat is worth 0.000 regardless of which member fills it", and on that basis withdrew `sevenplane`
as an addition. That conclusion generalised from ONE seventh member to ALL of them, and the generalisation
is wrong: which distance the seventh is bought on decides whether it pays.

| entry | the seventh, and what makes it distant | public |
|---|---|---|
| `sixstrong` | `band_span`, a near-twin of a seated member | 0.930 |
| `sixfresh` | `lr4e-5_ws`, shares no recipe with the five | 0.931 |
| `sixfresh v2` | identical to `sixfresh`, refreshed kit only | 0.931 |
| `sevengrade` | `grade_ladder` — distant in the OBJECTIVE (severity loss) | 0.931 |
| `sevenplane` | `through_dense` — distant in the INPUT PIXELS | **0.932** |

Objective distance ties. Pixel distance ticks. That is the cleanest separation this board has produced,
because the six seated members, the kit and the blend are held fixed across the last two rows and only the
seventh member's TYPE of distance differs.

**THE SIZE, STATED HONESTLY BEFORE ANYONE BUILDS ON IT.** +0.001 is one board increment and roughly a sixth
of the 0.0062 holdout floor; against the 1.5% house bar it is nothing, and against the 0.054 frontier gap
it is noise. This is an INSTRUMENT reading that discriminates two axes, NOT a result, and it must not be
planned around as though the axis were worth a percent. What it licenses is a direction — the decode-group
argument in `knee-7d1` (2.4x transfer) is the thing pixel-distance points at — not a number.

**IT ALSO PARTLY RE-OPENS THE SEVERITY READ.** `sevengrade`'s tie was explained two ways and this entry
removes one of them: "no seventh seat adds anything" is now false, so `grade_ladder`'s tie is evidence
about the SEVERITY OBJECTIVE's transfer to the board rather than about the count. That is a real
narrowing, and it makes the displacement entry the severity axis has been owed more informative than it
was an hour ago, not less.

## The eighth seat is not buyable from the mounted pool, and the reason is one sentence (2026-08-22)

No submission. This section is what a slot was NOT spent on, and why that is a finding rather than a
pause.

**FIRST, THE PIN WAS THREE HOURS STALE.** `comp/champion.json` still named `basedense_v1` (0.931, five
members) while `sevenplane` had held 0.932 since 02:45 UTC. `seat_value` reads its roster from that pin
precisely so no arm header can hold a stale copy — so every candidate priced this morning would have been
priced against a champion that no longer holds the board. `champion_check` passed throughout: it asserts
the pinned entry can be REBUILT, and it cannot know the entry is no longer the best one. Pin corrected,
principles carried forward, and the gap itself recorded there as a principle.

**THEN, BOTH RULERS OVER ALL SIXTEEN MOUNTED-BUT-UNSEATED MEMBERS**, against the true seven (0.8695):

| ruler | best candidate | reading |
|---|---|---|
| seat value, as an eighth | `aug_long` | +0.0004 |
| seat value, as a displacement | `aug_long` for `lr4e-5_ws` | +0.0005 |
| distance (nearest-neighbour) | `lora_r8_v3small_224` | 0.8235 |

**0 of 16 clear +0.0015**, the threshold under which this ruler has never picked a board tick, and the
best move is a third of it. That alone only repeats what 2026-08-21 said about a five-member champion.

**WHAT IS NEW IS THAT THE TWO RULERS ARE ANTI-CORRELATED HERE, AND THAT IS THE BOTTLENECK.** Ordered by
distance the four most distant candidates are `lora_r8_v3small_224` (0.8235), `dinov3_base_lora` (0.8286),
`crop80_tissue` (0.8422) and `deep24` (0.8428) — all far outside the seated pool's own internal spread
(0.8801–0.9143). Ordered by seat value those same four are the BOTTOM four (-0.0009, -0.0006, -0.0004,
+0.0001). Every mounted candidate is either **strong and a twin** or **distant and weak**, and the board
has already priced both halves: `sixstrong` seated a strong twin and lost a tick, `crossenc` and `maxdiv`
seated distant weak members (0.8268, 0.8345) and lost four and five. Under the 3x harm multiple an eighth
seat from this pool prices at roughly -0.002. It is a NEGATIVE-expectation slot, not a coin flip.

**THE ONE MEMBER THAT WAS BOTH IS THE ONE THAT WON.** `through_dense` sits at 0.8558 solo with a nearest
neighbour of 0.8922 — distant enough to sit at the loose end of the seated seven, strong enough not to be
charged for. That is the whole of what `sevenplane`'s +0.001 identifies, and it is a tighter statement
than "pixel distance pays".

**AND THE CORRELATION RULER DOES NOT EXPLAIN THE BOARD BY ITSELF** — stated because the previous section
leaned on distance. Across the three tested sevenths the ordering is not monotone: `through_dense` 0.8922
→ 0.932, `band_span` 0.9188 → 0.930, `grade_ladder` 0.9248 → 0.931. `grade_ladder` is the LEAST distant of
all sixteen candidates by this metric and it beat the more distant `band_span`. So what `sevenplane`
measured was distance in the MECHANISM (a different input tensor), and label-free correlation is at best a
proxy for it on three points. Do not promote the column to a predictor.

**CONSEQUENCE, AND IT IS A TARGET RATHER THAN A VERDICT.** The remaining slots are not held out of
caution; there is no permutation of what is mounted that is worth one. What buys the next tick is a
checkpoint that is BOTH — nearest neighbour under ~0.88 and solo at or above ~0.85 — and no arm has ever
been aimed at that pair. `vjepa_member` is the only one in flight that could be: a video-pretrained trunk
is the largest available step in what the encoder was fitted on, and its pre-registered pass bar (0.845)
is within a whisker of the strength half. Its distance is unmeasured and measurable the moment it lands.

### Correction, same morning: 79% of that distance column is solo score

The section above ordered the pool by nearest-neighbour correlation and concluded no permutation was worth
a slot. The ordering was confounded and the conclusion with it.

Across all 23 same-split checkpoints, **pearson(nearest, solo) = 0.890**. A weaker member's ordering is
noisier and therefore decorrelates from everything, so roughly four fifths of the distance column's
variance is quality wearing distance's clothes. `MemberDistance`'s docstring argues it cannot restate a
quality reading because it never opens the label table — that is true of the truth vector and not of the
predictions, and the leak arrives through them.

Regressing it out — `nearest = 2.270 * solo - 1.038`, residual sd 0.0133 — gives distance FOR STRENGTH,
and the pool reorders sharply:

| | raw nearest | residual |
|---|---|---|
| most redundant seated | `depth_regions_narrow` 0.9143 | `lr4e-5_ws` **+0.0173** |
| most distant unseated | `lora_r8_v3small_224` 0.8235 | `crop80_tissue` -0.0140, `aug_long` -0.0134 |

`lr4e-5_ws` and `depth_regions_narrow` read the same 0.9143 raw and are +0.0173 and +0.0107 residual. The
four members the raw column called most distant — the two DINOv3 members, `crop80_tissue`, `deep24` — are
also the four weakest, and three of them move to the middle once strength is removed. `dinov3_base_lora`
drops from most-distant-but-one to -0.0101, ordinary.

**WHAT THIS DOES NOT DO IS RESCUE THE RULER.** Neither form orders the three tested sevenths: raw gives
`through_dense` 0.8922 → 0.932, `band_span` 0.9188 → 0.930, `grade_ladder` 0.9248 → 0.931, and residual
gives -0.0123, +0.0127, +0.0187 for the same board scores. `grade_ladder` beats the more distant
`band_span` under both. Worse for the correction specifically: the ONE paired observation `MemberDistance`
was built to reproduce is `lr4e-5_ws` more distant than `band_span` (0.931 against 0.930), and the
residual **inverts** it (+0.0173 against +0.0127). So correcting the confound costs the ruler its only
supporting case. Both are unvalidated; they disagree; one slot separates them.

**AND IT DOES CHANGE THE CONCLUSION ABOVE.** The claim was that every mounted member is either strong and
a twin or distant and weak. On residual that is false: `aug_long` is -0.0134 at solo 0.8511, and it is
simultaneously the best seat value of all sixteen — as an add (+0.0004) and as a swap for `lr4e-5_ws`
(+0.0005). That is the profile of the member that took 0.932: `through_dense` was -0.0123 at 0.8558 with
+0.0008. `kneemri-twinswap` is built on it — `sevenplane`'s seven with `lr4e-5_ws` out and `aug_long` in,
same kit `cf6a99f`, count held at seven — and it is the first entry that removes a twin rather than adding
a stranger. **Held, not fired**: one slot is spent today already, it is early, and the entry is worth more
once `vjepa_member` has been priced on the same two rulers and the two can be ranked against each other.

### The third ruler breaks the tie, and it kills `twinswap` (2026-08-22)

`ordinal_signal` run over all 23 same-split members — severity concordance across 334,942 cross-level
pairs, labels file never opened. It contradicts the swap the residual correction just recommended:

| | ordinal (labels / pairs) | residual distance | seat value |
|---|---|---|---|
| `lr4e-5_ws` (seated) | 0.748 / 0.770 | +0.0173 | — |
| `aug_long` (challenger) | 0.738 / 0.767 | -0.0134 | +0.0005 |

The incumbent wins by 0.010 over labels, against a seat experiment whose resolvable spread was 0.007 to
0.016. `knee-7d1` carries a pre-registration from 2026-08-20 — take the next single-member swap where the
rulers disagree and decide it by the ordinal one — recorded there as UNRUN because no pool pair was
separated by a margin either ruler could see. It is runnable now, and applied here it says **do not fire
`kneemri-twinswap`**. The kernel stays built and unfired; a pre-registration that only ever confirms is
not one.

**AND THE SCAN FINDS THE PAIR THAT DOES EXECUTE IT.** Ordinal over the seated seven runs `calib_span`
0.759, `band_dense` 0.751, `through_dense` 0.750, `base_dense` 0.748, `lr4e-5_ws` 0.748,
`depth_regions_narrow` 0.743, `band_dense_long` 0.741. The highest-ordinal member NOT seated is
`grade_ladder` at 0.750 / 0.780 — above two of the seven, and above `depth_regions_narrow` by +0.007 over
labels and +0.011 over pairs, both inside what the ruler resolves.

`kneemri-gradeswap` is that displacement: `sevenplane`'s seven with `depth_regions_narrow` out and
`grade_ladder` in, same kit, count held at seven. It settles three things with one slot — the unrun
ordinal forward test, the severity displacement `sevengrade` was owed once `sevenplane` removed its
count confound, and the residual-distance ruler, which calls `grade_ladder` the most redundant member of
all 23 (+0.0187) and therefore predicts this entry LOSES. The three rulers point three ways at the same
seat, which is the cheapest arbitration this board has been offered.

**BOTH KERNELS ARE BUILT AND NEITHER IS FIRED.** One slot is spent today, it is early, and `vjepa_member`
lands around 08:35 UTC — it will be priced on all three rulers and ranked against these two before any
slot is spent.

#### 2026-08-22 — slot 2: `gradeswap` (fired 08:07 UTC, kernel v1)

Pushed as v1 and submitted the same minute; Kaggle reports 3 remaining today. The kernel ran the public
3-study sample in 71 s and projected **7.7 h** over the hidden set, so the score is not due until roughly
15:45 UTC. All seven members loaded and fingerprint-checked; `overlap=False` logged INERT because all
seven are whole-stack, which is the expected no-op rather than a silent drop. Every member asked for bf16
and the T4 cannot run it, so all seven served fp16 — the intended serving precision, logged per member.

**Gates before firing, all four:** `mount_check` 33 ok / 0 refused (gradeswap declares both dinov2
packages it loads); `champion_check` confirms `sevenplane` at 0.932 is still buildable; `witness_check
--device cpu` verified all seven of this entry's members (21 ok / 2 unverified / 0 refused across the
pool, and neither unverified member is seated here); `kit_check --mounted` reports 5 commits behind, which
is deliberate — the kernel imports only `kneemri.kaggle.kernel_run`, which the mounted `cf6a99f` kit
carries, and the five outstanding commits (the `core/data` rename, the paths config, and the three from
2026-08-21) change nothing this kernel reaches.

**WHY THIS FIRED BEFORE `vjepa_warm` LANDED, against the rule written at the top of this session.** The
line above says rank the arm against both built kernels before spending a slot. Two things make that
ordering wrong here rather than merely inconvenient:

1. `gradeswap` is an INSTRUMENT about rulers, not a claim about the trunk, so its read does not depend on
   the arm. It is the other way round — knowing which of the three rulers transfers is an INPUT to pricing
   `vjepa_warm`, which will itself be read on all three. Firing it first makes the arm's own reading
   interpretable; firing it second wastes that.
2. `vjepa_warm` cannot take a slot today regardless. It is a member, not an entry. Seating it means a kit
   rebuild, and the kit is five commits behind, so that is a refresh plus a full re-gate — hours, not
   minutes. Waiting reserved nothing for it.

Three slots remain, which is more than the arm can consume.

The pre-registration is unchanged from the kernel docstring, restated here because it was written before
the number: **above 0.932** confirms the ordinal ruler as the one that transfers and licenses re-ranking
every seat by it; **0.932** is a tie saying a displacement cannot be resolved at seven either; **below
0.932** refutes the ordinal forward test on its first honest trial and leaves the residual-distance ruler
standing — that ruler calls `grade_ladder` the most redundant member of all 23 for its strength (+0.0187)
and therefore predicts this entry LOSES a tick. Whichever way it reads, one ruler stops being quotable.

+0.001 is one board increment against a 1.5% house bar. This arbitrates three rulers; it is not a claimed
improvement and must not be planned around as a number.

**RESULT, 09:34 UTC: 0.933.** Above 0.932, so the pre-registration reads PASS on its own terms: the
ordinal ruler called this displacement correctly and re-ranking seats by it is licensed. A new best,
+0.001 over `sevenplane`.

Read the disagreement, which is the whole point of the instrument — the three rulers pointed three
different ways on ONE entry:

| ruler | what it said about swapping `depth_regions_narrow` -> `grade_ladder` | verdict |
| --- | --- | --- |
| ordinal concordance | grade_ladder ahead by +0.007 over labels, +0.011 over pairs — swap | **right** |
| binary seat value | +0.0001, a fifteenth of its own +0.0015 resolution — no signal | blind |
| residual distance | most redundant of 23 for its strength (+0.0187, pearson 0.890) — predicts a LOSS | **wrong** |

So the residual-distance ruler is now 0-for-2 in OPPOSITE directions on one day: it called `vjepa_warm`
the most distant member this project has ever produced and that member priced +0.0001, and it called
`grade_ladder` maximally redundant and that member bought a board tick. It measures what it claims
(the residuals really are far apart, or close); the INFERENCE from residual geometry to seat value is
what has failed twice. Stop quoting it as a selection rule.

Carry the bead's own caveat rather than dropping it: one trial cannot separate "the ordinal ruler
transfers" from "grade_ladder is simply a good member". A pass licenses re-ranking; it does not prove
the ruler, and the next use of it is a RE-TEST, not an assumption.

The second thing this settles is arithmetic, and it corrects a document filed two hours earlier. The
seat scan priced 63 candidates at +0.0005 or less and `interpretations/kneemri/2026-08-22_the_pool_is_exhausted.md`
concluded the pool was exhausted by comparing those offline numbers against the +0.0015 BOARD threshold.
`gradeswap` refutes that comparison by instance: +0.0001 offline produced +0.001 on the board. Offline
seats must be multiplied by the measured compression (`knee-7d1`, median +4.67 board points per holdout
point) before they meet a board threshold, which puts the best available eighth seat at ~+0.002 board.
The pool is flat, not empty. Both that interpretation and the root README are corrected.

---

## `accumseat` — 0.933, a tie, and the ordinal ruler goes back in the box (2026-08-22 20:42Z, ref 55698943)

`small_accum_s2026` displaces `lr4e-5_ws_s2026`, the champion's weakest ordinal seat at 0.770, for a
member at 0.784. Pre-registered: **≥0.934** the ordinal ruler transfers · **0.933** a tie about the SEAT
rather than the ruler · **≤0.932** it does not transfer.

**RESULT: 0.933.** The tie branch, and the ledger's own caveat is what makes it decisive rather than
merely uninformative. `gradeswap` passed on a +0.011 ordinal gap; this is +0.014, a LARGER separation,
and it bought nothing. A ruler whose two trials do not order by their own separation is not measuring
what it claims, so `experiments/README.md`'s written condition fires: the single pass was the
coincidence.

**All three rulers are now retired as selection rules.**

| ruler | record | status |
| --- | --- | --- |
| residual distance | 0-for-2, in opposite directions | dead since `gradeswap` |
| binary seat value | blind at its own ±0.0015 resolution | never usable at board resolution |
| ordinal concordance | passed at +0.011, tied at +0.014 | **dead as of this entry** |

No further submission slot may be spent on a one-for-one member swap justified by a ruler. What the two
ties DO establish is about the roster: two different displacements of the same seven both read 0.933, so
**the seven-member vote is flat to single-member exchange**. The remaining board budget belongs to the
member COUNT — an eighth seat, which no entry has ever tested — or to a member from an axis the roster
does not contain, and `knee-7d1`'s measured compression (+4.67 board points per holdout point) is the
arithmetic that prices either.

**Two 22.08 slots lapsed unused.** The machine slept from ~22:40Z to ~10:20Z and the UTC day closed with
them unspent. Recorded rather than glossed: the eighth-seat scan that would have chosen an entry was
launched at 22:36Z and died in its first member when the host went down.

## The eighth seat, scanned rather than submitted (2026-08-23 11:28Z, no slot spent)

The `accumseat` entry above named the remaining budget as **member count — an eighth seat, which no
entry has ever tested** — and recorded that the scan which would have chosen that entry died at 22:36Z
when the host went down. It was rerun card-free over all 66 same-split records, champion seven held.

**Flat.** Roster of seven reads 0.8694; the best eighth is `attend_true` at **+0.0004**, with
`attend_dense`, `crop80_span` and `small_batch6` tied behind it, and **59 of 59 candidates sit inside
the +0.0015 this ruler needs before it has ever picked a board tick**. `record_mix` prints the verdict
itself: *"best is +0.0007, inside +0.0015: this ruler cannot tell it from the incumbent."*

At knee-7d1's +4.67 board points per holdout point that prices at roughly **+0.002 board** — real, in
`gradeswap`'s class, and a sixth of the +0.007 that 0.933 → 0.940 needs. **An eighth seat drawn from
the existing pool is not the lever**, and no slot was spent finding that out.

That leaves exactly one of the two named budget items standing: **a member from an axis the roster does
not contain**. `final_fit` — 100% data, `--fold -1` — is on the card as this is written and is that
member: data volume is the only axis that has ever cleared the bar (`half_data`, −0.0234, 3.8× the
noise floor).

Also recorded, because it re-derived a retired conclusion: the ordinal sweep is now a committed command
(`python -m kneemri.evaluation.ordinal_signal --directory <processed> --seated <names>`) rather than a
shell loop over the single-checkpoint CLI, which is how the 23- and 48-member scans were produced. Its
first full run over 80 records reproduced the roster's annotated numbers exactly (`small_accum` 0.784,
`severity_weight` 0.783, `small_batch6` 0.781, weakest seat `lr4e-5_ws` 0.770) and independently
validates the measure — the smokes rank last at 0.41–0.55 and the five collapsed `--encoder-weights`
members sit at the floor (`mae_matched` 0.617, `encoder_adapt*` 0.63–0.67). **The ruler is still retired
as a SELECTION rule** by the entry above; the tool is kept as a diagnostic, not a chooser.

### Pre-registration: `kneemri-greedy7` — the greedy rule re-run on a pool that grew

`small_batch6 + band_dense_long + calib_span + through_dense + aprior_dense + depth_regions_narrow +
calibrated_targets`, offline **0.8712** over the 909-study fold-0 holdout against the champion seven's
**0.8694**. Same count, same `overlap=false`, same `RankAverage`, every member already in
`kneemri-members` — the entry costs a kernel push and no re-upload.

**THIS IS THE STANDING RULE, NOT A NEW BET.** `greedy4` ran this exact search over the then-mounted
sixteen, was pre-registered to land *at or below* 0.928, and came back **0.930 as a new champion** —
refuting the one-swap rule outright. This ledger's own conclusion from that was to *"make every future
entry greedy-optimal by rule."* Since then the mounted pool has grown by five members
(`small_accum`, `severity_weight`, `small_batch6`, `aprior_dense`, `focal_dense`, pushed 2026-08-22),
and the greedy re-run over 28 mounted records is what that rule now says to ship. It is also NOT the
move the `accumseat` entry banned: that ban is on a one-for-one member swap justified by a ruler, and
this moves four of seven seats on the joint metric itself.

**WHOLESALE, NOT A SWAP.** `band_dense`, `grade_ladder`, `base_dense` and `lr4e-5_ws` all leave;
`small_batch6`, `aprior_dense`, `depth_regions_narrow` and `calibrated_targets` all enter. Two entries
have now tied at 0.933 on single swaps of this same seven, which the ledger read as *"the seven-member
vote is flat to single-member exchange"*. Four seats at once is the test that reading cannot answer.

**PREDICTION, written before the run.** +0.0018 offline at knee-7d1's median +4.67 board points per
holdout point prices at about **+0.008**, i.e. 0.941 — which would clear the standing 0.940 target. I do
not expect the median; the compression is a median over a spread and this edge is a quarter of the
0.0062 noise floor.

- **≥0.936** — the greedy rule transfers at full roster width on a grown pool; every future entry is
  chosen this way and the remaining slots go to re-running it as members land.
- **0.934–0.935** — directional, consistent with compression, champion moves.
- **0.933** — a third tie, and now a decisive one: the seven-member vote is flat to WHOLESALE exchange,
  not merely to single-member exchange. That retires roster search entirely and leaves only new axes.
- **≤0.932** — greedy over the grown pool overfits the 909-study holdout. **Named mechanism: selection
  bias.** The search chose these seven by maximising the very number it is scored on, over 28 candidates
  and 909 studies; `greedy4` was chosen the same way over 16 and did transfer, so a loss here says the
  bias grows with the pool, which is a fact worth the slot.

**WHAT IT DOES NOT TEST.** It is silent about data volume: every member is fold-0-trained on 79% of the
corpus. `final_fit` is on the card as this is written and answers that separately.

### Pre-registration: `kneemri-finalfit` — the data-volume step, and the crash that hid its success

`gradeswap`'s exact seven with `band_dense_s2026` out and **`final_ws_e12`** in. Count, kit (`cf6a99f`),
vote, serving config and the other six identical. `final_ws_e12` is `band_dense`'s flags copied verbatim
plus `--fold -1` and `--ema`: 4,406 training studies rather than 3,497. **Training data volume, 79% ->
100%, is the only design difference.**

**WHY THIS AXIS.** Data volume is the only axis that has ever cleared the 1.5% house bar here —
`half_data` 0.8344 against `band_dense` 0.8578, a -0.0234 gap at 3.8x the offline noise floor. The curve
is CONCAVE, so the 80->100% step is 0.32 of a doubling and prices at about +0.0029 offline before the
last-epoch penalty.

**TWO VARIABLES, NOT ONE.** `--ema` is not removable: with `--fold -1` there is no best-epoch selection
and the trainer keeps the LAST epoch, so ema is the deliverable's only variance control. The read is
"data step + ema" and a null cannot separate them.

**NO OFFLINE NUMBER EXISTS OR CAN EXIST FOR THIS MEMBER.** It has seen every holdout study. It cannot be
seat-scored, curve-scored or distance-scored, and `knee-7d1`'s 4.67x compression does not apply — that
converts an HONEST offline delta and this member has none. The pre-registration is written on the board.

- **0.936+** — the free data step transfers. Licenses `experiments/arms/final_fit_six.sh` (~10 h of card,
  written and arm_check-clean, deliberately unrun). The only path still priced above the 0.933 -> 0.940 gap.
- **0.934-0.935** — directional, under the bar. Keep the member, do NOT license the refit.
- **0.933** — a tie. The data step does not survive the swap; ~10 h retired before it is spent. That is
  the cost-tiered shape working: 93 min of card decides ten hours of it.
- **0.932 or below** — suspect the SWAP, not the data. `band_dense` is the strongest single member
  (0.8578); the next test ADDS `final_ws_e12` as an eighth rather than displacing it.

**A HARNESS BUG, NAMED BEFORE THE RESULT.** The twelve-epoch fit SUCCEEDED and wrote its checkpoint, and
the run log ends in a traceback anyway: the arm's automatic `seat_value` step asked `member_curve.swap`
to price a candidate pool that was empty — because the only candidate is this member, and it is skipped
for having no holdout record. `max()` on an empty sequence. **Plumbing, not training.** A crash in a
reporting step is indistinguishable in the log from a training failure, and it cost this run its first
reading. `swap` now returns `no candidate carries a holdout record, so no seat can be priced`, with a
test, because every `_ff` member the gated refit produces would have hit the same line.

### Result: `kneemri-greedy7` reads 0.933 — the fourth tie, and roster search is retired (2026-08-23)

Ref `55713981`, **0.933**. That is the pre-registered `0.933` band verbatim: *"a third tie. The board is
decisively flat to WHOLESALE exchange within this pool, not merely to one-for-one swaps. Roster search is
retired entirely and the budget belongs to new axes only."* No re-argument is available — the band was
written before the number and the number landed inside it.

**What was exchanged.** Four of the champion's seven seats, not one. `band_dense`, `grade_ladder`,
`base_dense` and `lr4e-5_ws` left; `band_dense_long`, `through_dense`, `depth_regions_narrow` and
`calibrated_targets` took their places. Count (7), kit (`cf6a99f`), vote and serving config were held
fixed. The roster was chosen by `record_mix --greedy` over the 28 packaged members: **0.8712 offline
against the champion seven's 0.8694**, the first offline gap in the whole search above the seat ruler's
+0.0015 resolution.

**+0.0018 offline bought +0.000 board.** At `knee-7d1`'s median 4.67 board points per holdout point that
edge priced at about +0.008. It delivered nothing. This is now the fourth board reading of 0.933 across
`gradeswap`, `accumseat` and `greedy7`, with `sevenplane` at 0.932 — a spread of 0.001 across every
rearrangement of this pool anyone has shipped.

**The mechanism, named.** Not the harness. The offline number is real and was computed correctly; what
fails is its TRANSFER. The greedy roster maximises the very quantity it is then scored on, over 28
candidates on 909 studies, so a +0.0018 edge selected that way is inside the selection noise of the
search itself rather than a property of the members. `2026-08-23_the_ensemble_axis_is_exhausted.md`
predicted this from the other direction: the marginal seventh member is worth +0.0004 and the best
SUBSET peaks at four (0.8696 against seven's 0.8694). A pool with nothing left in it to select cannot be
rearranged into a result by any rule, however good — and greedy is the best rule available.

**What this does NOT rule out.** Nothing about member QUALITY or about axes absent from the pool. Every
one of the 28 candidates was fitted from the same encoder family on the same pixels. `final_ws_e12`
(data volume, ref `55715039`, pending) and `vjepa_member` (a tubelet 3D trunk, on the card) are both
members that did not exist when this pool was searched, and neither is answered here.

**What it forbids.** No further submission slot may be spent on a roster drawn from the currently
packaged pool, by ANY selection rule — greedy included, and greedy was the strongest argument the
project had. The `greedy4` precedent (pre-registered at or below 0.928, returned 0.930) is hereby scoped
to the pool it ran on: greedy transferred when the pool still had slack, and does not now that it does
not. An entry must change what is IN the pool.

### Result: `kneemri-finalfit` reads 0.934 — the first board movement in five entries, and it came from DATA (2026-08-23)

Ref `55715039`, **0.934**. That lands in the arm's `0.934-0.935` band, pre-registered verbatim as
*"directional"* with the instruction *"keep the one member and stop"*. So `final_fit_six.sh` — the six
remaining seats refitted at `--fold -1`, repriced at ~10 h — stays UNRUN. The band was written before
the number and the number landed inside it; there is no re-argument available and none is attempted.

**What changed.** One seat of seven. `gradeswap`'s roster with `band_dense` out and `final_ws_e12` in —
the same recipe fitted on **4,406 studies instead of 3,497, +26%** — with count, kit (`cf6a99f`), vote
and serving config held fixed. No offline number exists for it by construction: `--fold -1` holds out
nothing, so the member has seen every holdout study and can never be seat-scored, curve-scored or
distance-scored again.

**WHY THIS ONE IS WORTH MORE THAN ITS +0.001.** Five entries in a row read 0.933, 0.933, 0.933, 0.932 —
`gradeswap`, `accumseat`, `greedy7`, `sevenplane` — and every one of them was a REARRANGEMENT of the
same packaged pool, including `greedy7`, which maximised the offline mean by the best selection rule
available and still delivered +0.000. This entry is the first that changed what is IN the pool rather
than which of it was seated, and it is the first to move. That is a statement about the AXIS, not about
the size: roster search is flat, data is not.

**The size, stated honestly and not inflated.** +0.001 is ONE unit of the board's resolution and sits far
under the 1.5% bar. It is directional and nothing more. It is not a finding, nothing may be planned
around it, and no confirmation slot is owed to it. What it buys is a DIRECTION, and the direction was
already argued from the other side: `knee-krnd`'s data slope is concave — each doubling worth ~0.48 of
the last — and this is +26% on ONE member of seven, which is a small fraction of one doubling applied to
a seventh of the ensemble. A board reading of +0.001 is about what that predicts, which is the reason to
believe the mechanism rather than the number.

**What it does NOT license.** Not `final_fit_six`. Its own gate was 0.936 and the result missed it by two
board points; at ~10 h for six more members whose individual contributions must then be divided the same
way this one was, the ratio is poor and the band said so in advance. Not a data-QUANTITY programme
either: `knee-krnd` prices the next full doubling from 3,526 studies at ~+0.009 offline, and we do not
have another 3,526 studies to fetch.

**What it does license.** The reading that the live axis is what the encoder and the corpus CONTAIN, not
how the members are arranged — which is the same axis `raddino_dense` (knee-vp8q, queued behind vjepa)
attacks from the initialisation side rather than the sample-count side.

**Champion repinned.** `comp/champion.json` moves from `gradeswap` (0.933) to this entry; `champion_check`
is what proves the new pin still rebuilds.

### Result: `kneemri-weakseven` reads 0.908 — the slope was 2.2x too steep, and the holdout was never wrong (2026-08-24)

Submission 55734428, fired 05:55Z, read 07:14Z. **0.908**, inside the pre-registered 0.905-0.920 band,
which the kernel docstring wrote as *"the slope in the large is ~1-2, not 4.67 ... `board_transfer` must
be refitted or retired."*

This is the only entry in this file that was designed to lose, and it is the one that changed the plan.

**Why it existed.** Every board reading we owned sat in a 0.0043 offline band, against a 0.0062 holdout
noise floor. The recorded +4.67 board points per holdout point was therefore a slope fitted through
noise, unfalsifiable from its own data, and no amount of training could fix that — the missing quantity
was a point OUTSIDE the band. `record_mix` over the 28 packaged members picked the seven weakest solos
(0.8215-0.8462) for a joint 0.8562, the furthest offline a point can sit while every member stays
rebuildable from `comp/members.roster`. Count (7) and vote (`RankAverage`) held fixed so only member
QUALITY moved.

**The refit.**

| anchor pair | Δ offline | Δ board | slope |
|---|---|---|---|
| champion 0.8694 → 0.934 | 0.0132 | 0.026 | 1.97 |
| chain floor 0.8648 → 0.927 | 0.0086 | 0.019 | 2.21 |
| `board_transfer` over all 12 entries | — | — | **2.14** |

**The clause that survived, and it is the important one.** Inversions are 4 of 55 orderable pairs and
**0 of the 33 the holdout resolved at all** — every disagreement is under 0.0053. The competing
hypothesis was that the holdout is a poor guide above 0.868 and board gains live where it cannot see.
That is refuted: the holdout ordered a 0.0132 offline gap into a 0.026 board gap, monotonically and in
the right direction. Trust the SIGN above 0.0053; halve the MAGNITUDE.

**What it costs the plan.** 0.934 → 0.950 is +0.016 board = **+0.0075 offline** at 2.14, against a
measured offline ceiling of ~0.872 (+0.0026 over the champion). The requirement is ~3x the entire
remaining headroom of the audited space. Seat swaps, ensemble width, decode knobs and vote rules are not
flat — they are arithmetically short by 3x. The 4.67 constant had been keeping that whole family of
cheap arms looking viable; every offline delta this file priced since 2026-08-16 is worth 46% of what it
was sold for.

No slot bought more than this one, and it bought it by scoring 0.026 below the champion on purpose.

### Result: `kneemri-allfit` reads 0.937 — the all-data axis is real, sub-linear, and now spent (2026-08-24)

Ref `55736360`, submitted 07:35 UTC, **0.937**. A new best by +0.003 over `finalfit`, and the largest
single-entry movement since 2026-08-19. Kit `e7b65a0`, read from the kernel log's own `kit` line.

**The dose, which is the whole point of the entry.** Count (7), vote (`RankAverage`) and serving were
held fixed and the only thing that moved was how many seats trained on all 4,406 studies:

| entry | all-data seats | fold-0 seats | board |
|---|---|---|---|
| the 0.933 chain (`gradeswap`/`accumseat`/`greedy7`) | 0 | 7 | 0.933 |
| `finalfit` | 1 | 6 | 0.934 |
| `allfit` | 6 | 1 | **0.937** |

The first seat bought +0.001; the next five bought +0.003 between them, about +0.0006 each. That is the
**0.936–0.939 band**, pre-registered in `interpretations/kneemri/2026-08-24_the_all_data_dose_response.md`
before the number, whose consequence reads verbatim: *"Converting a seventh seat buys a seventh of a
diminishing quantity, so the 7.3 h train is NOT justified — the honest move is to stop here and record the
axis as real-but-spent."* So `band_dense_long_ff` is **UNRUN** and the card stays with the other project.
The seats are not independent: six all-data members largely re-learn the same extra 909 studies.

**This is +0.003, and the owner's bar is 0.02.** Directional, not a result. What it is allowed to license
is a DIRECTION — data volume moves this board where seven rearrangements of the same members did not —
and it is not allowed to license a plan built on the size.

**What it cost: the offline ruler, on this pool, entirely.** Six of seven seats held out nothing, so this
entry has no offline number of any kind. `board.jsonl` records it null, `board_transfer` names it in
`unmeasured()` rather than dropping it, and it enters no pair. Roster decisions from here are made against
the board alone at five reads a day.

### The record went stale for four days, and the correction I pushed yesterday was computed on the gap

`comp/board.jsonl` — the machine-readable half this file's header points at, and the file
`board_transfer` prices **every** experiment from — was last written on 2026-08-20 at `greedy4_v1`. Seven
entries existed in this prose and not in the record: `sevenplane`, `gradeswap`, `accumseat`, `greedy7`,
`finalfit`, `weakseven`, `allfit`.

**The consequence was not cosmetic.** Yesterday's refit was done BY HAND off this prose, and its first
anchor — *"champion 0.8694 → 0.934"* — pairs `gradeswap`'s offline with `finalfit`'s board. Two different
entries. `finalfit` has no offline number at all, by the same `--fold -1` construction recorded above, so
there was no pair there to take. That anchor read 1.97; the honest pair (`weakseven` 0.8562/0.908 against
`gradeswap` 0.8694/0.933) reads **1.89**. The `+2.14` that came out of it went into this file **and** into
`comp/members.roster` as a correction of the older `+4.67`.

**Backfilled, the tool reads +1.92** over 15 ensemble entries and 90 orderable pairs. The size of the
error is small — 2.14 against 1.92 — and that is exactly why it survived: an entity error whose numeric
consequence is inside the noise is invisible to every check except reading the two rows.

| reading | slope | fitted on |
|---|---|---|
| `knee-7d1`, 2026-08-16 | +4.67 | 5–6 pairs spanning 0.0043 offline — **inside** our own 0.0062 floor |
| by hand, 2026-08-24 | +2.14 | three anchors, one of them pairing two different entries |
| `board_transfer`, backfilled | **+1.92** | 15 entries, 90 pairs, 57 of them resolved |

**The inversion clause got STRONGER**: 4 of 90 pairs inverted, **0 of the 57 the holdout resolves at all**
(was 0 of 33). Every disagreement still sits under 0.0053.

**But note where the slope is NOT measured.** The four best entries — `allfit`, `finalfit`, `sevenplane`,
`accumseat` — are all one-sided and enter no pair, three of them because `--fold -1` erased their offline
number. So +1.92 is fitted entirely on entries at or below 0.933 and is an **extrapolation at the top of
the board**, where every remaining decision now lives.

**The rule that survives is unchanged and is the one that matters:** never read an offline seat against a
board threshold without the compression. The CONSTANT has now been wrong twice, at +4.67 and at +2.14, and
both times because it was quoted from prose rather than read from the tool. Read it from
`python -m kneemri.evaluation.board_transfer`. If the tool and a document disagree, the document is stale.

**Third stale-claim-in-a-live-document this week, and none of the three was caught by a gate.** The others
were the roster's `+4.67` and the champion pin lagging the board by three hours. The shape is identical
every time: a number is correct when written, the thing it describes moves, and nothing connects them. A
gate is buildable here — assert every `## ` result heading in this file has a matching `id` in
`board.jsonl` — and it is filed rather than written now, because the axis reading was the thing owed.

### The door-closing arithmetic is an extrapolation, and the backfill did not fix that (2026-08-24, knee-1p1i)

Line ~1962 above closes the whole cheap-arm family: *"0.934 → 0.950 is +0.016 board = +0.0075 offline at
2.14 ... Seat swaps, ensemble width, decode knobs and vote rules are not flat — they are arithmetically
short by 3x."* Two things have moved under it since, and only one of them was recorded.

**Restating it at the current numbers.** The champion is 0.937, not 0.934, and the constant is +1.92, not
+2.14. So the requirement is 0.013 board = **+0.0068 offline**, against a measured offline ceiling of
~0.872 — about +0.0026 over the champion pool. That is ~2.6x the remaining audited headroom rather than
3x. The verdict does not change; it is restated so that nothing downstream quotes 2.14 again.

**The defect the backfill did NOT fix, which is the actual finding.** `board_transfer` names it in its own
output: the four best entries — `allfit`, `finalfit`, `sevenplane`, `accumseat` — are ONE-SIDED and enter
no pair, three of them because `--fold -1` erased their offline number. The slope is therefore fitted
**entirely on entries at or below board 0.933**, and every use of it above 0.933 is an extrapolation —
including the use that shuts the door.

The risk is not symmetric. If the ratio STEEPENS near the top, the offline headroom needed for +0.013 is
smaller than +0.0068 and the audited space is not arithmetically dead. If it FLATTENS, the door is shut
harder than stated. There is a mechanism for either, so this is not wishful: the metric is a mean of
twelve independent AUCs, and late offline gains may concentrate in the hard labels (steepening) or in the
easy ones every member already gets right (flattening). We cannot tell, because the pairs that would tell
us are exactly the ones `--fold -1` destroyed. That is a second, undiscounted instalment of the price
`allfit` paid, on top of the one already recorded.

**And the fix is blocked by the same damage.** Extending the fit needs a fold-0-only pool meaningfully
above 0.933 — but 0.933 *is* the fold-0 ceiling as measured (`gradeswap`/`accumseat`/`greedy7` all read
it, greedy accumulation saturated four steps running, the seven-member vote is flat to wholesale
exchange). No rearrangement of existing fold-0 members clears it, so no slot can buy this. Only a BETTER
FOLD-0 MEMBER moves it, which needs card — which is what `ema_dense` (knee-56w5) is, and it is on the card
now. Its top band is simultaneously an EMA verdict and the first fold-0 member above the seated band since
the ceiling was declared.

**What to do with this in the meantime: nothing, loudly.** The honest position is that the arithmetic
still says the cheap families are short, and that its confidence interval is wider than it reads. It does
not license re-running an arm the offline ruler already called flat. It does mean that a family should not
be described as *arithmetically* dead in future write-ups without the words "on an extrapolated slope".

### Result: `kneemri-lastseat` reads 0.937 — the repeat holds, and the seat is not the variable (2026-08-24)

Ref 55739968, submitted 10:25Z, scored inside the hour. **0.937 — identical to `allfit`**, landing in the
pre-registered **0.936–0.939** band, the one the kernel called the expected outcome.

The exchange was `band_dense_long_s2026` (0.8587 solo) out, `small_batch6_s2026` (0.8598 solo — the best
single member this project has ever fitted) in, with count, vote, kit and the six-seat all-data dose all
held. So this is the first REPEAT at the top of the board, and it buys three things:

- **`allfit`'s +0.003 was the DOSE, not the seat.** Two entries differing in one seat read the same
  number, so the dose table in `interpretations/kneemri/2026-08-24_the_all_data_dose_response.md` stands
  without a caveat. That was the reading most at risk.
- **Board scatter across a single-seat exchange at 0.937 is at most one tick.** Every board-only decision
  from here inherits that error bar. The honest limit is the one the pre-registration already wrote down:
  two entries reading the same band BOUND the scatter, they do not measure it.
- **"The seven-member vote is flat to single-member exchange" now extends to the all-data pool**, where it
  had never been checked. It has now held across five entries and both pools.

Note what did NOT happen: the better member offline (+0.0011 solo) was neither the better seat nor the
worse one. In the fold-0 pool solo score and seat value are ordered *oppositely*; here they are ordered
not at all. Both readings say the same thing from different directions — **at seven members the seat
identity is not the variable**, which is exactly the seat principle the member-count curve already
carried, arriving now by a second route.

**What it costs the plan.** The seventh seat is retired as a search axis, the same way roster search was
retired at `greedy7`. Nothing left at the top of this board is bought by rearranging members that already
exist — which is the fourth independent statement of the data-bound verdict, and it points the remaining
card at the same place: a BETTER member, or a member of a kind this pool does not contain.

### Result: `kneemri-augseat` reads 0.936 — type does not carry at seven either (2026-08-24)

Ref 55743145, submitted 12:44Z. **0.936**, one tick under `lastseat`'s and `allfit`'s 0.937, landing in
the pre-registered **0.936–0.939** band whose name was written before the run: *type does not carry at
seven*.

The exchange held count, vote, kit and the six-seat all-data dose, and moved the single fold-0 seat from
`small_batch6_s2026` (0.8598 solo, the best member this project has fitted) to `aug_long_s2026` — a
member the offline ruler scores **0.0076 worse** and which had never been mounted in any entry. So the
variable was not seat identity this time but the *kind* of distance: a regularised member rather than a
better-or-worse one of the same construction.

It buys one reading and closes one door:

- **A member the holdout ranks clearly worse is worth the same seat.** Three entries in a row now read
  0.936–0.937 across a single-seat exchange, and the third moved an axis the first two did not touch.
  The seat is not the variable on the identity axis *or* the regularisation axis. Offline solo score
  keeps failing to predict seat value in every direction it has been asked.
- **Neither branch of `knee-avzz` fired.** `≥0.940` would have said the ruler is blind to the
  regularisation axis and would have licensed mounting on measured DISTANCE against the ruler's own
  verdict; `≤0.935` would have said the offline number was right about a killed member. 0.936 says
  neither. `farseat` is therefore **not submitted** — its whole case was the `≥0.940` branch — and the
  `vjepa_warm` question stays **open** rather than retired. Writing 0.936 into either branch would be
  reading the band the pre-registration explicitly reserved for "no".
- **Board scatter stays bounded at one tick** across three single-seat exchanges, which is the error bar
  every board-only decision from here inherits. It is still a bound, not a measurement.

**What it costs the plan.** The seventh seat is now retired twice over — once on identity (`lastseat`),
once on type (`augseat`). Nothing at the top of this board is bought by rearranging or re-regularising
members that already exist. The remaining card goes to a member of a kind this pool does not contain,
which is what `margin_seat` (a different objective) and `dissent_seat` (a different error structure)
are for.

## `peak8` — the offline peak, shipped: **0.932** (submission 55751365, kit `2cc4fc8`)

Eight members, the best-scoring subset the greedy holdout search has ever found (0.8731), and the first
board entry to carry `margin_seat_s2026`, `sided_pool_s2026` and `crop80_span_s2026` at all. It landed
in the **pre-registered kill band (≤0.934)**.

It is **0.0019 above `allfit` offline and 0.005 below it on the board.** The two rulers now disagree in
sign, and with the two earlier pairs the shape of the transfer is readable:

| offline | board | slope |
|---|---|---|
| 0.8562 | 0.908 | — |
| 0.8712 | 0.933 | ~1.67 |
| 0.8731 | **0.932** | **negative** |

**Offline greedy member selection is closed as a lever.** The transfer is real where members are weak
and **saturated** at the top of the range, so the `board_transfer` +1.92 constant must not be
extrapolated above ~0.933. A greedy peak is now evidence about the holdout, not about the board.

**What this entry does NOT say.** It is not a reading on member **count**. All eight seats are fold-0 —
an all-data dose of 0/8, against `allfit`'s 6/7 on the one axis ever measured worth +0.003. Count and
dose are fully confounded here, and the −0.005 is the dose difference's own size with the sign it
already had. Count-at-dark-dose has never been on this board and cannot be built until a **seventh dark
member exists**, which is what `knee-rndp` (`small_batch6_ff`) is for: six dark checkpoints exist, so
every entry to date has been six-dark-plus-one and the pool has never had a seventh to offer.

## 2026-08-25 — five entries, one tick of movement, and the record that did not hold them

Written on 2026-08-25 during the `knee-624j` record repair, from the submissions' own descriptions and
their scores read back off Kaggle. **These five had scored and appeared in NO record** — not this file,
not `board.jsonl` — which is the defect that bead exists for. `board_check --live` is now the gate.

| entry | ref | members | board | the one variable |
|---|---|---|---|---|
| `dark6` | 55756018 | 6 | 0.937 | all-data dose at six, no fold-0 seat at all |
| `dark7` | 55756369 | 7 | 0.937 | count of dark members, six → seven |
| `v3seat` | 55757184 | 7 | 0.936 | seventh seat's encoder FAMILY (DINOv3-base, strong reading) |
| `farseat` | 55758082 | 7 | **0.937** | seventh seat's ARCHITECTURE family (V-JEPA2, breaks slice-then-pool) |
| `votefit` | 55759764 | 6 | **0.938** | the vote: `LogitAverage` for `RankAverage` |

**`dark6` = `dark7` = `farseat` = `allfit` = `lastseat` = 0.937.** Five entries, one number. The seventh
seat has now been filled with a fold-0 member, a converted all-data member, a different encoder family
and a different architecture family, and it moved nothing in any of them. `v3seat`'s 0.936 is one tick
down and inside its own pre-registered FLAT band. **The marginal seat is inert regardless of provenance**,
which says the six already determine the answer.

**The all-data dose is spent at six, measured rather than extrapolated.** `dark6` against
`allfit`/`lastseat` says the fold-0 seat those two carried bought nothing; `dark7` against `dark6` says
the seventh dark member bought nothing either. `knee-rndp`'s seventh dark checkpoint was built and it
answered the question it was built for with a null.

**`votefit` 0.938 is the best score this project owns and it is not a result.** One tick over `dark6` on
an identical roster, inside the pre-registered 0.935–0.939 FLAT band, an order of magnitude under the
1.5% bar. The `≥0.940` branch did NOT fire, so the licensed `QualityWeightedRank` sweep stays unbuilt.
The vote was the last unvaried board axis; **no member-shaped or serving-shaped variable remains.**

**What the backfill did to the transfer constant.** `board.jsonl` held 34 of 58 submissions until this
repair. Fifteen scored all-ours entries were absent — `basedense_v1`, the entry `comp/champion.json`
pins, among them — and five of those carry an offline number. With them entered, `board_transfer` reads
**+1.72** where kernels shipped this week quote +1.82, +1.92 and +2.14. Every one of those was fitted on
an incomplete set. It also now reads **0 inversions in 108 resolved pairs**: where the holdout separates
two entries by more than its own noise floor, the board has never once ordered them the other way.

**`blendseat_v1` 0.939 — the corpus axis finally has a board number, and it is one tick.** `votefit`'s
identical six with `base_dense_ff` replaced by `merged_blend_s2026`, the only member trained on external
expert image reads (Rijeka). Vote rule, count, kit and serving config all held, so the single moving
variable is the CORPUS one seat was fitted on. It read **0.939** against `votefit`'s 0.938, inside its
own pre-registered 0.939–0.941 band. One tick is under the 0.0062 paired floor: this is DIRECTIONAL, not
a result. What it does say is narrow and worth keeping — an externally-fitted member is not harmful at
serving, which is the first evidence of that kind this project owns, every earlier merged-corpus number
having been offline. It does **not** license the 7 h `merged_blend_ff` refit on its own.

It is also the best board row this project owns, and `comp/champion.json` is pinned to it.

**`soloseat_v1` 0.925 — the count curve's anchor, re-dated, and the ensemble premium is half what a
stale anchor said.** `final_ws_e12` served alone: one member, `overlap=False`, the champion's kit, no
vote rule. It exists because the only single-member board number this project owned was `solo_v4` 0.914
from 2026-08-17, and every member built since is `--fold -1` with no holdout left to score on — so the
question "did member quality move in nine days?" had no offline answer and only a slot could settle it.

It read **0.925**, landing in its own pre-registered **0.920–0.928 DIRECTIONAL** band: +0.011 over
`solo_v4`, real in direction, under the 1.5% house bar as a single step. Two consequences, and the
second is worth more than the number.

**The `--fold -1` suspicion is cleared.** The lowest pre-registered band (≤0.913) would have indicted
all-data fitting at serving — no holdout to select on, EMA standing in for early stopping — and that
implicates FIVE of the champion's six seats. It did not happen. All-data members are sound at serving,
and the roster does not need auditing.

**The ensemble premium is +0.014, not +0.025.** `interpretations/kneemri/2026-08-26_the-deficit-is-the-member.md`
priced it off the 2026-08-17 anchor and read our six-member 0.939 as buying +0.025 over our own best
single. Against a current single it buys **+0.014**. The member deficit against the forum's self-reported
singles (0.938–0.942, topic 735304) narrows from 0.015–0.028 to **0.013–0.017** — still the largest
single gap on this board, still the axis every remaining card hour belongs to, but smaller and measured
rather than inferred from a nine-day-old row.

Caveat kept from the kernel docstring: `solo_v4` was served by `members=1`, which RANKS inside the
package rather than naming a member, so its identity is unrecoverable. This is "our best single THEN vs
NOW", not a matched swap.
