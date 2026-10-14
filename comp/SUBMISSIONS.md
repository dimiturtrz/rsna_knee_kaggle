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

**`jitterfit_v1` 0.937 — rigid test-time views are neither a lever nor a hazard, and the last serving
axis closes.** `votefit`'s identical six, served through `SpatialViews(0, -6)`: a rigid view jitter
applied at inference, with count, roster, vote rule (`LogitAverage`), kit, fold status and per-member
precision all held. `knee-wnam` called this the last unmeasured serving axis, and it was — every other
serving variable this board carries has had its own entry.

It read **0.937**, the same digit as `dark6` 0.937 and one tick under `votefit` 0.938 — squarely in its
own pre-registered **0.935–0.939 FLAT** band. What makes this worth a slot rather than a shrug is that
the pre-registration was TWO-SIDED and both sides are now refuted by the same number:

- **≥0.940 would have said the members are pose-brittle** — that a small rigid perturbation reveals
  variance the ensemble could average away. It did not happen. The seats are not pose-sensitive at this
  amplitude, so nothing is to be gained by averaging over views.
- **≤0.934 would have said the warp destroys slice-edge information** — that resampling the volume
  costs signal at the boundaries where a focal finding lives. It did not happen either. The warp is
  harmless.

A flat reading that kills a hoped-for gain AND clears a suspected hazard is a closed axis, not a null.
Rigid views are retired: not "untried", not "inconclusive", measured and inert. The remaining board
levers are member quality and supervision, which is where `blendsix` is pointed.

Offline null by construction — all six seats are `--fold -1`.

**`blendthree_v1` 0.939 — the floor-raise mechanism gets its first direct board test, at half dose, and
buys nothing.** The champion's six seating with three non-blend seats replaced by their own
blend-supervised twins (`blend_calib`, `blend_ladder`, `blend_ws`), each refitted on harmonised labels
and 12 / 0.2–0.8 geometry. Vote (`LogitAverage`), count (6), kit (`8243a95`), serving views and member
order are `blendseat`'s, untouched — the kernel diff is three lines and each is the same swap. This is
DOSE 3 of `knee-5z9r`'s dose curve, whose dose 1 is `blendseat_v1` 0.939.

It read **0.939** — not near the champion, *exactly* the champion — landing in the pre-registered
**0.936–0.939 FLAT** band.

Why that is worth writing down rather than filing as another flat entry: `knee-urvf`'s standing
prescription is that *"what raises the ensemble is raising the FLOOR of every member, or adding one that
is strong AND unlike."* Ten entries have now moved count, member identity, fold status, regularisation,
encoder family, architecture family, the vote rule, serving views and corpus, all inside 0.936–0.939 —
and this is the first entry to move the **supervision the members were fitted under**, which is the
first half of that prescription tested directly. At three seats of six it moved the board by 0.000.

That is a lowered prior on dose 6, not a refutation of it, and the band said so before the number: a
LINEAR dose effect reads flat at the midpoint, and only the endpoint separates "blend supervision does
nothing" from "three seats is not enough of it".

**The bead's own rule is not fired.** `knee-5z9r` sends this band to dose 6 at slot 5. Slot 5 is held by
owner directive (`knee-pubw`), and the owner beats the bead: dose 6 waits for a slot on 2026-08-27, and
is asked for before it is spent. The champion pin is unchanged — a tie does not displace a pin, and does
not need a slot to defend it.

Offline null by construction — every seat is `--fold -1`.

## `resseat` — the decode-path bet, slot 5 of 2026-08-26 (22:21Z, ref 55803955)

The champion's six with **one seat moved**: `lr4e-5_ws_ff` → `res392_ws_s2026`. Nothing else. Every
champion seat decodes at **336 px**, and 392 px is the only resolution the champion does not contain,
so the axis being moved is the **decode path** — the one diversity axis with a *measured* board
transfer. `knee-nas5` collapsed three members onto a single pixel group and paid 0.0059 offline for
0.014 on the board: a 2.4× compression, against the whole-record `board_transfer` median of +1.72.

Quality-neutral by construction: the outgoing twin reads 0.8525 solo, the entering one 0.8523. Count
held at six on board evidence — every 7-member entry we own caps at 0.937 (five of them), and both
0.939 entries are 6-member. The single fold-0 seat is `merged_blend_s2026`, which was already fold-0 in
both 0.939 entries, so no new fold-0 penalty is introduced.

**The offline ruler is blind here, and this entry says so first.** Eleven seating variations spanned
0.8687–0.8705 against a 0.0062 three-seed noise floor — the ruler cannot separate the field. This swap
read the joint-best 0.8705 at **+0.0001** over the champion's base, and it is a *proxy* number besides:
four of the six seats are `--fold -1` and were stood in for by their fold-0 twins to compute it at all.
That is not evidence, is not quoted as any, and `board.jsonl` records this entry's `offline` as **null**
— which is what `board_check` refused it for on the first try, correctly.

Pre-registered against the 0.939 champion, before the score:

| branch | reading |
|---|---|
| **≥ 0.941** | decode-path diversity is a live lever above the plateau; next slots sweep resolutions |
| **0.940** | directional only, under the bar; keep and re-test |
| **0.937–0.939** | **the most likely branch** — eleven entries on one plateau; the seating axis closes alongside vote, views, corpus and supervision |
| **≤ 0.936** | 392 px is actively worse *in company*; resolution mixing is a hazard, not a diversity source |

**SCORED 0.938** (2026-08-26 ~23:15Z). The **0.937-0.939 branch**, which is the one the entry named as
most likely. One point below the 0.939 champion, which is a sixth of the noise floor and reads as flat,
not as a regression: the swap neither cost nor bought anything the board can see.

That is the **twelfth** consecutive entry on the 0.936-0.939 plateau, and the reading the branch
pre-committed to now applies. **The seating axis is closed.** Eleven seating variations could not be
separated by the offline ruler (0.8687-0.8705 against a 0.0062 floor), and the board has now failed to
separate them either. Decode-path diversity is real where it was measured -- `knee-nas5` paid 0.014 on
the board for collapsing three pixel groups -- but ADDING one 392 px seat to five 336 px seats does not
buy it back, so the compression finding does not invert into a construction rule. Both 0.939 entries
remain 6-member and `comp/champion.json` stays pinned to `blendseat_v1`.

What this leaves: the last two gains came from a **vote rule** (`votefit`) and **better supervision**
(`blendseat`), not from a seat swap, and both non-seating mechanisms outperformed every seating one.
The next slots belong to mechanisms that change the PIXELS or the LABELS, not the roster -- which is
exactly what `knee-e6sj` (slice contiguity) is.

Fired on explicit owner authorisation for the last slot of the day, with ~1h40m left before the 00:00Z
reset. The kernel push was refused once with `Maximum batch GPU session count of 2 reached` — the cap is
per **account**, shared with the biohub/celltrack project, and no gate of ours asks before pushing
(`knee-mjqk`). Nothing of theirs was cancelled; a letter went to their inbox and a waiter took the first
slot that freed.

## 2026-08-27 · `kneemri-calibsix` — 0.941, the plateau breaks

**0.941.** A new best, +0.002 over `blendseat_v1` and `blendthree_v1`, and the first of twelve entries
to leave the 0.936–0.939 band.

The entry is the champion's six seating with `--calibrated-targets` on **every** seat. Nothing about the
roster moves: same six seats, same `LogitAverage`, same count, same serving views, same member order.
Five members are their champion twin's flags plus the one flag; `calib_span_ff` already carried it. So
this is not a seating variation — it is the **member-quality lane** tested at roster scale for the first
time, on the only term this project has measured positive twice (`knee-bs4l`: +0.0072 proxy / +0.0129
annotated at the span corner; +0.0061 at the dense-hold corner).

**Read it as directional, not as a finding.** +0.002 is under the 0.0053 Hanley–McNeil floor and well
under the 0.0062 three-seed floor. One board read cannot separate it from the plateau it topped.

**It came in BELOW its own pre-registration** (`knee-wrl6`, ~0.944 with `knee-ps8o`'s halved local slope
already applied), and that miss is the useful part. The 23% seat pass-through was measured for ADDING
one member to a roster; nothing had measured what happens when all six improve on the same axis at once.
Now something has, and the answer is that it does not compound: the pass-through does not improve when
the whole roster moves together.

**The arithmetic that follows.** At this pass-through the member-quality lane ceilings near 0.943. The
gap to first place is 0.013 (leader 0.952, board read 2026-08-27T05:57Z, rank 49 of 2486). So 0.950 is
not reachable by improving members, and the remaining routes are **external data** (`merged_dense_cf`,
on the card as this scored) and **stacking** (`knee-orjq` — whether directional gains ADD, which nothing
has ever tested and which the whole 0.950 plan assumes).

Witness-checked before the push: five seats ok, one UNVERIFIED (`calib_span_ff`, published before
2026-08-24), no defects. Kit `fec94e7`, kernel version 1. Slot 1 of 5 on 2026-08-27.

## 2026-08-27 · `kneemri-calibmix` — 0.941, count is not a lever

**0.941 — identical to `calibsix` at six seats.** Eleven seats, five more than the champion, same
score to the board's resolution.

The entry is `calibsix`'s six **unchanged**, plus the second fit of five of those six members
(`final_ws_e12`, `grade_ladder_ff`, `merged_blend_s2026`, `lr4e-5_ws_ff`, `through_dense_ff`). Same
`LogitAverage`, same `overlap: false`, same kit `fec94e7`. The single variable is COUNT, and the seats
added are deliberately the low-diversity kind — same recipe family, same decode path, differing only in
fit. It asks the one question `calibsix`'s read could not separate from its own member-quality gain:
does error-structure diversity from more seats buy anything once every seat is already calibrated?

**The pre-registered FLAT branch fired** (0.939–0.941). No.

**Two independent rulers now say the same thing.** Offline, the same window ran the seat-count curves on
the 103-member pool: the *honest* (quality-ordered) curve goes six 0.8720 → nine 0.8729 → seventeen
0.8736 → peak **0.8738 at twenty-four** — +0.0018 for four times the seats, under a third of the 0.0062
floor and inside the +0.0017 in-sample/held-out selection bias. The *diverse* (flat-distance) curve peaks
at 0.8650 across ninety-nine seats, **−0.0088 behind honest**, because flat-distance selection elects the
pool's worst members first (`focus_latmen` at seat 2 drops 0.8634 → 0.7501) and needs twenty-five seats
to match what one member does alone. Count plateaus; blind spread loses.

**What this does NOT close.** Both rulers are offline, and `knee-nas5` measured the decode-path /
pixel-group axis at 0.0059 offline against **0.014 on the board** — a 2.4× compression, the only
super-linear board transfer this project has measured, and the one axis neither selector can see. A
**targeted** pixel-group add (`res392_ws_s2026` at 392 px, `crop80_span_s2026` at crop 80, both already
in the mounted package, taking pixel groups 1 → 3) is untouched by either result. Note `resseat` tested
`res392` only as a SWAP, dropping a full-fit member for it, and read 0.938 — the pure ADD has never run.

Kit `fec94e7`, submission `55813714`. Slot 2 of 5 on 2026-08-27.

## `pixgroup_v1` — 0.940, the pure pixel-group ADD, and the entry that closes the seating axis

`calibsix`'s six calibrated seats unchanged, plus two pixel-group adds — `res392_ws_s2026` at 392 px and
`crop80_span_s2026` at crop 80 — taking the entry from one pixel group to three. Same vote, same overlap,
same kit. This was the one seat-lane mechanism `calibmix`'s close explicitly left open: `knee-nas5`
measured the decode-path / pixel-group axis at 0.0059 offline against **0.014 on the board**, a 2.4×
compression and the only super-linear board transfer this project has measured. `resseat` had tested
`res392` only as a *swap* and read 0.938; the pure add had never run.

**The pre-registered FLAT branch fired** (0.939–0.941), at 0.940 — one tick *below* `calibsix` at six
seats. The 2.4× multiplier did not reproduce as an add onto a calibrated base.

**Read together, five entries now close the axis.** blendseat 0.939 (7 seats) · blendthree 0.939 (7) ·
pixgroup 0.940 (8) · calibsix 0.941 (6) · calibmix 0.941 (11 names resolving to 5 recipes). A **0.002
range across five different seatings**, against a 0.0053 Hanley–McNeil floor. Count, fit-duplication and
now targeted pixel diversity all read the same number. **The seating axis is saturated for us at ~0.941.**

**What it does NOT close.** Both adds are weak solos chosen for disagreement (`res392_ws` scores 0.8523
alone). This prices *the add at this base*, not pixel diversity as a mechanism — a member that is both
pixel-diverse and *strong* has never been seated.

**Where it sends us.** The champion's own 0.938 → 0.941 climb came from calibration — member **quality**,
not seating. With five entries flat, the remaining card hours go to better single members. `knee-wbwq`
opens that lane: `fitted_encoder.py` has always been able to warm-start an arm from a fitted member, and
**0 of 131 arms ever used it**, so every member on disk is a cold fit while a third of the roster is one
decode path re-run with a different loss.

Kit `fec94e7`, submission `55816263`. Slot 3 of 5 on 2026-08-27.

## `honest12` — 0.937, and the offline seat selector is refuted as an instrument

The greedy-12 seating, fitted on half of 909 shared holdout studies and scored on the other half in both
directions. Offline peak **0.8751** at twelve seats against the champion five-twin shadow's 0.8701 —
**+0.0050**, the largest offline seating gain this project has measured. Seven of the twelve had never
been on the board, including `calib_dense_hold_s2026` (0.8634 solo, the best single member we own and the
greedy's first pick), which no board entry had ever held. The variable it isolates is member **quality at
the seat level** — the one thing `calibsix`'s calibration read and `pixgroup`'s diversity read could not
separate from count.

**The bottom rung of its own pre-registered ladder fired**: `<=0.938 the weak seats dilute`. At 0.937 it
is the worst of the four entries spent today and one tick below `resseat`.

**This is a sign flip, not a compression.** +0.0050 offline became −0.004 on the board. Every prior
board/offline mismatch in this project has been a *shrunken* gain — `board_transfer`'s +0.74 slope,
`knee-nas5`'s 2.4× the other way. A selector whose gains arrive on the board with the opposite sign is
not a weak instrument, it is a broken one, and **no further slot should be spent on an offline-greedy
seating.** The mechanism is legible: the greedy fits on 454 studies, which cannot resolve a seat worth
less than the 0.0062 three-seed floor, so it seats `mae_matched_s2026` (0.6735 solo) at the honest
curve's own dip and calls it a gain.

**Six entries now close the seating axis from both ends.** honest12 0.937 (12) · resseat 0.938 (6) ·
blendseat 0.939 (7) · blendthree 0.939 (7) · pixgroup 0.940 (8) · calibsix 0.941 (6) · calibmix 0.941
(11). The ordering is **inverse** to offline holdout gain.

> **PARTLY RETRACTED 2026-09-17 00:36Z — the mechanism argument stands, the two board readings do not.**
> The sound half of this closure is the one about the fit set: a greedy fitting on 454 of 909 studies cannot
> resolve a seat worth less than the 0.0062 floor, so **half-holdout greedy seating stays closed.** Both
> board readings offered as evidence are wrong.
> **There is no sign flip.** `honest12` (offline 0.8751 → board 0.937) against the twinfive shadow it beat
> offline (`twinfive_v1`, 0.8705 → 0.935) is +0.0046 offline → **+0.002 board, the same sign**. The top-band
> line predicts +0.0067, so it *under-delivered by 2.1 combined sd* — a shortfall, not a reversal. The
> "−0.004" compares `honest12` to `calibsix_v1`, a different entry it never claimed to beat.
> **The inverse ordering is unsupported.** `resseat_v1`, `blendseat_v1`, `blendthree_v1`, `pixgroup_v1`,
> `calibsix_v1` and `calibmix_v1` all carry `offline: null` in `comp/board.jsonl`. The sentence ranks their
> board values against offline numbers that do not exist.
> What re-opens is **designed** seating priced on the full 909 — see knee-5hbtz.

**What it does NOT close.** The members. `calib_dense_hold_s2026` at 0.8634 solo has still never been read
alone on the board, and seven of these twelve remain individually unpriced. A bad seating of good members
is not evidence about the members.

**Where it sends us.** Every remaining card hour and slot 5 to a better single member. `knee-wbwq` — warm
restarts from fitted encoders, which 0 of 131 arms have ever used — is the open lane.

Kit `fec94e7`, submission `55817896`. Slot 4 of 5 on 2026-08-27.

## calibsolo — 0.923 · one seat of the champion, to price the bonus

Seat 1 of `calibsix` (`final_ws_cf`) served **alone**. Same kit, same feed, same witness-clean member as
the six-seat champion, so the only thing that changed is the count.

**The ensemble bonus is +0.018.** 0.941 at six seats against 0.923 at one. Six prior seating entries span
0.937–0.941 and none of them could separate that bonus from seat count or member choice, because none of
them ever read one seat.

**It lands in the middle pre-registered band** (0.921–0.931: no reordering). Ensembling is not the whole
engine (that needed ≤0.920) and member quality is not the whole remaining lane (that needed ≥0.932). Both
halves stay live.

**No offline number, and the null is the honest entry.** `final_ws_cf.metrics.jsonl` carries a recipe and
no holdout mean. This seat has been on the board inside the champion for days and has never been read on
our own ruler — worth fixing before it is trusted further.

**Where it sends us, and it is not where slot 4 pointed.** 0.923 against `soloseat_v1`'s 0.925 says our
best board single has not moved. The same day's foreign read put `raptor_v4_swa` at **0.8938** on our
909-study holdout against our best member's 0.8634 — +0.030 of member quality that has never been on the
board at all. The +0.018 measured here is what a better seat 1 gets **added to**, not what it competes
with. See `interpretations/kneemri/2026-08-27_the-feed-was-the-whole-shortfall.md`.

Kit `a642ba8`, submission `55828490`. Slot 5 of 5 on 2026-08-27.

## raptor_solo — 0.923 · the foreign member alone, and the +0.030 gap does not exist

`raptor_v4_swa` — dreaddevelopment's published `raptor-knee-widedense` checkpoint — served **alone** on our
decode, our forward pass and our submission writer, after `foreign_ruler` proved our runner reproduces
their own published table. Same kit and same one-seat serving as `calibsolo`; only the checkpoint changed.

**0.923. Exactly `calibsolo`'s 0.923, on the same day.** Two members, two architectures, two training
corpora, one board score.

**This closes the largest lever on the planning list.** `knee-9g28` read raptor at 0.8938 on our 909-study
holdout against our best member's 0.8634 and priced the difference as +0.030 of member quality nobody had
put on the board. `knee-3uje` then argued it was a two-ruler artefact — 0.9152 against 0.9167 when both are
scored on the 58 gold-annotated studies, level. The board has now settled it in `knee-3uje`'s favour on the
only ruler that counts. It also removes `knee-rh4t`'s member-gap reading of the 0.938–0.942 forum singles:
whatever those posters have, it is not in the one foreign checkpoint we can actually run.

**What it opens.** This member is now the cleanest **diversity** probe this project owns, because its
quality is held equal by measurement rather than by assumption. Everything a seven-seat blend with it gains
is diversity, with member quality controlled — which is exactly what `raptorseat` was pushed to read.

**What it does not say.** Nothing about their unreleased `finespacing`/`arms` checkpoints. And two members
agreeing at 0.923 is as consistent with a shared **label-noise ceiling** (`knee-fuhq`) as with shared
quality — those two readings predict opposite things about every remaining member arm, and this entry
cannot separate them.

Kit `9eea9b8`, submission `55829913`. Slot 1 of 5 on 2026-08-28.

## raptorseat_v1 — **0.943, the new board best** · one foreign seat, and the monoculture is the plateau

Ref 55831424, submitted 02:28Z, scored 05:06Z (2h38m in the queue). Kit `b8d76a6`. The champion six
served exactly as `calibsix` serves them — `overlap=False`, `LogitAverage`, no views — then rank-blended
with `raptor_v4_swa` at `RaptorSeat`'s default 1/7.

**Pre-registered in the kernel docstring, before the number:** ≥0.945 diversity pays past member quality,
re-pin on that alone · **0.9425–0.944 real gain, next question is a second foreign arm** · 0.9395–0.9424
flat, does NOT license a fitted weight · <0.9395 diluting, closes the seating question.

**0.943 lands in the real-gain band.**

### Why this is the cleanest diversity test the project has run

Member quality is held equal *to three decimals, on the board's own ruler, on the same day*:

| entry | ref | what | board |
| --- | --- | --- | --- |
| `calibsolo` | 55828490 | one of our six, alone | **0.923** |
| `raptor_solo` | 55829913 | `raptor_v4_swa` alone, on our decode and our writer | **0.923** |
| `calibsix` | 55810756 | our six | 0.941 |
| **`raptorseat_v1`** | **55831424** | our six + that foreign member at 1/7 | **0.943** |

Every previous seating entry moved membership *and* quality together and could attribute its result to
neither. Here the seventh seat is worth exactly what our own best single is worth, so the +0.002 is
attributable to the architecture, the decode and the training corpus, and to nothing else.

### What it does and does not license

**+0.002 is below the owner's 1.5% bar. It is not a finding as a SCORE and must not be planned around as
one.** What it is, is the first non-flat reading in **twelve** on ensemble construction — seat count,
membership across five different sevenths, per-label selection (`label_combine`), the vote (`votefit_v1`),
overlap TTA, external corpus, all flat — and the axis it moved is the one axis the pool never varied.
See `interpretations/kneemri/2026-08-28_architecture-monoculture.md`.

It licenses precisely what its own band says and nothing more: **a second foreign arm**, seated by
`RaptorSeat.fleet` at 1/(6+n). It does NOT license a fitted weight (that branch was reserved for the
flat band and would cost a slot per value against a holdout every foreign arm has already seen).

**Do not seat all thirteen.** The inventory is four families, not thirteen lenses: 10× `coatnet_rmlp_2_rw_384`
@384, 1× `convnext_small`@336, 1× `convnextv2_base`@336, 1× `tf_efficientnetv2_l`@480. Ten coatnets at
1/16 each would outvote our six for arithmetic reasons and read flat on a mechanism that never got tested.
The fleet teacher pass + `teacher_distance` pick the distinct set first.

### The pin moved

`comp/champion.json` now pins this entry. `members` stays our six — `champion_check` holds that field to
what `comp/members.roster` can rebuild, and the foreign seat is rebuilt by a **Kaggle mount** of the public
CC0-1.0 `raptor-knee-widedense` dataset — so it is declared under `serving.foreign_seat` instead.

## 2026-08-28 · `kneemri-onepair` — 0.936, two seats buy 65% of what seven buy

`final_ws_cf` + the foreign `raptor_v4_swa` at 1/2 each. Ref 55834982.

**An instrument, not a champion attempt** — its expected score was below the pinned 0.943, and what it
buys is whether an ensemble's bonus comes from member COUNT or member DISTANCE. That decides where the
remaining slots and card-hours go, which is worth more than a tick on the board.

It is the fourth cell of a square whose other three were already on this ruler. The two solos agree to
three decimals, so member quality is held fixed and only count and distance vary:

| entry | seats | board | bonus over solo | per extra seat |
|---|---|---|---|---|
| `calibsolo` (`final_ws_cf` alone) | 1 | 0.923 | — | — |
| `raptor_solo` (the foreign member alone) | 1 | 0.923 | — | — |
| **`onepair`** | **2** | **0.936** | **+0.013** | **+0.013** |
| `calibsix` | 6 | 0.941 | +0.018 | +0.0036 |
| `raptorseat` | 7 | 0.943 | +0.020 | +0.0033 |

**The 6th and 7th members are worth ~0.0035 each; the 2nd distant one is worth 0.013 — 4× per seat.**
Two seats capture 65% of the seven-seat bonus. The gap to the champion is 0.007, under the owner's 0.02
bar, so a two-member ensemble is not *notably* worse than the six-member one at a third the runtime —
which is also the first thing this project has that speaks to the Efficiency LB (`knee-5bv`).

**The pre-registered 0.933–0.940 band fired**: distance is worth much more per seat than count, which
licenses a foreign-heavy roster via `RaptorSeat.fleet` over the distance-selected families.

**Stated limit, written before the number.** No all-ours two-member board entry exists, so the count term
at n=2 is inferred from the n=1 and n=6 endpoints. A reading inside 0.926–0.932 could not have separated
"distance is ordinary" from "our own pairs would do this too"; 0.936 sits *above* that band, which is what
licenses the distance reading — but the separating slot (a second all-ours pair) was deliberately not
spent, so the inference rests on the endpoints, not on a measured n=2 control.

**What it does NOT say.** It does not license dropping to two seats. `raptorseat` at 0.943 is still the
best row on the board and still the pin; 0.936 < 0.943 whatever the per-seat arithmetic says. The finding
is about where the NEXT seat should come from, not about how many to keep.

## 2026-08-29 · `kneemri-trioseat` — 0.938, and the prediction that missed it by 0.011

`final_ws_cf` + `grade_ladder_cf` + `through_dense_cf` at 1/3 each. Ref 55862930.

**The pre-registration named 0.925-0.929 and the board read 0.938.** That is the arm's own `>=0.938`
branch, written down as the surprise rather than the expectation. The reasoning behind the prior was
that all three seats sit in one pixel group — 20 slices, band 0.1-0.9 — so `knee-nas5`'s measured
pixel-group collapse should have eaten the count term and left the trio near its own first seat. It
missed by +0.009 to +0.013, several times the 0.0053 Hanley-McNeil floor. The prediction is what the
number refutes; the arm did what it was built to do.

| entry | seats | decodes | board | bonus over solo |
|---|---|---|---|---|
| `soloseat` | 1 | 1 | 0.925 | — |
| `onepair` | 2 | 2 | 0.936 | +0.011 |
| **`trioseat`** | **3** | **1** | **0.938** | **+0.013** |
| `calibsix` | 6 | ~3 | 0.941 | +0.016 |
| `raptorseat` | 7 | ~4 | 0.943 | +0.018 |

**What it actually measured.** Every other multi-seat entry on this board moves decode geometry and
supervision treatment together, so no entry before this one could separate them. These three seats are
matched on slices, band, resolution and backbone, and differ only in treatment — final weight-averaging,
the grade ladder, the through-plane read. Their +0.013 over solo is therefore a clean read of
supervision-treatment diversity alone, and it lands level with the +0.011 that `onepair` bought by
crossing an architecture boundary. Treatment diversity buys what architecture distance buys, for half
the decode cost.

**Which corrects how `knee-nas5` was being quoted.** That result constrains PIXEL groups; it was being
carried into planning as if it constrained SEATS, a strictly stronger claim it never measured. Three
seats in one pixel group are not one seat. The monoculture plateau survives — 0.938 is still under
`calibsix` 0.941 and `raptorseat` 0.943, so treatment diversity is a cheaper source of the same kind of
gain, not a replacement for the other four seats.

**Stated limit.** One board read, one hidden test set. The 0.002 over `onepair` is instrument noise and
nothing is claimed from it. The load-bearing quantity is 0.938 against a pre-registered 0.927.

**Efficiency track.** Three members, one decode, 0.005 behind a seven-seat nine-hour kernel. This is the
entry to pin there.

## 2026-08-29 · `kneemri-fleettwo` — 0.942, and the foreign-seating axis closes at one seat

`raptorseat_v1` (0.943) named exactly one follow-up in its own pre-registration: *whether a second
foreign arm adds again or whether one foreign decode is the whole of the effect.* This entry is that
follow-up — the champion seven plus `raptor_ft_effv2l480`, seated at 1/8 by the same `RaptorSeat.fleet`
rule, on one extra forward pass over the existing raptor decode.

| entry | our seats | foreign seats | board |
|---|---|---|---|
| `calibsix` | 6 | 0 | 0.941 |
| **`raptorseat`** | **6** | **1** | **0.943** |
| `fleettwo` | 6 | 2 | 0.942 |

**0.942 is the pre-registered FLAT branch** (0.9415-0.9444). Its text, written before the number: *one
foreign decode WAS the whole of the effect… closes the foreign-seating axis at one seat. Do not spend a
further slot widening it.* The remaining raptor families (`cnv2b336`, `cnn336`) are therefore not
queued. The gain `raptorseat` measured came from crossing the architecture boundary **once**, not from
distance accumulating across seats.

**Why the member was the most defensible one available.** `teacher_distance` scored all thirteen public
raptor checkpoints over 400 shared studies with `RankAverage.agreement` — no labels, no slot — and found
four families and nine copies (`v4_full` × `v4_full_swa` agree at 0.9955; they are one model twice).
`effv2l480` is the most isolated of the thirteen, nearest-neighbour agreement 0.8223. If distance
accumulated, this is the seat that would have shown it.

**Stated limit, pre-registered before the number.** The entry cannot attribute the null to the SEAT
versus to the MEMBER. Every raptor arm is `--fold -1`, so `effv2l480` has no measured quality at all —
its holdout sits inside its training set. 0.942 is equally consistent with "a second foreign arm adds
nothing" and with "this particular foreign arm is weak". Separating those costs a solo slot on
`effv2l480`, which was deliberately not spent then and is not worth spending now that the axis reads
flat under either reading. `raptorseat` is not retracted: `v4_swa` is board-measured at 0.923 solo.

## 2026-08-29 · kneemri-fourseat — 0.937, and the seating axis is exhausted

`fourseat_v1` (ref 55866570) added `small_batch6_ff` to trioseat's three seats, all four inside read A,
so the entry still cost one decode. It read **0.937** against trioseat's 0.938 — inside the
pre-registered 0.936-0.938 band, whose own text was *"the seat term is spent at three, do NOT run the
refits."* That branch fired, and the ~4-hour refit lane it gated is cancelled.

Every seating variant now has a board number, and they say one thing together:

| entry | seats | decodes | board |
|---|---|---|---|
| `soloseat_v1` | 1 | 1 | 0.925 |
| `onepair_v1` | 2 | 2 | 0.936 |
| **`trioseat_v1`** | **3** | **1** | **0.938** |
| `fourseat_v1` | 4 | 1 | 0.937 |
| `calibsix_v1` | 6 | 3 | 0.941 |
| `fleettwo_v1` | 8 | 4 | 0.942 |
| `raptorseat_v1` | 7 | 3 | **0.943** |

From three seats upward everything lands in 0.937-0.943 — a 0.006 span against a 0.0053 Hanley-McNeil
floor. The count term is real from one seat to three and indistinguishable from noise after that, on the
cheap decode budget and the expensive one alike. Adding a foreign architecture buys one step
(`raptorseat` +0.002) and a second one buys nothing (`fleettwo` flat).

**The consequence, which is the point of writing this down:** the seating axis is exhausted. Further
board movement has to come from a *better member*, not from another arrangement of the members already
trained. The eight whole-corpus read-A checkpoints found on disk are still worth having — but as
*candidates to beat*, not as seats to add.

**Efficiency track:** the pin stays `trioseat_v1`, 0.938 at one decode, 0.005 behind a seven-seat
nine-hour kernel. It is now known to be the cheap lane's ceiling rather than its first sample.

**Stated limit:** as with `fleettwo`, this cannot separate "a fourth seat adds nothing" from
"`small_batch6_ff` specifically adds nothing" — a `--fold -1` member has no offline quality number.

## `raptorweight_v1` — 0.944, the pin, and the weight is still not a lever

The table above closed at `raptorseat_v1` = 0.943 and declared the seating axis exhausted. One entry
landed after it and the doc did not say so for a day, which `board_check` is what caught (a row priced
in `comp/board.jsonl` and named nowhere here).

`raptorweight_v1` (ref 55875622, 2026-08-30) is `raptorseat_v1` with ONE float changed:
`RaptorSeat.weight` 1/7 → 1/3. Same seven members, same `LogitAverage`, same kit, same raptor
checkpoint, same rank-space join. It scored **0.944 — the best board row this project owns**, and
`comp/champion.json` pins it.

**The band label and the arithmetic disagreed, and the arithmetic won.** 0.944 − 0.943 is +0.001, under
this entry's own pre-registered `BOARD_RESOLVABLE` = 0.0015, and the band's 0.9435 edge is not an
attainable score at all since the board quantises to 0.001. So a 2.3× change in the foreign share moved
the board by one unresolvable tick. What is true is narrow: the pin moves because this is the best row,
NOT because the weight was shown to be a lever. With `fleettwo_v1`'s flat count reading beside it,
neither dimension of the foreign-seating axis has produced a resolvable move, and `knee-uco3`'s
inert-seat reading gains its second support.

**Do NOT sweep the weight to 1/2.** The gain branch's own instruction to do so was conditional on
clearing the resolution floor, which it did not.

**Stated limit:** this cannot separate "the weight is flat" from "THIS foreign member is inert at any
weight". Six of the seven seats are `--fold -1`, so no offline number exists or can exist for this pool.

## `greedyseat_v1` — 0.936, and the board priced an offline constant to +0.0001

`greedyseat_v1` (ref 55877491, 2026-08-30) is the first **procedurally selected** entry this project
has shipped. Seven seats named by a greedy forward search over 111 holdout records — not hand-picked:
`calib_dense_hold_s2026`, `merged_span_cf`, `labelwise_dense_s2026`, `blendhold_s2026`,
`finding_calib_dense_s2026`, `margin_seat_s2026`, `warm_calib_dense_s2026`. It was fired to buy back an
offline ruler, because `comp/champion.json`'s pin has six `--fold -1` seats and therefore no offline
number by construction.

It scored **0.936**, the bottom edge of the pre-registered 0.936–0.942 "as predicted" band.

**The instrument landed, and it validated an offline measurement on the board.** The two prior fold-0
anchors are 0.8712 → 0.933 and 0.8751 → 0.937, a slope of 1.026 board points per holdout point. Through
that line:

| offline fed in | predicted | board rounds to | actual | error |
|---|---|---|---|---|
| 0.8768 — the greedy's raw in-sample figure | 0.9387 | 0.939 | 0.936 | −0.0027 |
| 0.8740 — bias-corrected | 0.9359 | **0.936** | 0.936 | **+0.0001** |

The greedy selection bias was measured purely offline at **+0.0028** (111 records, seven seats,
`knee-jsgb`). The board independently prices the same correction at **+0.0027**. That cross-ruler
agreement is worth more than the row's score. It also disposes of an apparent **ruler inversion** —
offline up 0.0017 against the previous anchor while the board went down 0.001 — which is not an
inversion at all, but the greedy bias; correcting for it restores the ordering. `comp/board.jsonl`
records the corrected 0.8740, never the raw figure.

**Procedural selection does not beat hand-picking.** 0.936 against `raptorweight_v1`'s 0.944 is a 0.008
shortfall. The fold-0 all-data dose accounts for roughly half (−0.004/−0.005, previously measured),
leaving ~0.003–0.004 for "greedy is worse than curated" — present, but only a couple of ticks above the
0.0015 board resolution.

**Stated limit:** this cannot separate those two terms. No hand-picked fold-0 seven exists, so the
comparison crosses the fold boundary and the dose is an estimate, not a matched control. And a landed
anchor does not make the greedy ORDER a ranking: the search is in-sample over 111 members, and its
seats 10–11 go to the pool's two worst inside a plateau where +0.0024 buys twenty-five seats. The
champion pin does not move.

## `vjepaseat_v1` — 0.943, and the board says what the holdout said

`vjepaseat_v1` (ref 55895006, 2026-08-30, kit `53f81d3`) is `raptorweight_v1`'s exact eight-seat serving
with **one seat added**: `vjepa_warm_s2026`, the only non-`dinov2-small` member this project has ever
shipped. A V-JEPA2 ViT-L trunk warm-started from video pretraining, nearest-neighbour agreement **0.8119**
against the pool's 0.91–0.94, and the most distant member the project owns. Vote (`LogitAverage`), foreign
weight (1/3), kit and every other seat are the champion's, untouched.

It scored **0.943** against the matched control's **0.944**. That is −0.001, inside this entry's own
`BOARD_RESOLVABLE` = 0.0015 and an order of magnitude under the house bar. The honest reading is **flat**:
the most architecturally distant member the project owns bought nothing in the champion's own seating, and
cost nothing either. The pin does not move.

**What earns the slot is that this is a board-level confirmation of an offline seat measurement**, on the
axis the ledger has been arguing about for days. Offline, `swin_supervised` prices at seat value −0.0012
(band [−0.0020, −0.0003], best possible swap −0.0013) despite being the most decorrelated member the
project owns at 0.8346 nearest-neighbour. This entry runs that experiment on the board, with a *different*
distant member, and returns the same answer to the same resolution.

| ruler | member | distance | verdict |
|---|---|---|---|
| holdout seat value | `swin_supervised` | 0.8346 NN | −0.0012 |
| board, matched control | `vjepa_warm_s2026` | 0.8119 NN | −0.001 |

Two rulers, two members, one sign. **The non-DINOv2 lane is quality-bound, not diversity-bound**, and that
now rests on a board reading and not only on a holdout one. The consequence is a size: a member at this
correlation needs roughly **+0.017 to +0.020** solo over the lane's current 0.8331 to buy a seat rather
than sit inert in one. See `interpretations/kneemri/2026-08-30_the-non-dino-lane-is-quality-bound.md`.

**Clean mount, verified.** `run_check` on the kernel log reads `loaded 7 of 7 members`, and the log's own
line says `loaded and fingerprint-checked 7 of 7`. Unlike `farseat_v1`, this entry IS what it was named as,
so the flat is a reading about the member and not about a silent drop.

**What it cannot separate:** "a distant member is inert in this seating" from "*this* member is inert", and
both from "the eighth seat is inert regardless of who fills it" — `knee-uco3`'s inert-seat reading, which
gains a third support here and is now the most economical account of `raptorseat_v1`, `raptorweight_v1` and
this entry together.

## `holdswap_v1` — 0.944, exactly the champion's number, and the bands were anchored at the wrong score

`holdswap_v1` (ref 55900461, 2026-08-30, kit `53f81d3`) is `raptorweight_v1`'s exact seven-seat serving
with ONE seat exchanged: `lr4e-5_ws_cf` out, `calib_dense_hold_ff` in. Vote (`LogitAverage`), foreign
weight (1/3), count, kit and every other seat held. The entering member was named by a fold-0 swap table
as the highest-priced single-seat swap available, at +0.0022 seat value — above the entry's own
`BOARD_RESOLVABLE = 0.0015`. The question was not whether that member is good; it was whether a **fold-0
proxy can ORDER blind seats**, which is the only cheap instrument that would let the roster improve
without spending a slot per candidate.

**It read 0.944 against the matched control's 0.944. Not a small gain — no gain, the same number to the
board's full precision.**

### The pre-registered bands were mis-drawn, and their labels must not be quoted

The kernel wrote its bands against "the champion's 0.944" in prose and then anchored the arithmetic at
0.941. Its "a real gain over the champion, above the resolution floor" band opens at **0.9425 — below the
champion** — and its FLAT band (0.9415–0.9424) sits lower still. So this score lands numerically inside a
band labelled *a real gain* while being arithmetically identical to the control.

This is the **second** time in this project that a band edge was written at a score that could not mean
what its label said. `comp/champion.json` records the first, for `raptorweight_v1`, where the "real gain"
edge of 0.9435 was not an attainable score at all because the board quantises to 0.001. A pre-registration
is worth what its ARITHMETIC says, never what its label says — and the error is caught by subtracting the
control, not by reading the band. The reading taken here is FLAT, on the delta. The band structure is
discarded.

### What the flat buys, which is the whole value of the slot

**The fold-0 proxy does not order blind seats.** It priced this swap at the top of its table and the board
returned nothing, so the champion's six are interchangeable at this quality level and the cheap instrument
does not survive the `--fold -1` crossing. That closes the **single-seat swap axis**: no further slot goes
to exchanging one champion seat for a near-equal member, and the two blind refits queued behind it
(`merged_span_ff`, `labelwise_dense_ff`) are pool members rather than swap entrants.

### Third flat on the seating axis, and the pattern is the finding

`raptorweight_v1` moved the foreign SHARE by 2.3× for one unresolvable tick. `vjepaseat_v1` ADDED the most
architecturally distant member the project owns for −0.001. This entry EXCHANGES a seat for the
best-priced alternative for 0.000. Three different operations on the seating — reweight, add, swap — and
not one produced a resolvable move. `knee-uco3`'s inert-seat reading gains its **fourth** support and is
now the most economical explanation of all of them: at this pool's quality the *arrangement* of seats is
not a lever. What is left are the levers that change what a member SEES (corpus, decode) or WHAT IT IS
(trunk).

**Clean mount, verified.** `run_check` reads `loaded 6 of 6 members` and the log's own line agrees, so the
flat is a reading about the swap and not a silent drop. Kit clean, not DIRTY. **The champion pin does not
move** — 0.944 ties the pinned row, and `champion_check` pins the best row, not an equal one.

## `spanseat_v1` — 0.944, the third single-seat exchange in a row to read flat

`spanseat_v1` (ref 55908066, 2026-08-31, kit `53f81d3`, kernel `kneemri-spanseat`) is the seven-seat
serving again — the raptor member decoded first into a scratch directory, then six of ours on the
budget it left — with one seat moved against `holdswap_v1`: `calib_dense_hold_ff` out, `merged_span_ff`
in. Vote (`LogitAverage`), foreign weight, count, kit and every other seat held.

The point of the swap was KIND rather than fit. `calib_dense_hold_ff` and the seats around it are the
same recipe family reaching the same ceiling from slightly different directions; `merged_span_ff` is a
span member trained on the merged corpus, which is the closest thing the roster has to a member that
saw different data. The owner's framing is the one it was fired on: an ensemble is about diversity, not
about averaging seven numbers that are the same number.

**It scored 0.944 — flat, to the board's own quantum.** `holdswap_v1` read 0.944 and `vjepaseat_v1`
read 0.943 before it, each one seat away from the same six. Three consecutive single-seat exchanges at
the top, one of them a genuinely different kind of member, and none of them moved the board. Read
plainly, that is what a saturated seating looks like from the outside: at 0.944 the marginal seventh
seat is worth less than the board can resolve, and no reseating of this pool is the lever. It does not
say the members are equivalent — the ruler here is the board's 0.001 quantum, and three flats inside it
bound the effect rather than measuring it.

What it does say, and this is the part worth acting on, is that the next slot should not buy a fourth
seat exchange. Either the pool gains a member that is better rather than merely different, or the entry
changes shape (size, vote rule, or the corpus a member is fitted on) — a candidate that reads DISTINCT
on `member_axes` rather than one that reads far on rank correlation alone.

## 2026-08-31 — `stackseat_v1` closes the fit axis, and `peak8` prices greedy selection bias

Two entries went out inside an hour. The first is a clean result. The second was first written up here
as a refutation of the offline ruler; that write-up was wrong, and this section replaces it.

`stackseat_v1` asked the FIT axis. It is `raptorweight_v1` with `stack_increments_ff` in place of
`merged_blend_cf`, the champion's one remaining fold-0 seat. Count, kit and vote held; the single moved
term is HOW that seat was fitted — `--fold -1 --ema`, on the project's highest solo (`stack_increments`
0.8642, the only member distilled from `raptor_teacher`). **It read 0.944 against a control of 0.944:
exactly 0.000**, the pre-registered flat band, which the entry itself said would close the fit axis
alongside count and weight.

That is the fifth flat on the champion pool. Reweight (`raptorweight_v1` +0.001), add (`vjepaseat_v1`
-0.001), swap (`holdswap_v1` 0.000), count (`dark7_v1` flat), kind (`spanseat_v1` 0.944) and now fit
(0.000). Two axes are closed on this pool: neither WHO sits nor HOW they were fitted moves it. That is
also not anomalous — `interpretations/kneemri/2026-08-31_the_pool_is_redundant_not_the_task.md` bounds
any new member at 12.7% of the pool's remaining uncertainty, so flat is what the redundancy predicts.

`peak8` shipped the eight-member set `member_curve`'s greedy selects over the pool, RankAverage, all
fold-0, 8 of 8 loaded. **It read 0.932**, and the first version of this section called that a failure of
the offline ruler. It is not. Three things were wrong with that reading, and all three are checkable.

**The row was entered raw.** `knee-jsgb` closed on 2026-08-30 with a standing rule: an entry selected by
search enters `board.jsonl` bias-corrected, never raw. `member_curve.honest` measures that bias directly
at +0.0025 to +0.0031 for 7-12 seats over pools of 77-119 members, replicated across four logs.
Corrected, peak8 is 0.8704, which predicts 0.934 against an actual 0.932. The raw 0.8731 predicts 0.936,
and that 0.004 of phantom shortfall was the entire basis of the error. `board_check` does not yet assert
the rule, which is how a raw number got written the day after the rule was made (knee-wz81).

**peak8 is not the best offline set the project owns.** It is THIRD — behind `honest12` (0.8751 offline,
0.937 board) and `greedyseat_v1` (0.8740, 0.936) — and it is third on the board as well. The holdout
ordered all three correctly. The claim that the offline argmax is not a board argmax rested on peak8
being that argmax, and it never was.

**The ruler has never mis-ordered a pair it resolves.** `board_transfer`, refitted with this row present,
reports 11 of 421 pairs the board ordered the other way and **0 of 269** of the pairs the holdout actually
resolves at a gap above its 0.0053 floor. Both peak8 inversions print UNRESOLVED, at +0.0037 and +0.0036
against board gaps of a single tick. peak8 extended the clean run from 0/233 to 0/269 — it strengthened
the ruler's record rather than breaking it.

So `member_curve`'s greedy and `seat_value`'s argmax remain valid instruments and are reinstated; they
were retired here on a misreading, for the few hours between the two versions of this section. What peak8
actually bought is the confirmation that greedy selection bias is real, is about 0.0027 at this pool size,
and must be subtracted before a search-selected entry is priced — plus a defect worth fixing, that no gate
enforces the rule that would have caught it.

The general lesson is one the repo already writes down: an UNRESOLVED pair is not a refutation. The ruler
declined to order two pairs and was read as having failed them, and the conclusion drawn from that — that
offline selection could no longer justify a slot — would have moved every future member decision onto the
board, at five slots a day.

## 2026-08-31 — `champviews_v1` and `halfcham_v1`: the last two `Serving` knobs, both flat

Two instruments, three and a half hours apart, both on the champion's exact seating with one argument
moved. Between them they answer the last unasked questions about how an entry is SERVED, and the answer
is the same twice: nothing.

`champviews_v1` moved `Serving(views=SpatialViews())` and read **0.943** against a control of 0.944 —
inside the pre-registered flat band, and one tick down, which is under `BOARD_RESOLVABLE` (0.0015), so
the sign is not readable and the honest word is FLAT rather than a cost. `run_check` read 6 of 6 members
loaded with no `afford` drop, so this is not the `farseat_v1` defect wearing a null's clothes.

`halfcham_v1` moved `Serving(precision=torch.float16)` and read **0.944** — exactly the control. That
number is the one this entry had to work hardest to earn, because an entry reading exactly its control is
ambiguous between "fp16 is free" and "the knob was never connected", and the disconnected case has
already happened here: an earlier loadcheck wrapped `predict_member` in an OUTER autocast, the inner one
carried an explicit dtype and overrode it, and twelve arms dutifully reported `max|diff|` of exactly 0.0.
Only the log separates the two, and it was read before the slot was spent —
`serving: vote=LogitAverage views=identity only (default) precision=torch.float16`. The knob fired.

So fp16 serving costs nothing measurable on the board. That is worth having for a reason the score does
not show: the efficiency board scores wall-clock, and a faster session also widens the margin against
`Ensemble.afford`, which gives up the weakest member rather than the entry when a ceiling is threatened.
Half precision is the default for every later entry on those grounds alone.

**What the two rows close, together with the six seating flats.** All four `Serving` knobs are now
answered — vote (`votefit_v1`, 0.938, 2026-08-25, which is why the champion carries `LogitAverage`),
views (flat), budget (implicit in every entry) and precision (flat). Reweight, add, swap, count, kind and
fit are six flats on WHO SITS. Neither who sits nor how they are served moves this score, and no further
slot goes to either. It is also not anomalous: `2026-08-31_the_pool_is_redundant_not_the_task.md` bounds
any new member at 12.7% of the pool's remaining uncertainty, so flat is what the redundancy predicts.

What the record leaves open is the two axes no serving knob can reach: what a member SEES (corpus, decode
at TRAIN time) and what it IS (trunk). That is why the standing work is the feature-distillation chain
rather than another seating.

**Neither row licenses**: anything about the NUMBER of views (three is `SpatialViews`'s default, held,
never swept), about views on a member that does not read the whole stack — where overlap TTA is not
already known to be spent (knee-eeo) and the two axes would confound — about fp16 TRAINING (the standing
recipe is bf16 train / fp16 serve, and this asked only the serving half), or about bf16 serving, a
different exponent range that remains untested. `witness_check`'s eps was calibrated against
fitted-precision members and is not re-read by this result.

## 2026-09-01 — `lensseat_v1`: the pooling lens is on the competence scalar, and that is the sixth direction

`lensseat_v1` (ref 55940907) gave the champion's most-swapped seat to a different POOLING LENS.
`raptorweight_v1`'s seating, count, kit `ba241cf4`, order, `overlap=False`, `LogitAverage` and
`FOREIGN_WEIGHT=1/3` all held; `lr4e-5_ws_cf` out, `labelwise_dense_s2026` in — the only roster member
fitted with `--slice-pooling labelwise`, and the project's third-best solo at 0.8613. It read **0.944**
against a control of **0.944**: exactly 0.000, the pre-registered FLAT band.

The claim it was built to test was that a pooling FUNCTION is not a point on the competence scalar,
because it changes which slice evidence survives to the head. `competence_axis` reads 81.7–82% of what
separates 105 members as ONE scalar over 909 studies, and that is the mechanism under the five earlier
flats (`raptorweight` +0.001, `vjepaseat` −0.001, `holdswap` 0.000, `spanseat` 0.944, `stackseat` 0.000,
with `halfcham` closing precision alongside). The board says the scalar absorbs the pooling axis too.

**What this retires**: member selection from the CURRENT roster. No further slot goes to choosing,
weighting, counting or re-lensing members already in the pool.

**What it does not retire**: seating a member that is BOTH distant from the pool and above the lane's
size bar. Every seating flat on this record moved in a member that was either distant-and-weak
(`vjepaseat`) or strong-and-twin — `stack_increments` reads SCALED at rank 0.891, and across 26 unshipped
members priced against `honest12`, solo strength and distinctness are ANTI-correlated: only
`attend_dense` (rank 0.8916, solo 0.8569), `aprior_dense` (0.8811, 0.8557) and `focal_dense` (0.8887)
are DISTINCT while clearing the 0.8521 size bar. `distinct14` (ref 55943714) asks that question on a
fully-scoreable `honest12` base, where every member is matched-fold and no all-data dose confounds it.

**Not licensed**: reading this as "pooling does not matter" at TRAIN time. `labelwise_dense_s2026`'s own
0.8613 solo says the lens builds a competent member. What is refuted is that the lens buys ENSEMBLE
diversity against this pool.

## 2026-09-01 — distinct14: the member axis is closed, and this is the flat that closes it

Ref 55943714. Board **0.937**. Control `honest12` **0.937**. Delta **0.000**, dead centre of the
pre-registered FLAT band (0.936-0.938), whose own text — written into the kernel docstring before any
number existed — reads *"FLAT; closes the member axis, not just seating."*

The entry is `honest12` plus two seats: `attend_dense_s2026` (rank 0.8916, solo 0.8569) and
`aprior_dense_s2026` (rank 0.8811, solo 0.8557). They are the only two members in the pool that are
simultaneously DISTINCT against the roster and above the non-DINOv2 lane's size bar of 0.8521.
14 of 14 members mounted, verified by `run_check` before the slot was spent.

### Why this flat is not the seventh copy of the six before it

Every prior seating flat moved a member that was either distant-and-weak or strong-and-twin:

| entry | delta | the member it seated | the escape hatch it left open |
|---|---:|---|---|
| raptorweight | +0.001 | reweight, no new member | weight is not the axis |
| vjepaseat | −0.001 | `vjepa_warm` rank 0.8212, solo 0.8337 | 0.018 under the size bar |
| holdswap | 0.000 | swap at matched strength | no new information added |
| dark7 | flat | count | seats are not the axis |
| spanseat | 0.944 | `crop80_span` rank 0.8653, solo 0.8466 | 0.006 under the size bar |
| lensseat | 0.944 | pooling lens | lens is on the competence scalar |
| **distinct14** | **0.000** | **distinct AND above the bar** | **none** |

Each of the six had a live alternative explanation: the member was wrong, not the axis. This entry
removes that explanation by construction. It is the first one that could not be dismissed as a bad pick.

### The measurement underneath it

A sweep of 83 unpriced roster members against `honest12` classified 25 DISTINCT. Joining that column to
solo AUC gives an almost monotone anti-correlation — only **3 of 25** clear 0.8521, all at rank 0.88-0.89:

```
PASS  small_feat_fit_s2026        rank 0.8904  solo 0.8571
PASS  sided_pool_s2026            rank 0.8879  solo 0.8561
PASS  aprior_dense_s2026          rank 0.8811  solo 0.8557
under through224_s2026            rank 0.8719  solo 0.8505
under crop80_span_s2026           rank 0.8653  solo 0.8466
under vjepa_warm_s2026            rank 0.8212  solo 0.8337
under dinov3_base_lora_s2026      rank 0.8187  solo 0.8268
under lora_r8_v3small_224_s2026   rank 0.8176  solo 0.8217
under focus_mcl_s2026             rank 0.0445  solo 0.5332
```

`focus_mcl` is the proof of the shape: the most decorrelated object the project owns, scoring 0.533.
It is distinct because it is broken.

Kimi's Q-F research (`research/deep_dives/2026-09-01_decorrelation_without_solo_cost.md`, 19 sources) says this is
the **expected** shape near the frontier rather than an artefact of this pool: no published method buys
strong decorrelation without 1-3 points of solo cost, and Abe et al. 2024 measure diversity pressure
dropping ImageNet ensemble accuracy 71.49 to 70.16. Their letter pre-committed the reading: *"If
`distinct14` lands flat, the anti-correlation is effectively a law for our regime and the only open
levers are supervision and trunk family."* It landed flat.

### What is retired

Member selection from the current pool, entirely — reweight, add, swap, count, precision, pooling lens,
and now distance-with-quality. `small_feat_fit_s2026` and `sided_pool_s2026` were the natural follow-up
entry and are **not** being shipped: that would ask the answered question an eighth time.

### What is not licensed

- Not "ensembling is spent". This entry's base scored 0.937, which is 0.007 below the champion,
  so it measures the member axis on the weaker of the two bases. A 14-seat entry that cannot move
  0.937 says nothing about whether the same two members would move the champion.
- Not a refutation of `attend_dense` or `aprior_dense` as members. Both clear the size bar and stay
  on the roster.
- Not an argument against diversity. It is an argument that the pool has no more of it to spend.

### What remains

What a member IS (trunk family) and what it is TAUGHT (supervision). Those are the two levers Kimi's
letter names and the only two this flat does not touch.

## `fasttrio_v1` — 0.939, and the efficiency track gets an entry that costs the accuracy row nothing

Submitted 2026-09-01 20:32Z, ref `55947454`, kernel `kneemri-fasttrio`, kit `ba241cf4`. Three seats
(`final_ws_cf`, `grade_ladder_cf`, `through_dense_cf`) — `trioseat_v1`'s same-pixel-group trio, served
with two arguments moved and nothing reseated: `precision=torch.float16` and `vote=LogitAverage()`.
The owner selected it for the efficiency track.

**Read 0.939 against `trioseat_v1`'s 0.938.** +0.001, inside the pre-registered FLAT band
0.9365–0.9395 and under `BOARD_RESOLVABLE` 0.0015.

### What it settles

Both moved arguments were already measured free at SEVEN seats. `halfcham_v1` read its control's score
exactly in forced fp16. `votefit_v1` put `LogitAverage` above the `RankAverage` default that
`trioseat_v1` was carrying by omission. This says they are free at THREE seats too. The `>=0.940`
branch — the vote is worth more where there is least to average — did not fire. Averaging rule and
serving precision are flat against seat count.

### Why it cost the accuracy row nothing

`efficiency_read` verified the mechanic before the slot went: of 2,143 settled teams, 475 carry an
efficiency entry strictly BELOW their public best and **0** carry a better one. A deliberately cheap
submission adds an efficiency candidate rather than displacing the 0.944.

### Not licensed

Reading 0.939 as an accuracy result. It is three seats where the champion has seven, and it was built
to be cheap. The accuracy row stays 0.944, 0.010 behind a 0.954 leader.
On 2026-08-30 it stood at 45th of 3239; the board puts the same score at
**rank 77 of 3646** (read 2026-09-13T11:27:46Z) — the score did not move, the field did, which is the only thing a remembered rank
can never tell you. Thirty-two places in fourteen days, NINETEEN of them in the four days since the
2026-09-09T19:53Z read (58th of 3421), is the drift rate this line exists to make visible, and it is
holding: six more ranks in the nineteen hours since the 2026-09-12T16:54Z read (71st of 3602) is still
about a rank every three hours, so standing still costs eight ranks a day.

## corpusdose_v1 (0.943) + corpusfull_v1 (0.940) — the external-corpus dose curve

Two slots, 2026-09-01, refs 55948598 and 55949239, both on kit `ba241cf4`. They complete a four-point
curve in ONE variable: how many seats of a fixed-size ensemble are members trained on the merged
external corpus rather than on RSNA alone.

| external seats | entry | board |
|---|---|---|
| 1 | champion | 0.944 |
| 2 | `spanseat_v1` | 0.944 |
| 4 | `corpusdose_v1` | 0.943 |
| 6 | `corpusfull_v1` | 0.940 |

### What it settles

The curve is flat to two seats and BENDS DOWN after. The six-seat point is -0.004 against the champion,
above `BOARD_RESOLVABLE` 0.0015, which fires `corpusfull_v1`'s pre-registered `<= 0.9425` branch. More
external-corpus members is not more score; the external corpus buys its one seat and no more.

### The confound, declared before the slot went

Reaching six seats forced `grade_ladder_cf` and `through_dense_cf` — the two structurally most distant
members in the pool — OUT at the same time the external dose rose. So the -0.004 is dose-up AND
distance-down together, not dose alone.

### Not licensed

Reading the -0.004 as evidence that STRUCTURAL DISTANCE is a newly discovered lever. `distinct14`
maximized member distinctness and read 0.937, worse than both endpoints of this curve. Distance-up and
distance-down both underperform the champion, which is the signature of a CLOSED axis, not a discovered
one.

### What it does NOT close

External data itself. A trunk CONTINUED on a foreign corpus is a different mechanism from a member
TRAINED on one, and `ortho_dense` / `v3large_dense` (knee-440m) ask that question separately.

## `aucvote_v1` — 0.943, and the combiner axis reads flat where it actually ships

**Ref 55965832, 2026-09-02, kit `d350e987`.** The champion's six seats in the champion's order
(`final_ws_cf`, `calib_span_ff`, `grade_ladder_cf`, `merged_blend_cf`, `lr4e-5_ws_cf`,
`through_dense_cf`), `overlap=False`, raptor foreign seat at `FOREIGN_WEIGHT=1/3` — and ONE argument
moved: `LabelQualityRank` (per-finding AUC weighting of ranks) instead of `LogitAverage`.

**Read 0.943 against the champion's 0.944.** −0.001, under `BOARD_RESOLVABLE` 0.0015, which is the
pre-registered 0.9425-0.9455 FLAT branch.

### Why the slot went to a rule that had already measured null four times

All four nulls were offline, on fold-0 pools. The champion is six-of-six `--fold -1`, so no offline
ruler can see the seating we ship and none of those four contained it. This is the first reading of the
vote rule at the arrangement that goes to the board.

### What it closes

The combiner is not a lever at six seats. Ninety-three board rows had never once changed the rule by
which seats combine; this changes it and nothing moves.

### What it does not separate — declared before the slot

A flat cannot tell *the rule is inert* from *the twin-derived weights are too noisy to act on*. Five of
the six weights are fold-0 twins' holdout AUCs and `final_ws_cf` took the pool mean, because its
ancestor `final_ws_e12` is dark and no twin was ever trained. `stackvote_v1` (ref 55966986, StackedLogit
on the same seating) is the second point: it derives its weights by a different fit, so a flat there
makes the twin-noise reading much harder to hold.

### Not licensed

A third combiner read. `kneemri-rankvote` is built and `run_check`-clean and pre-registers at −0.0003;
two points already bound this axis inside 0.0015 and a third would spend a slot to confirm a bound.

### Correction, 2026-09-02T20:12Z — `aucvote_v1` closed nothing

The entry above claims this reading closes the combiner as a lever at six seats. That claim is wrong and
the entry could not have supported it. Measuring the table it actually shipped:

| entry | max deviation from equal share | mean deviation | negative weights |
|---|---|---|---|
| `aucvote_v1` (`PER_LABEL`) | **0.54 pp** | 0.10 pp | 0 of 72 |
| `stackvote_v1` (`STACK_WEIGHTS`) | **24.83 pp** | 5.35 pp | 1 of 72 |

`aucvote_v1` weights each of the six to within half a percentage point of an equal share. That is
arithmetic, not misfortune: the six members' per-finding AUCs all lie between 0.776 and 0.917, and a
quality weighting over numbers that close is the identity map. The file that was submitted was the
champion's file. 0.943 against 0.944 measures the board's own repeatability, not a vote rule.

What survives is narrower and still worth writing down: **AUC-derived weighting over a pool this uniform
cannot move a board.** That kills the cheap form of the idea. It does not touch weighting itself.

`stackvote_v1` is the one that tested it — a fitted table that puts one seat at 41% of a finding and
shorts another — and it read the same 0.943. A real tilt matching the equal-weight benchmark is the
ordinary result when the holdings are CORRELATED, which ours are: one trunk family, one corpus, one
decode. Weighting pays on dispersion AND low correlation, and this pool has neither.

Both entries inherit the same defect and therefore do not independently confirm each other: five of six
weight rows in each are fold-0 twins, because the shipped seats are `--fold -1` and have no holdout.
The axis that is actually closed is twin-derived weighting. Weighting by the shipped members' own
quality has never been measured and cannot be without retraining the six with a holdout.

## twinfive_v1 — 0.935 (ref 55969178, 2026-09-02 22:04Z)

The five fold-0 twins under `LogitAverage`, shipped as an INSTRUMENT: the first paired
offline-and-board point the record holds above offline 0.8700 (offline 0.8705, honest — every
holdout study was withheld from every member's own training).

**Do not read this as a flattening slope.** Its kernel docstring pre-registered 0.930-0.935 as
"the slope FLATTENS at the top"; that branch was mis-specified before the run (`knee-k6vwv`).
`peak8` already sits at offline 0.8704 -> board 0.932, so two offline points 0.0001 apart landed
0.003 apart on the board and no slope is separable from them.

**What the slot bought: the twin-to-shipped gap is 0.009** (champion 0.944 vs this 0.935) — nine
times the 0.001 spread across the three vote rules. `aucvote_v1` and `stackvote_v1` both fitted
their tables on exactly these twins and served the `_ff`/`_cf` seats, so their named confound is
real and an order of magnitude larger than the effect they measured. Confounded in turn by seat
count (five here, six there), which has read flat across seven entries.

See `comp/board.jsonl` for the full reading.

## monotone_v1 — 0.944 (ref 55967974, 2026-09-02 20:42Z)

The champion's six seats held byte-identical, with every target column **cubed** after blending.
Cubing is strictly increasing, so every within-column ordering survives while every probability
value moves. An identity test, not a comparison.

**It read exactly 0.944 — the champion's own score.** The metric is per-column rank-based.

**Closed forever:** temperature scaling, threshold shaping, isotonic/Platt fitting, per-label
probability rescaling, ECE minimisation as a shipping objective. Every monotone post-hoc transform
of a submitted column is worth zero on this board. No slot may be spent on one, and no offline gain
reported in ECE terms is a board gain.

**Not closed:** `--calibrated-targets` is a training signal, not a post-hoc transform — it changes
the column's ordering, which this says nothing about. The `_cf` members stay valid.

Three axes now read flat within 0.001: who SITS, how they are COMBINED, how the output is SHAPED.
The lever is in the members.

## `fiveship_v1` — 0.941. The 0.009 splits 1:2, and two thirds of it is the twin.

**Ref 55970549, 2026-09-03, kernel `kneemri-fiveship`, kit `6a52c849`. Five members, no offline number.**

The champion's own shipped members with `final_ws_cf` dropped: `calib_span_ff`, `grade_ladder_cf`,
`merged_blend_cf`, `lr4e-5_ws_cf`, `through_dense_cf`, `overlap=False`, `LogitAverage`. Exactly the
shipped counterparts of the five `twinfive_v1` shipped as fold-0 twins.

**THE SPLIT.** `twinfive_v1` read 0.935 against the champion's 0.944, and that 0.009 was confounded two
ways at once — twin-vs-shipped weights, and five seats against six. This entry holds the seats at five
and the weights at shipped, so the confound comes apart:

- champion 0.944 − fiveship 0.941 = **the SEAT term is 0.003**
- fiveship 0.941 − twinfive 0.935 = **the TWIN term is 0.006**

Pre-registered at 0.938-0.941 as "the two terms split evenly". They split **1:2**, which lands inside
that branch — and, importantly, NOT inside 0.942-0.944, the branch that would have made the twin gap
the whole story and the seat free.

**WHAT THE TWIN TERM COSTS US.** Every number this project infers from a fold-0 twin understates the
shipped pool by about **0.006 on the board**. That is most of the 0.008 gap to the leader. It is also
the gap that `aucvote_v1` and `stackvote_v1` were reading across: both fitted their vote rules on twins
and then served the shipped `_ff` / `_cf` members. Their flats stand as board readings of the combiner
axis, but they were never measuring the shipped pool's own combiner.

**WHAT IT DOES AND DOES NOT SAY ABOUT SEATING.** One seat out of six costs 0.003 — three board quanta,
so the seat-count axis is not exactly flat. But the seven entries that read flat all RESEATED at
constant count; subtraction is a different operation and this is its first reading. It does not reopen
reweight, swap, precision, lens or distance, and it is emphatically not a licence to add a seventh
seat: 0.003 is the price of removing the sixth, not the value of a seventh.

No offline number is recorded and none could be: four of the five are `--fold -1`, so every holdout
study is in-sample for part of the pool. Writing a twin's score into this row would poison
`board_transfer` the same way `raptor_solo`'s in-sample 0.8938 did.

`run_check` before the slot: loaded 5 of 5 members, no drop, no identity change. No card time.

## `holdfive_v1` — 0.936. The supervision axis is board-flat, and the label lane loses its card claim.

Pre-registered band **0.933-0.936 HIT**, and it is the branch that says stop. Five `--hold-annotations`
members against `twinfive_v1`'s five ordinary fold-0 members — same count, same fold-0 status, same
`LogitAverage`, same `overlap=False`, same kit machinery. One term moved: the supervision family. The
board moved **+0.001**, which is under `BOARD_RESOLVABLE` 0.0015 and therefore not a movement at all.

**THE OFFLINE GAP DID NOT TRANSFER, AND THE TRANSFER IS THE WHOLE READING.** Offline this pool reads
0.8729 against twinfive's 0.8706 — **+0.0023**, measured before the push by `vote_ceiling` over 909
shared fold-0 studies, split-half, overfit gap +0.0000. The fitted `board_transfer` slope of +1.55
predicted 0.938-0.939. It arrived at 0.936. So either the slope is wrong at this end of the record, or
this particular offline gap is not the kind of gain that transfers — and the second reading is the one
the de-confounding below supports.

**WHAT THE FLAGS ARE ACTUALLY WORTH, MEASURED THE SAME DAY AND FOR FREE.** The entry's own docstring
admitted its two supervision flags were fused: all five seated members carry `--calibrated-targets` as
well as `--hold-annotations`. A calib-only five — `calib_span_s2026`, `calibrated_targets_s2026`,
`merged_dense_cf`, `merged_labelwise_cf`, `merged_span_cf`, the fold-0 members carrying the calibration
flag and NOT the hold flag — reads **0.8724** by the identical estimator over the identical 909 studies.

| pool | offline |
|---|---|
| ordinary five (`twinfive_v1`) | 0.8706 |
| calib-only five | **0.8724** |
| hold + calib five (this entry) | 0.8729 |

So the +0.0023 splits **+0.0018 calibration / +0.0005 hold-annotations**. The flag the entry is named
for carries about a fifth of a gap that then transferred at zero. Both terms sit under the 0.0062
report-holdout noise floor individually; the split is the reading, not either number. The caveat is
real and bounds rather than isolates: the calib-only five is matched on FLAGS, not on members, so
member identity is confounded with the flag.

**WHAT THIS CLOSES.** The label-supervision axis is board-flat at full pool dose. That is the fourth
axis to read flat after seating (seven entries), combining (three vote rules) and shaping
(`monotone_v1` at exactly 0.944). The pre-registration named the consequence in advance and it stands:
this stops the label lane from consuming card. `reader_agreement`'s 108 confident false positives are
still a real defect in the labels, and the four `--hold-annotations` checkpoints still read +0.538 to
+0.700 against their controls on the gold cells — but those gains live inside 58 annotated studies,
1.3% of the corpus, and at pool scale on the board they are worth nothing measurable.

**WHAT IT DOES NOT CLOSE.** It reads five seats, not the champion's seven, so it says nothing about
whether the same change moves a 0.944. It is fold-0 throughout, so the twin caveat applies to its
absolute number — which is why every branch was written against `twinfive_v1` and not the champion.
And `--calibrated-targets` as a TRAINING signal is not what `monotone_v1` closed; `monotone_v1` closed
post-hoc monotone transforms, and the calibration flag remains the larger of the two supervision terms
even though neither transferred.

`run_check` before the slot: loaded 5 of 5 members, no drop, no identity change. `mount_check --live`:
ok. `kit_check`: 0 commits behind. One slot, INSTRUMENT, no card time.

### CORRECTION to the row above, made within the hour, and it inverts the reading.

The pre-registration was written against a `board_transfer` slope of **+1.55 board points per holdout
point**. That constant is not what the tool says and never was. Run against the record with this entry
in it, `python -m kneemri.evaluation.board_transfer` reports:

```
compression: median +1.25 over the WHOLE record — "the wrong number to price an arm with"
  among entries at or above 0.8678:  13 resolved pairs, median +0.97  [+0.94, +1.17]
  among entries at or above 0.8694:   3 resolved pairs, median +0.71  [+0.70, +0.89]
```

The +1.55 came from prose and was carried forward into a pre-registration, a kernel docstring and a
board row. `CLAUDE.md` already warns that this constant has been quoted wrong from prose twice; this is
the third time, and it is the first time it changed a verdict.

**AT THE CORRECT SLOPE THE ENTRY DID NOT UNDERPERFORM — IT LANDED EXACTLY ON IT.** This pool sits at
offline 0.8729, inside the `>= 0.8694` band where the fitted compression is **+0.71**. An offline gap
of +0.0023 therefore predicts **+0.0016 on the board**, i.e. 0.9366. It scored **0.936**. The band that
was labelled "the offline gain does not survive" is in fact the band where the gain survives precisely
as the top-end slope says it must.

**SO THE FINDING IS NOT ABOUT SUPERVISION. IT IS ABOUT THE CEILING.** What this entry actually measures
is the exchange rate at the top of the record, and the exchange rate is collapsing: +1.25 over the
whole record, +0.97 above 0.8678, +0.71 above 0.8694. Compression is not a constant to multiply an arm
by; it is a decreasing function of where the pool already sits, and every arm this project priced with
a single number was priced at the rate for a weaker pool than the one it would join.

**THE BAR THIS PUTS ON EVERY FUTURE ARM.** `BOARD_RESOLVABLE` is 0.0015. At +0.71, an offline gain must
exceed **0.0021** before it can move the board by one resolvable quantum. Every knob-sized offline
delta this project has been chasing — the 0.0005 hold-annotations term, the 0.0018 calibration term,
the +0.0006 `LogitAverage` margin over flat — is *provably* below the board's ability to see it, not
merely close to it. And the owner's 1.5% significance bar would need roughly **0.021 offline**, two
whole points of holdout AUC, which no knob and no seating has ever produced here.

**WHAT SURVIVES FROM THE ROW ABOVE.** The flag decomposition stands on its own offline measurement:
+0.0018 calibration, +0.0005 hold-annotations, calib-only five at 0.8724 over the identical 909
studies. What does NOT stand is the sentence "the offline +0.0023 did not survive to the board" — it
survived at exactly the rate the record supports. The label lane's card claim still fails, but for a
sharper reason than the one first written: not because supervision does nothing, but because its
offline size is under the 0.0021 the board can resolve at this pool's own position.

**AND THE INVERSIONS AGREE.** All 10 board-order inversions in the record are pairs the holdout left
UNRESOLVED (gaps of 0.0020-0.0037, under its 0.0053 floor); **0 of the 276 pairs the holdout actually
resolved were inverted by the board.** The ruler is not broken. It is being read below its resolution,
which is the same defect in a different coordinate.

## `ortho_solo_v1` — 0.907. The first non-DINOv2 point, and it cannot be read yet.

`ortho_dense_s2026` served alone (ref 55994354, 2026-09-03 21:00Z). DINOv3 ViT-L continued on 1.25M
knee slices, honest fold-0, **offline 0.8452 → board 0.907**. Shipped as an instrument on the RULER:
every pair in `board_transfer` comes from a DINOv2-family trunk, so the offline-to-board slope this
project prices every arm with has never once been fitted outside one family.

**IT DOES NOT YET LICENSE A FAMILY-SPECIFIC SLOPE, and the temptation to read one is the reason this
row exists.** A slope needs two comparable points and there is no comparable partner on disk. The only
other honest-offline solo is `solo_v1` (0.8518 → 0.868), three weeks and several kit revisions older,
on a different decode. Every solo scoring above 0.914 — `soloseat_v1` 0.925, `calibsolo` 0.923,
`raptor_solo` 0.923 — is `--fold -1` and carries no offline number at all. So 0.907 against 0.923 is
**not** a measured family penalty: those three saw the test-adjacent data this one withheld.

What the point IS good for: it is one honest `(offline, board)` pair from outside the family, and the
mount rehearsal for the ortho encoder dataset (`loaded 1 of 1`). A matched partner — one current-kit
DINOv2 member at a comparable offline, served alone at fold-0 — would turn it into a reading. That
partner has not been shipped.

## `foldfour_v1` — 0.936. Half a measurement; the other half is the number that matters.

Four seats, all `--fold 0` (ref 55994671, 2026-09-03 21:31Z): `calib_dense_hold_s2026`,
`calib_span_s2026`, `through_dense_s2026`, `small_batch6_s2026`. Four because only four members have
twins on both sides of the fold line.

Its partner `fitfour` is the identical seating at `--fold -1`. **The measurement is the delta between
the two board numbers, and neither means anything alone** — which is why this row states no branch of
its own. The branches live on the pair (knee-q18e8, and
`interpretations/kneemri/2026-09-03_the-0953-chain-is-short.md`).

**THE READING IS ASYMMETRIC, declared before the slot.** Four seats and a weaker pool are exactly
where a concave learning curve says the refit term is LARGEST, so the pair's delta is an **upper
bound** on that term at the champion's seven seats. Null or small (≤ +0.003) is decisive: it kills the
last step of the 0.953 path and saves roughly thirty hours of card. Large (≥ +0.004) is
**uninformative** — it is consistent with both a constant term and one that decays to nothing by the
time the pool is good.

No seating operation is involved: the seating is held fixed across both halves by construction.
`run_check` before the slot: `loaded 4 of 4 members`.

### `foldfour_v1`'s offline, measured after the slot: 0.8711

`record_mix` over the four records (no card, no inference pass): LogitAverage **0.8711**, RankAverage
0.8711, solos 0.8545–0.8634 (`calib_dense_hold` 0.8634, `small_batch6` 0.8598, `through_dense` 0.8558,
`calib_span` 0.8545). Four-of-four is the only subset of its size, so the bracket is a point.

**It lands the entry on the record's own line rather than off it**, which is worth stating because the
pair's delta will be read against this: `twinfive_v1` 0.8705 → 0.935, this entry 0.8711 → 0.936,
`holdfive_v1` 0.8729 → 0.936. Three current-kit fold-0 pools spanning 0.0024 offline and one board
tick — the compression the record already shows near this range, not an anomaly to explain.

## `fitfour_v1` — ref 55995214, 2026-09-03 22:34Z — **0.937**

The full-corpus half of the matched pair. Same four seats as `foldfour_v1`, every one refit at
`--fold -1`. No offline number and none is possible: each seat has seen all 909 holdout studies, which
is the arm rather than a gap in it. Board row is `scope: excluded` for that reason, so it never enters
`board_transfer`'s fit.

### The delta is +0.001, and the pre-registered decisive branch fired

`foldfour` 0.936 → `fitfour` 0.937. The branch declared before either slot was **≤ +0.003 kills the
last step of the 0.953 chain**. Four seats and a weaker pool are where a concave learning curve says
the refit term is LARGEST, so +0.001 is an **upper bound** on the term at the champion's seven seats,
not an estimate of it. The step had been carrying **+0.006**, budgeted from a single measured pair
(`twinfive` 0.935 → `fiveship` 0.941) and then treated as a constant.

The chain now reads 0.885 offline × 0.97 = 0.947, +0.001 refit = **0.948**, missing 0.953 by 0.005.
Board 0.953 needs offline ≈ 0.891. Every re-arrangement axis is closed in writing, so the remaining
+0.006 offline has to come from a better member.

Buys back roughly thirty hours of card, and retires the standing tax of keeping fold-0 and full-corpus
twins of every member. See
`interpretations/kneemri/2026-09-04_the-refit-step-is-dead-at-its-own-upper-bound.md`.

## spansolo_v1 — ref 55996812, 2026-09-04 00:12Z — 0.921

INSTRUMENT. The LOW anchor of `knee-l67zk`'s two-solo pair: `calib_span_s2026` served alone, current
kit, fold 0, offline 0.8545. The cell the board record never held — six solo operations exist and none
is a current-kit DINOv2 solo.

It exists to break a collinearity. `ortho_solo_v1` reads +0.02 above the DINOv2-fitted line, and offline
level and date are collinear across the whole record, so "the kit era carries it" and "ortholarge is
special" cannot be separated by the record itself.

Pre-registered before either score: **both ≥0.900** = the kit era explains it and ortholarge is not
special · **both ≤0.885** = it does not, and the in-domain trunk lane stays open whatever `ortho_stack`
reads offline · **0.886-0.899** = partial, report the split and close nothing.

This anchor read **0.921**, inside the ≥0.900 branch. **The pair is not complete and nothing is
concluded on one anchor** — the partner `kneemri-holdsolo` (`calib_dense_hold_s2026`, offline 0.8634)
ships at the next mark, and the pre-registration asks for both.

What is already visible, and stated as directional rather than as the reading: the two solos that
carry an offline number are `ortho_solo_v1` (0.8452 → 0.907) and this one (0.8545 → 0.921). The
DINOv2 solo is above the ortholarge solo on BOTH axes, and the two-point slope is ~1.5, which is the
record's own whole-record slope rather than the +0.97 operating-point one. Two points do not fit a
line and the second anchor is what this waits for.

`run_check` before the slot: loaded 1 of 1 members, no drop, no identity change. No card time.

## `kneemri-uncalib` — ref 56019970, 2026-09-04, board **0.937**

INSTRUMENT. Five seats, four of them swapped from the calibrated `_cf` build to the uncalibrated
`_ff` build with `merged_blend_cf` held fixed, so the calibration flag is the only variable against
`fiveship_v1` (0.941). Pre-registered before the slot: near 0.941 = calibration carries nothing and
fleet-wide refit is licensed · near 0.937 = the 0.006 is the calibration flag and refit alone is not
worth thirty hours of card.

**It read exactly the lower anchor.** The term that was budgeted as one constant is two:

| step | pair | delta |
|---|---|---:|
| pure refit, five seats | `twinfive` 0.935 → `uncalib` 0.937 | +0.002 |
| calibration, five seats | `uncalib` 0.937 → `fiveship` 0.941 | +0.004 |
| pure refit, four seats (independent) | `foldfour` 0.936 → `fitfour` 0.937 | +0.001 |

Two independent pairs put pure refit at or under +0.002, under this project's 0.0062 noise floor.
**Refitting the fleet for the refit is dead** — roughly thirty hours of card the 0.953 chain had
budgeted at +0.006 buys +0.002. Refit *with* `--calibrated-targets` keeps the +0.006, because
calibration is a flag on the same run and not a second run. The plan does not change; the reason
does. Any refit shipped without the flag throws 0.004 away.

Caveat on the control: `twinfive` is a fold-0 pool and this is a fold-full pool, so the +0.002 also
carries seat-for-seat build differences. The four-seat pair is the cleaner of the two and reads
smaller, not larger.

## `kneemri-colnull` — ref 56020638, 2026-09-04, board **0.500**

INSTRUMENT ON THE RULER, and it settles a question `comp/README` has carried as unmeasured since week
one. Constant 0.5 in every cell of every column. Three bands, pre-registered: **exactly 0.500** = a
plain mean of per-column AUCs, no offset, no non-AUC term · **0.495-0.505 but not 0.500** = a
tie-handling or prevalence term nothing here models · **anything else** = not a mean of per-column
AUCs, and every per-label plan in the ledger is priced on a formula that does not hold.

It read **exactly 0.500**. The metric is a plain unweighted mean over the twelve findings.

The consequence is that a per-label gain is now priceable without spending a slot: one column
improving by `d` moves the board by `d/12`. A finding lane must move its own column by ~0.018 before
the board's third decimal can show it at all, and by ~0.18 to be worth 1.5% of board alone. It also
makes `kneemri-colmcl` (ref 56020054) a clean weight read — with the mean confirmed, the drop from
constant-ing the MCL column is that column's real AUC minus 0.5, over twelve.

## kneemri-colmcl — ref 56020054, board 0.904

INSTRUMENT ON ONE COLUMN. The champion's six seats plus the raptor seat at 1/3, with the MCL column
overwritten by a constant 0.5 after the blend. It read **outside every pre-registered band** (0.918-0.928
expected, below 0.918 and above 0.928 the alternatives).

The harness was checked before the reading was believed, because a 0.040 drop is 96% of the largest drop
one column can produce and that is the shape of a bug:

- `run_check --log` on the downloaded kernel log reads `loaded 6 of 6 members`
- the scored `submission.csv` carries exactly the twelve `TARGETS` columns, with MCL at one unique value
- the kernel's `chosen` tuple and `FOREIGN_WEIGHT` are byte-identical to `kneemri-raptorweight`'s

So the arithmetic is forced, `colnull` having already fixed the metric as a plain unweighted mean:

    MCL board AUC = (0.944 - 0.904) x 12 + 0.5 = 0.980      interval 0.968-0.992 on rounding alone

MCL scores **0.8104 against our labels and ~0.980 against the board's**. It is the worst column of the
seven-member fold-0 rank pool and very nearly the best column on the board.

Consequences, in the order they cost:

1. The offline per-column map does not transfer in LEVEL — the board's twelve columns must average 0.944,
   the pool averages 0.8715.
2. On its only externally tested point the map does not transfer in ORDERING either. It inverts. The
   +0.0196 bottom-three headroom is not a number to plan against.
3. The other eleven board columns average 0.9407, and the whole remaining board headroom is 0.056. The
   owner's 0.953 target is 16% of everything left.
4. The gap is our SUPERVISION, not the model — and it lands on the column the four-reader census already
   calls second-worst-labelled at 50.0% radiologist agreement.

Next instrument: `kneemri-colbaker`, the same ablation on Baker's (the pool's strongest column at 0.9208).
`colmcl`'s own docstring named it as the refutation it could not make, and licensed it on exactly this
outcome.

## `kneemri-colbaker` — ref 56023244, board 0.903 — the per-column axis closes on the second endpoint

The same ablation as `colmcl`, on the opposite end of our own ordering: Baker's cyst, the pool's
STRONGEST column at 0.9208 offline, overwritten by a constant 0.5 after the blend. `colmcl` took the
weakest (0.8104). Between them they bracket a 0.110-wide offline spread.

    Baker's board AUC = (0.944 - 0.903) x 12 + 0.5 = 0.992      interval 0.980-1.004 on rounding alone
    MCL board AUC     = (0.944 - 0.904) x 12 + 0.5 = 0.980      interval 0.968-0.992

**The intervals overlap.** Two three-decimal board scores differenced and multiplied by twelve carry a
+/- 0.012 interval on the column, so this pair cannot order the two columns and never could have. That
RETRACTS the ordering half of `colmcl`'s reading — "on its only externally tested point the map inverts"
was one point read past its own resolution, and the second point does not rescue it in either direction.

What the pair DOES establish is level, and it is enough to close the lane:

1. **The offline spread is not there.** The two extremes of a 0.110-wide offline ordering both land at
   about 0.98 on the board. Whatever separates our columns offline is compressed by roughly an order of
   magnitude against Kaggle's labels — below the resolution of the only instrument that can see it.
2. **The weak columns are elsewhere and we cannot point at them.** The ten unread columns must average
   (0.944 x 12 − 0.992 − 0.980) / 10 = 0.9356. They carry the whole remaining headroom, and our holdout's
   map just failed to identify either end of it.
3. **The lane is unaffordable AND under-resolved.** One slot per column, twelve columns, each read no
   finer than 0.012 — against a column that must move 0.018 to shift the board's third decimal at all.

So the per-column axis is CLOSED, and it is closed on measurement rather than on price. Per-label work
still lands through `colnull`'s pricing rule (a column moving by d moves the board by d/12); what is
retired is the idea of choosing WHICH column to work on from our own holdout.

## `relabel7_v1` — 0.943 — corrected supervision is worth nothing at one seat in seven — 2026-09-05

Ref 56033961. The champion's six seats, unchanged in order, weight, overlap, combiner, kit and raptor
weight, with `errors_relabel_s2026` APPENDED as a seventh. `run_check` on the kernel log reads
`loaded 7 of 7 members`, so this is a real seven-member entry.

It is the only checkpoint the project owns fitted against a label table with 1,932 cells corrected at
the net-benefit-optimal budget of a confident-learning detector (Northcutt, Jiang and Chuang, JAIR
70:1373, 2021), and it was shipped after being declined offline. The decline used the 909-study holdout,
whose bias `colmcl` had measured that same day: our labels are false-positive-heavy (108 false positives
against 32 misses over the 58 radiologist-annotated studies), so a member fitted on CORRECTED labels is
scored by a ruler that still rewards the over-calling. Its −0.0056 solo was the sign the mechanism
PREDICTS. The claim was unfalsifiable offline by construction, which is exactly why it cost a slot.

**It read 0.943 against the champion's 0.944.** That is inside the board's own 0.001 resolution, and it
is the eighth consecutive flat in this pool — the first one that moved SUPERVISION rather than seating.

**What it refutes and what it leaves alone.** One member appended at a seventh of the vote. A real
per-member gain of up to about 0.007 would still land inside this flat, so what dies is the STRONG form
— corrected supervision buys a visible board step at one seat — and not the weak one. A relabelled
member at full weight, or a pool relabelled throughout, is a different question and this row cannot
close it. What the row does add is a price for asking it: the weak form needs most of the pool retrained
against the corrected table, which is card-days, and its own strong form has now read zero.

## `colhalf_v1` — pending — the third point on the ruler question — 2026-09-05

Ref 56037952, still scoring. Six members, the same set as `monotone_v1` (0.944), which is the control —
`submit_check` refused it as a known repeat and it was shipped with `--anyway` on that ground. Half the
twelve columns are served as a constant 0.5 after the blend, so under `colnull`'s rule the entry prices
six columns at once instead of one, at six times the leverage of `colmcl` and `colbaker` for the same
slot. Pre-registered before the slot was spent: **0.718-0.730** retires the per-label axis outright,
**0.755-0.785** licenses the full-corpus relabel, **0.731-0.754** needs one confirming arm, and anything
outside **0.718-0.785** is a harness defect to be found before the number is read.

## `v4solo_v1` — 0.922 — the label table lost to its own control — 2026-09-07

Ref 56070681, `kneemri-v4solo`, one member (`labels_v4hybrid_s2026`), instrument not a seating. The
control is `holdsolo_v1` (0.8634 offline, **0.928**): the same trunk, the same serving, one seat each,
and the only difference is the table the member was TRAINED on — `blend_hybrid_fx_v4b` here,
`report_labels_v2` there.

**It read 0.922, on the pre-registered low branch.** The kernel named `<=0.926` before the slot as "v4b
grading HURTS generalisation, and v2 stands as the shipped table". It landed four thousandths under
that line.

**What it settles is a disagreement between two offline rulers.** The v2 holdout of 881 read the v4b
student at −0.0049; the gold-58 twelve-mean read it at **+0.0131** and CLEARED its pre-registered bar.
The board sided with the holdout. The gold census shares an origin with the table it grades (knee-vrioa),
and this is the first held-out test of whether that shared origin was doing the work — it was. No
gold-58 census delta licenses a seat again without a board control beside it.

**What it does not touch.** The Lateral Meniscus cell (0.8273 → 0.9093, +0.0820, 2.7× the control
column's own excursion) is a claim about the TABLE, not about re-grading every column, and a per-label
repair of v2 is a different lever that this row prices nothing about.

## `cnnsolo_v1` — 0.915 — the gap is a constant of our ruler, not of a family — 2026-09-10

Ref 56142195, `kneemri-cnnsolo`, one member (`resnet_224_short_s2026`), instrument not a seating. The
first CNN this project has put on the board alone: offline **0.8527** midpoint / 0.8842 @ledger75.

**Two predictors were pre-registered against each other, before the slot.** Anchor-neighbour said
**0.920–0.923** — `labels_v4hybrid_s2026` sits at offline 0.8524, within 0.0003, and scored 0.922.
D283's flat-offset refit said **0.9135** (+0.0608 on the offline midpoint, sd 0.0035).

**It landed 0.915.** The realised gap is **+0.0623**, 0.4 sd from D283's constant and 0.0073 below the
neighbour. The DIRECTIONAL branch fired (0.900–0.918 was written as "the CNN seat stays a coin flip").

**What it settles.** The board-minus-offline gap survived transfer to an architecture family with no
board history at all — the strongest out-of-sample test the constant has had, since every earlier fit
was inside the transformer pool. The gap is a property of OUR RULER (a reader that invents 108
confident false positives in 696 cells against radiologists), not of any family's capacity.

**And it kills the anchor-neighbour method as a predictor.** Two members whose offline numbers agree to
four decimals landed 0.0073 apart on the board. The offline ruler cannot RANK across families even
where it agrees exactly; only the flat offset transfers. Three solo rows now price the same constant
from different families — `v4solo_v1` +0.0696, `holdsolo_v1` +0.0646, this one +0.0623 — a 0.0073
spread around D283's +0.0608, which is what the residual looks like when the level is right and the
family is not a term in the model.

**What it does not license.** 0.915 solo is above eight of the eleven solo rows in this file, so the
CNN is a real member and worth a seat test. Nothing here licenses spending card on a CNN FAMILY ahead
of the transformer pool: the branch that would have (>=0.918) did not fire.

## `cnnseat_v1` — 0.944 — family diversity was delivered and the board did not move — 2026-09-10

Ref 56147841, `kneemri-cnnseat`, seven members: the champion six plus `resnet_224_short_s2026`.
Mechanism-backed, not an instrument. `run_check` read 7 of 7 loaded; witness 0.00291 of a 0.0312 bound.

**Pre-registered, with the honest prior stated first** — the 7-seat mean on this board is 0.9365,
BELOW the 6-seat mean of 0.9378, and six prior seating entries had read flat. Branches: **≥0.947**
licenses a fast-CNN family programme · **0.944–0.946** is the ninth seating flat, closing the
single-CNN-seat question and NOT the CNN family · **<0.944** says a 0.915 solo does not survive
dilution at 1/7.

**It landed 0.944** — the middle branch, at the champion six's own board number to three decimals.

**Why this flat is worth more than the eight before it.** Every previous seating flat added a member
from the family the pool is already 273 deep in, so the standing explanation was correlation: the
binding constraint is diversity, not member quality, and the member-quality curve is flat anyway
(#4 0.8695 to #24 0.8610). This entry supplied exactly the missing thing — the first convolutional
member this project has honestly held out, from a family with no board history, at a solo 0.915 that
sits above eight of the eleven solo rows in `board.jsonl`. The mechanism was named in advance, it was
delivered, and the number did not move by one thousandth. **The seating axis is therefore closed
ACROSS families and not merely within one, and "the ensemble is correlation-bound" is no longer an
available explanation for the plateau.**

**What it does not say.** It does not refute the CNN family as a source of BETTER MEMBERS. It refutes
adding one to a fixed six at 1/7 weight — the eleventh distinct singleton to read flat that way. The
only seat-shaped question left is a SWAP rather than an addition, and it is a small one.

**Where it leaves the target.** The plateau arithmetic is unchanged and now more sharply pointed:
board ≈ offline + 0.063 across five fitted anchors, ensemble bonus +0.016–0.018, so **0.955 needs ONE
member at ~0.874 offline against our best honest ~0.866**. Seating cannot produce it, adding members
cannot produce it, and this entry spent a slot proving the strongest remaining version of the second.


## `matchpair_v1` — 0.933, ref 56150916, 2026-09-10 — the instrument that impeached the ruler

Shipped to measure the board's resolution floor at a matched-offline point. It measured the ruler.

`stackfit_s2026` against `holdsolo_v1`'s `calib_dense_hold_s2026`: identical on every recorded recipe
key, **0.0003 apart at midpoint** (0.8631 vs 0.8634). Pre-registered ~0.928 with the floor as the
first branch. It read **0.933** — the second branch, "offline does not predict the board at fixed
recipe". The reason is not unstable transfer. At **ledger75 the two members are 0.0090 apart**
(0.9039 vs 0.8949), and the board separated them by 0.005 in ledger75's direction.

Over the whole record (dropping `aprime_solo_v1`, board 0.506, a broken entry): midpoint resid sd
**0.0036**, corr 0.901; ledger75 resid sd **0.0015**, corr 0.982. Leave-one-out on ensembles n=34:
rmse 0.0029 vs 0.0017, ledger75 closer on **29 of 34**. See `experiments/README.md` and knee-f9aao.

**THE TARGET ARITHMETIC ABOVE IS MIDPOINT-FITTED AND IS SUPERSEDED.** "board ≈ offline + 0.063, so
0.955 needs one member at ~0.874 against our best honest ~0.866" was read through a ruler with twice
the blur. Refit on ledger75, n=41:

> **board = 1.1122 × ledger75 − 0.0673**, residual sd **0.0017**
> **0.955 needs an entry at ledger75 0.9192.** Best on record: `greedyseat_v1` 0.9040 (board 0.936),
> `honest12` 0.9012 (board 0.937). **The gap is +0.018 in ledger75 terms.**

That gap is ten times the ruler's own residual, which is the first time the target has been stated
with an instrument sharp enough to see it. It does not make the gap smaller — it makes it real.

**RETRACTED THE SAME EVENING BY `l75seat_v1`, below. Every ledger75-fitted number in this entry is an
in-sample fit and none of it survived its first out-of-sample test.**


## `l75seat_v1` — 0.939, ref 56151868, 2026-09-10 — the ruler lost its only out-of-sample test

Built as the clean test of the entry above. `greedyseat_v1` (board 0.936) was the same greedy over the
same 226-member pool with ONE variable moved: the cut the greedy maximises. Midpoint scored the new
eight at +0.0007 over the old — dead flat, ~0.936. Ledger75 scored it at +0.0098 — **0.949**. So for
once the two rulers made DIFFERENT predictions about a point neither had seen, and the slot bought a
decision instead of a number.

**It read 0.939.** The pre-registered 0.934–0.939 branch is the one that fired: *midpoint was right*.
The ruler with a quarter of the in-sample residual (sd 0.0015 vs 0.0036, LOO rmse 0.0017 vs 0.0029,
closer on 29 of 34 ensembles) missed by **three times as much** — 0.010 against midpoint's 0.003 —
the first time it was asked about a point it had not been fitted on.

**The mechanism of the error.** Ledger75 was fitted board-from-offline OVER THE RECORD, and the record
is what its residual was measured against. A recut that reorders the same members and is then chosen
BY that recut is not a second observation; it is the fit reading itself back. The sd 0.0015 was an
artefact of the fit, not a property of the ruler.

**What is now settled.** The repo default cut stays **midpoint**. knee-f9aao's labelling rule — a
scalar carries statistic, population and cut, `nn_corr@pool251`, `solo@ledger75` — stands on its own
merits and never licensed a ruler swap. The **P0 ledger75 migration bead closes UNBUILT**, and every
target arithmetic quoted in ledger75 terms above is void; the midpoint arithmetic it superseded is
back in force.

**And it is the TENTH seating entry to fail to beat 0.944** — seven seats, eight seats, cross-family,
greedy at two different objectives. The pool is the constraint, not the arrangement. Nothing about
seating, selection, weighting or re-lensing is worth another slot until a BETTER MEMBER exists.

## `resnetlong_v1` — 0.907, ref 56152701, 2026-09-10 — the board resolved a delta a third the size of our noise bar

**One member, `resnet_224_long_s2026`, offline 0.8460 midpoint on the 909.** It is the matched long
sibling of `resnet_224_short_s2026` (offline 0.8527, board 0.915, and a seat in `cnnseat_v1` 0.944):
the same resnet50d recipe at **48 epochs instead of 12**, with architecture, family, corpus, reading,
seed and holdout all held. One variable moved.

**The pre-registered branch `<= 0.908` fired.** Point estimate was 0.905.

| | offline (midpoint, 909) | board |
|---|---|---|
| `resnet_224_short_s2026` | 0.8527 | 0.915 |
| `resnet_224_long_s2026` | 0.8460 | **0.907** |
| delta | **-0.0067** | **-0.008** |

Slope **~1.19** against the fitted `board_transfer` **+1.55** — same sign, same order, and inside what
a two-point read can separate.

**What that costs us, and it is the point of the entry.** Our 0.0062 holdout "noise floor" is a
statement about **the 909-study holdout's RESOLUTION**, not about what is real. The board resolved an
offline delta roughly **one third the size of that bar** and put it where the ruler predicted. So a
half-percent arm is not a null by construction — it is a reading our holdout cannot take and the board
can. Two consequences, both immediate: sub-bar arms become worth running **and worth ranking**, and the
CLAUDE.md rule that *increments below the bar still compose* now has board evidence behind it instead
of an argument. What this does NOT do is re-open any arm dismissed on a **mechanism**; it re-opens the
ones dismissed on "inside the floor, therefore nothing".

**On the arm itself, the schedule question is closed.** 48 epochs LOSES to 12 on this recipe, on both
rulers, in the same direction and at the same size. That is a measurement, not a flat. It closes the
long-schedule question for the CNN family and says nothing whatever about the family's value —
`resnet_224_short_s2026` is a seat in the 0.944 champion-equalling entry, and this row says only that
its schedule was already right.

**Slot accounting.** Fifth of five for 2026-09-10, spent 22:40Z against a 23:30Z mark. The owner's
"land the last submission by midnight" was met with 3h20m to spare.

## `cleanseat_v1` — 0.942, ref 56168691, 2026-09-11 — the corrected ruler's preference did not transfer, and its sign went the other way

Six seats chosen by greedy T1 on the **corrected** label ruler, licence-clean, sharing **zero members**
with the champion's six — the first selection this project has run that returned a different SET rather
than a permutation of the same one. Members: `stack_increments_s2026`, `chainfour_s2026`,
`labelwise_extended_s2026`, `blendhold_s2026`, `stack_increments_base_s2026`, `resnet_224_long_s2026`.

Offline, the corrected ruler preferred it and the old ruler could not see the difference: **T1 0.8843 vs
the T0-era selection's 0.8817 (+0.0026)**, while on T0 the same six read 0.8763 against 0.8773. D343
rescored the gap with 94 blind human adjudications substituted (T1a +0.0026) and on a one-sided
correction (+0.0025) — three correction variants, one answer. Both numbers are in-sample by
construction, which is why the board row carries `offline: null`.

**The board read 0.942 against the champion's 0.944, and the pre-registered `<0.9425` branch fired.**

Say the magnitude before the verdict: **−0.002 is inside what this board resolves**, and 0.942 sits
0.0005 under a boundary that was chosen before the slot was spent. So the finding is the **sign**, not
the size. The +0.0026 the corrected ruler promised did not arrive, and what did arrive pointed the other
way.

**What this closes.** D340's ruler-error finding — the v2 ruler mismeasures members by +0.0010 to
+0.0140 with a member-dependent sign — is real offline and **does not license reseating**. The
corrected-ruler correction rung of D333 stays closed. And "the seven seating flats were an artefact of a
biased ruler" is now **refuted rather than merely unproven**: this is the eleventh seating entry to fail
to beat 0.944 and the first with a genuinely disjoint member set, which was the strongest remaining form
of the seating hypothesis. The member axis stays shut; the pool is the constraint.

**What it does not close, and it matters for the same evening.** This priced a SELECTION made on a
corrected ruler. It did not price the corrections themselves as label **operations**. `goldseat`
(`discord_mask_s2026`) and `corrseat` (`v2_corrected_s2026`) each swap a single seat and are chosen on
the blind-adjudicated **gold** ruler, not on T1; their question is whether MASKING or SUBSTITUTING the
discordant cells produces a better MEMBER — the one axis this row leaves open. One caution does
transfer, and it is written here before their scores land: **an offline ruler that named a preference
has already failed to transfer it once tonight.**

## `goldseat_v1` — 0.944, ref 56172032, 2026-09-11 — the gold ruler named a preference too, and the board declined it as well

The section above ends with a caution written before this score landed: *"an offline ruler that named a
preference has already failed to transfer it once tonight."* It has now failed twice, and the second
ruler was the strongest one this project owns.

One seat moved out of the champion six. `lr4e-5_ws_cf` OUT — AUC against blind hand-adjudicated
verdicts **0.5781**, the worst of the champion's six on that ruler, and the member the v2 ruler ranked
joint-best of the entire pool. `discord_mask_s2026` IN — **0.6982**, rank 1 of everything on disk. All
469 gold cells were adjudicated blind with the key sealed until the last verdict was written. A
**+0.0970** cut-free lead, and the board moved by **zero**: 0.944 against the champion's 0.944.

**Say the magnitude before the verdict.** 0.944 against 0.944 is a null, not a loss. D368 puts the
board's estimability floor at ~0.003 with the score quantized to 0.001, so this row cannot separate "the
gold ruler's preference is worth nothing at the operating point" from "it is worth less than the board
can see". What it does refute is the **size**: +0.0970 on the ruler bought at most +0.0005 on the board.

**What this closes.** Twelfth seating entry to fail to beat 0.944, and the second in one evening. Eight
board rows now read exactly 0.9440 across nine distinct seatings. The seating axis has been tried on
three rulers — v2, the corrected T1, and now the blind gold adjudication — and all three named a
preference the board declined. The gold one was the strongest remaining form of the hypothesis, because
it is the only ruler that was **not fitted on the same labels the members were trained under**. Seating
is saturated on every ruler available.

**The one thing that did resolve, and it is worth the slot by itself.** D367 priced `discord_mask`'s
marginal add at **−0.0029** on the v2 ruler and **−0.0019** on the corrected T1, while the gold ruler
called it the best member on disk. Two rulers in direct contradiction about the same member. The board
says **neither**: it says zero. The proxy's negative did not transfer any more than the gold's positive
did — which means the disagreement between them was never about the board at all. A label-table member
is scored by the v2 proxy under the very labels it declines to use, so that cost is now measured as an
**artefact** rather than a finding, and it should stop being quoted against label-table members.

**What it does not close.** Nothing about the mask **mechanism**. `discord_mask` writes nothing where
readers disagree, and D363/D365 give that a real target: the substitution family's addition direction is
78.6% wrong, 21 of 22 errors land on OA columns, and a lone dissent against v2 is 0.850 wrong where it
**asserts** against 0.017 where it **withholds**. A member that cannot make that error is still the
right shape. This row says only that swapping one seat for it inside a saturated seating does not move a
saturated number. Its sensitivity cost stays **unreadable** at 32 affirmed cells.

## `oofsolo_v1` — 0.926, ref 56065873, 2026-09-07 — priced in `board.jsonl`, and until now named nowhere here

This entry has carried a row since the day it scored and never had a section, which is the hole
`board_check` names. It is the entry that **caught the fold-0 leak**, so the omission was not a small
one.

The argument ran entirely on this file's own solo calibration. Solos of this family transfer
offline → board at about 0.79 board points per offline point: `spansolo_v1` 0.8545 → 0.921 and
`holdsolo_v1` 0.8634 → 0.928. `oof_teacher_s2026` read **0.8681 offline, above `holdsolo`**, and both
anchors extrapolated it to 0.932. It scored **0.926** — *below* `holdsolo`, a pass-through of 0.37 from
the low anchor and **negative** from the high one, against a pre-registration naming ≤0.928 as "the
teacher's fold structure is the first suspect". It was, and the code said so: `ReportFolds.holdout` puts
fold 0 in every `--fold k` training set.

**Amended 2026-09-11, and the amendment matters more than the original.** The size of the leak was
re-measured on a genuinely **matched** pair. `oof_teacher.sh` and `seat_clean_distill.sh` are identical
`run_train` blocks apart from `--teacher-targets` — leaky `teacher_oof_corpus.npz` against clean
`teacher_clean.npz` — and they read **0.8685 vs 0.8634 @holdout909 midpoint**. The transferred leak is
**+0.0051**, which is the entire margin by which that member led the pool.

D374 had read the same term as ~zero off `seat_distill_lr4e5` against the 20-slice clean student, a pair
confounded on **slice count and band** — both attributes D371 prices positive. So D374 was wrong in its
argument, and so was its retraction, which cited the same confounded pair from `oof_teacher.sh`'s own
header. Only the void's **conclusion** survives both. The matched pair is now the only correct
measurement in that file, and the header should say so.

**The honest pool maximum is 0.8649 (`clean_calib_s2026`), not 0.8685.** Every entry built after this
amendment is seated against that number.

**What it bought.** An un-leaked ruler, and a flag — `--exclude-fold` — that can state "drop this fold
from TRAINING without putting it in the holdout", which nothing in this repo could express before. It
also seeded the finding that closed the night: teacher exposure is a **fourth honesty filter** that none
of the three enforced ones reads, and twelve members of the packaged honest pool fail it.

## `tablepool_v1` -- 0.944, ref 56172710, 2026-09-12 -- the largest pool change we have ever shipped, and it moved nothing

Three seats arrive at once (`discord_mask_s2026`, `v2delonly_span_s2026`, `labelwise_depth24_s2026`) and
two depart. That size was the whole design: D368 fitted board score on member presence across the 83
scored entries that name their members, and three numbers came out of it -- the board is **quantised to
0.001**, the statistical floor of that design is ~0.003-0.008, and the combined minimum detectable member
effect is therefore **~0.003**. Measured seat-swap effects in this project run **0.001-0.002**. So no
single seat swap in this project's history has ever been *measurable* on this board, and the only moves
the board has ever resolved (0.005+) came from pool upgrades and label-table changes.

This was a pool upgrade, chosen on the blind gold-adjudicated ruler rather than on v2 -- the only ruler
we own that was not fitted on the labels the members trained under. **It read 0.944.**

State the magnitude before the verdict: 0.944 against the champion's 0.944 is a **null**, not a loss.
What the arm pre-registered for exactly this outcome fired: *"the pool axis closes with seating, leaving
solo member quality as the only class with a non-zero hit rate."* This is the **thirteenth** seating entry
to fail to beat 0.944 and the **ninth** board row reading exactly 0.9440.

**What it does not close.** Nothing about `discord_mask`'s mask **mechanism** -- its sensitivity cost is
still unreadable at 32 affirmed cells. And nothing about member **quality**, which is the one class this
row explicitly hands the remaining hit rate to.

## `honestsix_v1` -- 0.938, ref 56173554, 2026-09-12 -- and `honestfar_v1` -- 0.941, ref 56173880 -- the pair IS the measurement

Two entries, one UTC day, **identical six members**, and exactly one difference: `honestfar` blends the
raptor foreign seat at `FOREIGN_WEIGHT = 1/3` and `honestsix` does not. Neither row is to be read alone.

The six are `clean_calib_s2026`, `chainfour_s2026`, `stack_increments_s2026`,
`stack_increments_base_s2026`, `calib_dense_hold_s2026`, `seat_clean_distill_s2026` -- chosen by **solo
rank** over the 296-member packaged honest pool, not by a search scored on the same 909 studies, so
`MemberCurve.honest`'s +0.0025-to-+0.0031 search-optimism correction does not apply and none is taken.
`honestsix` carries a real offline number, **0.8737 @holdout909 midpoint**; `honestfar` carries **none by
construction**, because its seventh seat is `--fold -1` published weights that `foreign_overlap` measures
at **98.6% in-sample** -- the exact provenance that voided `raptor_solo`'s 0.8938 on 2026-08-29.

**THE DIFFERENCE IS +0.003. Say the number rather than rounding it either way.**

The pre-registration named `-0.003 to +0.003` as a **null** inside D368's estimability floor, and
`>= +0.004` as the threshold that would re-open the foreign axis as a **member** axis. +0.003 is the top
edge of the null band and **one thousandth below** the reopen threshold. **The null branch fires; the
foreign axis does not re-open.**

The null is the more useful of the two outcomes, in the arm's own words: it says the champion's foreign
seat **has been carrying approximately zero for weeks**, and that every entry built around protecting
that seat was protecting nothing. That closes a thread this project has never been willing to close. It
is equally **not** `<= -0.004`, so the champion is not indicted into a rebuild without the seat either.
The honest reading is that this pair cannot resolve the foreign join's board value in either direction --
and that what the null indicts is **the seating work, not the seat**.

**What `honestsix` says on its own.** 0.938 is **-0.006** against the champion: an all-honest our-side
pool chosen by offline solo rank does *not* reach the champion. The transfer arithmetic is worth
recording -- 0.8737 offline -> 0.938 board, against the solo-family anchors `spansolo` 0.8545 -> 0.921 and
`holdsolo` 0.8634 -> 0.928. That is an *ensemble* sitting almost exactly on a **solo**-fitted line: its
whole +0.009 of offline lead over the best honest solo (0.8649) bought about what a solo of that offline
value would have bought, and no more.

**What the pair does not say.** It does not price the foreign seat at a different `FOREIGN_WEIGHT`, and it
does not price it inside the champion's own six rather than these six. One weight, one pool, one pair of
board reads at a resolution of 0.001.

## `labelseat_v1` -- 0.944, ref 56186455, 2026-09-12 -- the seat gate's floor was CONFIRMED, not assumed

Priced in `board.jsonl` since it landed and named nowhere here until now, which is the same hole
`oofsolo_v1` sat in. `board_check` reports it on every run; this closes it.

The champion six plus `labelwise_depth24_s2026` as a seventh seat, `raptor` unchanged at 1/3. One line
different from the champion, and the champion carried NO labelwise member, so this is a SOURCE change and
not a reseating.

**The flat IS the result the slot was bought for.** Four bands, pre-registered before the spend:

| band | meaning | fired |
| --- | --- | --- |
| >= 0.947 | refutes the +0.0040 seat gate as a shipping gate | no |
| 0.9455-0.9469 | a real gain | no |
| 0.9425-0.9455 | FLAT — confirms +0.0040 as the floor | **yes** |
| < 0.9425 | `seat_value`'s SIGN becomes untrusted | no |

`knee-gcgbk` predicted exactly the third band. `seat_value` priced this seat at **+0.0008
[+0.0001, +0.0017]** and called it DISTINCT on all three axes while its own gate read 0 of 3 clear
+0.0040. The board now says a seat the offline ruler likes at +0.0008 buys nothing measurable — which
CONFIRMS +0.0040 as the shipping floor instead of leaving it an assumption, and it is the fourth
independent line landing on 0.944.

**What it does not say.** Not that labelwise pooling is worthless — that ONE labelwise member at ONE seat
in THIS pool is. The offline ruler could not have told us either way, which is why the slot was spent as
an INSTRUMENT and declared as one in its submission message.

**No offline number BY CONSTRUCTION**, the same reason as `honestfar_v1`: the raptor foreign blend sits at
`FOREIGN_WEIGHT` 1/3 and `foreign_overlap` measures its provenance 98.6% in-sample — the provenance that
voided `raptor_solo`'s 0.8938 on 2026-08-29. So the row is `scope: excluded` from the `board_transfer`
fit: it carries no (offline, board) pair. Its payload is the DIFFERENCE against the champion, which shares
its other seven exactly.

## `slotseat_v1` -- 0.944, ref 56192818, 2026-09-12 -- the CROSS-SITE ruler cannot pick a board tick either

The champion six plus `slot_dropout_s2026`, one line different. This entry was the first PREDICTION in a
week rather than an instrument, and what makes its flat informative instead of another reseating is that
**it was chosen although the offline ruler scored it at nothing**:

| ruler | prices `slot_dropout_s2026` |
| --- | --- |
| `seat_value --against cleanseat_v1`, cut 0.5 | **-0.0001** [-0.0008, +0.0006], DISTINCT (rank 0.9051, js 0.0190, maha 3.618) |
| cross-site, against its own control `band_dense_s2026` | **+0.0090** |

Every previous candidate was seated because the OFFLINE ruler liked it. This one was seated on the
disagreement, and the bet on the second ruler was written down before the slot was spent.

| band | meaning | fired |
| --- | --- | --- |
| >= 0.947 | the cross-site ruler PICKED A TICK the offline ruler scored at zero — a new selection instrument | no |
| 0.9455-0.9469 | a real gain, smaller than the ruler's own scale | no |
| 0.9425-0.9455 | FLAT — the cross-site ruler cannot pick a board tick either | **yes** |
| < 0.9425 | the seat is actively harmful | no |

**What this retires.** The cross-site ruler joins the offline ruler as a thing that cannot select a seat
the board will pay for. What survives of `slot_dropout` is an EXPLANATION of why offline flats are
uninformative — not a selection rule, and no longer an argument for spending a slot. The two rulers
disagreed by 0.009 on this member and the board sided with neither.

Fifth independent line to land on 0.944.

**No offline number BY CONSTRUCTION**, same as `labelseat_v1` and `honestfar_v1`: the raptor blend sits at
`FOREIGN_WEIGHT` 1/3 and `foreign_overlap` reads its provenance 98.6% in-sample. `scope: excluded` from the
`board_transfer` fit — the row carries no (offline, board) pair.

## `hidose_v1` -- 0.944, ref 56193747, 2026-09-12 -- the dose PAIR reads NULL, and that closes the family

`slotseat_v1` with ONE line changed. The six champion seats are in the champion's own order, untouched;
the serving is byte-identical; the seventh seat moves from `slot_dropout_s2026` to
`slot_dropout_hi_s2026`. Nothing else in the kernel differs.

The two seventh seats are the SAME OPERATOR AT TWO DOSES, and the cross-site ruler separates them
clearly:

| seventh seat | slot dropout | cross-site gain vs control | own holdout909 | board |
|---|---|---|---|---|
| `slot_dropout_s2026` (slotseat_v1) | 0.25 | +0.0090 | -- | 0.944 |
| `slot_dropout_hi_s2026` (hidose_v1) | 0.40 | +0.0132 | 0.8551 | 0.944 |

### The pre-registration, written into slotseat's own submission message before either score landed

The read was the PAIR, `hidose − slotseat`, not either entry against the champion:

- `>= +0.004` — the cross-site dose difference TRANSFERS to the board.
- `±0.003` — NULL. ← **fired, at exactly 0.000**
- `<= -0.004` — the dose OVERSHOOTS and the higher setting costs something the cross-site ruler cannot see.

Both entries read 0.944. The pair is **0.000**.

### What that means, and what it does not

A **+0.0042 measured cross-site difference between the two doses is invisible on the board.** That is
not a statement about noise swamping a small effect: the cross-site ruler is the instrument this project
built precisely because "the real Kaggle test dataset is NOT our dataset", and the board is the only
cross-site measurement that pays. The ruler separated the doses; the thing it was built to predict did
not.

Taken with `slotseat_v1` -- which was itself flat against the champion -- the **whole `slot_dropout`
seventh-seat family is inert at board resolution**, and it closes on two points plus a designed contrast
rather than on a single flat. That is the second half of the retirement `slotseat_v1` began: the offline
ruler cannot pick a board tick, and now the cross-site ruler cannot rank two doses on one either.

It does NOT say the operator is worthless. `knee-kiss9` separately measured, at full budget on two
seated recipes, that `slot_dropout 0.40` earns no seat -- which is a different claim on a different
ruler, and the two agreeing is why the family stops here rather than getting another arm.

### Why there is no offline number

Two-pass raptor serving, `FOREIGN_WEIGHT` 1/3, raptor rows 98.6% in-sample. `scope: excluded` from the
`board_transfer` fit, for the same reason every raptor entry is.

## `farswap_v1` -- 0.934, ref 56194309, 2026-09-12 -- distance loses to quality, on the third point

Declared an **INSTRUMENT** in its own submission message, with the prior stated as negative before the
spend. It was not sold as a champion beat and it did not become one.

`kneemri-allfit` scored 0.937 with six dark members plus `band_dense_long_s2026` (0.8587) -- a
DINOv2-small member on the decode axis, a sibling of four seats already in the room. This entry held all
six dark seats byte-identical, held the count at seven, held `RankAverage`, held `overlap=False`, and
moved ONLY who sits seventh: to `lora_r8_v3small_224_s2026` (0.8217), the pool's only non-DINOv2 member.
One variable, and that variable is architectural distance at fixed count.

| entry | seventh seat | seat solo | board |
|---|---|---|---|
| `allfit` | `band_dense_long_s2026` | 0.8587 | 0.937 |
| `farswap_v1` | `lora_r8_v3small_224_s2026` | 0.8217 | **0.934** |

### The pre-registration

- `>= 0.941` — distance beats quality despite a -0.037 solo gap. A NEW selection rule, re-opening the
  cross-family diversity axis.
- `0.935-0.940` — FLAT. The seventh seat is inert at this count, which answers `knee-uco3`.
- `<= 0.934` — quality dominates distance, consistent with `crossenc` and `maxdiv`. ← **fired**

### The answer, and why it is worth the slot it cost

**-0.003 against allfit.** A -0.037 solo gap is not bought back by being the only non-DINOv2 seat in the
room. This is the THIRD independent point on the same axis: `crossenc` and `maxdiv` each added a distant
low-solo member and each lost. An axis that closes on three designed points is closed differently from
one closed on a suspicion, and this is the one the ledger can now cite.

The instrument also did the thing instruments are supposed to do and rarely do here: it made a
prediction that could have been wrong in a useful direction. `>= 0.941` would have been a new selection
rule at a moment when both of this project's rulers had just been retired as seat selectors.

`run_check` read **loaded 7 of 7 members** before the slot was spent -- the exact census its sibling
`farseat_v1` failed silently for five days, shipping six of seven under a name that was then written
into the ledger as closing this very axis.

## `warmchair_v1` -- 0.944, ref 56194153, 2026-09-12 -- the first all-shippable 0.944

Six members, `RankAverage`, `overlap=False`. Pre-registered bands were `0.946+` re-pin /
`0.9435-0.9455` FLAT / `<=0.943` the all-data confound is the first suspect. It landed **0.944**, so
the FLAT band fires.

What it is, seat for seat against `stackseat_v1` (the pinned champion, also 0.944):

| seat | stackseat_v1 | warmchair_v1 | ledger75 |
|---|---|---|---|
| 1-3 | `final_ws_cf`, `calib_span_ff`, `grade_ladder_cf` | identical | -- |
| 4 | `stack_increments_ff` | `labelwise_dense_s2026` | 0.8964 |
| 5 | `lr4e-5_ws_cf` | `warm_calib_lowlr_s2026` | 0.8910 |
| 6 | `through_dense_cf` | identical | -- |

Two seats moved, both from a DARK foreign member to one of ours with a real holdout number, and the
score did not move. `stack_increments_ff` is fitted `--fold -1 --ema` and carries no holdout number by
construction; `lr4e-5_ws_cf` is likewise dark. So this entry replaces the two seats nobody can price
with the project's third-best solo member and a seated 0.8910, at zero board cost.

**What it does NOT do, stated because the plan said it would.** The pre-registration said a
licence-clean tie should REPLACE the pin. That argument does not apply here on inspection:
`champion_check` passes on `stackseat_v1`, and its six members carry no licence tag -- the
unshippable `merged_blend_cf` sits in `raptorweight_v1`, not in the pin. So the re-pin premise was
wrong, and the pin stays where it is. What warmchair buys is DARKNESS, not licence: it is the first
0.944 every one of whose seats has a number the project measured itself.

**And it is a fourth flat at 0.944.** With `hidose_v1`, `slotseat_v1` and `labelseat_v1`, four designed
seat changes in a row have landed on the same value. The board reports at 0.001, which bounds their
true spread below that -- it does not license a search inside it.

## the corpus lever closes -- `knee-q1pp0`, the subtraction this project had never taken

`base224_drop40_long` landed: **0.8552 mid / 0.8865 ledger75, peak e17 of 24**, clearing
`base224_control`'s 0.8451 12-epoch peak by +0.0101 and NOT peaking on its last epoch, so the budget
is no longer binding and a verdict is readable.

The matched subtraction, both sides converged, one variable (the corpus):

| arm | scheme | epochs | operator | LEDGER | LEDGER75 | peak |
|---|---|---|---|---|---|---|
| `mergedk_drop40_long_s2026` | mergedk | 24 | 0.40 | 0.8566 | 0.8853 | e17 |
| `base224_drop40_long_s2026` | canonical | 24 | 0.40 | 0.8552 | **0.8865** | e17 |

**delta = -0.0012 at ledger75.** The `<= 0.000` band fires: the corpus lever is dead at the dose
operating point and the second-site axis closes.

**The mechanism, which is the part worth keeping.** At 12 epochs the same pair read
`mergedk_drop40` 0.8805 vs `base224_control` 0.8754 = **+0.0051 for the merge**. Give both sides the
full budget and our own corpus gains **+0.0111** while the merged one gains **+0.0048**. The extra
site was substituting for training budget, not raising a ceiling -- more data made the fit converge
sooner, and every corpus number this project has quoted (+0.0039 mergedk, +0.0043 merged3) was read
at the epoch where that head start is still visible. The lever was real and it was not what it looked
like.

`base224_drop40_long`'s seat against `cleanseat_v1` at ledger75 is **+0.0002 [-0.0006, +0.0010]** --
no seat, which is expected for a plain recipe and is not what the arm was bought for.

**What this does NOT close.** KMAR-50K remains the only merge whose licence ships, and the axes reading
on `mergedk_drop40_long` is still DISTINCT (rank correlation 0.8914, js 0.0212, maha 3.390) -- a second hospital
does produce a genuinely different member. What is now closed is that the difference is worth anything
on either ruler we own.

## `tripleseat_v2` -- 0.946, ref 56232529, 2026-09-14 -- one resolvable step off the 0.944 band, and the pin moves

`stackseat_v1`'s seats with `grade_ladder_cf` and `lr4e-5_ws_cf` out and three labelwise
`drop40_long` fold-0 bases in (`labelwise_depth24_drop40_long_s2026`, `labelwise_ext_drop40_long_s2026`,
`labelwise_dense_drop40_long_s2026`, second shelf `kneemri-members-2`); seven of ours under
`LogitAverage`, `overlap=False`, raptor at 1/3. Kit `c3effa61`. `run_check` on the downloaded log:
7 of 7 loaded, no drop line.

Pre-registered bands: `0.947+` proxy transferred / `0.9455-0.9469` real gain, re-pin / `0.9425-0.9454`
flat / below refutes the proxy. It landed **0.946**: +0.002 over 0.944 on a 0.001 board, one step past
`BOARD_RESOLVABLE = 0.0015`, and the first entry above the flat 0.944 band. The fold-0-twin proxy
projected ~+0.0084; about a quarter landed, so the sign transferred and the size did not. Directional:
under the 0.0053 Hanley-McNeil floor and the 1.5% bar, one read. Offline is null by construction (four
dark seats plus raptor). The pin moves to it; `champion_check` now reads both shelves, since this is the
first pin seating second-shelf members.

## `doseseat_v1` -- 0.946, ref 56236942, 2026-09-14 -- FLAT: the drop40 dose is saturated at three

INSTRUMENT. `tripleseat_v2` with `through_dense_cf` out and `labelwise_drop_drop40_long_s2026` in (drop40
seats 3 -> 4); count, kit `c3effa61`, `LogitAverage`, `overlap=False`, raptor at 1/3 held. `run_check`:
7 of 7 loaded, no drop line. Fold-0-twin proxy +0.0011 [+0.0003, +0.0021] (ledger75, n=6 twin), expected
board move ~+0.0003 at the observed quarter transfer.

Pre-registered bands vs 0.946: `0.948+` dose unsaturated / `0.947` directional / `0.9455-0.9465` FLAT /
`0.945` dark seat worth more / below refutes the twin sign for dark seats. It landed **0.946**, FLAT. Pin
stays on `tripleseat_v2`. `through_dense_cf` is not load-bearing either way. With the next swap also a
single-seat move, the seat axis reads saturated at board resolution: stop reseating.

## `warmseat_v1` -- 0.945, ref 56238286, 2026-09-14 -- DILUTION: a fourth drop40_long seat subtracts

MECHANISM. `tripleseat_v2` plus `warm_calib_lowlr_drop40_long_s2026` as an extra seat (kernel `kneemri-warmseat`,
c5419a0a). Fold-0-twin ADD proxy +0.0007 [+0.0001, +0.0012] (ledger75, 909 rows). Pre-registered bands vs 0.946:
>=0.948 re-pin, 0.947 directional, 0.946 flat, <=0.945 dilution. Read **0.945**: dilution band. With
`doseseat_v1` flat, the drop40_long family is saturated at three seats, and a positive twin proxy of this size did not transfer in
sign. Pin stays on `tripleseat_v2`. Further same-family seats are not a slot; the next gains must come from better
members (full-corpus refit `fullseat_v1`, higher-resolution crops), not from more of them.

## `fullseat_v1` -- 0.946, ref 56246750, 2026-09-15 -- FLAT: one whole-corpus refit seat is under board resolution

MECHANISM. `tripleseat_v2` with `labelwise_ext_drop40_long_s2026` swapped for `labelwise_ext_drop40_long_ff`, the same
recipe fitted `--fold -1 --ema` on the whole corpus (kernel `kneemri-fullseat`, 0ed3d579). No offline number by
construction. Pre-registered bands vs 0.946: >=0.948 re-pin, 0.947 directional, 0.946 flat, 0.945 tick down, below
that stop full-fitting. Read **0.946**: flat. One seat's extra fifth of data does not resolve at a 2/3 share over
seven; this does NOT close full-fit. The dose read is `fullseat2` (ext AND depth24 refits, 7df0eaab), with
`labelwise_dense_drop40_long_ff` training behind it for the three-seat entry. Pin stays on `tripleseat_v2`.

## `fullseat2_v1` -- 0.946, ref 56250751, 2026-09-15 -- FLAT: two whole-corpus refit seats, dose series flat

MECHANISM. `tripleseat_v2` with `labelwise_ext_drop40_long_s2026` AND `labelwise_depth24_drop40_long_s2026` swapped
for their `--fold -1 --ema` refits (kernel `kneemri-fullseat2`, 7df0eaab). No offline number by construction.
Pre-registered bands vs 0.946: >=0.948 re-pin, 0.947 directional, 0.946 flat, <=0.945 stop full-fitting. Read
**0.946**: flat. Dose series 0 / 1 / 2 refit seats = 0.946 / 0.946 / 0.946. The extra fifth of data on the drop40
seats does not resolve on the board; the third refit (`labelwise_dense_drop40_long_ff`) is not run. Pin stays on
`tripleseat_v2`.

## `tripleseatclean_v1` -- 0.946, ref 56259030, 2026-09-15 -- INSTRUMENT: licence-clean mount reproduces the champion

MECHANISM. None; an instrument. `tripleseat_v2` byte-for-byte with the mounts trimmed to what the eight seats load
(dinov2-small, members, raptor, kit; DINOv3/resnet50d/dinov2-base removed; kernel `kneemri-tripleseatclean`,
38eae2f4, knee-z76qf). Pre-registered: must read 0.946. Read **0.946**: the champion is rebuildable from
licence-clean mounts alone. Pin stays on `tripleseat_v2`.

## `noraptor_v1` -- 0.945, ref 56263073, 2026-09-15 -- INSTRUMENT: the foreign seat's share, bounded not measured

MECHANISM. None; an instrument. `tripleseat_v2`'s eight seats minus `raptor_v4_swa`, the one foreign member, leaving
seven all-ours seats and the same blend (kernel `kneemri-noraptor`). Pre-registered: >=0.945 the foreign seat carries
nothing; 0.944 directional; <=0.943 a real share. Read **0.945** against the champion's 0.946 -- one tick, which IS
the board's resolution, so this bounds raptor's contribution below 0.001 rather than measuring it. Pin stays on
`tripleseat_v2`. Its pair `nolabelwise_v1` asks the same question of the labelwise trio.

> **THE BOUND ABOVE IS WRONG BY 2x, AND THE PAIRED READING BELOW IS WORSE (2026-09-16).** 0.001 is the board's
> reporting RESOLUTION; its BLUR — the spread over which two entries certify nothing about each other — is
> **0.0020**. So a one-tick gap does not bound raptor under 0.001. It bounds nothing: 0.945 and 0.946 are the
> same reading, and raptor's share is consistent with anything up to about two ticks. The pre-registration was
> written against the wrong scale, so the `>=0.945` branch it fired is not the branch it meant to fire, and this
> instrument is UNRESOLVED rather than answered. What it does still establish is a ceiling — raptor is not worth
> three ticks, because `nolabelwise_v1` shows what three ticks looks like.

## `nolabelwise_v1` -- 0.943, ref 56264081, 2026-09-15 -- INSTRUMENT: the labelwise trio's share, measured

MECHANISM. None; an instrument, the pair to `noraptor_v1`. `tripleseat_v2`'s eight seats minus the three
`labelwise_*_drop40_long_s2026` seats, leaving five (kernel `kneemri-nolabelwise`). Pre-registered: >=0.945 the
trio carries nothing; 0.944 directional; <=0.943 a real share. Read **0.943** against the champion's 0.946 --
**-0.003, three board ticks**, so the <=0.943 branch fired and the trio's share is MEASURED rather than bounded.

~~Read as a pair, the two instruments say where this champion's load sits. Dropping the ONE foreign seat costs under
a tick (0.945); dropping the THREE labelwise seats costs three (0.943). Per seat that is roughly 0.001 each for the
labelwise trio against under 0.001 for raptor, so the labelwise family is what the entry is made of and the foreign
seat is nearly free. Consequence for the licence axis: a licence-clean entry gives up about one board tick, not the
entry -- which is what `tripleseatclean_v1` (0.946, same eight seats on licence-clean mounts) already showed from
the other side.~~ Pin stays on `tripleseat_v2`.

> **THE PAIRED READING IS WITHDRAWN (2026-09-16), on all three of its steps.**
>
> **The per-seat division.** -0.003 over three seats was divided by three to get 0.001 a seat. That assumes the
> three seats' contributions ADD, and additivity in this project is not an assumption, it is a question that has
> been measured and answered NO — `interpretations/kneemri/2026-09-16_grand_fragments_reverse.md` finds every
> leave-one-out in the grand family sub-additive, with three fragments predicting +0.0075 together and delivering
> +0.0035 and +0.0049. Dividing a three-seat drop by three is that same error one level up: it reads an
> interaction term and prints it as a main effect, per seat.
>
> **The comparison.** Even granting additivity, 0.001-a-seat against under-0.001-a-seat compares two quantities
> that are both inside the 0.0020 blur. Nothing separates them. -0.003 itself is 1.5 blurs — directional, not a
> measurement, which is the standard this file applies to every offline number and did not apply here.
>
> **The licence corroboration is false.** `tripleseatclean_v1` is NOT "the same eight seats on licence-clean
> mounts". Its member list is byte-identical to the pin's, `raptor_v4_swa` included. It is a reproducibility
> instrument — it shows the board returns 0.946 twice for one seat set, which is worth having and is the
> evidence the 0.946 ties are real — and it says nothing whatever about licence-clean mounts. So the licence
> axis has no corroboration from either side here, and "a licence-clean entry gives up about one board tick"
> is unmeasured.
>
> What survives the pair: the labelwise trio carries more than the foreign seat. The direction, and only that.

## `pricetop_v1` — ref 56275593, board 0.938, 2026-09-16

INSTRUMENT (knee-hr499), slot 1 of 5. Six priceable seats chosen by an offline blend SEARCH over the
909-study holdout: `margin_seat_s2026`, `warm_calib_lowlr_drop40_long_s2026`, `blendhold_s2026`,
`labelwise_drop_drop40_long_s2026`, `oof_teacher_s2026`, `labelwise_depth24_drop40_long_s2026`.
Pre-registered `>=0.943` / `0.940-0.941` / `<=0.938`, centre 0.940. **The `<=0.938` branch fired.**

**The branch's written cause is wrong, and the error is one this file already documented.** That branch
reads *"compression at the top is lower or the curve has gone flat, 0.955 needs MORE than +0.011
offline."* It does not follow, because the prediction was fed the **raw** offline 0.8785 and this is a
SEARCH entry. `greedyseat_v1` recorded the same mistake and its fix (knee-jsgb): a searched blend enters
`comp/board.jsonl` bias-corrected. At the measured greedy bias +0.0028 the honest offline is **0.8757**,
and the two fold-0 search anchors read:

| entry | offline (corrected) | board |
|-------|--------------------|-------|
| `greedyseat_v1` | 0.8740 | 0.936 |
| `pricetop_v1`   | 0.8757 | 0.938 |

This paragraph read "at the top band's +0.8" until 2026-09-16 23:0xZ, a constant quoted from prose and
never from the tool. `board_transfer` read live at 22:33Z reports the ≥0.8679 band at **+1.07 over 65
pairs** against a whole-record **+1.09** — flat, not saturating — and the ≥0.8711 band at +1.71 over TWO
pairs, `DIRECTION ONLY`. At +1.07 the corrected value predicts 0.938 and the board read 0.938. So the
instrument did not find flat compression; it priced the search bias **on the board**, independently of
the offline measurement, for the second time. The slope between the two anchors is +1.18 on n=2 — an
anchor pair, not a refit, and indistinguishable from the fitted +1.07 at that n.

**The second reading is the one that matters for 0.955.** Every board row at or above 0.943 carries
`offline: null` **by construction** — the pinned `tripleseat_v2` 0.946 included — because their seats are
fitted `--fold -1`/cf and hold nothing out. The entire offline-priceable population tops out at 0.938.
That is not a saturated ruler; it is two **disjoint** populations. No maximisation of the 909-study blend
reaches the champion from inside the priceable set, and the all-data dose — not the search — is what
separates them. A further slot spent searching that holdout harder is a slot spent inside the capped
population.

## `foldseven_v1` — ref 56287691, board 0.944, 2026-09-16 — and the three-entry decomposition it forced

Seven seats, `kneemri-foldseven`, offline 0.8767, `selection: designed`.

**The blocker it created.** `pricetop_v1` reads 0.8757 → 0.938; `foldseven_v1` reads 0.8767 → 0.944.
**One offline point, six board points.** That residual is 3x the 0.0020 blur `board_transfer` reports and
wider than the entire offline span its top stratum is fitted over, and it survives the recut: in Q5,
`foldseven` 0.8167 → 0.944 over `pricetop` 0.8189 → 0.938, an inversion in the ruler. Until it is
assigned, no arm in this band can be priced through the transfer curve at all — which is every arm we
now run.

The difficulty axis is **closed** and is not the explanation: split at the median hard-share over the
same 714 resolved pairs, gain made HARD transfers +0.75 and EASY +0.72. Flat. There is no difficulty
correction to make between our holdout and the board.

Fitting a slope through 0.006 of scatter needs ~10 more pair points at one slot each — a bad ratio, so
the pair-point instrument was abandoned mid-plan and the three remaining slots of 2026-09-16 went to a
decomposition instead. Each entry varies exactly ONE thing and holds the ruler still:

| entry | what moves | what it measures |
|-------|-----------|------------------|
| `kneemri-foldseven-rerun` | nothing — same seats, same reading | pipeline run-to-run scatter, NEVER measured |
| `kneemri-eightseat` | +1 seat, +0.0005 ledger75 | board response to a sub-blur ruler move |
| `kneemri-swapseat` | different seats, +0.0001 ledger75 | seat identity at a fixed reading |

All three are labelled INSTRUMENT in their submission messages; none is a rank attempt, and a move in
our favour on any of them must not be pinned. **The rerun is read FIRST** — it is the error bar, and the
other two are unreadable without it. If the rerun does not return exactly 0.944, every closure in this
file priced on a 0.002-0.004 board delta is retired, including several that read as settled.

### 56288555 — foldseven re-ship (INSTRUMENT, leg 1) · board **0.944** · read 2026-09-17 00:00Z

**The clause above did not fire.** The rerun returned exactly 0.944, so every closure in this file priced
on a 0.002-0.004 board delta stands. Byte-identical to `foldseven_v1` (ref 56287691) — same seven seats,
same kit `c3effa61`, nothing moved — and `entry_offline --ruler midpoint` independently re-derived its
offline as 0.8767, foldseven's number to the digit, which is the check that the pool really was the same
one. **The board's run-to-run scatter for an identical entry is 0.000** at the competition's own 0.001
resolution. Recorded `scope: excluded`: a re-ship carries no new pair, and entering it would weight one
point twice in a fit that has just been shown to be sensitive to exactly that.

### 56288513 — eightseat (INSTRUMENT, leg 2) · board **0.944** · read 2026-09-17 00:00Z

Foldseven's seven plus `labelwise_depth24_s2026`, an eighth seat chosen FOR being negligible. **State the
cut**: `seat_add` prices it at +0.0005 @ledger75, and at midpoint — the cut `board.jsonl` records — the
same seat is worth **+0.0002** (0.8769 against 0.8767). Both are sub-blur. Board: flat.

So **a sub-blur offline move buys exactly nothing**, measured against an error bar of zero rather than an
assumed one. That is stronger than "the board is noisy below 0.0020": it says the transfer curve is FLAT
there, and it is only sayable because leg 1 landed first. The thirteenth seating entry to read 0.944.

**The unintended result is larger than the intended one.** `eightseat_v1` is the first designed,
honestly-priced entry ever to sit above the top band's floor — the two that had been there were
greedy-selected on the holdout and were voided the same evening (knee-52ldh). Adding it took the ≥0.8705
band from +1.45 over three pairs marked DIRECTION ONLY to **+1.44 over six resolved pairs [+1.39, +1.96]**,
and the ≥0.8678 band from +1.05/62 to +1.23/77. The band this project ships in was never unpriceable; it
was empty. Consequence, filed as knee-xz903: the gap to the leader (0.946 → 0.957, +0.011 read
2026-09-17 00:13Z) converts to **+0.0076 offline** — above the 0.0062 noise floor, well below the 1.5%
bar, so first place is a chain of composing sub-bar increments and not a single mechanism.

**Leg 3, 56288717 (swapseat), is still scoring.**

### 56288717 — swapseat (INSTRUMENT, leg 3) · board **0.944** · read 2026-09-17 00:19Z

`foldseven_v1` with `through_dense_s2026` replaced by `clean_calib_drop40_long_s2026` — six of seven seats
shared. `run_check` reads **7 of 7 members loaded**, which the kernel states as the condition for this entry
to read anything at all. Offline 0.8763 at midpoint, **−0.0004** against foldseven (the +0.0001 that chose
the swap is @ledger75 — different cut, not a different answer). Board: flat.

#### The three legs only mean anything together, and what they make is a series

Leg 1 measured the board's identical-entry scatter at **0.000**, which is what turns legs 2 and 3 from
"flat, probably noise" into nulls. Against `foldseven_v1` (0.8767 → 0.944), all at midpoint:

| entry | seats shared with foldseven | Δ offline | Δ board |
|---|---|---|---|
| `eightseat_v1` | 7 of 8 (one added) | +0.0002 | **0.000** |
| `swapseat_v1` | 6 of 7 (one swapped) | −0.0004 | **0.000** |
| `pricetop_v1` | 1 of 6 (near-disjoint) | −0.0010 | **−0.006** |

Every offline delta here is below the 0.0062 holdout noise floor — our ruler cannot separate any of these
pools. The `pricetop`/`foldseven` blocker — "one offline point, six board points", which has gated pricing
every arm in this band — is **resolved and was misframed**: there is no rogue seat to find, because the two
entries share exactly one member.

##### RETRACTED 2026-09-17 00:36Z: there is no pool-overlap effect, and the arithmetic that found one was wrong

This section first read the table above as **"board movement at a fixed reading tracks POOL OVERLAP, not
pool quality on our ruler"**, and drew from it that `board_transfer`'s slope applies within a family and
not across one. **Both claims are withdrawn.** They rested on the single `pricetop`/`foldseven` pair, read
without removing the offline trend, which is the same un-detrended comparison this file has now made wrong
three times.

Fit the 50 ensemble rows that carry both numbers and the top band is a **tight line**: over the 15 rows at
offline ≥ 0.8700, slope **+1.45** with residual sd **0.0022**. Against that line `pricetop_v1`'s residual is
**−0.0019 — under one sd.** It is an ordinary point, not an anomaly, and no pool-identity effect is visible
anywhere in the record. The "three times its own blur" above compared a board delta to the board's
*rounding*, never to its *scatter about the transfer line*, which is the only denominator that could make it
mean something.

What the detrended fit gives instead is worth more than the retracted claim:

- **The gap converts and cannot be luck.** 0.011 board to the leader is **4.9 residual sd**, or **+0.0076
  offline** at the top slope. A real gain that size will show; scatter will not deliver it.
- **A board delta under ~0.004 between two of our own entries is within 2 sd of the line and is not
  evidence.** That is a standing rule for reading this file, and it is what the earlier reading violated.
- **Designed seating buys RESIDUAL, and residual is not level.** The three highest residuals in the top
  band are `swapseat_v1` +0.0032, `foldseven_v1` +0.0026, `eightseat_v1` +0.0023 — all designed, all
  honestly priced on the full 909. **All three also read 0.944.** Sitting above the transfer line at a
  given offline reading is not the same as moving the board, and writing "designed seating is the thing
  that is working" (this line, 00:36Z) read the first as the second. To land 0.947 the OFFLINE number has
  to move +0.0076; no seating of this pool does that, because the pool is what bounds it.
  Four independent measurements now say the same thing and none of them is new:
  seat COUNT is not an axis (1/2/3/4/6 seats read 0.914/0.919/0.913/0.918/0.911, non-monotone over a
  0.008 span — `interpretations/kneemri/2026-08-16_the_grafted_members_became_a_cost.md`); the seat AXIS
  is saturated (`interpretations/kneemri/2026-08-25_the-seat-axis-is-saturated-and-the-member-is-0.028-behind.md`
  — that title's `0.028` is WITHDRAWN as a two-ruler artefact and the saturation half is what stands; the
  correction that replaced it, "our member is level with theirs", is itself over-read and was corrected
  2026-09-17 — n=58 at SE 0.012 cannot resolve the +0.0076 offline the board gap implies. The 896-row
  answer was already on disk: our best member reads 0.8959 @ledger75 against the honest local refit of
  his recipe at 0.8791 (`comp/members.declines:827-833`), so against the recipe we have replicated our
  member LEADS by +0.0168 and that comparison is closed. What is open is the ABSOLUTE member question —
  making ours better — which is where the card belongs);
  half-holdout greedy seating cannot resolve a seat below the 0.0062 floor (the surviving half of the
  closure at `:2450`); and the 0.946 tier reads **flat across five different one-seat perturbations** of
  `tripleseat_v2` — drop40 dose 3→4, `through_dense_cf` removed, 0/1/2 full-fit refits, mount trimmed —
  which bounds that tier's true spread under the board's own 0.001. **The next gain has to come from what
  a member IS**, which is the 2026-08-16 sentence arrived at again by a different road.

Re-derive the fit before quoting any number in this subsection; adding a row moves it, which is this file's
standing failure mode. See knee-5hbtz, which also re-opens part of the seating closure at :2450.

## `ffsolo_v1` — 0.944, ref 56296698, 2026-09-17 — one member lands two ticks under the seven-seat champion

Pre-registered instrument, landed in its own modal band, and the band it landed in is not the interesting
part. `labelwise_depth24_drop40_long_ff` is the `--fold -1 --ema` refit of the best recipe here, +26%
studies over its fold-0 twin — the largest data change available without new data. The twin reads ledger75
0.9053; carried over the +1.09 transfer slope that projects a solo board of **~0.943**, and the refit
**read 0.944**. FLAT (the 0.9405-0.9454 branch): the extra fifth of the corpus is worth about +0.001 to one
model, inside the 0.0022 residual sd. So more data does not make a better member either, and the data axis
closes at the member level and not only inside a saturated ensemble — knee-sexix's closure stands as
written rather than being re-scoped.

**The slope predicted a board score to one thousandth.** Its only previous solo anchor was `matchpair_v1`;
it now has two, and `matchpair_v1`'s own retraction the same morning (the `offline75` ruler seam,
knee-36xad) makes the second one load-bearing.

**And the number that matters is the comparison nobody pre-registered.** A SINGLE member reads 0.944; the
seven-seat `tripleseat_v2` reads 0.946. The best solo on this board before today was `matchpair_v1` at
0.933 — but that was `stackfit_s2026`, and the +0.013 ensemble premium read off it was a fact about a weak
member, not about ensembling. Priced against a GOOD member the ensemble is worth **+0.002**: two ticks on a
board that resolves to 0.001, and below the 0.0062 offline floor that half-holdout greedy seating already
could not see past. The 0.011 to the leader is therefore a **member** gap on the arithmetic as well as in
the prose above, and no reseating of this pool reaches it. Seven board flats across one-seat perturbations
were saying this; this entry gives it a number.

Re-derive the fit before quoting any number in this subsection.

## `l75solo_v1` — 0.941, ref 56309162, 2026-09-17 — the honest solo, and the 0.005 to the champion splits in two

INSTRUMENT, pre-registered, and it landed on its point prediction. The best solo this project owns
shipped alone with no selection step: `labelwise_depth24_drop40_long_s2026`, fold-0 trained, so its
ledger75 **0.9053** is a reading on studies it never saw.

PREDICTED **0.9414** off the solo-only transfer `board = -0.1217 + 1.174 * offline75` (resid sd 0.0019,
n=7), 2sd `[0.9377, 0.9452]`, modal branch `0.938-0.945`. READ **0.941** — inside 0.0004 of the point,
the third consecutive thousandth-exact call that slope has made. Re-derive the fit before quoting it.

**What this buys that `ffsolo_v1` could not.** `ffsolo_v1` at 0.944 is the SAME RECIPE refit on the full
corpus, so the two entries are a matched pair differing in exactly one thing — whether the holdout fifth
was in training. Full-corpus refit is therefore worth **+0.003** board to one member, measured directly
rather than inferred against a prediction. The champion at 0.946 is eight seats over that 0.944 member,
so the ensemble premium is **+0.002**.

The 0.005 between this entry and the champion decomposes as **+0.003 data and +0.002 ensembling**, and
neither term is reachable by reseating. The 0.011 to the leader stays a MEMBER gap.

## `extffsolo_v1` — 0.945, ref 56312235, 2026-09-17 — the refit term reproduces, and the ensemble premium falls to one tick

INSTRUMENT, pre-registered in knee-22nk5, and it cleared its top band. The question was whether the
**+0.003** whole-corpus refit term REPRODUCES on a second recipe. That term had been measured by exactly
one matched pair (`l75solo_v1` 0.941 / `ffsolo_v1` 0.944, same recipe, differing only in whether the
holdout fifth was trained on) while sitting load-bearing in the inversion that prices every open arm.

PREDICTED **0.9432**. The honest twin `labelwise_ext_drop40_long_s2026` reads **0.9045** ledger75; carried
over the solo transfer `board = -0.1217 + 1.174 * offline75` (resid sd 0.0019, n=7) that gives 0.9402, plus
the +0.003 term under test. READ **0.945** — inside 0.0018 of the point, so inside one residual sd.

**The term reproduces at n=2, and reads slightly larger here**: +0.0048 against `depth24`'s +0.003. Carry
~+0.004. The inversion stands: 0.947 needs +0.0021 offline75, 0.955 needs +0.0089.

**The larger reading is the ensemble premium.** The best SOLO on this board is now **0.945** against the
seven-seat champion `tripleseat_v2` at **0.946**. That premium was read as +0.013 off `matchpair_v1` (an
artefact of `stackfit_s2026`'s weakness), then +0.002 off `ffsolo_v1`, and is now **+0.001** — one board
tick, at a board that resolves to 0.001, which bounds it below what this record can measure at all.

**A RANK INVERSION, AND IT IS SCATTER.** This entry's honest twin is the WEAKER of the two
(`labelwise_ext_drop40_long_s2026` 0.9045 against `labelwise_depth24_drop40_long_s2026` 0.9053), and the
stronger one has ALREADY shipped in refit form — it is `ffsolo_v1`, ref 56296698, which read 0.944. So the
weaker member scored one tick HIGHER on the board. That inversion is 0.001 against a solo-transfer residual
sd of 0.0019 and a board resolution of 0.001: it is the smallest difference this record can represent, and
reading a mechanism into it would be reading scatter. The two members are board-indistinguishable.

The consequence is for the PLAN, not for the members: there is no untried refit of a better member waiting
on disk. Both twins are shipped and both land at 0.944-0.945. The 0.011 to the leader remains a MEMBER gap,
and no reseating of this pool reaches it.

## `extsolo_v1` — 0.937, ref 56315030, 2026-09-17 — the refit term is not a constant, and the solo line missed high

The held-out half of `extffsolo_v1` (0.945): the same recipe, `labelwise_ext_drop40_long_s2026`,
differing in one thing — the fifth of the corpus held out. So the pair measures the full-corpus refit
term DIRECTLY on a second member, where the first pair (`l75solo_v1` 0.941 / `ffsolo_v1` 0.944) had
measured it on `labelwise_depth24_drop40_long`.

**Term = 0.945 − 0.937 = +0.008, against that pair's +0.003.** Two members, 2.7× apart, and a dose
series of 0/1/2 full-fit seats (`tripleseat_v2` / `fullseat_v1` / `fullseat2_v1`) that reads
0.946 / 0.946 / 0.946 — flat — on a seat of this very member. A refit is therefore a MEASUREMENT to be
taken per member, never a step size to multiply by. Every plan ending "refit at `--fold -1` and ship
it" loses its coefficient.

This retracts `extffsolo_v1`'s own claim that the term reproduces at n=2 and that ~+0.004 could be
carried. That reading was taken against a PREDICTED honest twin (0.9402), because the twin had never
shipped. This entry is that twin, measured.

**And the solo transfer line missed for the first time.** `board = −0.1131 + 1.165 · offline75`
(resid sd 0.0016, n=8) predicted **0.9403**, 2 sd [0.9371, 0.9435]; the read is **0.937**, −2.1 sd,
outside the band, after three consecutive thousandth-exact calls. One point outside a 2 sd band does
not break a ruler. What makes it worth writing down is WHERE it sits: this is the second-highest
offline75 ever shipped, nothing above 0.9053 had ever tested the line, every open plan extrapolates it
there, and the miss is in the direction that costs. Treat a prediction above ~0.904 as an UPPER BOUND
until a second point at the top reads on the line.

## headlrsolo_v1 — 0.937, ref 56316099, 2026-09-18 — the transfer line has no slope where we ship

`headlr_ship_s2026`, offline75 **0.9059**, the highest single member on disk. Pre-registered on the LINE
rather than on its knob (`--head-lr` 2e-4 is board-invisible at +0.0006 ledger75). The `<= 0.939` branch
fired.

Three readings now sit at the top of the solo line:

| member | offline75 | board |
|---|---|---|
| `labelwise_ext_drop40_long_s2026` | 0.9045 | 0.937 |
| `labelwise_depth24_drop40_long_s2026` | 0.9053 | 0.941 |
| `headlr_ship_s2026` | 0.9059 | **0.937** |

Their offline75 spread is **0.0014**, well inside the 0.0062 paired floor — on our own ruler these three
members are a TIE. Their board spread is 0.004, two board-blur units. The line fitted on the seven points
below 0.900 (`board = -0.1217 + 1.1743 * offline75`, residual sd 0.0019) over-predicts all three, by
0.0034 / 0.0004 / 0.0051 — three of three LOW, and −2.7 sd at this point.

This does not prove a bend: three tied members cannot measure a slope. It proves the weaker and more
expensive thing — **above offline75 0.904 the slope's centre may not be used, only its upper bound**, and
our holdout ruler no longer discriminates between the best members we own. Every plan shaped "raise
offline75 by X and collect 1.1X of board" is unpriced at the top of the range, which is the only place we
ship. The one term still measured board-to-board is the refit (+0.008 / +0.003), and it is unaffected by
this because it never passed through the offline ruler.

## `spanff_v1` — 0.926, ref 56330671, 2026-09-18 — the refit term is ADDITIVE, and there is no ceiling at 0.9445

INSTRUMENT, pre-registered before the slot, and it settled the one live question the 2026-09-18 audit left
open: is the whole-corpus refit term ADDITIVE, or a CEILING that pulls every refit toward ~0.9445 regardless
of its base? ADDITIVE predicted **0.925–0.929**, CEILING predicted **0.943–0.946**. It read **0.926** — inside
ADDITIVE, about 9 sigma from the other branch, against a median board blur of 0.0010.

**A weak base refits to a weak result.** `calib_span_s2026` shipped 0.921 as `spansolo_v1`; its whole-corpus
refit reads 0.926. Member quality survives the refit intact, and the refit does not wash out the differences
between bases. The prior was inverted to ADDITIVE ~0.8 **before** shipping, on four whole-corpus solos found at
0.923 / 0.925 / 0.944 / 0.945 with no attractor between them; the slot confirmed a reading already taken from
the record rather than discovering one, and was labelled that way in its own submission message.

**The term now has three points.**

| base | base board | refit board | term |
|---|---|---|---|
| `labelwise_depth24_drop40_long` | 0.941 | 0.944 | +0.003 |
| `calib_span` | 0.921 | **0.926** | **+0.005** |
| `labelwise_ext_drop40_long` | 0.937 | 0.945 | +0.008 |

Mean +0.005, and the spread is **not** explained by base strength — the weakest and the strongest base sit at
opposite ends of it.

**What it does not buy.** Additivity confirms the arithmetic of the gap rather than opening a lever. The best
honest base this project owns carries a board-equivalent 0.941, and a bankable +0.005 lands 0.946 — exactly the
standing champion. Reaching the 0.957 leader still requires an honest base near 0.916 ledger75 against our best
at 0.9078. The 0.011 stays a MEMBER gap, which is what `ffsolo_v1` and `l75solo_v1` each concluded independently.

## `dmlbff_v1` 0.943 · `long36_v1` 0.936 · `dmlbfflong36_v1` 0.938 — refs 56340982/85/99, 2026-09-18 — neither refit term is a lever

Three solo whole-corpus refits, each pre-registered in its bead before shipping.

| entry | member | bead | pre-registered | board | vs its non-term refit |
|---|---|---|---|---|---|
| `dmlbff_v1` | `dml_b_ff` | knee-8bmi2 | 0.947+ / 0.944-0.9469 / <0.944 | 0.943 | -0.002 vs `extffsolo_v1` 0.945 |
| `long36_v1` | `depth24_ff_long36` | knee-lz8pd | 0.947+ / 0.944-0.946 / <=0.943 | 0.936 | -0.008 vs the 24-epoch depth24 refit 0.944 |
| `dmlbfflong36_v1` | `dml_b_ff_long36` | knee-bxp8h | P(clears 0.946) ~0.20 | 0.938 | below both parents |

The peer term reads -0.002, inside the top band's ~0.004 two-sd scatter: it buys nothing on the board. The
36-epoch anneal reads -0.008, about twice that scatter: on a whole-corpus refit, where no holdout stops the
schedule, the longer anneal HURTS. Stacked, they land below both parents. The champion `tripleseat_v2` stands at
0.946, and the rank read live at 21:36Z is 68 of 3962, leader 0.957. The gap is still a member gap, and neither
term closes any of it.

## `headlrff_v1` — 0.938, ref 56348362, 2026-09-19 — the refit term does not carry the highest-offline75 base

INSTRUMENT, pre-registered in knee-k7292: 0.940-0.945 the term carries / >=0.946 clears the champion / <=0.939 it
does not. `headlr_ship_s2026` read 0.937 solo; its whole-corpus refit `headlr_ship_ff` reads 0.938, +0.001, inside
the top band's ~0.004 two-sd scatter. The refit term now has four points, +0.001 / +0.003 / +0.005 / +0.008, and
the best offline base drew the smallest. It is not a constant a plan can add to a base.

## `noemasolo_v1` — 0.944, ref 56391333, 2026-09-20 — EMA is not the expired flag the bead argued, and it was hiding a refit-term artefact

INSTRUMENT, pre-registered in knee-f2mj3: `>=0.950` a real lever licensing a programme / `0.946-0.949` flat /
`==0.945` dead flat / `<=0.944` EMA was load-bearing. **READ 0.944, the bottom branch.** This is a one-flag A/B on
the board — `labelwise_ext_drop40_long_noema_ff` against its byte-identical twin `extffsolo_v1` (0.945), the same
recipe at `--fold -1` with `--ema` ON — so the delta is **−0.001** for removing it.

One tick is the board's own resolution, and the bead was right that a ~0.004 two-sd scatter could not resolve a
~0.005 effect. What makes it readable is that **an independent honest ruler landed the same morning and agrees on
the sign.** knee-0ed8f put the matched fold-0 contrast on the card (the ship arm verbatim plus `--ema`, one
variable, control already on disk):

| member | LEDGER | LEDGER75 | epoch | rows |
|---|---|---|---|---|
| `labelwise_ext_drop40_long_ema_s2026` | 0.8687 | **0.9073** | 23 | 909 |
| `labelwise_ext_drop40_long_s2026` | 0.8672 | 0.9045 | 23 | 909 |

**+0.0028 for ema ON**, inside knee-0ed8f's ±0.0053 matched-null band but positive, and the board's −0.001 for ema
OFF carried at the +1.071 solo slope is −0.0009 to −0.003. Two rulers, one sign.

**`matched_axes` had the sign backwards.** It read ema `False > True` at +0.0049 over 264 recipe-carrying members.
That estimate was cross-member at n=2 and single-seed, drawn from a corner where every fold-0 member is ema OFF and
every `_ff` refit is ema ON — ema perfectly confounded with fold, which is the confound knee-0ed8f was written to
break. It broke it in the direction that says: **keep `--ema` on every `_ff` refit.** The shipping configuration was
already correct, and knee-f2mj3's premise — that the flag's ensemble-shape justification expired when the entry
became a solo — is refuted on measurement rather than left open.

**AND THE CONFOUND WAS IN A CONSTANT WE PLAN WITH.** `extsolo_v1`'s row prices the whole-corpus refit term on this
recipe at **+0.008** (0.945 − 0.937) and concludes the term is "2.7x apart" from depth24's +0.003 and has "no single
value". That pair crosses the ema axis: `extffsolo_v1` is ema ON, `extsolo_v1` is ema OFF. The matched ema-ON twin
is this morning's arm at 0.9073, which the solo transfer (n=10, slope +1.071, intercept −0.0307, resid sd 0.0021)
prices at board **0.9413** — so `extffsolo_v1`'s term is **+0.004**, not +0.008. The fully-measured ema-OFF pair is
`extsolo_v1` 0.937 → `noemasolo_v1` 0.944, **+0.007**. Four bases now read +0.001 (headlr), +0.003 (depth24), +0.004
(ext, ema-on), +0.007 (ext, ema-off). Still not a constant — but the outlier that drove the "no single value"
retraction was an ema misattribution, and the working carry of ~+0.004 survives it.

**No new best member, and the ladder is unchanged.** 0.9073 is second on the honest ledger75, behind `dml_b_s2026`
at 0.9078, and the top eight span 0.0033 — the saturation of `interpretations/kneemri/2026-09-20_pool_exhaustion.md`
§1 stands. The champion `tripleseat_v2` holds 0.946; rank read live 11:42:49Z is 70 of 4037, tied with 13, leader
0.957, gap 0.011. Both beads close and the standing entry is untouched.

## `deepsolo_v1` — 0.941, ref 56405965, 2026-09-20 — serving past the fitted depth COSTS, and the offline ladder predicted the wrong sign

`extffsolo_v1`'s member (`labelwise_ext_drop40_long_ff`) byte for byte, band held at its fitted
0.1–0.9, served at **32** slices instead of the fitted 20. One constant moved. It landed on the
pre-registered bottom branch — the one the kernel itself named as the branch that would retire the
whole idea — at **0.941 against 0.945, four board ticks down.**

`run_check` on the downloaded log reads `loaded 1 of 1 members`, so this is a reading of the SERVING
and not a `KernelRun` member drop wearing a score (the `farseat_v1` failure).

**The offline ruler called the sign wrong, which is the part worth keeping.** The ladder this entry
extrapolated from — `band_dense_s2026` through `holdout_backfill --served-slices`, band held, reading
0.8859 / 0.8877 / 0.8888 ledger75 at 12 / 20 / 24 — is monotone increasing and projected +0.0024 at
32. The board read −0.004. The entry named one suspect before it ran: that ladder was measured on a
`canonical`-scheme member and this member is `extended`, so the per-slice information curve may not be
scheme-invariant. The second suspect is the head — 32 slots is further outside the twenty-slot
distribution `attend` was fitted over than anything the offline ladder touched, so the pooling is being
asked to weight slots it has never seen.

**Consequence, and it is a slot saved.** `kneemri-deep40solo` is built and COMPLETE at 40 slices and
**will not be shipped**: 40 is strictly further out of distribution than 32 on an axis now measured
negative, so its expected value sits below this row. The board has measured the descending half of this
ladder four times (`onegroup_v1` 0.942 at 12, `cheapdecode_v1` 0.925 and `cheapwide_v1` 0.928 at the
224 px / 12 corner, this row at 32) and the ascending half once, negative. **Serve every member at its
fitted depth.**

## basinsoupff_v1 — 0.946, ref 56406877, 2026-09-20 — a SOLO ties the seven-seat champion, and weight souping transfers to the full-data line

`basinsoup_ff_k4_s2026` is the uniform weight average of four full-data checkpoints: the shipping solo
`labelwise_ext_drop40_long_ff` (= `extffsolo_v1`, board 0.945) and three 4-epoch `--fold -1` seed-children
forked off it with `--init-scope all` at backbone-lr 2e-5, seeds 301/302/303. One checkpoint, one forward
pass, no new recipe. `run_check` on the downloaded kernel log reads `loaded 1 of 1 members`.

**It landed on the middle pre-registered branch.** The kernel named `>=0.947` as the op transferring,
`0.946` as a tie under the board's own blur, and `<=0.945` as the family closing on its fourth
explanation. It read 0.946.

**What is new.** This is the first solo this project owns to reach 0.946. It ties `tripleseat_v2` — seven
members, three read groups, 56 slice-reads a study — with a single member behind a single decode. On the
efficiency ladder the solo dominates that champion by construction, and nothing about a seating can make
seven trunks cheaper than one.

**What it does not do.** 0.946 does not beat 0.946. The pin does not move, the standing entry is
untouched, and the +0.001 over its own parent sits inside the 0.0022 top-band residual sd — directional
only, not a finding, and not a delta to plan around.

**What it settles.** Weight souping transfers from the fold-0 line to the full-data line. Fold-0 measured
+0.0016 (parent 0.9073 → `basinsoup_k3` 0.9089); the board paid +0.001 on the `_ff` line. Same sign, same
order, two different rulers. The op is the only weight-space operation this project has measured positive
and it is now confirmed twice.

**The bar it still misses, stated on the ruler that can resolve it.** The solo board-transfer fit
(n=10 single-member entries, slope +1.071, intercept −0.0307, resid sd 0.0019) puts board 0.955 at
offline75 **0.9201**. The best fold-0 fit the project owns is `dml_b_s2026` at 0.9078. The gap to first
place is **+0.0123 offline75**, and souping's measured size is +0.0016.

## 2026-09-20, slots 4 and 5 — the fold penalty, measured

`basinsoupffk3_v1` (ref 56407615, board **0.945**) and `basinsoupfold0_v1` (ref 56407707, board
**0.943**) are one matched pair, shipped as instruments. Same K=3 soup shape, same recipe; the only
difference is that one line's children saw the 909 held-out studies and the other's did not.

**The fold penalty is 0.002** — twice the board's own 0.001 resolution, and a fifth of the 0.011 gap to
first place. A fold-0 `ledger75` number is therefore **board-predictive at a −0.002 offset**, which is
what the slot bought: the honest fold-0 shelf can now rank candidates against the board without a slot
being spent to find out.

It also retires `knee-xjxdn`. The apparent fold-0/`_ff` anti-correlation was three points spanning
0.0033, under the 0.0053 paired floor; the direct matched pair the bead never had reads +0.002 in the
ordinary direction.

Second reading, from the K=3 row: dropping the anchor parent out of the soup costs 0.001, so the
children carry the soup rather than the parent being averaged back in. Directional only — inside the
0.0022 top-band residual sd.

## 2026-09-21, slot 1 — ff-of-soup, and weight space closes in both orders

`dmlbffsoup_v1` (ref 56409058, kernel `kneemri-dmlbffsoup`, member `basinsoup_dmlb_ff`) read **0.940**
against a pre-registered ~0.947. The bottom branch, and by a margin that is a reading rather than scatter:
−0.007 from the fit is 3.2× the 0.0022 top-band residual sd.

The operation was ff-of-soup — average first on the fold-0 line, then give the average the corpus — the
opposite order to the standing champion `basinsoup_ff_k4_s2026` (soup-of-ff, 0.946), and an order that had
no checkpoint on disk before 2026-09-20.

**Mechanism.** A soup sits at an averaged point whose value *is* the averaging. Four epochs of gradient at
`--backbone-lr 2e-5 --init-scope all` walk it off that point, and nothing re-averages afterwards, so the
soup is spent rather than compounded. soup-of-ff survives precisely because the averaging is LAST.

**Second reading, not separated by this row.** The −0.002 fold-penalty offset was fitted on the `basinsoup`
K=3 pair, and using it to rank two dml_b-family bases asked a 0.0003 offline difference to order two board
entries — a tenth of the residual sd. The offset is real on its own family (2× board resolution, matched
pair) and was over-extended here.

**Consequence.** The weight-space family is closed in both orders: round 2 already read SATURATES (the soup
recovers its basin's ceiling and cannot raise it, while divergence rose 34× and bought nothing), and this
row closes the remaining operation. Champion pin unchanged — `basinsoup_ff_k4_s2026`, board 0.946, rank 74
of 4068, gap 0.011 to the 0.957 leader.

## soup16_v1 — 0.944, ref 56489294, 2026-09-23 — serving BELOW the fitted depth costs too

INSTRUMENT on knee-uy8z0. `basinsoupff_v1`'s single member (`basinsoup_ff_k4_s2026`, 0.946) served at
`SharedRead(16, (0.1, 0.9))` instead of its fitted 20 slices, band held. `run_check`: loaded 1 of 1.
Pre-registered: >= 0.946 the shallow rung is free; <= 0.944 the fitted depth is optimal from both sides.
Read 0.944, the bottom branch. The -0.002 is ~1 top-band residual sd, so it is directional on its own;
it is the ladder that carries the verdict: 12 slices 0.942 (`onegroup_v1`), 16 0.944, 20 fitted 0.946,
32 0.941 (`deepsolo_v1`). Serve every member at its fitted depth; the decode saving is not free.

## headlrffp_v1 — 0.939, ref 56493655, 2026-09-23 — stopping a refit at its twin's peak buys nothing visible

INSTRUMENT (knee-yatne). `headlr_ship_ffp` is `headlr_ship`'s `--fold -1` refit stopped after e17 via
`--stop-after-epoch`, the 24-epoch anneal kept, so the only difference from `headlrff_v1` (e23, **0.938**)
is which epoch ships. The holdout twin `headlr_ship_s2026` decays 0.0027 midpoint from its e17 peak to e23.

Pre-registered: >= 0.941 the stop transfers; 0.939-0.940 directional; <= 0.938 no. It read **0.939**:
+0.001, the board's resolution and under one top-band residual sd. The midpoint decay does not reach the
board as a usable gain, so last-epoch refits stand and the champion soup's children are NOT re-cut at a
peak epoch. Gates: submit_check new set, roster/kit/mount/champion clean, witness 1 ok (the submission
message says UNVERIFIED — wrong, the witness had not finished when it was written), run_check 1 of 1.

## dmlbsoup_v1 — 0.944, ref 56500295, 2026-09-23 — the dmlb family is not what lost; the order was

INSTRUMENT. `basinsoup_dmlb_k7_s2026` (fold-0 soup, 0.9096 ledger75, the best fold-0 checkpoint owned)
served solo, so the soup-then-serve order matches `basinsoupfold0_v1` (0.943) and only the family changes.
It separates the two readings of `dmlbffsoup_v1`'s 0.940: family or order.

Pre-registered: >= 0.944 the order was the cause and a dmlb soup-of-ff is licensed; 0.943 directional;
<= 0.942 family closed. Read **0.944**, the top branch by the letter — but +0.001 is the board's resolution
and half a top-band residual sd, so what it establishes is the NEGATIVE: the dmlb family is not worse, and
the 0.940 was ff-of-soup losing to soup-of-ff, as `basinsoupff_v1` (0.946) vs `basinsoupfold0_v1` (0.943)
already showed on the other family.

**The licence is not spent.** A dmlb soup-of-ff means refitting seven children with `--fold -1` and souping
them: ~10-14 h of card (UNVERIFIED per-child time) for an expected ~0.947 against the 0.946 champion — a
+0.001 knob inside scatter, against a 1.5% bar. Recorded as licensed and unqueued. Gates: submit_check new
set, mount/kit/roster/champion clean, run_check 1 of 1, witness UNVERIFIED (a soup).

## soupraptor_v1 — 0.944, ref 56504501, 2026-09-23 — raptor dilutes our top member

INSTRUMENT (knee-3mamq). `basinsoup_ff_k4_s2026` + `raptor_v4_swa` at weight 1/2. 0.944 against the soup
served alone at 0.946: −0.002, the pre-registered "≤0.946" branch. The two-architecture lift seen by
lower entries does not survive at our top — a 0.946 member averaged with a weaker foreign reader loses.
knee-ir6d3's gate parks the ensemble lane on this read. `soupraptor23_v1` (weight 2/3, ref 56505465)
is still scoring; by this result its expected read is lower still.

## soupraptor23_v1 — 0.939, ref 56505465, 2026-09-23 — the raptor-weight curve is monotone down

INSTRUMENT. Same two members as `soupraptor_v1`, raptor weight 2/3. Curve over raptor weight:
0 → 0.946 (soup alone), 1/2 → 0.944, 2/3 → 0.939. Monotone decreasing: at our top the foreign reader
only dilutes, so no raptor weight rescues the ensemble lane (knee-ir6d3, parked).

## triseat_v1 — 0.952, ref 57001137, 2026-10-09 — a third family over the public 0.950 wave is worth two ticks

The public 0.950 recipe (nartaa's CoAtNet + goodpjw2008's ConvNeXt reader, rabari9999's V1 per-finding
shares, 444 teams tied there) with tripleseat's seven seated at 1/3 of a final rank blend. Pre-registered
`>= 0.952` = a third family adds; it read 0.952, rank 129 of 5567 read 2026-10-09 (silver range), from 850 at 0.946.
Kernels pinned to the pre-2026-10-09 Kaggle image (the new py3.13 image breaks transformers on
tokenizers 0.23.1). No offline number: both public arms trained on every labelled study. Ours7 alone
read 0.945 and the wave 0.950, so the gain is complementarity, not strength. Next: `kneemri-trihalf`
(ours at 1/2) asks whether the third family is under-weighted.

## trihalf_v1 — 0.952, ref 57005577, 2026-10-09 — the share is flat from 1/3 to 1/2

`kneemri-trihalf` is `kneemri-triseat` with our seven at 1/2 of the final rank blend instead of 1/3, every
other byte equal. It ties triseat at 0.952. At the board's 0.001 resolution the curve is flat across that
range, so the second tick does not depend on the exact share; it bounds the two entries' gap below 0.001
and does not license a finer share search. Triseat stays pinned (earlier submission, same score). Next on
the curve is mirrorseat (ours against nartaa alone at 1/2), which asks whether the ConvNeXt reader still
earns its seat once ours is in the blend.

## mirrorseat_v1 — 0.953, ref 57008651, 2026-10-09 — the ConvNeXt reader is redundant once ours is in

`kneemri-mirrorseat` is our seven rank-blended at 1/2 against nartaa's public 0.949 CoAtNet alone, with no
ConvNeXt reader. It read 0.953, one tick over triseat and trihalf (both 0.952, both carrying the reader).
Pre-registered `>= 0.952` = the reader is redundant; it fired, and the entry has one fewer family to mount.
The reader lifted nartaa alone by one tick on the public board (0.949 → 0.950); our seven supply the same
complementarity and more of it. One tick is directional (top-band residual sd ~0.002), not a finding.
No offline number: nartaa trained on every labelled study. `run_check` read 7 of 7. Re-pinned as champion.
Next: tiltseat/tritwo were built on the triseat wave; the open question is now the share against nartaa
alone (1/2 is the only point read).

## mirrorpair_v1 — 0.952, ref 57017195, 2026-10-09 — nartaa's second checkpoint adds nothing

`kneemri-mirrorpair` keeps mirrorseat's half for our seven and splits nartaa's half evenly between his 384
CoAtNet and his unseated 224-crop checkpoint (ours 0.50, 384 0.25, 224-crop 0.25, rank blend). It read
0.952, one tick under mirrorseat. Pre-registered `<= 0.952` = close; it fired. Two checkpoints of one recipe
on one corpus are one family, so splitting the seat dilutes the stronger checkpoint without adding an axis.
`run_check` read 7 of 7. No offline number. Champion unchanged (mirrorseat 0.953). The foreign-seat axis is
now flat on share, on reader, and on a second checkpoint of the same family.

## mirrorthird_v1 — 0.953, ref 57023373, 2026-10-09 — ours at 1/3 holds the champion's score

`kneemri-mirrorthird` is mirrorseat with the share moved: our seven at 1/3, nartaa's 384 CoAtNet at 2/3, rank
blend. It read 0.953, level with mirrorseat at 1/2. Pre-registered `0.952-0.953` = flat; it fired, so the
share is flat from 1/2 down to 1/3 at board resolution (1/2 0.953, 1/3 0.953; the triseat-wave 1/3 and 1/2
both read 0.952). That flat is the useful reading for the FINAL PICKS, not for the public score: at an equal
public lead a lower ours share has lower private variance against the 0.950 pile (medal-risk table,
`interpretations/kneemri/2026-10-09_medal_risk_vs_foreign_share.md`: ~0.08 at 1/3 against ~0.18 at 1/2), so
mirrorthird replaces mirrorseat as the pick-1 candidate. The pin stays on mirrorseat (a tie is not a better
entry). `run_check` read 7 of 7. No offline number. Next point: `kneemri-mirrorquarter` (ours 1/4), whose
flat branch keeps the lower share.

## mirrorquarter_v1 — 0.952, ref 57030644, 2026-10-10 — ours at 1/4 drops one tick; the share floor is 1/3

`kneemri-mirrorquarter` is mirrorseat with our seven at 1/4 and nartaa's 384 CoAtNet at 3/4, rank blend. It
read 0.952, one tick under mirrorthird (1/3) and mirrorseat (1/2), both 0.953. Pre-registered `<= 0.952` =
the optimum sits at 1/3 or above; it fired. One tick is inside the top band's ~0.002 residual scatter, so the
drop is directional, not a measured slope; but the pre-registration chose the branch before the number and
the branch says stop lowering. The board's share curve is 1/4 0.952, 1/3 0.953, 1/2 0.953. Pick 1 stays
`mirrorthird_v1`: the lowest share that holds the best score. The share axis is spent; no further point is
worth a slot. `run_check` read 7 of 7. No offline number (nartaa trained on all studies). Shipped as a
known repeat of the member set (`submit_check --anyway`), since only the share moved.

## mirrortiltthird_v1 — 0.953, ref 57037388, 2026-10-10 — the per-finding profile sits below board resolution

`kneemri-mirrortiltthird` is mirrorthird with a per-finding share fitted on the holdout: ours 1/2 on ACL, MCL,
Lateral Meniscus, PF OA and Synovitis, 1/4 on Fracture, and 1/3 elsewhere. It read 0.953, the same as
mirrorthird. The pre-registered `0.952-0.953` branch, "flat", fired. The holdout's +0.006 per-finding gain is a
replica's number and does not show at the board's 0.001 resolution. It carries more private risk than flat
1/3 (0.14 against 0.09), so pick 1 stays `mirrorthird_v1`. `run_check` read 7 of 7. Slot 3 goes to `trisota`,
not `mirrortilt`, because the tilt did not clear 0.954.


## trisota_v1 — 0.952, ref 57046536, 2026-10-10 — public share tables do not transfer under a third family

`kneemri-trisota` put `triseat`'s ours7 at 1/3 over rabari9999's `sota-0-954` wave: nartaa 384 with the 224 crop
at 0.3 inside the CoAtNet side, and goodpjw2008's reader with its per-finding share table re-tuned on the public
board. It read 0.952, the same as `triseat_v1` and `trihalf_v1`, and one tick under `mirrorthird`. The pre-registered
`0.952-0.953` branch fired: a share table fitted on the public board without us does not carry once a third family
is seated, so the public-retune axis closes and pick 2 stays `mirrorseat`. Every reader-in-the-wave entry reads
0.952, and every nartaa-alone one 0.953. The reader is redundant once ours is seated (the 562-study readeroof
pricing says the same offline: half-half minus ours −0.0026). No offline number; nartaa trained on every study.

## oaifree_v1 — 0.944, ref 57051234, 2026-10-10 — the reader adds nothing on the board; OAI-free hedge below the 0.948 bar

`kneemri-oaifree` blended ours7 at 1/2 with goodpjw2008's ConvNeXt reader (ImageNet init, competition-report labels),
with no nartaa: the first entry carrying no OAI-trained weights. It read 0.944. That is the score of every ours-only
seven-seat entry (`foldseven`, `eightseat`, `swapseat`, all 0.944), so the reader's half is worth zero on the board.
It sits below `basinsoupff_v1` (0.946), which is OAI-free too. The amended pick-2 rule's `<= 0.947` branch fired.
No OAI-free blend clears 0.948, so pick 2 is the owner's call between `basinsoupff_v1` (0.946, OAI-free insurance)
and `mirrorseat` (0.953, upside). The 562-row readeroof pricing had the half-half at −0.0026 under ours alone, and
the board reads it flat, inside its 0.001 resolution. The reader's +0.0104 lead over the replica pile does not survive once ours is seated.
Members ship from kit 5fea6702.

## twosoupmirror_v1 — 0.954, ref 57055272, 2026-10-11 — two weight soups replace the seven as the ours third

`kneemri-twosoupmirror` kept mirrorthird's nartaa 384 seat at 2/3 and made the ours third out of two weight soups from
different families, `basinsoup_ff_k4_s2026` (board 0.946 alone) and `basinsoup_dmlb_k7_s2026` (0.944 alone), 1/6 each in
rank space. It read 0.954 against mirrorthird's and mirrorseat's 0.953, the project's first entry above 0.953, and the
pre-registered `>= 0.954` branch fired: it is the pick-1 candidate. One tick sits inside the top band's ~0.002 residual sd,
so the reading is directional: the soups are at least as good an ours side as the seven, on two checkpoints instead of
seven, which also halves the kernel's runtime risk. The offline analog (+0.0015) was priced against a weak nartaa replica
and is sign-only. Both KernelRuns loaded 1 of 1, kit 5fea6702.

## heliosthird_v2 — ERROR (blank), ref 57054372, 2026-10-10 — the SKM-TEA instrument crashed on the hidden rerun

`kneemri-heliosthird` v2 seated heliosli's public S6 model at 1/6 beside nartaa 384 (2/3) and ours7 (1/6) to price what
the 09-15 SKM-TEA ruling costs us. It passed the public test (run_check 7/7, S6 3/3), sat PENDING ~4.6 h and came back
COMPLETE with a blank score and Kaggle's "unhandled error while rerunning your code". The hidden-rerun log is not served,
so the cause is unknown; the S6 stage is foreign code we did not write and the only new piece against twosoupmirror.
The instrument prices nothing: the ruling stays UNPRICED (not "costs ~0"). Not re-shipped — the S6 family can never be a
pick, so a second slot buys only the instrument reading, and that is worth less than a licence-clean slot today.

## heliossoup_v1 — 0.955, ref 57063827, 2026-10-11 — S6 at 1/6 beside nartaa and the soups

`kneemri-heliossoup` seats heliosli's public S6 model (`heliosli/rsna-knee-compact-v1`, 4-fold ConvNeXt-small; the
author's README invites blending) at 1/6. nartaa 384 holds 2/3 and the twosoupmirror soups take 1/12 each, rank space.
heliosthird_v2 crashed on the hidden rerun, so the outer cell now tolerates S6's PROBE_TOO_MANY_FALLBACKS exit. That
exit writes a complete mean-filled table first, so the run continues on it. The public run read S6 exit 0 and 0 fallback.
It scored 0.955 against twosoupmirror's 0.954 (rank 239 -> 116). One tick is inside the top band's ~0.002 residual sd,
so the reading is directional. It was filed as an instrument under the 09-15 ruling. The owner revoked that ruling on
2026-10-11, and the entry is now pick-eligible.

## s6third_v1 — 0.956, ref 57063964, 2026-10-11 — S6 at 1/3: the share curve climbs one tick per point

Same producers, with S6 at 1/3, nartaa at 1/2 and the soups at 1/12 each. It scored 0.956, rank 87. The S6 share curve
now reads 0 -> 0.954, 1/6 -> 0.955, 1/3 -> 0.956. The curve is monotone, so S6 really did run on the hidden rerun; a
mean fallback would have read near 0.954. Two consistent ticks are still about one residual sd each, so this is
directional. It is the champion pin and the S6-side pick-1 candidate. The next point is `s6half`, with S6 at 1/2.

## s6half_v1 — 0.955, ref 57077161, 2026-10-11 — S6 at 1/2: the share curve turns over at 1/3

Same producers, with S6 at 1/2, nartaa at 1/3 and the soups at 1/12 each. It took about four hours on the hidden rerun
and scored 0.955, one tick under s6third. The curve now reads 0 -> 0.954, 1/6 -> 0.955, 1/3 -> 0.956, 1/2 -> 0.955,
which is the pre-registered "peaked at 1/3" branch. Each step is still about one residual sd, so the peak's position is
directional. What it settles is that S6 does not want to lead the blend. The 2/3 point (`s6twothirds`) is dropped.
Slot 4 goes to `s6probit`: the probit combiner at the s6third shares, the peak.

## s6probit_v1 — 0.956, ref 57087369, 2026-10-11 — probit fusion at the s6third shares: flat

Same producers and shares as s6third (nartaa 1/2, S6 1/3, soups 1/12 each). Only the combiner changed: per-label ranks
are fused through `ndtri` instead of averaged. It scored 0.956, equal to s6third. That is the pre-registered "inside
resolution" branch, and it agrees with the 0.954 tie between `twosoupprobit` and `twosoupmirror`. Two flat reads at two
share points retire the combiner axis. Rank average stays the default. `s6sevenprobit` is dropped, and nothing gets
paired with probit. The next slots go to the ours-share axis (`s6equal`) and to decorrelating pick 2 (`s6seven`).

## s6seven_v1 — 0.956, ref 57095414, 2026-10-12 — the seven in place of the dmlb soup at the s6third shares: ties the champion, pick 2

Same shares as s6third: nartaa 1/2, S6 1/3, and ours at 1/6. Ours is now basinsoup_ff_k4 at 1/12 plus mirrorthird's
seven, logit-averaged, at 1/12, instead of the two soups. Every run carried its own Budget (knee-s6n3v), and run_check
read 8 of 8. This is kernel v2; v1 was not submitted. It scored 0.956 against s6third's 0.956.

It ties the champion with a different ours seat: the pre-registered decorrelated pick-2 branch. Pick 2 = s6seven_v1; pick 1 stays s6third_v1 unless a later entry reads >=0.957. s6sevenequal (S5) stays live if s6equal (S4) also reads >=0.956.



## s6oaifree_v1 — 0.953, ref 57102245, 2026-10-12 — the OAI-free hedge, S6 1/2 beside ours with no nartaa seat: LIFT, the OAI-free hedge moves into the 0.95 pile

heliosli S6 at 1/2, basinsoup_ff_k4 at 1/4 and mirrorthird's seven (logit-averaged) at 1/4, blended in rank space. There
is no nartaa seat, because nartaa's CoAtNet is OAI-trained and OAI eligibility still has no host ruling. Forum thread 748276
argues that the NIH/NDA geo-block makes OAI not reasonably accessible under rule 2.6.b, the same logic that banned KneeCoT
(knee-ll0of). S6's manifest is convnext_small in22k trained on the competition pool, so the entry is OAI-free end to end.
Pre-flight passed: run_check read 8 of 8 members, and S6 exited 0 at 182 s. Re-deriving the blend locally from the
kernel's own seat tables reproduced submission.csv with max |diff| 0.0. It scored 0.953 against basinsoupff_v1's 0.946, the
best OAI-free entry before it.
**0.953 is the >=0.950 branch.** S6 carries the OAI-free line +0.007 over basinsoupff_v1 (0.946), 0.003 under the OAI-bearing s6third (0.956), and 0.001 under the public 0.954 mass. The OAI-free pick 2 becomes s6oaifree_v1; s6oaifree23 (S6 2/3, same members) is armed with --anyway to read the share direction.


## s6dense2_v1 — 0.956, ref 57108981, 2026-10-12 — S6 at eval stride 2, HALF the windows: flat, the stride axis closes

Same producers, shares and blend as s6third (nartaa 1/2, S6 1/3, soups 1/12 each, rank space). The one change: S6's
`--eval-stride` patched to 2. It was built believing s6third ran S6 at the source's `.get` default of stride 3, so this was
meant to be DENSER. The mounted `rsna-knee-compact-v1` manifest says `eval_stride: 1`, and the visible logs agree: 528.7
windows/study for s6third against 262.0 here. So the entry that shipped averages half the windows. The docstring records
both the original pre-registration (wrong premise) and the corrected read.

It scored 0.956, equal to s6third. Under the corrected read that is the flat branch: S6's window density between stride 1
and stride 2 does not move the board at 0.001 resolution, and s6third's hidden S6 was not deadline-bound in any way that
a sparser pass recovers. Two consequences. The `*dense2` follow-ups (`s6oaifreedense2`, `s6sevendense2`), gated on
>=0.957, are retired. And stride 2 is a free ~half cut of S6's hidden runtime (~3.3 h -> ~1.7 h projected), which is the
headroom a fourth producer would need inside the 9 h limit.

## s6mavit_v1 — 0.955, ref 57115920, 2026-10-12 — hengck23's MaViT as a fourth producer at 1/8: DILUTES, MaViT closed as a seat at 1/8

s6dense2 (0.956) with every share scaled by 7/8 and hengck23's MALA-T MaViT reader (MIT, `hengck23/hengck23-rnsa-knee`
`00000008.pth`, loaded `weights_only=True`) at 1/8: nartaa 7/16, S6 7/24, soups 7/96 each, rank space. MaViT runs as a
subprocess from `dimiturnt/mavit-seat-src` before S6, inside the slack stride 2 freed. Kaggle version 2, the source in the
repo; v1 was the same blend with the nartaa table bound by a walrus, which `submit_check` could not resolve (UNKNOWN, exit
2), so v2 spells the path literally and `mavit_sub.csv` maps to the `hengck23_mavit` seat.

Evidence it was shipped on: `kneemri-mavitgold` scored the checkpoint on the 58 organiser-gold studies at macro 0.9202,
1.77 s/study on T4, against `band_dense_hold_s2026` on the gold 58 at 0.8956, paired +0.0247 [+0.0048, +0.0469] (`annotated_readings`).
Its lift sits where ours is weakest: Effusion 0.980 vs 0.852, Synovitis 0.823 vs 0.752, Contusion 0.942 vs 0.896,
Lateral OA 0.826 vs 0.787; ours leads PF OA and ACL. Its training data is unstated, so gold58 may be in-sample for it,
which is why the share is 1/8 and the board, not gold58, decides. Visible run: 0 MAVIT FAILED, S6 PASS, 0 fallbacks,
blend recomputed from the five frames to 1e-16.

PRE-REG (s6dense2 0.956, resolution 0.001, top-band residual sd 0.0013 @ledger75): >= 0.957 lifts and becomes pick 1,
the 1/4 point (`kneemri-s6mavit25`) earns the next slot; 0.956 flat closes MaViT at 1/8 as a seat; <= 0.955 dilutes.
RESULT: **0.955**, one below s6dense2's 0.956 at the board's 0.001 resolution (inside the top band's ~2 sd of 0.0026, so direction only). The gold58 lead (+0.0247 paired) did not transfer: either gold58 was in-sample for hengck23's weights (training data unstated) or scaling the 0.956 core by 7/8 cost more than a fourth reader adds. The 1/4 point (`kneemri-s6mavit25`) and `s6r7mavit` stay on the shelf unshipped; the R7 line takes the slots (`s6r7_v1`, ref 57130546, 2026-10-13).

## s6r7solo_v1 — 0.957, ref 57132795, 2026-10-13 — heliosli's retrained R7 bundle in S6's seat: LIFTS, new champion

`kneemri-s6r7solo`: s6dense2's shape with the heliosli seat swapped from S6 to the retrained R7 bundle
(`8c7b666e27dc0e8f`, the producer behind heliosli's public 0.957 notebook): nartaa 1/2, R7 1/3, soups 1/12 each, rank
space. run_check 2 of 2, blend recomputed from its own frames to 1e-16.

PRE-REG (s6dense2 0.956): >= 0.957 lifts and becomes pick 1; 0.956 flat; <= 0.955 R7 worse than S6.
RESULT: **0.957**, one above every s6-family entry and pinned in `comp/champion.json` (kit `5fea6702`). One tick at the
board's 0.001 resolution is direction, not a finding (top-band residual sd 0.0013 @ledger75), but it moves us from the
0.956 tier (~rank 424 of 5895 at the 2026-10-14 08:24Z read) into the 0.957 tier (~rank 119). A pure copy of heliosli's notebook
(R7 + nartaa 0.5/0.5, no seat of ours) reads 0.956 by its own authors, so our soups at 1/6 are what this entry adds.
Same-day follow-ups pending: `s6r7_v1` (S6 1/6 + R7 1/6, ref 57130546) and `s6ourquarter_v1` (soups 1/8 each, ref
57132831). The slot after them is pre-registered: `kneemri-s6r7soloquarter` if ours at 1/4 lifts, else
`kneemri-s6r7light` (R7 1/4, nartaa 7/12 — the untested side of the share, since S6 at 1/2 read 0.955 against 0.956 at 1/3).

## s6r7_v1 — 0.957, ref 57130546 · s6ourquarter_v1 — 0.956, ref 57132831 — 2026-10-13: neither moves the board

- `s6r7_v1` (S6 1/6 + R7 1/6 in the heliosli seat, soups 1/12 each, nartaa 1/2) reads **0.957**, tying `s6r7solo_v1`.
  Splitting the heliosli seat between its two producers neither adds nor costs at 0.001; `s6r7solo` keeps the pin
  (same tier, reached first).
- `s6ourquarter_v1` (soups 1/8 each, S6 0.30, nartaa 0.45; pre-reg vs `s6third` 0.956) reads **0.956**: FLAT. Our share
  at 1/4 instead of 1/6 does not lift, so `kneemri-s6r7soloquarter` (the stacked follow-up, gated on >= 0.957) is retired.
- Slot 4 goes to `kneemri-s6r7light` (R7 1/4, nartaa 7/12) per the tree written before these read.

## s6r7light_v1 — 0.956, ref 57148827, 2026-10-13 — R7 at 1/4, nartaa 7/12: one tick under, R7 1/3 stays

`kneemri-s6r7light`: `s6r7solo`'s shape with the R7 share cut from 1/3 to 1/4 and the freed 1/12 given to nartaa
(R7 1/4, nartaa 7/12, soups 1/12 each, rank space). run_check 2 of 2, blend recomputed from its own frames to 1.1e-16.
The untested side of the heliosli share: S6 at 1/2 had read 0.955 against 0.956 at 1/3, so the step is downward.

PRE-REG (B = `s6r7solo_v1` 0.957): > B lifts, pin it, slot 5 = `kneemri-s6r7sixth` (R7 1/6, nartaa 2/3); = B flat, 1/4
and 1/3 bracket a flat top and the share axis closes; < B, 1/3 is the optimum. Either non-lift sends slot 5 to
`kneemri-tblendsolo` v2, the efficiency-track instrument (knee-p844l).
RESULT: **0.956**. The pre-registered < B branch fires: R7 1/4 reads 0.956 against 0.957 at 1/3, the same shape as the S6 curve (1/6 .955, 1/3 .956, 1/2 .955), so both heliosli readers peak at a 1/3 share and `s6r7solo_v1` stays the pin. Slot 5 goes to `kneemri-tblendsolo` v2 and `s6r7sixth` is not shipped (it was gated on a lift). A 0.001 tick between two of our own entries is inside 2 sd of the top band's residual
scatter, so this reads as public RANK, not as a better model; the 0.957 tier holds ~118 teams with ties broken by
submission time, and 0.958 holds 78.

## s6r7solo224_v1 — 0.957, ref 57158477, 2026-10-13 — nartaa's unused 224-px checkpoint inside his half: flat

`kneemri-s6r7solo224`: `s6r7solo` with one stage added. `nartaa/rsna-knee-publication-swa-weights-20261007` ships two
CoAtNets and every public blend, ours included, read only the 384-px one; the second,
`raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` (sha256 3394fdd7…), is the same recipe at 224 px on a 0.8
centre crop of the field (0.945 alone). Scored by nartaa's own 224 script (`dimiturnt/nartaa224-src`, diffed against
our mounted 384 copy: crop, resolution, output name and no mirror pass are the only differences) and folded into his
half as 0.65 x rank(384) + 0.35 x rank(224), the split varshithakolupuri's public notebook fixed before any run. R7
1/3 and the soups 1/12 each unchanged. run_check clean (loaded 2 of 2 members, R7 exit 0 at 443 s).

PRE-REG (B = `s6r7solo_v1` 0.957): > B the second field of view carries signal the 384 reader misses, new pin; = B
below board resolution at 0.175 of the vote, and the reading is public RANK inside the 0.957 tie; < B the 224 reader
dilutes the 384 one at this share.
RESULT: **0.957**. Flat: the 224 view at 0.175 of the vote sits below board resolution. Public rank read live afterwards was 119 of 5954 in an 84-way tie at 0.957, so it bought no rank either. The 224 checkpoint stays a candidate only as a composition lever (s6r7flip224); alone it is closed at this share.

## s6r7flip_v1 — 0.957, ref 57167502, 2026-10-14 — R7 read with its own mirror TTA: flat

`kneemri-s6r7flip`: `s6r7solo` byte-for-byte except one anchor in heliosli's R7 source, `if manifest.get('flip_tta'):`
→ `if True:`, which passes the worker's shipped `--flip-tta`. The R7 bundle trained with `mirror: 0.5` (coronal/axial
width flip, sagittal slice reversal, Medial ↔ Lateral outputs permuted), so the second view is in its training
distribution; the worker averages the two passes in logit space and takes MCL from the plain pass. The manifest ships
`flip_tta: false`, so every public R7 blend, `s6r7solo` included, read the reader on one view. Blend unchanged:
nartaa 1/2, R7 1/3, soups 1/12 each, rank space. run_check clean (loaded 2 of 2 members, R7 exit 0 at 414 s, FLIP_TTA_READY in both workers).

PRE-REG (B = `s6r7solo_v1` 0.957): > B the mirror view carries signal, new pin; = B below board resolution at a 1/3
rank share, s6r7solo stays; < B the mirrored pass (or its state read-out through a stacker fitted on plain passes)
costs signal, or the 8.3 h deadline bit and unreached studies fell back to the mean.
RESULT: **0.957**. The = B branch fires: the mirror view at a 1/3 rank share sits below board resolution and `s6r7solo_v1` keeps the pin (same tier, reached first). Live read 2026-10-14 08:26Z: rank 127 of 5958, a 95-way tie at 0.957, leader 0.966. Pending to scored took 3.4-3.6 h (04:52Z submit; PENDING at 08:16Z, COMPLETE at 08:26Z), against the R7 deadline at started + 8.3 h, so the deadline fallback cannot have filled the table wholesale. Slot 2 goes to `kneemri-s6r7stride1` per the rule written before this read (flip > B would have sent `s6r7flipstride1`); `s6r7flipres384` drops with it. A 0.001 tick between two of our own entries is inside 2 sd of the top band's residual
scatter, so it reads as public RANK, not as a better model.

## r7half_v1 — 0.956, ref 57178693, 2026-10-14 — R7 1/2 + nartaa 1/2, the soups' sixth dropped: one tick under, the soups still decorrelate

`kneemri-r7half` (v1): `kneemri-s6r7solo` with both weight soups removed, R7 (heliosli's bundle 8c7b666e27dc0e8f, eval
stride 2, rc tolerance) 1/2 + nartaa 1/2, rank fusion via `RaptorSeat.combined`. This is the composition heliosli's
teammate jingshuifu published on 2026-10-14 as `rsna-knee-raptor-multiview`, a "private candidate" with no score.
Our R7 share curve read 1/4 0.956 and 1/3 0.957; 1/2 was never read. Kaggle run clean: nartaa seat 3 rows, R7 exit 0
at 403 s, PASS 3/3, fallback 0. `run_check` refuses on "no member census printed" by construction (no member of ours
is served), and its other refusals read clean by hand.

PRE-REG (B = `s6r7solo_v1` 0.957): > B the soups were dilution once R7 sits beside nartaa; = B the soups' sixth is
below board resolution and r7half is the cheaper equal (pick-2 candidate, different composition); < B the soups still
decorrelate.
RESULT: **0.956**. Under B by one tick, the direction the 0.956 prior predicted: dropping the soups loses rather than gains, so the soups' sixth still decorrelates slightly against R7 + nartaa. Pick 2 therefore stays with `s6r7_v1` (0.957, different producers), not r7half. Board read 17:37Z: rank 135 of 5995, tied with 108 others at 0.957; leader 0.966 (12:26Z it was 131, so the tie keeps filling from below). Full-LB download 17:37Z: silver line (rank 299) 0.956 with 1087 teams at or above it, gold (rank 21) 0.963. A 0.001 tick between two of our own entries is inside 2 sd of the top band's residual
scatter, so it reads as public RANK, not as a better model.


## s6r7stride1_v1 — 0.957, ref 57185827, 2026-10-14 — R7 at its manifest's own eval stride 1: flat, stride 2 stays

`kneemri-s6r7stride1` (v2): `kneemri-s6r7solo` with one more R7 anchor, `str(manifest.get('eval_stride', 3))` →
`'1'` (was `'2'`). Members, weights (nartaa 1/2, R7 1/3, soups 1/12 each) and rank fusion are s6r7solo's; no flip.
run_check clean (loaded 2 of 2 members, R7 exit 0 at 429 s, no FLIP_TTA_READY); submission bit-identical between v1
and v2; the R7 table moves against stride-2 plain by up to 0.31 on sampled studies, so the lever landed. Stride 1 is
what R7's own manifest declares; every s6r7* entry, the 0.957 pin included, ran 2 through the anchor `s6dense2`
(knee-1oxsy) introduced, which HALVED S6's windows against its manifest's stride 1 and read flat (0.956 = 0.956). So
this asks the same question of R7: an INSTRUMENT, nothing tuned, expected flat.

PRE-REG (B = `s6r7solo_v1` 0.957): > B unlike S6, R7's sparser read had cost signal; = B stride is below board
resolution and stride 2 stays as free runtime (expected); < B stride 2 was better for this blend.
RESULT: **0.957**. The = B branch fires, as expected: R7's denser read moves its own table (up to 0.31 on a sampled cell) but not the blend at board resolution, the same answer S6 gave, so stride 2 stays as free runtime and the R7 eval-stride axis closes at 0.001. Pending to scored took ~4.1 h (14:38Z submit, COMPLETE by 18:46Z), the slowest R7 entry yet but inside the 8.3 h deadline. Full-LB read 18:27Z (n=5996): ours rank 136 at 0.957; gold #21 0.963, silver #299 0.956, >=0.958 113 teams, >=0.957 226. Flip and stride have now both read flat on R7, so neither is a lever to compose onto a res384 > B read: composing a flat lever buys nothing the board can see, and the 10-15 slot after such a read goes to a mechanism of its own, not to `s6r7flipres384` or a res384+stride1 build. A 0.001 tick between two of our own entries is inside 2 sd of the top band's residual
scatter, so it reads as public RANK, not as a better model.

## s6r7res384_v1 — 0.957, ref 57192860, 2026-10-14 — nartaa's 384 reader at its full 384 px view: flat, = B

`kneemri-s6r7res384` (v2): `s6r7solo` except nartaa's scoring source is patched on two anchors, `INPUT_RES = 320` →
`384` and its shape assert `(94,3,320,320)` → `(94,3,384,384)`, both via split-unpacking that refuses a source holding
either anchor zero or twice. His source trains and scores `coatnet_rmlp_2_rw_384` on a 320 px centre crop of a native
384 px, 140 mm box (`INPUT_MODE = "crop"`), so the outer ~12 mm per side is never read; this reads it at the same pixel
scale (crop, never resize), 1.44x the field of view. Blend unchanged: nartaa 1/2 (mirror TTA), R7 1/3, soups 1/12
each, rank space. nartaa infer 16.9 s vs 13.0 s. run_check clean on v2 (loaded 2 of 2 members, R7 exit 0 at 392 s).

PRE-REG (B = `s6r7solo_v1` 0.957): > B the clipped periphery carries signal, new pin, and the 224 reader's full-view
variant becomes worth building (a SCALE change there, `CROP_FRAC = 0.8` before resize, so priced lower); = B below
board resolution, read the rank; < B the train/test field-of-view mismatch costs more than the periphery adds.
RESULT: **0.957**. Ties `s6r7solo_v1` and `s6r7_v1` at the competition's 0.001 resolution: the 1.44x field of view neither carries board-visible signal nor costs it, so neither the periphery hypothesis nor the train/test mismatch is visible here. Not a new pin; the 224 reader's full-view variant is NOT armed (its gate was > B). A 0.001 tick between two of our own entries is inside 2 sd of the top band's residual
scatter, so it reads as public RANK, not as a better model.
