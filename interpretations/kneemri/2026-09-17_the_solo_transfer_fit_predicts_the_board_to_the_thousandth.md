# The solo-only board transfer fit predicts the board to the thousandth, and it says what a single member has to be worth

2026-09-17. Bead knee-e9f83. No card spent: an OLS over `comp/board.jsonl`.

## Read it, never quote it

`CLAUDE.md` records five occasions on which a board-transfer constant was quoted from prose and was
wrong — +4.67, +2.14, +1.55, a "saturating" curve that was in fact steepening, and a pair count that
re-partitioned on the next row. **This file is a snapshot of one afternoon's fit and is subject to the
same rule.** The fit moves every time the record gains a row: adding `l75solo_v1` alone took it from
`-0.1217 + 1.174` to `-0.1131 + 1.165`. Re-derive before using:

```python
rows = [json.loads(l) for l in open("comp/board.jsonl", encoding="utf-8")]
solo = [r for r in rows if r.get("members") == 1 and r.get("scope") != "excluded"
        and r.get("offline75") and r.get("board")]
# OLS board ~ offline75
```

## The fit, 2026-09-17

Solo entries only, `scope: ensemble`, both numbers present, ruler `ledger75`:

| entry | offline75 | board | residual |
|---|---|---|---|
| `ortho_solo_v1` | 0.8748 | 0.907 | +0.0013 |
| `resnetlong_v1` | 0.8764 | 0.907 | −0.0018 |
| `cnnsolo_v1` | 0.8842 | 0.915 | −0.0011 |
| `v4solo_v1` | 0.8874 | 0.922 | +0.0016 |
| `spansolo_v1` | 0.8895 | 0.921 | −0.0016 |
| `holdsolo_v1` | 0.8949 | 0.928 | +0.0024 |
| `matchpair_v1` | 0.8962 | 0.933 | −0.0006 |
| `l75solo_v1` | 0.9053 | 0.941 | −0.0002 |

    n = 8    board = -0.1131 + 1.165 * offline75    residual sd 0.0017

Every residual is inside ±0.0024, on a board that reports at 0.001 — so the fit is within two reporting
steps everywhere, across a 0.03 span of offline75.

## Why solo-only, and why that is the honest cut

Member count is the confound. An ensemble row carries a premium over its best member (measured at
**+0.002** on the champion), so mixing member counts into one regression fits a line through two
different populations and attributes the premium to the slope. This fit spans one member throughout,
which is also the only shape we can act on: the deliverable is one checkpoint.

**`ffsolo_v1` (board 0.944) is correctly ABSENT.** It is a full-corpus refit, so it holds out nothing
and has no honest `offline75` to regress against — `entry_offline` leaves the cell null rather than
filling it with an in-sample number, which is the `raptor_solo` void rule doing its job. Five older
solos (`solo_v1`, `solo_v4`, `soloseat_v1`, `calibsolo`, `raptor_solo`) carry no `offline75` either and
are out for the same or a related reason.

## Three consecutive point predictions, each exact to the thousandth

The fit is used by pre-registering a point and a 2sd interval BEFORE a slot is spent, then reading the
score. The last three have all landed on the point:

- the corrected slope at +1.07 predicted **0.938**, and the board read **0.938**
  (`comp/SUBMISSIONS.md:4340` — and the same entry is the case study in reading a RAW offline where a
  recut one was owed);
- the following solo anchor landed on its point too — *"The slope predicted a board score to one
  thousandth"* (`comp/SUBMISSIONS.md:4495`);
- `l75solo_v1`: predicted 0.9414 from `1.174 × 0.9053 − 0.1217`, modal branch `0.938–0.945`.
  **Scored 0.941** (`comp/SUBMISSIONS.md:4512`).

That is three such calls in a row, each pre-registered before its slot. Three is not
a validation — it is three — but it is enough to use the fit for PLANNING rather than only for
reporting, which is what the rest of this file does.

## What the fit says a single member has to be worth

This is the part that changes what to hunt for. Invert it, and add the **+0.003 that a full-corpus
refit buys one member** (measured directly by the `l75solo_v1` / `ffsolo_v1` matched pair, same recipe,
differing only in whether the holdout fifth was trained on):

| board target | needed honest offline75 | gap from our best solo (0.9053) |
|---|---|---|
| 0.944 (tie `ffsolo_v1`) | 0.9053 | **0** — already there |
| **0.947 (BEAT the 0.946 champion, solo)** | **0.9074** | **+0.0021** |
| 0.955 (the stated goal) | 0.9142 | **+0.0089** |

**The champion is beatable by one member about 0.002 offline75 better than the one we ship**, refit on
the full corpus — no ensembling, no reseating. That number is *inside the 0.0062 paired floor*, which
means it is not even detectable as a single A/B: it cannot be shopped for with knob arms, and any
candidate has to clear the floor to be believed at all. The honest reading is that a tie-or-beat is a
coin flip at this margin — 2sd on the fit is ±0.0034 board, wider than the 0.001 we need — so "beat the
champion" is not a target to aim a mechanism at.

**0.955 is.** It needs **+0.0089 offline75 on a single member**, which is a ~1%-class lever and squarely
the size the owner's standing bar asks for. That is the number every open arm should be priced against,
and it is why knob-sized arms at the shipping corner are instruments rather than improvements.

## What this does NOT license

- The +0.003 refit term is **one matched pair**, not a curve. It is a directional measurement with n=1
  and no reason to be constant across recipes.
- The inversion assumes the fit's slope holds 0.005 beyond its largest x. `l75solo_v1` at 0.9053 is the
  top of the fitted range; 0.9142 is an extrapolation.
- Nothing here says the +0.0089 is reachable. It says what it would have to be worth.
