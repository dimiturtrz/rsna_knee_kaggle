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

## The refit term is SOLO-ONLY, and the board already measured that twice

The +0.003 above is a statement about a LONE member. Inside an ensemble the same swap has been run as a
dose series, and it is flat at every dose:

| entry | refit seats of 8 | board | ref |
|---|---|---|---|
| `tripleseat_v2` | 0 | 0.946 | 56232529 |
| `fullseat_v1` | 1 | 0.946 | 56246750 |
| `fullseat2_v1` | 2 | 0.946 | 56250751 |

Three entries, identical but for how many seats were swapped to their `--fold -1 --ema` whole-corpus
refits, all reading the same thousandth. A tie at one value bounds the true spread below 0.001, so one
refit seat is worth **< 0.001 board** in an eight-seat entry against **+0.003** as a solo — and two seats
buy no more than one.

The mechanism this points at: the extra holdout fifth teaches a lone member something its seven
neighbours already supply between them. Refitting is a way of recovering information the ensemble is
already recovering by averaging, so the two do not add.

Consequence for the hunt, and it is the reason this section exists: **the champion cannot be lifted by
refitting its members.** That has been tested at two doses with every gate green. The solo path is the
one where the refit term is live, which makes the inversion below the correct place to spend it — a solo
at 0.9074 honest offline75 would be refit into 0.947 against a 0.946 that refits into 0.946.

## What this does NOT license

- The +0.003 refit term is **one matched pair** at the solo end, not a curve. It is a directional
  measurement with n=1 and no reason to be constant across recipes. The three ensemble rows above do NOT
  give it a second solo measurement; they bound the term in a DIFFERENT population (1 seat of 8) where it
  reads under 0.001.
- The inversion assumes the fit's slope holds 0.005 beyond its largest x. `l75solo_v1` at 0.9053 is the
  top of the fitted range; 0.9142 is an extrapolation.
- Nothing here says the +0.0089 is reachable. It says what it would have to be worth.

## AMENDMENT, 2026-09-17 23:25Z — the title is now too strong, and the miss is at the top

`extsolo_v1` (ref 56315030) shipped `labelwise_ext_drop40_long_s2026`, offline75 **0.9045**. The n=8
fit above predicted **0.9403**, 2 sd [0.9371, 0.9435]. It read **0.937** — below the band, −2.1 sd,
ending three consecutive thousandth-exact calls.

One point outside a 2 sd band in nine is what a 2 sd band is for. What makes this one load-bearing is
**where** it sits. Every call the fit got right was taken between 0.875 and 0.896. Everything this
project plans is an extrapolation ABOVE 0.904, and until tonight exactly one row lived up there
(`l75solo_v1`, 0.9053 → 0.941). There are now two, 0.0008 apart on the x axis and **0.004 apart on the
y axis** — twice the residual sd, at essentially the same abscissa.

So the top of this line is one hit and one miss and cannot discriminate between:

* **scatter** — the board resolves to 0.001 and a 0.004 spread at fixed offline75 is within what two
  members of equal offline quality can differ by on a hidden split; or
* **a bend** — the line flattens above ~0.904, i.e. offline75 gains stop converting where we actually
  ship, and every plan that multiplies by 1.1 is over-predicting its own payoff.

The second reading is the expensive one and it is not yet refuted. Refit with the ninth point:
`board = -0.0705 + 1.1164 * offline75`, **residual sd 0.0019** (was 0.0016) — the slope fell from
1.165 to 1.116 on one row, which is itself a statement about how little constrains it.

**Operationally, until a third point at the top:** a predicted board above ~0.904 offline75 is an
**upper bound**, not a centre. Do not price a plan's payoff off this slope alone, and do not quote the
gap to first place (+0.012 offline75) as though its conversion were known.

`kneemri-headlrsolo` is that third point: `headlr_ship_s2026`, offline75 **0.9059**, the highest this
project owns, pre-registered on the LINE rather than on its member.

## SECOND AMENDMENT, 2026-09-18 00:28Z — the title is now wrong, and the top of this line is retired

`headlrsolo_v1` (ref 56316099) shipped `headlr_ship_s2026`, offline75 **0.9059**, the highest single
member this project owns. It read **0.937** — the same board as `extsolo_v1`'s 0.9045 member.

The top of the line is now three points: 0.9045 → 0.937, 0.9053 → 0.941, 0.9059 → 0.937. Their offline75
spread is **0.0014**, well inside the 0.0062 paired floor, so on our own ruler these three members are a
**tie**; their board spread is 0.004. The line fitted on the seven points below 0.900
(`board = -0.1217 + 1.1743 * offline75`, residual sd 0.0019) over-predicts all three — 0.0034 / 0.0004 /
0.0051, **three of three LOW**, −2.7 sd at the newest point.

What this establishes, and what it does not. It does NOT establish a bend: three tied members cannot
measure a slope, and the sign test on n=3 is weak. It DOES establish that **above offline75 0.904 the
slope's centre may not be used — only its upper bound** — and that our holdout no longer separates the
best members we own. The original title ("predicts the board to the thousandth") held for the body of the
range and is false at its top, where every shipping decision is actually made.

Consequence for planning: every plan shaped "raise offline75 by X and collect ~1.1X of board" is unpriced
where we ship. The one term still measured board-to-board is the **refit** (+0.008 / +0.003), and it is
untouched by this because it never passes through the offline ruler — see
`2026-09-17_the_refit_term_is_per_member.md`.
