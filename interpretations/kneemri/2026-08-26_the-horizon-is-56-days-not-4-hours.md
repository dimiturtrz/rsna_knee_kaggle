# The horizon is 56 days, not four hours — and the climb rate says 0.950 arrives without a step change

**2026-08-26 22:46Z.** `kaggle competitions list -s rsna-knee --csv` plus `board_read --live` plus a fit
over `comp/board.jsonl`. No slot, no card, three minutes.

## The fact nothing in this repo held

```
deadline 2026-10-22 23:59    teamCount 2469    userRank 46
```

**56 days remain.** At five slots a day that is roughly **280 submissions**, not five. The deadline
appears in no doc, no bead, no kernel and no interpretation — every plan this project has made was made
against a four-hour horizon and a five-slot budget, because the four-hour horizon is what the recurring
prompt says and nobody asked the competition.

## Our own climb, fitted

Running-best public score, one point per day, from the first day the pool stabilised:

| date | best | date | best |
|---|---|---|---|
| 08-18 | 0.925 | 08-23 | 0.934 |
| 08-19 | 0.929 | 08-24 | 0.937 |
| 08-20 | 0.931 | 08-25 | 0.938 |
| 08-21 | 0.931 | 08-26 | **0.939** |
| 08-22 | 0.933 | | |

| window | slope | days to +0.011 |
|---|---|---|
| all 9 days | **+0.0016/day** | 6.7 |
| last 5 | +0.0016/day | 6.9 |
| last 3 | +0.0010/day | 11.0 |

**0.950 is one to two weeks away at the rate we are already moving, against 56 days of runway.** Mild
deceleration is there (mean daily gain 0.0020 over the first four days, 0.0015 over the last four), and
even the pessimistic 3-day slope lands inside the window with 45 days to spare.

## Every recent gain is below the noise floor, and they compose

Of **57 scored submissions, 18 set a new best** — a 31.6% hit rate. But since 2026-08-19 the sizes
collapse:

```
band_v1      0.925 -> 0.928   calibspan_v1  0.928 -> 0.929
greedy4_v1   0.929 -> 0.930   basedense_v1  0.930 -> 0.931
sevenplane   0.931 -> 0.932   gradeswap_v1  0.932 -> 0.933
finalfit_v1  0.933 -> 0.934   allfit_v1     0.934 -> 0.937
votefit_v1   0.937 -> 0.938   blendseat_v1  0.938 -> 0.939
```

**Eleven consecutive new bests, none of them larger than +0.003.** Not one clears the 0.0062 three-seed
noise floor on its own. Read individually every one of them is a knob, and CLAUDE.md's own rule says a
knob-sized delta is directional and never a finding.

Read as a chain they are **+0.014 against the 08-18 baseline** — and the same rule says increments below
the bar still compose, and a chain that sums past the bar IS a result. That is what this project has
actually been doing for eight days: not finding a lever, running a **ratchet**. The 31.6% hit rate is the
ratchet's efficiency, and it is high.

**This is the reading that changes the spend.** The loop has repeatedly wanted a step change because each
individual submission looked like noise. The board says the process is working at +0.0016/day and the only
thing it lacks is days — of which it has 56.

## The leader, on two points

Leader **0.952 on 2026-08-25** and **0.952 on 2026-08-26**. Flat across the only two reads we hold.
**Two points is not a slope** — state that plainly rather than fitting it. Our rank moved 48 → 46 while
the field grew 2373 → 2469, so we gained on the board while the board grew under us.

The gap is **0.013**. Our own slope covers it in eight days *if the leader stays still*, which is the one
assumption here with no evidence behind it. That, not our own rate, is the open question — and it is
answerable for free: `board_read --live` costs no slot, and a leader-score series is the instrument this
project does not yet keep.

## What it changes

1. **Slots are not the scarce resource. Card hours and ideas are.** ~280 remain. The rule that has
   governed all day — spread them, save the last — is a *daily* rule and correct as one; it should never
   again be read as a global scarcity that argues against firing an informative arm.
2. **Do not abandon the ratchet for a step change.** The ratchet delivers 0.950 inside the window. A step
   change is worth pursuing *in parallel* — it is what closes on FIRST place rather than on 0.950 — but
   not by stopping what is working.
3. **Seating is the ratchet's cheapest fuel and it is nearly spent.** Nine six-member entries span 0.014
   in total, and the last two gains came from a vote rule (`votefit`) and better label supervision
   (`blendseat`), not from a seat swap. The two most recent non-seating mechanisms are the two that moved
   it, which agrees with everything the forum and
   [our own gold-label reading](2026-08-26_our-model-is-past-its-teacher.md) said today.
4. **Start a leader series.** One `board_read --live` a day, appended rather than overwritten, turns the
   only unevidenced assumption in this note into a measurement within a week.

## The denominator, said out loud

Nine daily points for our slope; **two** for the leader's. The slope is fitted on running-best, which is a
monotone series — it cannot go down, so its fit is optimistic by construction and a flat week would not
show as a decline, only as a bend. The deadline and the rank are Kaggle's own API and are exact. Treat the
"6.7 days" as an *order of magnitude that comfortably fits inside 56*, never as a date.
