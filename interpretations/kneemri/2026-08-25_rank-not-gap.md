# We are 48th of 2,373, not "0.014 behind" — the board instrument was reading one page

2026-08-25 · `python -m kneemri.kaggle.board_read --live` · CPU, no slot, no card

## What was measured

```
rank 48 of 2373, tied with 7 at 0.938; leader 0.952, gap 0.014, taken 2026-08-25T13:39:48+00:00
```

Team `Dimitar Terziev` / `dimiturnt`, 0.938, 58 submissions — the third-highest submission count on the
board. Above us: Brandon Low 0.952, MKhlystun 0.948, qkrtkddnjs 0.947, Scott Willis 0.947. Then a wall:
**forty-two teams inside 0.936–0.940**, seven of them on our exact score.

## What it corrects

`board_read` was built yesterday to end the practice of quoting the gap from prose, and it shipped with
the defect it existed to prevent. It called `competition_leaderboard_view`, which returns the board's
**first page — twenty rows**. A team at rank 48 is structurally absent from that page, so `ours` was not
read from the board at all; it was passed in from `champion.json` and subtracted from the page's best
score. The output — `leader 0.952, ours 0.938, gap 0.014` — is arithmetically correct and answers a
question nobody asked. Every note since has quoted it as though one team stood between us and the top.

Forty-seven do. And the shape of those forty-seven is the whole planning story:

- **A 0.014 gap across a thin board is one rival to pass.** Across this plateau it is the entire field.
- **The plateau cuts both ways.** Seven teams share 0.938 to the board's precision, and the board
  resolves ~0.0015. One board increment here buys tens of places — the same increment near the top buys
  almost nothing, because the top is where the board is thinnest.
- **The house bar (~1.5% absolute) and the distance to first place are now the same number.** That is
  new information: it says no knob closes this, and it says the remaining distance is exactly one
  result of the size we have agreed is worth calling a result.

## The part that is worse than a bug

**The fact was already reachable and nobody ran the tool.** `kneemri/kaggle/efficiency_read.py`
downloads the full accuracy leaderboard (`competition_leaderboard_download` — same call, also no slot)
and its `standing(team)` already prints `accuracy rank …`. It was written earlier the same day, for the
efficiency track, and the accuracy rank rode along as a byproduct nobody read.

So this was not a missing capability. It was a capability in a module named for a different question,
while the module named for THIS question answered it wrongly. That is the failure mode CLAUDE.md's
"audit → file → plan → work" exists to stop, and the audit step here was one `grep -l leaderboard_download`.

## What changed

`board_read` now downloads the **whole** board and reports a `Standing` — rank, field, and how many
teams are tied with us — with `gap` demoted to a property. Our score is read off the board rather than
trusted from `champion.json`; the two should agree, and a disagreement is worth hearing about. `ours` is
found by `TeamMemberUserNames`, never `TeamName`, because a team name is display text the owner may
change between reads. `comp/leaderboard.json` gained `rank` / `field` / `tied`, so `champion_check` —
which already refuses a read older than two days — now refuses a stale *rank* rather than a stale gap.

One incidental defect found on the way: the CSV Kaggle ships carries a BOM, so decoded as plain `utf-8`
its first column is named `﻿Rank` and every row raises `KeyError: 'Rank'` against a header that reads
correctly in a log. `utf-8-sig`.

## Provenance

Beads: `knee-8g5w` (this, closed), `knee-1kn5` (the two downloaders, open). The board file it writes is the source; do not quote the numbers above from this page
once they are a day old — the standing decays every day five slots are spent, ours and every rival's.
