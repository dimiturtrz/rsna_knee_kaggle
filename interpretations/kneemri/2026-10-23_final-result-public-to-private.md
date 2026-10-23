# Final result: what the private board said about the public one

**Result.** Bronze. Both final picks — `s6r7pfheavytrio_v1` (ref 57431093, manual) and Kaggle's auto pick
among the 0.958 entries — scored **0.943 private** against **0.958 public**. Public board at close: 0.958,
rank 288 of ~6,400 teams, leader 0.970. The private rank is not served by the CLI; the medal is as Kaggle
shows it on the team page.

**Denominator.** The submissions API returns the 20 most recent entries (2026-10-19 04:47Z to
2026-10-22 19:29Z), all scored on both splits. Two of them (`flatown_v1`, `flatown2x_v1`, public 0.668)
were broken column-order entries and are left out of the fit; the other 18 are the population below.

| public | private | n | entries |
|---|---|---|---|
| 0.958 | 0.943 | 7 | heavyflat, sixthflat, heavytrio (pick), flatown4x, seventhflat, tenthflat, alleven |
| 0.958 | 0.942 | 1 | reader split 0.40/0.60 |
| 0.957 | 0.942 | 2 | row lever (column order fixed), trio60 |
| 0.956 | 0.941 | 2 | r7oaifreedual, trioeven |
| 0.955 | 0.940–0.941 | 2 | r6oaifree, r7oaifreedual |
| 0.954 | 0.938 | 1 | r7oaifreedual |
| 0.953 | 0.935 | 1 | reader split 0.30/0.70 |
| 0.950 | 0.930 | 1 | nartaa 384/224 at 0.80/0.20 |
| 0.942 | 0.922 | 1 | reader split 0.60/0.40 |

**What it says.**

1. **The private split kept the public order exactly.** No pair of our entries swaps order between the two
   boards. Every knob that lost on public lost on private too, and by more: the slope is about +1.3
   (0.943 − 0.922 over 0.958 − 0.942), so the private split spreads our entries *wider* than the public one.
   The fear that the 0.958 pile was public noise that private would
   reshuffle — did not hold for our own entries. The final-week knobs were not over-fitted to public.
2. **The drop is a level shift, not a defect of any one entry.** At the top it is a uniform −0.015; it grows
   to −0.020 at public 0.942. A constant offset over every structure we shipped (flat line, trio, soup
   shares, OAI-free variants) is a property of the private split's difficulty, not of the picks.
3. **The 0.958 tie was split by hidden precision**, as expected for a board that reports three decimals: seven
   0.958 entries read 0.943 and one (reader split 0.40/0.60) read 0.942.
4. **Pick choice did not matter.** The best private score among these 20 is 0.943, reached by seven
   entries, both picks among them. No unpicked entry in this window would have scored higher.

**What it does NOT say.** It says nothing about the field. The medal fell from what the board suggested
because other teams' public-to-private drop was smaller than ours, or their order changed, and neither is
visible from our own submissions. Entries before 2026-10-19 are not in the API's window, so whether an
older, structurally different entry held up better on private is untested here.
