# v3 speaks less and misses more, and the ruler moves six of eight seats

2026-09-06. Zero card. Two readings taken the moment the corpus reader finished, by
`experiments/reader_end_chain.sh`.

## The corpus

The reader walked **4407 studies**, finish reason `stop` for all 4407 — no truncation, no rate-limit
tail — and wrote both sidecars. The chain refused it on the first run with `REFUSED: 4407 distinct
studies, expected 4406`: the guard's constant had been carried over from prose, while the reader's own
census reports `total = 4407` on every one of the twelve findings. The constant now cites the sidecar.
That is the third time in this project a number quoted from prose disagreed with the artefact that
produced it.

## Reading one: agreement against the radiologist

**Denominator: 696 annotated cells over 58 studies.** That is the whole radiologist-annotated set, and
v3 read every one of them, so this is a rate over the corpus that matters and not over a subset that
happened to parse.

| | false positives | misses | wrong |
|---|---|---|---|
| v2 (recorded) | 108 *confident* | 32 | 140 |
| v3 | 47 (all) | 74 (UNK 45, NO 29) | **121** |

**The −19 is not a like-for-like win and is not reported as one.** The two halves moved in opposite
directions, and the v2 figure counts only *confident* false positives while the v3 figure counts all of
them — so the FP columns are not the same measurement. What survives the mismatch is the direction: **v3
speaks less and misses more.**

Whether that is an improvement depends on what a label is for. For training, a miss is a mislabelled
negative and a false positive is a mislabelled positive; neither is free, and the class balance decides
which hurts more.

### The silence has less headroom than the noise

**221 of 696 cells (31.8%) are silent.** The radiologist calls **20.4%** of those positive, against
**41.1%** where the reader spoke. So the reader's silence is genuinely informative — it is not
uniformly distributed over the truth — and a *perfect* silence-filler would fix at most **45** cells,
while the over-calling that already exists costs **47**. Any silence-filling heuristic that is less
than perfect is very likely a net loss. This closes the "fill the UNKs" idea at the arithmetic, before
any card is spent on it.

Silence is concentrated where the report language is least standardised: Synovitis 41, Baker's 31,
Fracture 26, Lateral OA 25.

### Per finding

Net (over minus under) is dominated by three cells, and they do not point the same way:

| finding | over | under | net |
|---|---|---|---|
| Contusion | 14 | 3 | **+11** |
| ACL | 5 | 1 | +4 |
| PF OA | 0 | 8 | −8 |
| Synovitis | 3 | 16 | **−13** |
| Effusion | 2 | 16 | **−14** |

Effusion and Synovitis are under-called and Contusion is over-called. A per-finding threshold is the
obvious response and is exactly the kind of fit-the-loss-to-the-weakness move the project bans — the
fix belongs in the prompt or in balanced data, not in a per-label knob.

### Three knob-sized reads, three intervals containing zero

- English 80.0% against non-English 76.3%: **+0.037 [−0.079, +0.145]**
- graded positive calls 77.0% against plain 82.9%: **−0.059 [−0.200, +0.094]**
- hedged 83.3% against plain 78.9%: **+0.045 [−0.227, +0.223]**

Every interval crosses zero. Directional at most; none is a finding, and none licenses a language or
hedging branch in the reader.

## Reading two: what the ruler does to the seat order

`ruler_swap` holds the predictions fixed and swaps the truth table underneath — the different question.
109 members over 909 shared studies, 8 seats deep. The candidate table covers **1.0000 of the 10908
cells**, so nothing below is a coverage artefact.

**Seated-set churn 0.750** — every one of the eight seats moves position, and six of the eight *names*
change. `chainfour_s2026` goes from seat 1 to seat 8; `labelwise_dense_s2026` from 6 to 1.

The blend reads 0.8768 on the recorded ruler and 0.9018 on the candidate. **That 0.025 is not a gain.**
It is the same predictions read by a more agreeable ruler; it prices the RULER, not the members. Quoting
it as an improvement would be the `board_transfer` error again.

What the churn says is sharper than any level: **at three-quarters churn, a seating argument built on
the recorded ruler is an argument about the ruler.** Every closed seating experiment on this project was
decided under the recorded ruler. This does not reopen them — the axis is closed on seven board flats,
which are board readings and not ruler readings — but it does say that the offline ORDER those
experiments produced was never the stable object it looked like.

## What this does not rule out

- **Labelling with v3 anyway.** This measures agreement on 58 studies; it says nothing yet about whether
  a member trained on v3 labels scores better. That is a card experiment and it is untouched.
- **The under-called pair.** Effusion and Synovitis are under-called by 30 cells between them, and both
  are findings whose report language is unstandardised. A prompt change aimed at the LANGUAGE (not at
  the threshold) is a live mechanism.
- **The ruler itself.** Which of the two tables is closer to the radiologist is not answered here;
  `ruler_swap` deliberately does not ask.
