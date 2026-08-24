# Architecture diversity is closed, and the cell it left empty is strong-and-distant

2026-08-24. Sense-making of eight of our own arms, not a proposal. Beads: `knee-2f0q`, `knee-bdhb`,
`knee-1xsu`. Ledger: `experiments/README.md` §`cnx_dense`, §`v3base_dense`, §`vjepa_warm`, §`mae_matched`.

## The question this answers

Mixing architectures is the standard prescription for an ensemble, and the champion is seven members of
one family — six DINOv2-small plus one DINOv2-base. The intuition that this is the obvious gap is a good
one. It is also the intuition this repo has already tested three separate ways, and it did not pay. That
is worth writing down properly, because the reason it did not pay is more useful than the fact.

## What we actually measured

Seat value is measured by `record_mix` over the 909-study shared holdout with the roster held fixed, and
`+0.0015` is the threshold at which a seat has ever been resolvable on the board.

| member | trunk | solo | seat value | verdict |
|---|---|---|---|---|
| `base_dense` | DINOv2-**base** (768 wide) | 0.8512 | the one paying seat | seated; produced the 0.931 |
| `v3base_dense` | DINOv3-base, full adaptation | 0.8455 | **−0.0007** (sixth), **−0.0005** (swap) | tie |
| `cnx_dense` | ConvNeXt — not a transformer at all | 0.7927 | **−0.0042** | no |
| `vjepa_warm` | V-JEPA2-L — 3D, video-pretrained | 0.8337 | **+0.0002** | no |
| `mae_matched` | DINOv2-small, domain MAE pretrain | 0.6735 | **−0.0041** | no |

Three trunk families, one of them outside the transformer class entirely, and the axis closes on the
ledger's own words: *a different inductive bias is not a different error structure.*

## Why a trunk swap cannot reach the error structure

Look at what those arms hold FIXED rather than at what they move. Every one of them shares the same slice
sampling, the same 2D-slice-then-pool architecture, the same five-level targets, the same BCE objective and
the same head. The only thing that moves is the trunk's weights.

The errors an ensemble needs to be uncorrelated are made downstream of all of that. If two members are
asked the same question, with the same supervision, over the same pixels, pooled the same way, they will be
wrong about the same studies — and a different pretraining corpus does not change which studies those are.
It changes how the features are spelled, not what the model is being optimised toward.

That also re-reads `base_dense`, the one member that DID pay. Its seat was originally read as *a different
encoder family brings different errors*. `v3base_dense` holds its recipe and swaps only the family and gets
a tie — so the seat `base_dense` won was probably bought by its **768 width against a pool of 384-wide
members**, i.e. capacity, not diversity of any kind. The one data point that looked like architecture
diversity paying was not that.

## The empty cell

Sort every member by (solo strength, distance from the seated pool):

```
                      near the pool          distant from the pool
  strong        every DINOv2 member          ← EMPTY, always has been
                (seats saturate at 7)
  weak          not worth building           vjepa_warm    0.8337, seat +0.0002
                                             mae_matched   0.6735, seat -0.0041
```

`knee-053z` pre-registered the requirement as DISTANT-AND-STRONG: nearest seated neighbour under 0.88 and
solo over 0.85. `vjepa_warm` cleared the distance clause by a mile — nearest seated neighbour 0.8210
against a seated spread of 0.8801–0.8911, the first genuinely different prediction family this project has
produced, by 0.059 — and missed the solo clause by 0.017. It bought +0.0002.

`mae_matched` shows the other failure mode, and the diversity ruler caught it cleanly: nearest neighbour
0.4333. That is not a distant member, it is a broken representation whose predictions are nearly
uncorrelated with everything. Uncorrelated-because-broken and uncorrelated-because-complementary look the
same to a correlation and completely different to a scoreboard.

**Distance is necessary and demonstrably not sufficient.** What failed in both cases was not the distance
ruler — it was the inference from distance to seat.

## The decomposition that says what to do instead

Ambiguity decomposition, the standard result for why ensembles work:

```
ensemble error  =  mean member error  −  ambiguity
```

Every arm above moves exactly one of those two terms and lets the other go where it likes.

- `vjepa_warm` raised ambiguity a great deal and raised mean member error at least as much. Net +0.0002.
- `v3base_dense` / `cnx_dense` left ambiguity essentially flat, because of the shared-pipeline argument
  above. Net −0.0007 and −0.0042.

The quadrant is empty because **nothing here has ever optimised both terms at once.** Diversity has always
been a side effect we hoped for — pick a different trunk, then measure afterwards whether the errors
happened to decorrelate — rather than something the member's own loss was asked to produce.

## What this licenses, and what it does not

It does **not** license another trunk. That is three measurements deep and the axis is closed; a fourth
family would be re-running a settled experiment.

It licenses changing the **objective** rather than the encoder, because the objective is where the shared
error structure actually lives. Two constructions follow, and both are cheap because the plumbing exists:

1. **Negative-correlation training** (`knee-2f0q`). Put the correlation penalty in the member's own loss so
   it is trained to be accurate AND to disagree with the pool, jointly instead of in sequence. This is the
   direct implementation of "move both terms". `DistilledObjective` already loads the pool's per-study
   predictions and adds a term pulling the student TOWARD the ensemble's ordering; the decorrelation term is
   the same loader with the opposite sign.

2. **AUC-margin training** (`knee-bdhb`). A member optimised on the metric itself rather than on BCE is
   strong by construction and differently-wrong by construction, because BCE and an AUC margin disagree
   about which studies matter — BCE spends capacity on calibration that a rank-based metric cannot see.

Both produce a member that is strong *and* distant for the same reason, which is the cell nothing has
occupied. That is the argument for spending card on them ahead of a fourth encoder.

## The falsification

If a decorrelation-trained member lands distant-and-weak again — vjepa's quadrant — then ambiguity cannot
be bought at this pool's operating point by any means, and the seat axis closes for good rather than just
for trunks. That is a genuine possibility and it is the outcome that would retire the whole line, so it is
worth stating before the run rather than after.

One live measurement bears on this and has not read yet: `kneemri-farseat` seats `vjepa_warm` — the most
distant member we have — in the champion's seventh seat. If the board pays for distance that the offline
ruler priced at +0.0002, the ruler is under-pricing ambiguity and the case above gets stronger. If it does
not, the +0.0002 was right and distance is being correctly valued at nearly nothing.
