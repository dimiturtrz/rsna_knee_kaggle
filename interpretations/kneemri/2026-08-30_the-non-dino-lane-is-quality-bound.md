# The non-DINO lane is the last branch open, and it is bound by QUALITY, not by distance

**2026-08-30 · knee-m69u · knee-8009 · knee-7ybf · reading three results already on disk, no new card time**

## The lane is the only one left, and that is a result rather than a preference

`stack_increments` closed the other branch. It banked two increments that each reproduce against their own
control — the label-table swap (+0.0034) and distillation (+0.0035) — predicted **+0.0069** additive against
`calib_dense_hold_s2026` 0.8634, and delivered **+0.0008**, inside its own pre-registered null band. They do
not sum because they act on the same thing: what the model is asked to ORDER. They compete for one pool of
headroom. The six-arm roster retrain that additivity would have licensed is not armed, and the two remaining
increments are not re-armed.

What that leaves is one sentence in the ledger: *it leaves the non-DINOv2 trunk lane standing as the only
remaining branch of `knee-m69u`.* Which is also where the owner has been pointing for days.

## The lane's problem is the opposite of the one it has been treated as having

`swin_supervised` is the cleanest read of it, because it was measured on both axes.

| | value | the seated pool |
|---|---|---|
| solo holdout | 0.8331 | ~0.8634 at the top |
| seat value against `greedyseat_v1` | **−0.0012**, band [−0.0020, −0.0003] | — |
| nearest-neighbour correlation | **0.8346** | 0.8772 – 0.9225 |
| mean correlation | 0.8132 | — |

It is **the most decorrelated member this project has produced**, by a clear margin, and it was still refused
a seat — best possible swap −0.0013. So the non-DINO trunk already supplies the diversity the seat ruler pays
for. It is refused purely on solo strength.

**And the board has now said the same thing, with a different member.** `vjepaseat_v1` (ref 55895006,
2026-08-30) is `raptorweight_v1`'s exact seating with one seat ADDED — `vjepa_warm_s2026`, a V-JEPA2 ViT-L
at 0.8119 nearest-neighbour agreement, the most distant member the project owns. Vote, foreign weight, kit
and every other seat held. It read **0.943 against the control's 0.944**: −0.001, inside the entry's own
0.0015 board resolution. `run_check` confirms 7 of 7 members mounted, so this is a reading about the member
and not a silent drop of the `farseat_v1` kind.

| ruler | member | nearest-neighbour | verdict |
|---|---|---|---|
| holdout seat value | `swin_supervised` | 0.8346 | −0.0012 |
| board, matched control | `vjepa_warm_s2026` | 0.8119 | −0.001 |

Two rulers, two different members, one sign, at the same resolution. The claim below is no longer an
offline one.

That inverts the standing reading of this lane. Every argument for it so far has been a diversity argument —
"even if a single model doesn't outperform, an ensemble with it may" — and that argument is now *confirmed and
spent*: the distance is there, it has been bought, and it is not enough on its own. The deficit left to attack
is score.

## Which makes the target a size, not a direction

Seated members sit near 0.8634 solo at 0.877+ correlation. A member holding swin's ~0.83 correlation needs
roughly **+0.017 to +0.020 of solo score over 0.8331** to buy a seat instead of costing one.

Pre-registering that number now is what makes this lane cheap to run: the required gain and the house
significance bar coincide. A result big enough to seat is automatically big enough to report; a result too
small to report is automatically not worth a slot. There is no ambiguous middle to argue about afterwards.

No knob this project has ever turned moves a trunk that far. That is the honest constraint on what to run next,
and it disqualifies most of the obvious list.

## The confound nobody has separated

| trunk | best solo | pretraining it arrived with |
|---|---|---|
| DINOv2-small | 0.8964 | large-scale self-supervised |
| vjepa2l | 0.8343 | self-supervised video |
| swinbase | 0.8331 | IN22k supervised |
| cnxbase | 0.8186 | ImageNet supervised |

The gap has been read as **architecture**. It has never been separated from **pretraining**. Every non-DINO
trunk in this project arrives with ImageNet or IN22k weights and no medical stage; the DINOv2 trunks arrive
with the largest self-supervised pretraining any of them has seen. Two explanations, one measurement, and the
project has been quoting the first.

This matters because only one of the two explanations has a lever attached. If it is architecture, the lane is
worth 0.83 and closes. If it is pretraining, the missing stage is buildable, needs **no labels** — which is
also the external-data route `champion.json` names as one of only two paths above 0.943 — and is the only
mechanism on the table sized to the +0.017–0.020 the seat costs.

## But first: the lane is nine runs, and all nine wear the DINOv2 recipe

Before spending anything on architecture or pretraining, count what the claim rests on. Every run this
project has ever done on a trunk that is not DINOv2, smokes excluded:

| run | trunk | best | epochs | still climbing | backbone_lr | one_cycle | layer_decay |
|---|---|---|---|---|---|---|---|
| `cnx_own_recipe` | cnxbase | 0.8186 | 12 | no | 4e-4 | True | 1.0 |
| `cnx_dense` | cnxbase | 0.7927 | 12 | **yes** | 4e-5 | True | — |
| `cnx_mid_rate` | cnxbase | 0.7177 | 1 | yes | 1.2e-4 | True | 1.0 |
| `swin_supervised` | swinbase | 0.8331 | 12 | no | 1e-4 | True | 1.0 |
| `swin_wholetop` | swinbase | 0.8274 | 12 | no | 1e-4 | True | 1.0 |
| `swin_calib_dense` | swinbase | 0.8224 | 10 | no | 4e-5 | True | 1.0 |
| `vit_in21k` | vitin21k | 0.7847 | **5 of 12** | **yes** | 4e-5 | True | 1.0 |
| `vjepa_member` | vjepa2l | 0.8343 | 10 | **yes** | 4e-5 | True | 1.0 |
| `vjepa_warm` | vjepa2l | 0.8337 | 6 | no | 4e-5 | True | 1.0 |

Nine runs, against well over a hundred on DINOv2. Three of the nine never converged. And the recipe is a
**constant** down the whole lane: `one_cycle` True everywhere, `layer_decay` 1.0 everywhere, `unfreeze_last`
6, `head_lr` 1e-3, `patience` 0, `ema` False — every one of them inherited from the recipe this project tuned
on DINOv2-small over a hundred runs.

Two consequences, and the second is the one that changes what to run.

**Three quoted numbers are floors, not readings.** `vjepa2l`'s 0.8343 is a run still climbing on its last
recorded epoch; `vit_in21k`'s 0.7847 is five epochs of a twelve-epoch recipe. Both are quoted in this project
as what those trunks are worth. They are what those trunks had reached when the run stopped. The trunk table
in the previous section inherits that defect and should be read with it.

**The schedule axis has never been tried at all on this lane, and rate is only the visible half of it.** A
higher peak rate is what the two rate pairs point at, but `one_cycle` at a twelve-epoch budget peaks near
epoch 2 and anneals from there, so a trunk that needs more steps at a higher rate gets neither: it is walked
up past its usable rate early and annealed away from it before it converges. That is the exact shape
`cnx_dense` shows — monotonically climbing through epoch 11 and never turning over. And `layer_decay` 1.0 is
the same omission from the other side: layer-wise decay is the standard way a convolutional or hierarchical
trunk takes a higher trunk rate without destroying its early layers, the flag exists and is used on DINOv2
(`band_dense_llrd`), and it has never once been set on a non-DINO trunk.

So there are now **four** candidate explanations for the non-DINO deficit — architecture, pretraining, rate,
and schedule — and the project has been quoting the first while only ever testing under one setting of the
other three.

## The shortlist, ranked by plausible gain over wall-clock

This is the "look away from the DINOs" list, and the ranking is the point: the naive reading of that request
is item 4, and item 4 is the wrong spend.

**1. `cnx_warm_anneal` — queued, ~20 min, zero new code.** Six epochs from `cnx_own_recipe`'s peak encoder at a
quarter of the parent's rate. Pre-registered at ≥0.8286 for "the trunk had more". It is the cheapest possible
probe of whether a non-DINO trunk's number is its architecture's worth or its recipe's, and it is already on
the card behind `recal_dense`. It is also the matched control `knee-7ybf` needs, so nothing in this lane should
be armed before it reports.

**2. Give the lane a recipe of its own (`knee-23wa`) — the cheapest of the four explanations to eliminate, and untested.**
One arm on `swinbase`, the trunk that already carries the diversity, moving the three flags the lane has never
moved: a peak rate the trunk's own runs prefer (1e-4, not DINOv2's 4e-5), `layer_decay` below 1.0 so the
higher rate does not destroy early layers, and a schedule that is not `one_cycle` at a 12-epoch budget. Cost
is real — ~16 min an epoch, so ~3.2 h — and it must be priced as an INSTRUMENT rather than as a seat play:
+0.011 is the visible size of the rate half alone, under the +0.017–0.020 a seat needs. What it buys is the
licence to re-read the whole lane. Nine runs and three quoted floors currently rest on the assumption that a
DINOv2 recipe transfers; if it does not, every non-DINO number in the ledger is re-opened, including the two
trunks that have only ever seen 4e-5. Single-variable discipline still applies — moving three flags at once
answers "does a fitted recipe help" and NOT "which flag did it", and the follow-up split is the price of
starting with the cheap joint arm.

**3. Feature distillation as a pretraining stage — `knee-8009`, the only item sized to the target.** Match a
non-DINO trunk's embeddings to a DINOv2 trunk's on knee MRI, then fine-tune. It is the one lever that attacks
the confound directly rather than adding another point to it. An instrument version exists with **zero new
code** — high `--distill-weight`, then `fitted_encoder`, then a warm-started labelled fit — and decides whether
the real build is worth it. Falsification stated in advance in the bead: a gain that arrives while the ~0.83
correlation is LOST is not a win, because the lane is only worth anything while both hold.

**4. Warm restarts of `swinbase` and `vjepa2l` from their own peak checkpoints — ~20 min each, gated.** Same
mechanism as item 1 on the two trunks that scored above ConvNeXt. Explicitly **not** armed until item 1 clears
+0.010; `cnx_warm_anneal`'s own pre-registration is what licenses them, and running them first would spend an
hour to learn what twenty minutes is about to say.

**5. A timm trunk — EfficientNet-B0, ResNet34 — and why it is last, not first.** This is what "a non-DINO
shortlist" naively means, and it is the wrong next spend. Cost first: the trainer's variant registry
(`kneemri/models/backbone.py`) is HF/AutoModel-based, and timm appears only on the *inference* side
(`raptor_model.py`, `foreign_member.py`), so a timm-only trunk needs a new family there rather than a flag. But
the real objection is not cost. Such a trunk arrives ImageNet-only, so it walks straight back into the
pretraining confound — it adds a fifth row to the table above and cannot separate its two columns. A
competitor's ablation puts EfficientNet-B0 at 0.7815 OOF, below every trunk we already own. It tests
architecture, which is the question we cannot currently answer, using the one design that guarantees the answer
stays ambiguous. Hold it until item 2 reports.

**Retired, with the reason, so they are re-opened by argument and not by re-quoting a number.** RAD-DINO —
chest-radiograph pretraining, tried cold at 0.8141 against a 0.8512 control, **−0.0371** (knee-7l4e); note this
is the one item that *did* test the pretraining axis, and it failed on domain, not on the idea. RadImageNet —
Keras `.h5` with no PyTorch port, 2–4 h of conversion whose failure mode is a silent weight mismatch. CoAtNet —
it IS `raptor`, foreign `coatnet_rmlp_2_rw_384`, board 0.923, matched the same day by our own `calibsolo` 0.923.

## What this does not claim

Nothing here has been measured tonight. Every number above was already on disk; what is new is the crossing of
them, and one pre-registered target. The lane could still close: if item 1 lands under 0.8226 and item 2's
instrument moves nothing, then 0.83 is what these architectures are worth on this corpus and the diversity we
bought with `swin_supervised` has nowhere to be spent. That is a real outcome and it should be written up as
one, not quietly left open.
