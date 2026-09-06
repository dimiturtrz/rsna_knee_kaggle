# The foreign teacher buys solo quality and spends diversity — and the trade is monotone in the weight

2026-09-06. Subject: distilling the out-of-sample foreign teacher, which reads 0.8929 fold-matched, into
our own band_dense student (0.8578), read at two teacher weights. Instruments: `price_served_base8.sh` (seat on the served base of 8) and
`kneemri.kaggle.seat_value` / `member_axes` (the three-column cell). Logs: `logs/seat_mix07.log`,
`logs/axes_mix07.log`. Card: 73 min each arm; zero slots.

## What the flag does, because the arm that used it had it wrong

`DistilledObjective.__call__` (`kneemri/training/objective.py:411`) is

    super().__call__(focused) + self.distill_weight * self.imitation(focused)

— **additive**. The label term is at full strength at every setting, and `--distill-weight` scales only the
teacher's term on top of it. There is no mix, and no setting of this flag turns the labels off. So the two
arms are not "labels 0.7" and "labels 0.3": they are one label objective with the teacher at 0.3 and at 0.7.
`foreign_distill_mix07`'s own docstring reasoning ("at 0.3 the label term holds 0.7 of the loss, so the
ordering that survives is ours") is wrong about the mechanism, and is corrected here and on knee-bq62a.

## The two points

| | teacher 0.3 (`foreign_distill`) | teacher 0.7 (`foreign_distill_mix07`) | direction |
|---|---|---|---|
| solo holdout mean AUC | 0.8595 | **0.8657** | up (+0.0062) |
| js divergence from the roster | 0.0190 | **0.0138** | **down**, and below the 0.0150 boundary |
| member_axes cell | SCALED | **TWIN** | toward the pool |
| seat on the served base of 8, 1x | — | **−0.0005** [−0.0010, +0.0001] | negative |
| the same seat at 6x | — | −0.0040 [−0.0060, −0.0022] | negative, monotone |

Control for the solo column: `band_dense` = 0.8578, the same recipe with no teacher at all. The seat curve
falls monotonically from 1x to 10x (−0.0005 to −0.0064) and split-half picks 1x on both halves, reading
−0.0005 and −0.0006 out of sample. There is no weight at which this member is worth a seat.

## The mechanism, stated so a later use is re-tested rather than assumed dead

The teacher is a strong foreign model, and its ORDERING of our studies is nearer our own pool's consensus
than our student's label-fit is. So each increment of teacher weight pulls the student **toward** the pool,
not away from it — and every column moves together: solo up (the teacher is good), js down, rank up, maha
down (the student is becoming the pool's average opinion), seat down (an average opinion adds nothing to
the average).

That is the trade written plainly: **distillation from a teacher inside your own pool's opinion buys solo
quality and spends diversity.** For a solo submission that is a gain. For a seat on an eight-member vote it
is the one thing a seat cannot use, which is why a member that is better alone is worse in the ensemble
here — the same shape the ledger already records for whole-stack members and overlap TTA.

## What this closes, and what it does not

Against the pre-registration (seat ≥ +0.003 = distillation is a member factory; < 0.0011 = the supervision
axis closes on a readable ruler): **0.7 reads −0.0005 and closes.** `foreign_distill_mix30` (weight 3.0,
ten times the original, queued behind dissent) is the endpoint instrument that says whether the line bends
back at a strength where the teacher overwhelms the label term rather than re-weighting it. Its own
pre-registration predicts a null on this evidence, and it is being run as an instrument, not as a candidate.

Not closed: distillation from a teacher OUTSIDE the pool's opinion (the mechanism above is about this
teacher's relation to this pool, not about distillation); and the solo-quality gain itself, which is real
and is the right lever for anything scored alone rather than seated.
