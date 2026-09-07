# The distillation gain, and how much of it was fold 0's labels

2026-09-07. What `kneemri.evaluation.teacher_witness` measured, what it does not say, and the two arms
that finish the reading. Every distilled fold-0 number this project owns was produced under teachers
fitted on fold 0, so this is a correction to a family of readings rather than to one run.

## The defect

`oof_fold_k` (k=1..4) was fitted at `--fold k`, which holds out fold k and trains on every other fold —
**fold 0 included**. A student distilled from their targets and then SCORED on fold 0 is mimicking four
functions that each memorised the labels it is being graded against. `oof_teacher_s2026` read +0.0107
over its control, `seat_distill_lr4e5` read +0.0107, and both numbers were quoted as a treatment effect
for days.

The board is what caught it first, without knowing why: `oofsolo_v1` scored **0.926** from an offline
read ABOVE `holdsolo_v1`'s, which scored **0.928**. An offline ordering that inverts on the board is a
ruler failure, not a member failure.

## The instrument

A plain correlation cannot see this. A leaky teacher's fold-0 output is close to the truth, so a good
student agrees with it for the innocent reason that both are good. Correlating **residuals** `p - y`
per label removes exactly that: what is left is the part of each reading the label does not explain, and
two functions whose MISTAKES line up on studies neither was scored on are related by something other
than the label.

Two controls, on two axes, because either alone is ambiguous:

- **`band_dense`** — same architecture, same fold, no teacher in its loss. It fixes the scale that
  "related" is read against.
- **the four CLEAN teachers** (`--exclude-fold 0`) — they never saw fold 0, so a student that tracks the
  leaky set's residuals and not theirs is tracking memorised labels rather than the family's shared
  inductive bias.

It cost no training run: both students were already on disk with their fold-0 records, and only the
eight teachers' fold-0 outputs were missing — 909 studies × 8 checkpoints, minutes, taken in the gap
between the last clean teacher and the student that reads them.

## The reading

    seat_distill_lr4e5   macro AUC 0.8633       band_dense   macro AUC 0.8578

    seat_distill_lr4e5  vs leaky   mimicry +0.9096      vs clean   mimicry +0.9004
    band_dense          vs leaky   mimicry +0.8844      vs clean   mimicry +0.8879

    student   leaky - clean   = +0.0092
    control   leaky - clean   = -0.0035
    ------------------------------------
    the leak path                +0.0127

The difference-in-differences is the only form that survives both controls, and its sign is not
ambiguous: the student prefers the teachers that saw its holdout, and the undistilled control on the
same fold prefers the other set.

**What it does NOT say.** That the whole +0.0107 is leak. The student's mimicry of the CLEAN set is
already +0.0125 above the control's, so most of what distillation transferred is the family's shared
function rather than fold 0's labels. On this reading the leak is roughly a third of the transfer.

**The caveat that is not hidden.** The clean teachers are weaker (below), and a weaker teacher set has
noisier residuals, which shrinks a correlation on its own. The control bounds that — `band_dense` reads
both sets and produced −0.0035, so noise-from-weakness alone does not manufacture +0.0092.

## The starvation confound, and a correction to my own claim

Removing fold 0 from a teacher's training set removes ~900 of ~3,500 studies, so the clean teachers are
weaker BY CONSTRUCTION and a clean student's miss is ambiguous between "the leak was the gain" and "the
teachers got worse". Priced directly, each teacher on its own holdout:

    fold 1   clean 0.8572   leaky 0.8578   -0.0006
    fold 2   clean 0.8524   leaky 0.8567   -0.0043
    fold 3   clean 0.8607   leaky 0.8740   -0.0133
    fold 4   clean 0.8611   leaky 0.8695   -0.0084
    -----------------------------------------------
    mean                                   -0.0067

An earlier note recorded this confound as reading ~zero. That was **fold 1 alone**, and fold 1 is the
cheapest of the four by an order of magnitude. The confound is −0.0067, not ~0, and the middle band of
`seat_clean_distill`'s pre-registration does not collapse the way that note claimed: a clean student at
0.861 is consistent with both stories, and the witness above — not that number — is what separates them.

The general shape of the mistake is worth more than the number: **one fold of four, read as the price.**
A rate quoted from the subset that happened to be finished is the same defect as a rate quoted from the
subset an English regex could parse.

## What the two arms finish

The leak can be removed two ways, and they fail differently, which is why both run:

- **`seat_clean_distill`** — weaken the TEACHERS (`--exclude-fold 0`), keep the held-out fold. Pays the
  −0.0067 handicap, and reads the term on a ruler we own. Running; ~1.2 h.
- **`seat_oof5_distill`** — remove the HOLDOUT (`--fold -1`), keep the teachers at full strength. Five
  blocks, each predicting the fold it held out, cover all 4,406 studies out-of-fold, so there is no
  evaluation set for a memorised label to leak into. Its ruler is the BOARD, by construction. Armed
  behind the first.

A miss on the starved arm cannot refute the full-strength one. That is the whole reason the four-row
table above was worth taking before either number arrived.

## The first one landed: the leak transferred ~zero AUC

03:20Z, twelve epochs, `convergence_check` clean, peak e9.

    seat_distill_lr4e5   leaky teachers, saw fold 0         0.8633
    seat_clean_distill   clean teachers, --exclude-fold 0   0.8634
    band_dense           no teacher, same fold              0.8578

**−0.0001 to remove the leak**, paid out of teachers that are −0.0067 weaker by construction. The
witness and the score are both right and they are not in conflict: the student demonstrably preferred
the teachers that had seen its holdout (+0.0127 diff-in-diff of residual mimicry), and that preference
bought it nothing. The mimicry rode along; the +0.0107 is the treatment.

The general shape, and it is the second time this document records one: **a leak visible in the
residuals is not automatically a leak in the score.** The witness answers "is it there", never "is it
the number" — only a re-run without it answers that, which is why the arm was worth 1.2 h of card
against a witness that had already come back positive.

What stays open: `oof_teacher`'s 0.8681 and the `oofsolo` board inversion (0.926 from an offline read
above `holdsolo`'s 0.928) are a different corner and are NOT explained by this. `seat_oof5_distill`
still runs — full teacher strength, `--fold -1`, board as ruler — and it is the only reading that can
price the term where it will actually be spent.

## The FOREIGN teacher is the same leak, by the same mechanism

Added 02:55Z, from mastermind, and it is a correction to a comparison rather than to a member. The
`foreign_distill` family distils from a teacher built out of **their** five released DINOv3 fold models
(`experiments/README.md:8408`, `teacher_foreign_corpus.npz`, all 4,406 studies). Their fold-k model was
fitted on their other four folds — which hold ~80% of OUR fold 0, with labels. So a student scored on
our fold 0 is again mimicking functions that encode the labels it is graded against. Different corpus
split, identical mechanism.

**What that costs.** `foreign_distill_mix07` (0.8657) and `oof_teacher_s2026` (0.8681) sit 0.0025 apart,
and that near-tie was read as "the teacher's SOURCE does not matter — a foreign family and our own buy
the same thing". Both halves of that comparison are leaky, so the reading is not a null about source; it
is two measurements of possibly the same leak. **Source is unpriced, not settled.** No arm may be
planned on the premise that a foreign teacher is interchangeable with ours.

The only clean full-strength reading of the term remains `seat_oof5_distill` on the board, and the only
clean held reading remains `seat_clean_distill`.

## What was written off on the strength of this

Five distilled members (`seat_distill_lr4e5` 0.8633, `foreign_distill_mix07` 0.8657,
`foreign_distill_calib` 0.8641, `foreign_distill` 0.8595, `foreign_dissent` 0.8538) are declined in
`comp/members.declines` under ONE argument rather than five, and the decline says in writing that it is
lifted by the two arms above and never by re-quoting the number in its own line.

Bead: knee-zwcyk (witnessed). Arms: `experiments/arms/teacher_clean.sh`, `seat_clean_distill.sh`,
`seat_oof5_distill.sh`. Instrument: `kneemri/evaluation/teacher_witness.py`.
