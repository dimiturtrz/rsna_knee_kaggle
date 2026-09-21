#!/usr/bin/env bash
# bead: knee-nchjt
# reading: transduce_dmlb_c4*.pt
#
# WHAT THIS ARM IS
#   Transduction, finally runnable: the 909 holdout studies are put into TRAINING under their TEACHER's
#   own predictions and scored against their TRUE readings. `--pseudo-fold 0` is the split half and
#   `pseudo_dmlb_fold0.csv` (kneemri.training.pseudo_table) is the label half. Two seed-children off
#   `dml_b_s2026`, nothing averaged, one checkpoint each.
#
# WHAT IT MEASURES, AND WHAT IT CANNOT
#   knee-tta1 measured naive self-training with a LINEAR head under zero shift: 40 cells, none clearing
#   +0.0005, the delta shrinking as the pseudo tail grew -- circularity, since pseudo-labels read off a
#   ridge lie along that ridge's own decision direction. Two suspects were named. Suspect (1), that the
#   head was linear, is what this arm simulates: a fine-tuned MEMBER is nonlinear everywhere. Suspect
#   (2), that shift is zero, CANNOT be simulated here -- our holdout is drawn from the same corpus, so
#   the level reads ~0 whatever the truth is on the real test set. This closes (1) or nothing.
#
# THE CONTROL IS ALREADY ON DISK, which is what makes this cheap enough to be worth running.
#   The six `basinsoup_dmlb_c40*` children are this recipe with `--pseudo-fold` absent and nothing else
#   changed: same parent, same --init-scope all, same 4 epochs, same 2e-5, same ema, same batch. They
#   read 0.9061 / 0.9062 / 0.9062 / 0.9064 / 0.9065 / 0.9065 ledger75 -- mean 0.9063, sd 0.00017. A
#   4-epoch continuation off this parent therefore COSTS -0.0015 against the parent's 0.9078, and
#   comparing a pseudo child to 0.9078 would read that cost as transduction's. The comparison is
#   against 0.9063, and the control's own spread is the floor.
#
# PRE-REGISTERED, against the control mean 0.9063 on ledger75, fold 0, 909 studies
#   >= +0.0010   a nonlinear head escapes the circularity the ridge could not. Suspect (1) is retired
#                as the reason knee-tta1 read null, and the real-test version -- where suspect (2)'s
#                shift is nonzero -- becomes a slot with a mechanism behind it.
#   -0.0010 to +0.0010   knee-tta1's null reproduces with a nonlinear head. Suspect (1) is retired the
#                other way: the head was never what killed it, the circularity is the method's. The
#                last open axis closes on its own instrument.
#   <= -0.0010   pseudo-labels actively harm a member fit. Same closure, harder.
#   Six control seeds at sd 0.00017 make +/-0.0010 a ~6 sd band. This is the rare arm whose control is
#   tight enough that two seeds resolve it.
#
# ZERO SLOTS. Fold 0, priced on ledger75, never on the board. Decisive in every direction without one.
#
# deliberate: every flag is `basinsoup_dmlb.sh:72-78` byte for byte EXCEPT --pseudo-fold, --labels,
#             --eval-labels and the seed. That is what makes those six children a matched control
#             rather than a similar one.
# deliberate: --eval-labels "$LABELS". Without it the run would be SCORED against the pseudo table too,
#             and would be measuring agreement with its own teacher -- the one way this arm produces a
#             number that means nothing. Precedent: blendema224.sh:88.
# deliberate: --seed 407 408, fresh, so these are not any earlier child re-run under a new name.
# deliberate: the pseudo table is BUILT here rather than required, so the arm is reproducible from the
#             teacher's record alone and cannot silently read a stale table someone else wrote.
# deliberate: --confidence 0.5 on the pseudo cells. `__conf` weights the loss, and the base table's
#             value is confidence in a REPORT READING the teacher never performed; carrying it over
#             would weight a model's guess by the certainty of a sentence. 0.5 is the midpoint of the
#             column's range -- a declared neutral, not a tuned value, and not a sweep.
#
# CARD. 2 children x 4 epochs x ~350 s = ~50 min, plus ~5 min to price. Nothing else of ours may share
#   the card.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

PARENT="$MODELS/dml_b_s2026.pt"
require "$PARENT"

TEACHER="$MODELS/dml_b_s2026.holdout.npz"
require "$TEACHER"

ARMPY="uv run --no-sync --project $REPO python"
PSEUDO="$PROCESSED/labels/pseudo_dmlb_fold0.csv"

$ARMPY -m kneemri.training.pseudo_table \
  --base "$LABELS" --record "$TEACHER" --confidence 0.5 --out "$PSEUDO"

for SEED in 407 408; do
  run_train "transduce_dmlb_c$SEED" \
    $RECIPE --epochs 4 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 2e-5 --slices 24 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
    --calibrated-targets --hold-annotations --ema \
    --batch 4 --accumulate 3 --slot-dropout 0.40 \
    --init "$PARENT" --init-scope all --seed "$SEED" --pseudo-fold 0 \
    --labels "$PSEUDO" --eval-labels "$LABELS" --out "$MODELS/transduce_dmlb_c$SEED.pt"
done

$ARMPY -m kneemri.evaluation.ledger75 --holdouts "$MODELS" \
  --only transduce_dmlb_c407 transduce_dmlb_c408 \
         basinsoup_dmlb_c401 basinsoup_dmlb_c402 basinsoup_dmlb_c403 \
         basinsoup_dmlb_c404 basinsoup_dmlb_c405 basinsoup_dmlb_c406 dml_b_s2026
