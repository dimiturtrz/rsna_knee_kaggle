# Legacy arm commands, recovered from `/tmp` on 2026-08-11

Every number in the ledger predating `experiments/` was produced by a script in `/tmp`. Those
scripts are not code we run again — the arms that mattered were rewritten under `experiments/arms/`
— but they are the only record of WHICH COMMAND produced a reported number, and a number whose
command is unrecoverable cannot be rerun, matched against, or refuted. `$D` is the processed data
root and `$L` the label CSV, as `experiments/lib.sh` now resolves them through `core.paths`.

A record, not a home: do not add to this file, add an arm. Transcribed AS RUN, so a flag here may no
longer exist — `package_members --pattern` below was replaced on 2026-08-16 by `--roster
comp/members.roster`, after two local member directories stamped one live dataset id with different
members. Copy a line from here to READ it, not to run it.

## armU.sh
```bash
uv run python -u -m kneemri.training.train --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --variant base --backbone-lr 4e-5 --slice-pooling max --labels "$L" --out "$D/compose_base_s2026.pt" 2>&1 \
  | grep --line-buffered -E "epoch |best |rror|emory"
```

## armV.sh
```bash
uv run python -u -m kneemri.training.train --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --backbone-lr 4e-5 --slice-pooling max --labels "$L" --out "$D/compose_small_s2026.pt" 2>&1 \
  | grep --line-buffered -E "epoch |best |rror|emory"
```

## arms.sh
```bash
uv run python -u -m kneemri.training.train --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --backbone-lr 4e-5 --labels "$L" --out "$D/lr5x_d6_s2026.pt" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"
```

## chain.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep -E "epoch |best |rror"; }
```

## chain10.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain11.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain12.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain13.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain14.sh
```bash
uv run python -u -m kneemri.evaluation.ruler_check --labels "$L" --confident "$C" --checkpoints $CK 2>&1 | tail -22
```

## chain15.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain16.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain17.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain18.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain19.sh
```bash
uv run python -u -m kneemri.evaluation.attribution --checkpoint "$C" --labels "$L" --studies 400 2>&1 | tail -30
```

## chain2.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror"; }
```

## chain20.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain21.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror|emory"; }
```

## chain3.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror"; }
```

## chain4.sh
```bash
  uv run python -u -m kneemri.training.train --fold $F --epochs 20 --one-cycle --weight-floor --variant base --batch 12 \
    --labels "$L" --out "$D/v3_fold${F}.pt" 2>&1 | grep --line-buffered -E "best |rror"
uv run python -m kneemri.kaggle.package_members --out comp/members --pattern "v3_fold*.pt" 2>&1 | tail -3
```

## chain5.sh
```bash
uv run python -u -m kneemri.training.train --fold -1 --epochs 20 --one-cycle --weight-floor --variant base --batch 12 \
  --labels "$L" --out "$D/full_base20.pt" 2>&1 | grep --line-buffered -E "epoch |best |rror|studies:"
uv run python -m kneemri.kaggle.package_members --out comp/members --pattern "full_base20.pt" 2>&1 | tail -3
```

## chain6.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror"; }
```

## chain7.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror"; }
```

## chain8.sh
```bash
run() { uv run python -u -m kneemri.training.train "$@" 2>&1 | grep --line-buffered -E "epoch |best |rror"; }
```

## chain9.sh
```bash
uv run python -u -m kneemri.training.train --fold 0 --epochs 24 --one-cycle --weight-floor \
  --variant large --batch 12 --labels "$L" --out "$D/large24_s2026.pt" 2>&1 \
  | grep --line-buffered -E "epoch |best |rror|emory"
```

## noisefloor.sh
```bash
  uv run python -m kneemri.training.train --fold 0 --epochs 8 --one-cycle --weight-floor --seed $S \
    --labels "$D/labels_pilkwang.csv" --out "$D/nf${S}_fold0.pt" 2>&1 | grep -E "best |rror"
uv run python -m kneemri.evaluation.member_curve --overlap --labels "$D/labels_pilkwang.csv" \
  --checkpoints "$D/nf2026_fold0.pt" "$D/nf7_fold0.pt" "$D/nf13_fold0.pt" 2>&1 | grep -E "alone|member\(s\)|rror"
```

## pslice.sh
```bash
uv run python -m kneemri.training.train --fold 0 --epochs 8 --one-cycle --weight-floor --slice-pooling max \
  --labels "$D/labels_pilkwang.csv" --out "$D/permax_fold0.pt" 2>&1 | grep -E "epoch |best |rror"
```

## q2.sh
```bash
  uv run python -m kneemri.training.train --fold 0 --epochs 8 --one-cycle --weight-floor --slice-pooling $MODE \
    --labels "$L" --out "$D/pool_${MODE}_fold0.pt" 2>&1 | grep -E "epoch |best |rror"
```

## q3.sh
```bash
uv run python -m kneemri.training.train --fold 0 --epochs 8 --one-cycle --weight-floor --anatomical-prior \
  --labels "$L" --out "$D/prior_fold0.pt" 2>&1 | grep -E "epoch |best |rror"
```

## q4.sh
```bash
  uv run python -m kneemri.training.train --fold 0 --epochs 8 --one-cycle --weight-floor --seed $S \
    --labels "$L" --out "$D/nf${S}_fold0.pt" 2>&1 | grep -E "best |rror"
uv run python -m kneemri.evaluation.member_curve --overlap --labels "$L" \
  --checkpoints "$D/pool_channels_fold0.pt" "$D/nf7_fold0.pt" "$D/nf13_fold0.pt" 2>&1 | grep -E "alone|member\(s\)|rror"
```

## q5.sh
```bash
uv run python -m kneemri.training.train --fold 0 --epochs 8 --one-cycle --weight-floor --anatomical-prior \
  --labels "$L" --out "$D/prior2_fold0.pt" 2>&1 | grep -E "best |rror"
```

## q6.sh
```bash
uv run python -m kneemri.training.train --fold 0 --epochs 8 --one-cycle --weight-floor --variant base --batch 6 \
  --labels "$L" --out "$D/base_fold0.pt" 2>&1 | grep -E "epoch |best |rror|Error"
```

## q7.sh
```bash
uv run python -m kneemri.training.train --fold 0 --epochs 24 --one-cycle --weight-floor \
  --labels "$L" --out "$D/ep24_fold0.pt" 2>&1 | grep -E "epoch |best |rror"
```

## q8.sh
```bash
  uv run python -m kneemri.training.train --fold 0 --epochs 8 --one-cycle --weight-floor --seed $S \
    --labels "$C" --eval-labels "$L" --out "$D/conf${S}_fold0.pt" 2>&1 | grep -E "best |rror"
```

