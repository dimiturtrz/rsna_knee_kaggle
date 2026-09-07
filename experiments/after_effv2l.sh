#!/usr/bin/env bash
# Wait for cell C to release the card, then fine-tune EfficientNetV2-L on HIS recipe, fold 0 held out.
#
# WHY THIS MEMBER. Every seat we own is a ViT or a hybrid reading dense windows; the foreign block's
# spearman against our core is 0.776 for the EfficientNet family and 0.848 for CoAtNet, so this is the
# least DINO-like error structure available on disk. The only measured lever in this project's top tier
# is family diversity (raptorseat +0.002, raptorweight +0.001, both board-confirmed), and every member
# of that block so far is IN-SAMPLE on our studies. This is the first EfficientNet fitted with the same
# 896 studies held out that everything else is read on, so its seat delta is a number and not a hope.
#
# WHY AN `ARM DONE` GREP AND NOT A PID WAIT. Cell C is launched by `after_cellc.sh`, whose pid does not
# exist until A' exits, so there is no pid to pass at arming time. `after_cellc.sh:49` writes
# `ARM DONE repl2x2_c` into `logs/repl2x2_c.full.log` after its `wait` returns; that string is the only
# signal that exists before the fact. Grepped from an EXISTING marker line, not derived from a
# checkpoint name (`CLAUDE.md`: the file carries the checkpoint spelling, the marker text does not).
#
# WHY 384 AND NOT 480. dread's released `raptor_ft_effv2l480` is a 480 px arm. A' at 384 sat at 28.9 GB
# of 31.8 on this card and effv2l is 117M params against coatnet_rmlp_2's 73M; 480 at bs 8 does not fit,
# and bs 4 would change the BatchNorm batch, which is a second variable in a cell that exists to read
# one. So this is HIS recipe at OUR resolution, and the comparison it licenses is against A' (same res,
# same fold, same 896), never against his published 480 number.
#
# WHY NO PLUMBING SHIM, AFTER ONE WAS ORDERED. D13 §1 said his `build_backbone` routes an EfficientNet
# name into the ViT branch and dies on `dynamic_img_size`. Measured, on his own predicate: for
# `tf_efficientnetv2_l.in21k_ft_in1k`, `hybrid=False` and `is_vit=False` — "vit" is not a substring of
# "efficientnetv2" — so it takes the `global_pool="avg"` branch and `timm.create_model` builds it,
# 117.2M params. The token would have been a no-op. The name that DOES take the ViT branch is
# `efficientvit_*`, a different family we are not running. `external/` stays unedited and `foreign_run`
# keeps its single announced shim.
#
# PRE-REGISTERED, solo OOF-896 under the LEDGER ruler (say the set name before the number):
#   < 0.840   with gold >= 0.90 — harness signature, not a member reading. Do not seat it.
#   0.840-0.865  v4-class on 19.5% fewer studies. Expected. His effv2l480 read 0.8703 IN-SAMPLE
#             against v4's 0.8915, so ~0.02 under A' is the null.
#   >= 0.875  better than v4 held-out; the family is a lane and our own cross-backbone fine-tunes
#             become the week's work.
# SEAT gate, and it is the one that decides a slot: S-delta minus the raptor-4/9 control >= +0.005,
# equivalently the delta with it seated INSIDE the foreign block at the shared 1/3. Reweighting alone
# reads +0.0058 HIS / +0.0037 LEDGER with NO new member, so a bare "+0.0055 SHIP" carries zero
# information. Read corr(raptor) first: above 0.90 the solo score is irrelevant.
# COLUMNS (D14, pre-registered before the number): wins LatOA/Fracture +0.03-0.06 over the DINO core,
# MedOA/PFOA +0.00-0.02, ligament/meniscus behind, Synovitis -0.05 or worse. A Synovitis loss here is
# HIS GEOMETRY (12-slice ROI crop, 3-slice stacks) and not the trunk — that is what makes the
# per-column geometry arm real. An Effusion sign flip between rulers is UNPRICED, not a loss.
#
# PRICE: ~1.6 h at 16 epochs by A's s/epoch scaled for params. No submission slot of its own.
set -u

REPO="D:/personal_projects/rsna_knee_kaggle"
PY="$REPO/.venv/Scripts/python.exe"
SCRIPT="$REPO/external/raptor/knee-mri-training-the-twelve-finding-model.py"
LOG="$REPO/logs/raptor_ft_effv2l_f0.full.log"
HOLDER=62660

# GATED ON THE DEVICE, not on an `ARM DONE` marker. Cell C released the card hours ago; the arm this
# now queues behind is `hisfluid_hold`, which was launched straight from the repo tree rather than
# through a watcher, so `lib.sh` wrote its markers into a snapshot log this tree cannot name. The card
# itself can be asked, and a pid that no longer appears in `nvidia-smi` has released it whatever log it
# wrote. `card_check` reads the same source.
echo "waiting for pid $HOLDER (hisfluid_hold) to release the card" >> "$LOG"
until ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qx "$HOLDER"; do sleep 30; done
echo "card released by $HOLDER at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

cd /d/personal_projects/rsna_knee_kaggle || exit 1
export PYTHONPATH="$REPO"
nohup "$PY" -m kneemri.experiments.foreign_run \
  --script "$SCRIPT" \
  --arch tf_efficientnetv2_l.in21k_ft_in1k \
  --res 384 --epochs 16 --bs 8 --k 12 --k_eval 24 --grad_ckpt --ckpt timm \
  --tag effv2l_f0 \
  --labels "D:/data/external/raptor/labels/labels_llm_soft.parquet" \
  --folds 1 --fold 0 --fold_file "$REPO/external/raptor/fold0_909.json" \
  --workers 4 >> "$LOG" 2>&1 &
echo "effv2l_f0 launched as pid $! at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
wait
echo "ARM DONE effv2l_f0" >> "$LOG"
