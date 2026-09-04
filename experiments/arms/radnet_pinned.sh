#!/usr/bin/env bash
# radnet_pinned.sh — the same arm as `radnet_family`, with the defect that voided it removed.
#
# READ `radnet_family.sh` FIRST. Every word of its argument still stands: the empty medical/supervised
# cell on the pretraining grid, the 1.35M-image RadImageNet corpus, the seven-million-parameter honest
# expectation of a weak solo, the measured HALF normalization, the 352 px read forced by DenseNet's
# activation bill. None of that is repeated here, and none of it is what changed.
#
# WHAT CHANGED IS ONE THING, AND IT IS NOT A FLAG. `radnet_family` scored 0.6716 and the ledger read
# that as the medical/supervised cell coming back empty. It was not a measurement of that cell.
# `knee-cj40f`: all 121 BatchNorm layers of that run — `norm0` and the entire frozen prefix included —
# finished at `num_batches_tracked = 226711`. Freezing pins PARAMETERS; `running_mean` and `running_var`
# are BUFFERS that the forward pass rewrites in train mode regardless. So the frozen prefix held its
# conv weights at RadImageNet while the statistics those weights consume were replaced by knee-MRI ones
# inside the first epoch, and being frozen is exactly why it could not compensate. `StudyModel.train`
# now re-pins them on every call, which is the only place it can be done: `nn.Module.train()` recurses
# into every child, so the trainer's own `model.train()` at the top of each epoch undid any one-shot
# `eval()` — which is how the defect survived a whole arm without a single error.
#
# THE ARM IS THEREFORE A MATCHED A/B WITH n=1 ON EACH SIDE, and every flag below is `radnet_family`'s
# verbatim for that reason. The delta against 0.6716 is the price of the BatchNorm defect, which is a
# number this project does not have for any arm and which prices the re-run of anything else that ran
# a frozen BN prefix.
#
# THE SECOND READING IS THE FROZEN PROBE. The same checkpoint's frozen-features ridge probe reads
# 0.8197 — computed in eval mode, therefore on the statistics the weights were actually trained with,
# which is why it is a full point and a half above the fitted score. A probe on frozen features is a
# FLOOR for fine-tuning the same trunk, not an estimate of it: it fits a linear map where fine-tuning
# fits the trunk. So a BN-correct fit landing BELOW 0.8197 would mean fine-tuning is destroying features
# a ridge could already read, and that is its own finding.
#
# PRE-REGISTERED, against the 0.8521 seat bar and the two numbers above:
#
#   >= 0.8521   THE CELL IS REAL AND BUYS A SEAT. Medical/supervised pretraining is competitive with
#               the incumbents at a twelfth their capacity, and `knee-nj4l`'s architecture-monoculture
#               reading has its first genuinely different member. Price the mix, then the slot.
#   0.8197-.852 BN WAS THE WHOLE DEFECT. The trunk is real and under the seat bar on its own, which is
#               where every bought seat in this pool started. Diversity candidate; priced by mix, not
#               by solo. Do NOT read the shortfall as the cell being empty.
#   0.6716-.819 BN WAS PART OF IT. Something else also costs — capacity, or the 352 px read, or the
#               corpus being CT-and-ultrasound-heavy. The delta from 0.6716 still prices the defect.
#   <= 0.6716   BN WAS NOT THE DEFECT. The cell is empty on its own merits, `radnet_family`'s reading
#               is restored, and the monoculture needs a different route. This is the branch that
#               closes the family, and it is the only one that does.
#
# WHAT THE SCORE DECIDES ABOUT THE *NEXT* CNN ARM, which is the decision this arm really gates. The
# supervised-ImageNet VGG-16 / ResNet-50 arm was going to be queued behind this one on the strength of
# the empty cell alone. Kimi's Q-DI says that is the wrong trigger, and gives a measured one.
#
# The only paper that measures architecture-family decorrelation directly is NVIDIA's *Exploring
# Diversity in Neural Architectures for Safety* (IJCAI-ECAI 2022): CNN+ViT ensembles are measurably
# more diverse than intra-architecture ones, and the weaker-but-different branch wins — ConvNeXt-B +
# ViT-B beats ConvNeXt-B + ConvNeXt-XL even though ViT-B is the less accurate model. That is this
# arm's whole thesis, measured, in the adjacent domain.
#
# The SAME paper carries the refutation, and it is the part that changes the plan: high diversity does
# NOT pay when the branches differ much in accuracy. "The metrics capture how diversely models
# classify, not the potential of the ensemble... These two objectives coincide when models perform
# similarly, while a discrepancy in accuracies causes them to misalign." MLP-Mixer Large is ~5 points
# below Base, measures as diverse, and ensembles worse. So distance is necessary and not sufficient,
# and `knee-y59c`'s anti-correlation of distance with solo strength is bounded rather than open-ended.
#
# THE GATE, therefore, on this arm's fitted score against incumbents at ~0.85:
#   >= ~0.82    the CNN axis is live. Compute per-case error correlation (Cohen's kappa or the
#               Q-statistic) between this member and the mean ViT-incumbent prediction on the holdout —
#               card-free — and queue VGG-16 / ResNet-50 only if that correlation is genuinely low.
#   <  ~0.80    DO NOT SPEND A SECOND CNN ARM. The literature says a branch more than ~5 points off
#               the incumbents is noise in the mix however different its errors look, and no
#               score-weighted average rescues that gap.
#
# NOTE WHAT NOBODY HAS MEASURED, so this arm is not read as confirming something already known. There
# is NO published leave-one-branch-out ablation removing only the CNN from a medical-imaging ensemble:
# the +0.031 in the Nilay/Thapar COVID-CT result is a total heterogeneous-ensemble gain across three
# inference paradigms and is not a per-branch number. There are no error-consistency figures for CNN
# vs ViT on a shared medical target, and no MSK-MRI point at all. The kappa this arm can compute would
# be the project's own first measurement on that axis, which is worth more than its solo score.
#
# VERIFY THE PIN FIRED BEFORE BELIEVING ANY OF THEM. The pin is unconditional — no flag to typo — but
# `pin_frozen_norms` returns a count that `model.py` currently throws away (knee bead filed), so a run
# that pinned nothing looks exactly like one that worked. This arm's mechanism check is direct and needs
# no plumbing: read `num_batches_tracked` out of the FIRST epoch's checkpoint. Unchanged from the
# pretrained value means the pin held; 226711-shaped drift means this arm is void the same way its
# predecessor was, and the score below is not a fact about RadImageNet.
#
# NO SEAT IS PRE-REGISTERED, for `radnet_family`'s reason — no branch above predicts clearing the bar,
# and the two that would buy one buy it through a mix, which is a card-free measurement rather than a
# slot.
#
# IT IS A FLAG-FOR-FLAG TWIN OF `radnet_family`, AND `arm_check` IS RIGHT TO SAY SO (knee-7uoq). Its
# rule is that two arms differing in nothing but their `--out` spend a card slot to produce a second
# name for one recipe, and that no comparison between them means anything. That rule holds because the
# flag line is normally the whole of what an arm controls. Here it is not: the variable under test is a
# CODE change — `StudyModel.train` re-pinning frozen BatchNorm — which no flag can express, because it
# is a bug fix rather than an option, and making it one would invite running the broken branch again.
#
# So the twin is deliberate and the flags MUST match; anything else would confound the fix with a
# recipe difference and forfeit the matched A/B. What a reader needs instead of a flag diff is a way to
# tell the two checkpoints apart after the fact, and there is one that is a property of the artefact
# rather than of a note: `radnet_family_s2026.pt` carries `num_batches_tracked = 226711` in all 121 of
# its BatchNorm layers, and `radnet_pinned_s2026.pt` must carry the pretrained value in every frozen
# one. That is the same read this arm's mechanism check performs, and it is why the check is not
# optional bookkeeping.
#
# twin-of: radnet_family
#
# bead: knee-cj40f
#
# Also serves knee-nj4l (architecture monoculture) and knee-nz34 (encoder-family pricing).
#
# --rank-weight 0 IS WHAT "VERBATIM" ACTUALLY MEANS HERE, added 2026-09-04 after the first launch of
# this arm died on `run_setup.py:392`. `RECIPE` carries `--rank-weight 1.0`, and at `--batch 1` the
# ranking term forms no pairs at all, so `radnet_family` descended PLAIN CROSS-ENTROPY under a ranking
# arm's name — it is one of knee-tj88y's sixteen. The guard that now refuses that combination did not
# exist when it ran. Matching it means matching what it DESCENDED, not what its flags claimed, so the
# term is switched off explicitly and the A/B keeps its single variable. Passing `--pair-memory` instead
# would turn the ranking term on for one side only and move two variables at once, which is precisely
# the mistake that voided `ortho_stack`'s first launch.
#
# deliberate: --rank-weight (0, to match radnet_family's actual objective — see above)
# deliberate: --variant
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train radnet_pinned \
  $RECIPE --rank-weight 0 --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack \
  --variant radnet121 --backbone-lr 4e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 352 \
  --vram-gb 26 \
  --labels "$LABELS" --out "$MODELS/radnet_pinned_s2026.pt"
