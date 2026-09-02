# bead: knee-7d53
# The aggregation axis STACKED — the three flags that each moved the focal findings up, in one member.
#
# WHY THIS ARM AND NOT ANOTHER PIXEL ONE. Fourteen arms read on run_comparison's per-label ruler on
# 2026-08-27 split cleanly by WHAT THEY CHANGED, not by how much they changed it. Every arm that gave
# the model MORE INPUT lost on the six model-bottlenecked findings: res392 -0.0022, intensity392
# -0.0076, crop80_span -0.0098, slices16 -0.0206, slotmean -0.0159, focal_pool -0.0159. Every arm that
# changed HOW THE SLICES ARE AGGREGATED gained: sided_pool +0.0076, attend_dense +0.0059,
# depth_regions +0.0016, through_dense +0.0005. Four positives, four different mechanisms, one family.
#
# The mechanism that explains the sign of both halves: a thin focal structure (ACL, a meniscal tear,
# the MCL) lives on two or three slices of twenty, so a pooling that averages over depth divides its
# evidence by the slice count. Adding slices makes that WORSE — which is exactly why slices16 is the
# single worst arm of the fourteen — and adding pixels does not touch it at all. Only a pooling that
# can concentrate on the slices where the structure is can recover it.
#
# THE STACK IS THE TEST. Each of the three below was measured ALONE against the same control and each
# is individually inside the noise floor. The question this arm asks is whether they COMPOSE, because
# they act at three different points: `--through-plane` before the encoder sees a slice, `attend`
# across the slice axis, `--pool sided` across the slots. Nothing about them shares a mechanism, so
# partial composition is the honest prior.
#
# PRE-REGISTERED. Best member this project owns is small_batch6 at 0.8598 solo; attend_dense reads
# 0.8569, sided_pool 0.8561, through_dense 0.8558. If the three compose fully this lands near 0.865
# and is a new best member; ~0.859-0.862 is directional and seatable; at or below 0.857 the three do
# not compose and the aggregation axis is a set of alternatives rather than a stack. It is a MEMBER
# bid, not a knob: the seat ruler prices a new best member into the champion five at roughly its solo
# margin, and member quality is the term this project's own board history says is binding.
#
# ATTENTION IS ABSENT ON PURPOSE and this is the OTHER half of the same test. model.py:322 takes the
# attend path and never reaches `over_depth`, so `--slice-pooling attend` and `--slice-regions` cannot
# both act — a region count passed beside attention is silently ignored. The two flags are rival
# answers to the same question (which depth does the evidence live at), so the stack forks here rather
# than choosing on the weaker of two solo readings. depth_regions is the weaker one (+0.0016 focal
# against attend_dense's +0.0059), which is why this arm is the second of the pair, not the first.

source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train fuse_regions \
  $RECIPE --epochs 12 --slice-pooling max --slice-regions 4 --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --pool sided --through-plane \
  --labels "$LABELS" --out "$MODELS/fuse_regions_s2026.pt"
