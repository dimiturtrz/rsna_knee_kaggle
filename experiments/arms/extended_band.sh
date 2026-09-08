#!/usr/bin/env bash
# extended_band.sh — the nine-slot cache, on the one arm in the tree whose control is EXACT.
#
# bead: knee-q2yz7
# reading: extended_band_s2026.holdout.npz
#
# THE AXIS. D180 closed member generation in (solo, nnCorr): fifteen distinct lineages share the
# top-20 residual tail with no axis in common, every recipe knob dies against its own permutation null,
# and the break-even for a seat is +0.032 solo or -0.093 nnCorr from the nearest real member. The one
# thing the surface has no column for is what a member SEES. This arm changes that and nothing else.
#
# WHAT CHANGES, MEASURED RATHER THAN ARGUED (researcher D182, off the two masks on disk, no decode):
#
#   canonical six, extended mask vs canonical mask    IDENTICAL, 0 cells differ
#   extra slots filled  COR_FLUID_NOFS / AX_T1 / AX_FLUID_NOFS   841 / 564 / 566  of 4,407
#   studies gaining >= 1 extra slot                   1,508 = 34.2%   (1,052 / 449 / 7 gaining 1 / 2 / 3)
#   mean filled slots per study                       4.568 -> 5.015  (+0.447)
#   studies with zero canonical slots that gain one   0
#
# The denominator is the whole corpus, not a fold. D181 first quoted 9.9%, which is a rate over SERIES
# (2,421 of 24,371) and is the wrong denominator for sizing an arm; the per-study rate is 34.2%.
# `slots.py:79-93` states the imaging behind it: 83,297 slices no member on the roster has ever seen,
# and none of it localisers - coronal PD without fat suppression is a standard meniscus sequence and
# axial T1 reads patellar cartilage.
#
# STRICTLY ADDITIVE IS THE PROPERTY THAT MATTERS. `EXTENDED_SLOTS` appends and never inserts, and the
# mask on disk confirms the routing agrees cell-for-cell. So this is not a re-decode of the same input
# under different rules: it is every current member's input plus three channels. A flat read therefore
# cannot be blamed on having disturbed the six, which is the failure mode every other SEE knob has.
#
# ONE VARIABLE, AND THE CONTROL IS EXACT. Byte-identical to `band_dense` (ledger 0.8578 @midpoint on
# the 909 frame, confirmed twice - recomputed off band_dense_s2026.holdout.npz and recorded at
# experiments/README.md:309) except `--scheme extended`. Researcher D184 4: of the 79 members at this
# exact geometry on the default trunk, `band_dense` is the ONLY one whose run_train line is recipe +
# geometry + --labels and nothing else. No distillation, no calibrated targets, no EMA, no held
# annotations, no encoder weights, no slot dropout, no batch override. So the delta is attributable to
# the three channels and to nothing else, which is why this is the arm rather than an extension of
# `labelwise_dense` (0.8959 @ledger75, the pool's best solo, but it differs from band_dense by three
# flags and would confound the reading). If this axis reads positive, the labelwise extension is the
# follow-up that produces the shippable member.
#
# DO NOT ADD `--anatomical-prior` (D184 0). `SlotAffinity.requested` (anatomy.py:230-248) RAISES when
# n_slot != 6, so an extended arm carrying it dies at construction - the second hardcoded-six site after
# ortho_diffusion. Threading the arm's own slot tuple through would be a small change, and the ledger
# says not to bother: `aprior_dense` is exactly this control plus that flag and reads 0.8557, -0.0021,
# with the mechanism read that the warm start washed it out.
#
# THE HEAD TAKES NINE SLOTS. `n_slot` is a constructor argument threaded model.py:182/204/226/239 into
# slot_head.py:54 and :62 (`slot_emb = randn(n_slot, hidden)`), sourced from the cache rather than from
# a constant. The ONE hardcode is `ortho_diffusion.py:163`, `for index, slot in enumerate(SLOTS)`, so the
# ortho family cannot be fed by this cache - this arm is not ortho and does not touch it.
#
# HARNESS CHECK, BEFORE THE VERDICT AND NOT AFTER. The printed recipe must name "scheme": "extended",
# and the loaded array's slot axis must be 9. A nine-slot cache silently falling back to the six-slot
# one is invisible in the score and would read as a clean null on the whole SEE axis.
#
# WHAT THE THREE CHANNELS CARRY (D184 1, the prior's own affinity function over EXTENDED_SLOTS). Seven
# cells at 1.00 across seven findings: MCL and both menisci via COR_FLUID_NOFS, PF OA via AX_T1,
# Effusion / Synovitis / Baker's via AX_FLUID_NOFS. The sharpest fact: PF OA has NO 1.00 cell among the
# canonical six - its primary plane is axial with needs_fluid False, and the only canonical axial slot
# mismatches on sequence at 0.50. AX_T1 is the one slot in the nine that reads patellofemoral OA at
# primary plane and matched sequence. Price it as one finding of twelve. ACL, both OA columns, Contusion
# and Fracture LOSE - the extras are all non-fat-suppressed and marrow oedema is invisible without it.
#
# AND WHO THE 34.2% ARE (D184 2, report_labels_v2 at the midpoint, Fisher two-sided). The gainers are the
# corpus's SICKER third, not its easy third: nine of twelve findings more prevalent, Lateral OA 1.90x and
# ACL 1.53x, both p<1e-4, and the ordering survives on the 909 holdout frame with the same signs. But
# they are NOT the rare-positive third - Synovitis 1.01 (p 0.888) and Fracture 0.85 (p 0.174) are flat, and
# Fracture has ~20 positives on the 909 gainer frame, so no per-column claim on those two is readable here.
#
# THE CONFOUND, STATED BEFORE THE DELTA (D184 3). Protocol extension is indication-driven: AX_T1 studies
# carry 1.54x the Synovitis rate and 0.42x the Baker's rate. Under the canonical scheme the six-column
# mask has no cell for their presence; under the extended scheme that presence BIT is itself an input. So
# a positive delta is attributable to three channels of PIXELS or to three bits of PROTOCOL METADATA that
# correlate with pathology. The separator costs no training and is run only if the arm reads positive:
# re-score with the three extra channels zeroed and the mask left intact.
#
# AND THAT CONFOUND IS NOW PRICED, BEFORE THE CARD (D185, card-free, on THIS arm's own 909 frame at the
# midpoint). The three presence bits fitted ALONE against the twelve columns are above chance - cross-fit
# macro 0.5431, in-sample ceiling 0.5895 - so the metadata channel is real, and `COR_FLUID_NOFS` carries
# almost all of it (ACL 0.593, Lateral OA 0.576). `AX_T1` alone is at or below chance nearly everywhere and
# its two most negative columns are Baker's 0.446 and PF OA 0.472, which is the indication effect pointing
# opposite to the affinity argument, measured rather than argued. But the quantity that decides the arm is
# what the bits add ON TOP OF THE MEMBER: a per-column logistic on [logit(band_dense), bit0, bit1, bit2],
# 5-fold cross-fit, same rows, same cut, reads member 0.8578 -> 0.8557, delta -0.0021. The member column
# reproduces the ledger's band_dense to four decimals on a path that never read the ledger.
#
# So the channel is real and ALREADY SPENT: the pathology those bits mark is pathology the canonical six
# already show. A positive extended delta cannot be charged to three bits of protocol. The zeroed re-score
# is therefore NOT mandatory - it stays a follow-up conditional on a positive read, because -0.0021 bounds
# the POST-HOC channel only and a trunk trained on those channels can exploit bit-by-image interactions a
# four-parameter logistic cannot represent.
#
# THE 20-SLICE ROUTING IS THE 12-SLICE ROUTING (D185 3, read off the live decode's progress file at 3,600
# of 4,407). Canonical six, 20sl-extended vs 20sl-canonical: 0 cells differ. Extras, 20sl vs 12sl: 0
# studies disagree, per-extra filled 699/467/443 identical, gainers 34.3% against 12sl's 34.3% on the same
# rows, index order identical. D182 called slot routing's independence from slice depth an argument; on
# 3,600 studies it is a measurement, and this header is quoting the right corpus at itself.
#
# PRE-REGISTERED (M188 3, written before the decode finished), against band_dense 0.8578 @midpoint:
#   >= +0.0015 add into honest12 @ledger75   the SEE axis is live, and the plane scheme (~3 lines in
#              slots.py) and the mm-FOV multiscale scheme follow in that order.
#   +0.0000 to +0.0015   the three channels are real input and inert evidence. The axis reads like every
#              other member axis and no further decode-scale cost is paid for it.
#   negative   nine slots dilute the slot attention faster than the extra channels inform it. That is a
#              fact about the HEAD, not the input, and `--slot-pooling attend` is the follow-up rather
#              than a different scheme.
#
# THE FIRST LAUNCH DID NOT FIT ON THE CARD AND PAGED. Run 1 (2026-09-08 12:13Z) derived 32.5 GB against a
# 31.8 GB card, printed `not waiting for VRAM ... no co-tenant can free enough`, and started anyway — which
# is what `device_budget.await_headroom` documents as the failure it cannot refuse: "Windows pages over
# PCIe instead of raising, so a run that starts into a held card does not fail — it crawls at a tenth of
# the throughput." Measured: epoch 0 took 4064 s against its control `band_dense`'s 361 s, a factor of
# 11.3, for a recipe whose extra work is +10% filled slots. 12 epochs would have been 13.5 h against the
# ~2.5 h priced below, on a card holding nothing else (`card_check`: one process, ours). Epoch 0 landed
# 0.7379 against the control's 0.7430, so nothing was bought for it either.
#
# THE FIX KEEPS THE COMPARISON EXACT: `--batch 6 --accumulate 2`. `DeviceBudget.needed` scales the
# activation term linearly in batch (`step = (activation/fitted) * (batch/FITTED_BATCH)`), so halving the
# batch halves the part that overflows while the effective batch stays 12 — identical to the control's,
# with no batch-norm in this stack to make accumulation differ from a wide step. It is the one change that
# makes the recipe fit without moving a variable the reading depends on. Run 1's holdout is kept as
# `extended_band_s2026_run1.holdout.npz` and is NOT a member: it is a 1-epoch paging crawl, not a result.
#
# PRICE: ~2.5 h, the same as its control. Cache built 2026-09-08 06:39Z,
# images_336px_20sl_0.1-0.9band_extended.npy, (4407, 9, 20, 336, 336) uint8 = 89.6 GB. No submission slot -
# the reading is offline against a control already on the ledger.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train extended_band \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --scheme extended \
  --batch 6 --accumulate 2 \
  --labels "$LABELS" --out "$MODELS/extended_band_s2026.pt"
