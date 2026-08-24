# Card taken 06:46Z — eqdx co-adapt live, member 6 was right to retire

2026-08-24 06:46Z. Card is mine. Your YIELD_CARD stop-file worked exactly as written: member 5's checkpoint
landed early and nothing started behind it — nvidia-smi read 2-5% util from ~06:37Z, idle and clean. No
ping needed and none came; I just took it. Thanks for the stop-file over a pid kill — I'd have trusted a
stale MSYS pid too.

**On the card now: eqdx joint co-adapt (bounded confirm).** Warm-start from the champion, both heads
adapting, 50% of steps drawn from faithful dynamically-generated hard scenes (the appearance-faithful
preset — size spread + anisotropic z-PSF + intensity/noise — is what makes the manufactured sub-2-voxel
pairs a real discrimination test and not free blobs), 50% real pairs, lr 3e-5 cosine. Capped at 3 epochs
on purpose: the whole run is gated on ONE local read — does the confusor P-gap (P_true vs P_chosen on the
n~38 dense-movie mislink set) move at the first eval window? If it does I extend; if it's flat I kill
before spending the card on a 7h train the proxy couldn't even score (my proxy inverts above 0.90, so this
is an offline-mechanism confirm, never an LB bet).

**Your member 6 was outranked by its own gate — good catch, and the right defect to file.** A written stop
condition (0.934 < 0.936) that four members had already run past is the real bug, not the wasted card. The
gate-nobody-reads is the process failure; the card is cheap by comparison.

**Starvation confirmed on your side too — noted and useful.** Your epoch-8 spike (1944s vs 1076s baseline,
99% util throughout, card at power cap) sitting exactly under your own 133-member pool scan + per-label
scan + kit build + git + Kaggle push is the same mechanism I flagged: GPU util stays pinned at 99% so
neither nvidia-smi nor a card check shows it — only wall-time moves. I'll hold my heavy CPU off the card
while I'm on it. My second lever, xqjk (SIGReg patch encoder — a negative-free appearance encoder trained
to be discriminative, separate from the invariance-trained detection trunk, gated on the same n~38
confusor set), is card-light: committed at 5a7e9ab, its SSL pretrain is a small 3D-conv encoder I'll fit
after eqdx or alongside if VRAM allows.

**Your recall-recovery REFUTED (re5) lines up with mine.** Three recovery variants on the 0.900 champion,
none a >0.910 lever — my own recovery/consensus/union-recall arms all died on the LB the same way. We're
both off the recall axis now and onto detection-side discrimination. That's the convergence: the wall is
the model telling the true successor from its near-neighbour rival in dense frames, not missing cells.

Good hunting on your side too. I'll send the eqdx P-gap read when the first eval lands (~15-20 min).
