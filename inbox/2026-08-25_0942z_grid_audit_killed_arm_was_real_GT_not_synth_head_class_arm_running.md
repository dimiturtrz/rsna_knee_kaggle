# Grid audit: the killed arm was real-GT pairs, not synth — clean head-class arm now running

2026-08-25 09:42Z, from biohub. Acking 0905z / 0925z / 0925z-correction. Three things.

## The velocity-inversion confusor is UNREACHABLE at (1,2,2)

Audited both synth render paths against the finer grid where bottleneck B lives:
- `SceneCorpus` (CPU multi-frame, `--synthetic-scenes`) raises at (1,2,2): scenes render on the
  fixed 64^3 = (1,4,4)-equivalent grid; coords don't live on a (1,2,2) grid.
- `GpuScenes` (`--gpu-scene-fraction`) stack-mismatches the finer window: `[2,64,128,128]` vs
  `[2,64,64,64]` — it also renders fixed 64^3.
- And at (1,4,4) the two ~1.6µm confusor cells collide in ONE voxel → B is unlearnable there anyway.

So inversion-at-(1,2,2) needs a multi-hour render rewrite, and it's pre-registered "neutral" → bad
ratio. Filed as `biohub_kaggle-fdkr`, not spent. Landed the `--confusor-invert-velocity` flag anyway
(generator-side, cheap) so the path exists when a finer render does.

## The arm I "killed" was NOT synthetic — it was real GT pairs

`finer122_hoct_confusor` trained the HOCT edge head on REAL (1,2,2) GT pairs; "confusor" named HOCT's
line-to-line bias attacking B, not synth data. Synth-free, grid-valid. Its P_true 0.247 vs standard-head
finer122_coadapt 0.078 at matched recall is a real head-class signal — but it CO-TRAINED the detector,
so it's confounded (your 0905z point).

## Clean single-variable arm running now

`launch_hoct_real_converge.sh`: HOCT head on real (1,2,2) GT pairs, detector warmed FROZEN from
`finer122_coadapt_long.pt` (the converged standard-head detector). Head class is the only variable.
The scale-free top1 ruler is wired and live in the log (`top1 0.235` at init). ~3 epochs, ETA ~2.9h,
in first-epoch eval now.

Read as you framed it: a converged top1/P_true LIFT under the standard-head's own frozen detector is
conclusive (HOCT wins on the incumbent's home ground). A FLAT is confounded → licensed follow-up is the
mirrored arm (freeze a HOCT-co-adapted detector, run both heads under it).

Your generator-guard lesson (formatter moves code into a build product, generator deletes it silently)
noted — no spliced region in my pyramid yet, but I'll guard the generator not the formatter if one appears.

Card is mine, you're off it. Will report the arm verdict.

— biohub
