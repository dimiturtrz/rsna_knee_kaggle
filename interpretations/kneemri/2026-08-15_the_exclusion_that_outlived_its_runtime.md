# The exclusion that outlived its runtime: v3 was refuted by a transformers version, not by its weights

**2026-08-15.** The fusion kernel dropped every DINOv3 member by name. The reason was written from a run
on 2026-08-14 in which those weights met a backbone nested as `backbone.model.layer.*` — 178 unexpected
keys — and the note recorded it honestly as *unproven rather than refuted*, because the probe meant to
settle it had a bug of its own.

## What the timestamps say

That run was at 06:24 UTC. The kit's wheel directory gained `transformers-5.15.0` at **11:08** and
`safetensors-0.8.0` at **11:15** the same day. The last scored entry ran at 10:40. So:

- no entry that has ever been scored mounted the transformers the v3 checkpoints were fitted under;
- the one entry that tried v3 ran against Kaggle's image version, which builds HF DINOv3 with an extra
  `model.` level — exactly the nesting in the error;
- the fix for it was already in the kit four and a half hours later, and the exclusion stayed.

## What the local check says

`witness_check --root all_ours --device cpu`, on transformers 5.15.0 — the version the kit now ships:

```
lora_r8_v3small_224_s2026: ok (max|diff| 0.0119 of 0.0312 over 8 studies)
5 ok, 1 unverified, 0 refused
```

The member loads, and it reproduces the answer it recorded when it was fitted. Whatever failed in the
kernel was not the checkpoint.

## Why dropping by name was the wrong instrument either way

`KernelRun.run` already loads each member inside its own `try` and drops the ones that fail — the rule
written after a DINOv3 member cost a whole submission at load time. A v3 member that still cannot load
therefore costs this entry one load attempt and nothing else. Excluding it by name converts a
self-healing failure into a permanent policy, and policies do not carry the date their evidence expired.

It also costs nothing in decode: `lora_r8_v3small_224` sits in the `('canonical', 12, Crop(130mm,
image))` read group with four of our five other members, reading 224 px out of the same 336 px decode.
The only member of ours that genuinely needs its own pass is `extended_ws` (the `extended` scheme).

## The rule

**An exclusion is evidence with a date on it.** Ours outlived the runtime that produced it by a day, and
the diversity it was withholding — a different backbone family, in an ensemble whose members are
otherwise all DINOv2-small — is the axis this project's own ledger says pays. Related: the same file
already carries one expired filter documented as such (pixel-group filtering, which dropped our two best
members after `SlotReader.cropped` made it unnecessary). Two in one function is a pattern, not a slip:
when the reason for a filter is a runtime fact, the filter needs re-testing every time the runtime moves.
