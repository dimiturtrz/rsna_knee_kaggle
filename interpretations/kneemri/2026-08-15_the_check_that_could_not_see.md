# The check that could not see: a pathspec anchored one level too deep

**2026-08-15.** `kit_check --mounted` — the gate written so that no slot is ever spent on code the repo
had already fixed — reported the mounted kit as current on every run since the package restructure. It
was 53 shipped-package commits behind.

## What happened

`main` anchored git at `Path(__file__).resolve().parents[1]`. Before `d02b6f6` that was the repository
root, because the module sat at `kneemri/kit_check.py`. The restructure moved it to
`kneemri/kaggle/kit_check.py`, so the same expression now named the **package** directory. The
judgement itself was unchanged and still correct:

```
git -C <anchor> log --oneline <kit-commit>..HEAD -- kneemri core
```

A pathspec resolves against where git runs. Anchored at `kneemri/`, `kneemri core` means
`kneemri/kneemri` and `kneemri/core` — two paths that do not exist. No commit ever matched, the drift
list was always empty, and the check exited 0 while printing a commit hash that was two days old.

## Why it is the same failure class as the nine before it

The value was recorded (`kit.json` carried the real commit, and the check printed it), and the consumer
read something else. The printed line even *contained* the evidence — `at 4faac3d, 0 commit(s) behind`
names a commit whose age contradicts the verdict beside it — and the shape of the output is what made
it unreadable: a number that is always zero looks like a clean check rather than a broken one.

The bead this closes (`knee-wgx`) was written on 2026-08-15 from a run that DID report the drift. That
run named `python -m kneemri.kit_check`, the pre-restructure path, which still anchored correctly
because the module was one level up. The same check under its new name said the opposite, and the
disagreement went unnoticed because only the pessimistic answer was surprising enough to write down.

## The fix, and the property it buys

`KitCheck` now asks git for the toplevel instead of counting directory levels:

```python
self.repo = Path(self.asked("rev-parse", "--show-toplevel"))
```

That cannot drift when a file moves, which is the property the old expression lacked. The regression
test asserts the thing that was actually broken — a path INSIDE the repository answers the same as its
root, and both answer non-empty for a kit that is genuinely behind. A test of the judgement alone
(which existed, and passed throughout) could never have caught this, because the judgement was never
wrong: only its anchor was.

## What it cost, and what it did not

Nothing measurable. The one entry submitted against the stale kit (`kneemri-fusion` v10, 2026-08-15
21:06 UTC) was pushed and run BEFORE the kit was rebuilt, so it ran the code its own log names, and
the kit is current at `0be5553` from 21:11 onwards. The cost was the two days in which the check was
believed rather than true.

**The transferable rule: a gate that has never failed is not evidence of health.** Give every check a
case it is known to fail, and run that case, or the check is only a comment that executes.
