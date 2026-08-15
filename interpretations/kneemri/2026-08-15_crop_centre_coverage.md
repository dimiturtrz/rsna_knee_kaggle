# Which crop centre keeps more knee — measured, and it is not enough to rebuild a cache for

`python -m kneemri.crop_coverage --sample 400 --workers 20`, 2026-08-15.

## What was asked, and why not an AUC

`TissueCentre` has been plumbed through six places and opt-in since 2026-08-14, and NOTHING has ever
been trained under it. knee-yrf's evidence was an offset distribution over 60 studies (median 7 mm, p90
18 mm, max 32 mm against a 160 mm FOV) from a scratchpad script that was never committed — an argument
that the image centre *could* be wrong, never a measurement that the tissue centre is right.

There is no ground-truth joint centre in this corpus, so "closer to the truth" is not decidable. Training
under each centre and comparing costs a GPU that is not free. What IS decidable label-free: how much of
the imaged tissue each centre's box keeps. That is the mechanism the strategy exists for — a centre is
worth something only if its box contains more knee.

## The numbers

Denominator first: **1848 series, from 400 studies sampled of 4407 in the corpus** (9.1% of studies,
every 11th by corpus order — a property of the order, not of a seed). Share of a series' Otsu tissue mass
retained inside the crop box, same threshold that locates the centre so the mask cannot disagree with
what it judges.

| crop | median | p10 | worst | series |
|---|---|---|---|---|
| image@130 | 0.8495 | 0.7299 | 0.4537 | 1822 |
| tissue@130 | 0.8860 | 0.7576 | 0.4524 | 1822 |
| image@80 | 0.4024 | 0.2795 | 0.1423 | 1848 |
| tissue@80 | 0.4175 | 0.3042 | 0.0929 | 1848 |

Paired over the same series, bootstrap 1500× at 95%:

- 130 mm: **+0.0294 [+0.0276, +0.0312]** over 1822 series
- 80 mm: **+0.0233 [+0.0215, +0.0251]** over 1848 series

Both exclude zero. The effect is REAL and it is 2-3 percentage points of tissue mass.

## What it says

**1. Tissue centring works, in the direction claimed, and by a knob-sized amount.** The 130 mm prediction
was that the two centres would be near indistinguishable because there is ~15 mm of margin. They are not
indistinguishable — but +0.029 of retained mass is the size of a rounding choice, not of a design change.
Against a bar of 1.5% absolute AUC this predicts nothing, and it costs a rebuild of 99 GB of cached
arrays plus a refit of every member on them. **Not worth it at 130 mm.** knee-yrf closes as measured-and-
not-promoted, with the default staying `image`.

**2. The tail moves the WRONG way, which is the finding.** At 80 mm the tissue centre's worst series
retains 0.093 against the image centre's 0.142 — it is better on median and p10 and *worse* at the
extreme. Mechanism: the Otsu centroid is a mass centre, so a bright coil, subcutaneous fat, or the far
limb in frame drags it off the joint, and when it fails it fails deeper than a fixed centre can. The
median-over-slices already guards against one bad slice; it does not guard against a series that is bright
in the wrong place throughout.

**3. knee-1nn's premise does not survive either centre.** The 80 mm crop was going to be centred on tissue
because "a 40 mm half-extent has no margin for an 18 mm error." At 80 mm BOTH centres retain about 40% of
tissue — an 80 mm box against a 160 mm FOV is a quarter of the area, so most of what is discarded is
discarded by design and not by mis-centring. Tissue centring buys 2.3 pp of that back and pays for it with
a worse tail. So the tight crop is not gated on the centre: knee-1nn can read 80 mm about the IMAGE centre,
which needs no cache rebuild, and the multi-scale hypothesis is tested on resolution rather than on
centring.

## What this cannot say

Retained tissue is not retained JOINT. A box that drifts toward the calf keeps tissue while losing the
knee, and this measurement would score that as a win. It bounds the failure from one side only: a centre
that keeps LESS tissue cannot be the one keeping more joint. That direction is the one the decision needed,
and it came back "tissue keeps more, slightly" — so the honest reading is that tissue centring is not
REFUTED, it is merely too small to pay for a cache rebuild. If a cache is ever rebuilt for another reason,
take the tissue centre with it; do not rebuild for it.

## What was NOT ruled out

The strategy stays in the code, opt-in, at zero cost. Two uses it might still earn:
- a centre that is a joint DETECTOR rather than a mass centroid — the tail failure above is an argument
  for a better locator, not against locating;
- an ensemble axis: two members reading the same series about two different centres are diverse in the
  input, and diversity has beaten member quality on this corpus six leaderboard entries running.
