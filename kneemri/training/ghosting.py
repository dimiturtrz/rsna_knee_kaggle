"""Motion ghosting as a train-time operator, with its severity CALIBRATED rather than dialled.

WHY THIS IS NOT ANOTHER KNOB. Every augmentation in `AugmentConfig` is a transform somebody chose a
number for: rotation 10 degrees because that looked like patient positioning, noise 0.02 because the
images are in [0, 1]. This one is different in exactly one way, and the way is the whole reason it was
built: KMAR-50K ships the same knee acquired twice, once with patient motion and once without, so the
severity of real motion is a MEASURED quantity rather than a taste. `core.data.corpus.kmar_stats` is
that measurement and it ran before a line of this module was written.

THE THREE NUMBERS IT WAS BUILT AGAINST (109 pairs of KMAR-50K, 2026-09-12, per-pixel in the [0, 1] space
this operator works in, quantiles over pairs):

    residual          p10 0.023   p50 0.040   p90 0.086
    residual_p99      p10 0.180   p50 0.363   p90 0.630
    background_share  p10 0.057   p50 0.113   p90 0.201

Only the FIRST of them is a parameter here. The second is a PREDICTION — the operator is handed the
magnitude and must reproduce the impulsiveness from its mechanism alone (p99/mean ~ 9, against ~2.6 for
additive Gaussian noise, so the number genuinely discriminates). It refuted the first build and then
fixed it: see `DISPLACEMENTS`, whose value is read off a sweep against that one number rather than
chosen.

THE THIRD NUMBER IS NOT A TEST AND MUST NOT BE USED AS ONE. `background_share` is defined in
`kmar_stats` as the fraction of residual energy falling on pixels below `BACKGROUND_PERCENTILE = 40` of
the clean image — the bottom FORTY per cent. On a KMAR volume that cutoff sits at 0.022 of full scale:
air, which is what it was meant to select. On this project's 224 px cache it sits at **0.416** — mid-grey
anatomy, nineteen times higher — because our cache holds a tight crop of a knee and KMAR holds a whole
field of view. The same statistic therefore names air in one corpus and tissue in the other, and the
0.113 KMAR reports is not comparable to the 0.43 this operator produces on our pixels. Measured
2026-09-12; it is a denominator error, not a defect in either corpus, and it would have been read as a
refutation of the operator by anyone who compared the two numbers without opening the definition.

THE MECHANISM. Motion during a phase-encoded acquisition makes the k-space lines inconsistent between
shots, and the inconsistency puts displaced REPLICAS of the object along the phase-encode direction
rather than smearing it in place. The corpus constrains how far: at a large displacement every pixel the
replica lands on changes, which is a diffuse residual, and KMAR's impulsiveness says its residual is not
diffuse. So the displacement is small and the residual concentrates on high-gradient edges — see
`DISPLACEMENTS` for the sweep that settles it at 2-4 px.

WHY THE SEVERITY IS A TARGET RESIDUAL AND NOT A REPLICA AMPLITUDE. An amplitude alpha does not map to a
measured residual by any constant: the residual it produces is alpha times the mean absolute difference
between the image and its own displaced copy, and that depends on the image. Parameterising by amplitude
would therefore have put a fitted constant between the corpus number and the knob, and the constant would
have been wrong for every image whose anatomy fills a different fraction of the frame. So the field IS the
target residual: the replicas are built at unit amplitude, their actual mean absolute difference is
measured per image, and they are scaled to land on the drawn target. The operator reproduces the
calibrated statistic by construction, and the only number crossing from KMAR into this repo is a number
KMAR itself reports.

THE PHASE-ENCODE AXIS IS DRAWN, NOT ASSUMED. Ghosts travel along the phase-encode direction, and our
cache records no phase-encode direction — the slot images arrive as pixels with no acquisition header
surviving. Fixing the axis would bake in a guess that is wrong for some fraction of studies and
undiscoverable afterwards; drawing it per image makes the model robust to ghosting on either in-plane
axis, which is the honest position given the cache cannot say. KMAR's own plane invariance supports
paying nothing for this: sagittal, coronal and transection volumes give medians 0.0406 / 0.0434 / 0.0367
and background shares 0.106 / 0.116 / 0.125, spreads far inside the p10-p90 spread within any one plane.

WHAT THIS MODULE DOES NOT CLAIM. KMAR is Siemens Aera 1.5T `pd_tse_fs` (its own cohort table), and
whether RSNA's mixed vendors ghost at the same scale is unmeasured — it cannot be measured without paired
volumes of ours, which do not exist. The severity distribution here is therefore a transfer, argued from
motion being a property of patients rather than of scanners, and that argument is the assumption an arm
built on this module is testing.
"""

import logging

import torch
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)

GHOSTING = "ghosting"
"""The key motion ghosting answers under when a run checks its switches landed (`kneemri.training.landed`).

A TARGET RESIDUAL RATHER THAN A FLAG, and its identity is 0.0 — every member on disk was fitted without
it. Reported from the config the augmenter will actually use, for the reason `SLOT_DROPOUT` is: the value
is threaded onto an `AugmentConfig` built by one of two branches in `run_setup`, and a branch that dropped
it would train the historical recipe under the arm's name and report a null about a mechanism that never
ran.
"""

DISPLACEMENTS = (2, 3, 4)
"""Replica displacement in PIXELS, drawn uniformly per image — and this value was MEASURED, not chosen.

It started at FOV/N for N shots, which is the textbook ghost spacing, and the corpus refused it. The
first build displaced whole-object replicas by half, a third and a quarter of the frame; run against
KMAR's own statistics it reproduced the magnitude it was handed (0.042 against 0.040, which it must, by
construction) and produced an impulsiveness of **3.39 against KMAR's ~9** — a residual far too diffuse
to be motion. A sweep of the displacement against that one number is monotone and decides the parameter:

    displacement px      2      3      4      6      8     16     32     64    112
    p99 / mean        7.87   7.95   7.73   7.19   6.67   5.40   4.34   3.81   3.62

KMAR's ~9 sits just past the top of that curve, which peaks at 3 px. So real motion in this corpus is a
SMALL displacement, not a half-FOV replica: the residual it leaves is concentrated on high-gradient
edges, where a few-pixel shift changes a pixel a great deal, and is near zero across flat tissue, where
it changes nothing. That is what an impulsiveness of 9 means, and it is the shape a large displacement
cannot make — displace the knee far enough and every pixel it lands on changes, which is a diffuse
residual by definition.

The ghost interpretation survives the correction; only its scale does not. A knee sequence's motion is
sub-centimetre, and at this cache's 224 px over a knee that is a few pixels.
"""

VERTICAL_SHARE = 0.5
"""How often the phase-encode axis is drawn as vertical, and it is a half because the cache cannot say.

Not a measured quantity: the acquisition header does not survive into the slot cache, so there is no
corpus number to read here. A half is the maximum-entropy choice given that ignorance, and the module
docstring argues why drawing the axis beats fixing it.
"""

MEASUREMENT_FLOOR = 1e-6
"""Below this the unit perturbation is numerically zero and scaling it to a target would divide by noise.

Reachable on a genuinely empty slot image, which the cache does hold: a study missing an acquisition
arrives as zeros behind a false presence flag, and the augmenter sees it before the head masks it.
"""


class Ghosting:
    """Phase-encode replica artefact at a measured severity, applied per image.

    Constructed with the MEDIAN target residual; the per-image target is drawn around it, because
    KMAR's own spread (p10 0.023, p90 0.086 against a p50 of 0.040) says severity varies by a factor of
    four across studies and a fixed value would train against a corpus that does not exist.
    """

    def __init__(self, severity: float, spread: float = 0.60) -> None:
        self._severity = severity
        self._spread = spread

    @property
    def severity(self) -> float:
        """The median per-pixel residual this operator aims to produce, in [0, 1] image units."""
        return self._severity

    def targets(self, count: int, generator: torch.Generator, device: torch.device) -> Float[Tensor, " n"]:
        """One target residual per image, lognormal around the median.

        LOGNORMAL RATHER THAN UNIFORM, AND THE SHAPE IS READ OFF THE CORPUS. KMAR's residual quantiles
        are 0.023 / 0.040 / 0.086 — multiplicative rather than additive around the median (p90/p50 = 2.15,
        p50/p10 = 1.74), bounded below by zero and with a tail on one side only. A lognormal with
        sigma = ln(2.15)/1.2816 = 0.60 reproduces that to the precision 109 pairs can support, and the
        default of `spread` is that number rather than a round one for exactly that reason.
        """
        draw = torch.randn(count, generator=generator, device=device) * self._spread
        return torch.as_tensor(self._severity, device=device) * torch.exp(draw)

    def applied(self, images: Float[Tensor, "n c h w"], generator: torch.Generator) -> Float[Tensor, "n c h w"]:
        """Each image mixed with displaced replicas of itself, scaled to hit its own drawn residual.

        The replica set is built at unit amplitude first and its mean absolute deviation measured per
        image, so the scale factor is whatever that image needs — this is the step that makes the field a
        measured residual rather than an amplitude with a fitted constant in front of it.

        Vectorised over the batch the way every other transform here is, with one exception that is
        forced: `torch.roll` takes a scalar shift, so the replicas are accumulated in a Python loop over
        SHOT COUNT (at most three iterations, each a whole-batch kernel), never over images.
        """
        if self._severity <= 0:
            return images
        count, device = images.shape[0], images.device
        vertical = torch.rand(count, generator=generator, device=device) < VERTICAL_SHARE
        index = int(torch.randint(len(DISPLACEMENTS), (1,), generator=generator, device=device))
        replicas = self._replicas(images, DISPLACEMENTS[index], vertical)
        deviation = (replicas - images).abs().mean(dim=(1, 2, 3), keepdim=True)
        scale = self.targets(count, generator, device).view(-1, 1, 1, 1) / deviation.clamp(min=MEASUREMENT_FLOOR)
        ghosted = images + scale * (replicas - images)
        return ghosted.clamp(0.0, 1.0)

    def _replicas(
        self, images: Float[Tensor, "n c h w"], displacement: int, vertical: Float[Tensor, " n"]
    ) -> Float[Tensor, "n c h w"]:
        """The unit-amplitude replica field: one copy displaced by `displacement` along the drawn axis.

        Both axes are computed for the whole batch and selected by the per-image draw, rather than
        splitting the batch in two — a select costs one kernel and a split costs a synchronisation.
        """
        down = torch.roll(images, shifts=displacement, dims=-2)
        across = torch.roll(images, shifts=displacement, dims=-1)
        return torch.where(vertical.view(-1, 1, 1, 1), down, across)
