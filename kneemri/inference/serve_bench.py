"""Does compiling the encoder pay at SERVING time? `python -m kneemri.inference.serve_bench`.

TRAINING ALREADY ANSWERED THIS FOR ITSELF — measured -21% per step for 2.2 s of compilation, which is
why compilation is the training default. Serving is a different question with the same words in it, and
the difference is what this measures:

    * training runs one shape for thousands of steps; a submission runs a handful of chunks, so a
      compile that costs seconds has far fewer steps to amortise over.
    * training compiles ONE model; a submission holds every member of the package, and each one pays its
      own compilation.
    * `Ensemble.forwarded` halves its rows on `torch.OutOfMemoryError`. Under compile that exception
      arrives WRAPPED (dynamo re-raises as `InternalTorchDynamoError`), so the retry that exists to keep
      a T4 entry alive would stop catching the thing it was written for.

The third item is why this file measures rather than adopts: the speed has to be worth reworking a
belt that has already saved a submission once. Measured 2026-08-17 on this 5090, res392_ws_s2026
(dinov2-small, 6 slots x 12 slices at 392 px):

    4 studies    eager  188.2 ms   compiled  143.4 ms   -24%   compile  9.5 s   payback 212 passes
    48 studies   eager 2269.2 ms   compiled 1767.6 ms   -22%   compile 13.0 s   payback  26 passes

So the steady state matches training's, and it is still not worth it: the 13 s is paid PER MEMBER and
repaid only after 26 kernel-sized chunks — about 1,250 studies that member predicts — and a package
holds six. Serving stays eager (knee-h90h).

MEASURE IT ON THE HOST THAT MATTERS, and this one does not. Kaggle serves on a T4 (sm_75, fp16 tensor
cores, no bf16 ones) and this is a 5090; the RATIO here is an upper bound on what the kernel would see,
because a faster card has less kernel-launch overhead for compilation to remove. A win here is a
question for the kernel, a loss here settles it.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from core import cli
from kneemri.inference.members import Member, MemberPackage

logger = logging.getLogger(__name__)

STUDIES = 8
"""A chunk's worth of shape, not a chunk's worth of studies. The kernel sends 48, which on a 448 px
whole-stack member is 3,456 images a pass — minutes of bench for a ratio that eight studies already
give, since both arms pay the same size."""

REPEATS = 5
"""Enough that one scheduler hiccup is not the reading. The arms are compared on the MINIMUM, so more
repeats can only sharpen it."""


class ServeBench:
    """One member's forward, timed eager against compiled.

    IT HOLDS A DEVICE AND A MEMBER, not the `Ensemble` whose forward it times — the forward arrives as
    a callable. A timing harness that imported the ensemble would be a second consumer of a module
    already at the god-module fan-in bound, to learn two things (a device and a shape) that it can be
    handed instead.
    """

    def __init__(self, device: torch.device, member: Member) -> None:
        self.device = device
        self.member = member

    def pixels(self, studies: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthetic uint8 of exactly the shape the kernel sends, and its all-present mask.

        RANDOM PIXELS ARE HONEST HERE and would not be for a score. A forward's cost is set by shapes
        and dtypes, which are the member's own (`slot_count`, `slices`, `read_size`) — nothing in the
        graph branches on the values. Reading real studies would add a decode to both arms equally and
        make the bench need a corpus to answer a question about kernels.

        WHICH IS WHY A RESOLUTION IS PRICED BY RESIZING THE MEMBER, not this tensor: `Member.resized`
        moves the size the model resamples to as well as the shape allocated here, and only the pair
        of them times what a member fitted at that size would cost (knee-d2fc).
        """
        shape = (studies, self.member.slot_count, self.member.slices, self.member.read_size, self.member.read_size)
        images = torch.randint(0, 256, shape, dtype=torch.uint8, device=self.device)
        mask = torch.ones(studies, self.member.slot_count, device=self.device)
        return images, mask

    def timed(self, run: Callable[[], object], repeats: int) -> float:
        """Seconds for the fastest pass, warmed first so compilation is not charged to the arm.

        THE WARM-UP IS THE MEASUREMENT'S WHOLE POINT and also its biggest lie. A compiled arm spends
        seconds building the graph on its first call, so timing that call measures compilation; timing
        only the ones after it measures the steady state and pretends the compile was free. Both numbers
        matter, so `main` reports the warm-up separately rather than folding it in.

        MINIMUM, not mean: every source of noise on a shared card ADDS time, so the fastest pass is the
        one least contaminated by the co-tenant.
        """
        run()
        self.synchronise()
        fastest = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            run()
            self.synchronise()
            fastest = min(fastest, time.perf_counter() - started)
        return fastest

    def synchronise(self) -> None:
        """Wait for the card, because a CUDA launch returns before the work does.

        Without this both arms measure how fast Python can queue kernels, which is precisely the cost
        compilation removes — the bench would report a win that the submission never collects.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def graphs(self) -> int:
        """How many distinct graphs dynamo built — one is the good case, many means it recompiled.

        A serving forward is not the training loop's single fixed shape: the final chunk of a test set
        is short, and every shape is a fresh compile. That is a cost the steady-state number cannot
        show, so it is read directly rather than inferred.
        """
        return int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))  # noqa: SLF001 - the only reader


@dataclass(frozen=True)
class ServeBenchReport:
    """The two timings as a table — the rendering half of what `ServeBench` used to be.

    Split for the reason the other reports were: timing needs a card, a loaded member and a forward to
    call, and rendering needs the four numbers they produced. Held together, the payback arithmetic —
    the line the decision actually rests on — could not be exercised without loading a member.
    """

    eager: float
    compiled: float
    warmup: float
    graphs: int

    def lines(self) -> list[str]:
        """The two arms, what compilation cost to get there, and how many chunks pay that back."""
        eager, compiled = self.eager, self.compiled
        saved = eager - compiled
        payback = self.warmup / saved if saved > 0 else float("inf")
        return [
            f"eager      {eager * 1000:8.1f} ms/pass",
            f"compiled   {compiled * 1000:8.1f} ms/pass   {(compiled / eager - 1) * 100:+.0f}%",
            f"compilation{self.warmup:8.1f} s          pays back after {payback:.0f} passes",
            f"graphs     {self.graphs:8d}              (>1 means a shape recompiled)",
        ]


def main() -> None:
    """Time one member's serving forward eager against compiled, on synthetic pixels of its own shape."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--root", type=Path, default=None, help="a package to time; default is the one that ships"
    )
    command.parser.add_argument("--member", default="", help="which member to time (default: the strongest)")
    command.parser.add_argument("--studies", type=int, default=STUDIES)
    command.parser.add_argument("--repeats", type=int, default=REPEATS)
    command.parser.add_argument(
        "--read-size",
        type=int,
        default=0,
        help="time the forward at THIS size instead of the member's own (`Member.resized`). The "
        "predictions are then meaningless and the cost is not: it is what a member fitted at that "
        "size would spend, which is how a resolution arm is priced before it is trained.",
    )
    arguments = command.parsed()

    from kneemri.device import VisibleCards  # noqa: PLC0415 - CLI-only
    from kneemri.inference.ensemble import Ensemble  # noqa: PLC0415 - CLI-only
    from kneemri.models.backbone import ENCODERS  # noqa: PLC0415 - CLI-only

    package = MemberPackage.at(arguments.root, arguments.paths)
    fitted = package.named([arguments.member])[0] if arguments.member else package.strongest(1)[0]
    device = VisibleCards().device()
    encoder = ENCODERS[fitted.variant]
    # LOADED AS FITTED, PRICED AS ASKED. The fingerprint is computed at the member's own resolution, so
    # loading a resized one raises `WeightsError` — correctly: at 420 px this member differs from its
    # stored fingerprint by 0.086 against a tolerance of 0.002, which is the guard noticing exactly the
    # meaningless predictions this bench is willing to buy. Resizing after the load keeps the guard
    # honest for every other caller and still moves the size the encoder runs at.
    model = package.load(fitted, encoder.load(), encoder.dim, device)
    ensemble = Ensemble(package, {fitted.identifier: model}, device)
    member = fitted.resized(arguments.read_size) if arguments.read_size else fitted

    bench = ServeBench(device, member)
    images, mask = bench.pixels(arguments.studies)

    def forward() -> object:
        """The timed unit — the same call both arms make, so only the encoder differs between them."""
        return ensemble.predict_member(member, images, mask)

    logger.info(
        "%s: %d studies x %d slots x %d slices at %d px",
        member.identifier,
        arguments.studies,
        member.slot_count,
        member.slices,
        member.read_size,
    )

    eager = bench.timed(forward, arguments.repeats)
    started = time.perf_counter()
    model.compile_encoder()
    forward()
    bench.synchronise()
    warmup = time.perf_counter() - started
    compiled = bench.timed(forward, arguments.repeats)

    command.report(ServeBenchReport(eager, compiled, warmup, bench.graphs()).lines())


if __name__ == "__main__":
    main()
