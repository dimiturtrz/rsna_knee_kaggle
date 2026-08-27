"""Per-diagnosis attention over the acquisition slots of one study."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

_PRESENT = 0.5
"""Midpoint of the {0, 1} presence mask — a slot below it is absent and leaves the softmax."""

_ABSENT_LOGIT = -1e4
"""What an absent slot's attention logit becomes: negligible after softmax, finite in fp16."""

REPORT_DIMS = 64
"""Width of the report-embedding target — see `kneemri.embeddings.report_embeddings` for why it is reduced."""

_HIDDEN = 256
"""Width the slot embeddings are projected to before the per-diagnosis attention."""

_SHARED_READING = 3
"""Rank of slot features that every diagnosis reads identically — "b s d", with no diagnosis axis."""

_DROPOUT = 0.2
"""Applied to the attended context, where 4,407 studies meet a twelve-way readout."""


class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences — cruciates sagittally, collateral ligaments and the
    meniscal body coronally, patellar cartilage axially — so pooling the slots identically would dilute
    the one carrying the evidence with the five that do not. Every diagnosis therefore gets its own
    query, and absent slots are masked out of the softmax so it renormalises over what the study
    actually contains rather than being fed a zero vector.

    Which sequence shows which finding is supplied as a prior on the attention logits rather than
    learned from scratch — see `anatomy`. The supervision here is one label vector per study and never
    says where in the study the finding was seen, so learning the mapping costs signal that is better
    spent elsewhere, and the mapping itself is not in doubt.

    The aggregation is deliberately this small. The label is attached to the study, so nothing in the
    supervision says which part of a study carries the finding, and a richer aggregator would have no
    signal to learn that from. Where the supervision is coarse, so is the head.
    """

    def __init__(
        self,
        dim: int,
        n_slot: int,
        n_out: int,
        prior: Float[Tensor, "o s"] | None = None,
        report_dims: int = REPORT_DIMS,
    ) -> None:
        super().__init__()
        hidden, p = _HIDDEN, _DROPOUT
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        # A parameter, not a buffer: the anatomy is where the query should START looking, and the data
        # is allowed to overrule it. Zeros when absent, so the prior costs nothing when it is not used.
        self.prior = nn.Parameter(torch.zeros(n_out, n_slot) if prior is None else prior.clone())
        # Predicts the study's REPORT embedding from the same slot features the diagnoses read. It is
        # always built and costs one small Linear, so no call site has to ask whether report
        # supervision is switched on; an unused head simply receives no gradient.
        self.report = nn.Linear(hidden, report_dims)

    def per_diagnosis(self, x: Float[Tensor, "b s *o d"]) -> Float[Tensor, "b s o d"]:
        """The slot features AS EACH DIAGNOSIS SEES THEM — one reading shared by all twelve, or twelve.

        The head reads a diagnosis axis whether or not the slice pooling produced one. `channels`,
        `max`, `attend` and `depth` all hand up a single vector per slot, which every finding then reads
        identically; `labelwise` pools the slices once per finding and hands up twelve. Expanding the
        first case is a VIEW and changes no arithmetic — the einsums below contract the same numbers in
        the same order — so every member on disk computes exactly what it computed before.

        The one place that knows the two shapes, so nothing downstream has to ask which pooling ran.
        """
        return x.unsqueeze(2).expand(-1, -1, self.query.shape[0], -1) if x.dim() == _SHARED_READING else x

    def attention(self, h: Float[Tensor, "b s o h"], mask: Float[Tensor, "b s"]) -> Float[Tensor, "b o s"]:
        """How much each diagnosis reads from each slot: a per-diagnosis softmax over the present ones."""
        att = torch.einsum("bsoh,oh->bos", h, self.query) / self.hidden**0.5
        att = att + self.prior.unsqueeze(0)
        return att.masked_fill(mask.unsqueeze(1) < _PRESENT, _ABSENT_LOGIT).softmax(-1)

    def context(self, x: Float[Tensor, "b s *o d"], mask: Float[Tensor, "b s"]) -> Float[Tensor, "b o h"]:
        """What each diagnosis has read from the study, one vector per finding, before it becomes a logit.

        Split out of `forward` rather than recomputed, because it is the LAST representation that still
        has an axis a loss can shape: `out` collapses each of these vectors onto a single learned
        direction, and everything a contrastive term wants to say about "torn looks unlike intact" is
        about the vector, not about its projection onto that one direction. `forward` composes the two,
        so every member on disk computes exactly what it computed before.
        """
        h = self.proj(self.per_diagnosis(x)) + self.slot_emb.unsqueeze(1)
        return self.drop(torch.einsum("bos,bsoh->boh", self.attention(h, mask), h))

    def logits(self, ctx: Float[Tensor, "b o h"]) -> Float[Tensor, "b o"]:
        """Each diagnosis's context read out by its own row of the readout — a per-finding dot product."""
        readout = self.out
        return (ctx * readout.weight.unsqueeze(0)).sum(-1) + readout.bias

    @override
    def forward(self, x: Float[Tensor, "b s *o d"], mask: Float[Tensor, "b s"]) -> Float[Tensor, "b o"]:
        """Attend each diagnosis's query over the present slots and read out one logit per diagnosis."""
        return self.logits(self.context(x, mask))

    def report_vector(self, x: Float[Tensor, "b s *o d"], mask: Float[Tensor, "b s"]) -> Float[Tensor, "b r"]:
        """Predict the study's report embedding from the mean of its PRESENT slots.

        Averaged over the present slots rather than over the per-diagnosis contexts: a report describes
        the whole study, so its target is a property of the study and not of any one finding's
        attention. Masked, for the reason the attention is masked — an absent acquisition must not pull
        the average toward whatever an unfilled slot happens to hold.

        Averaged over the DIAGNOSIS axis too, and for the same reason: under `labelwise` pooling a slot
        holds twelve readings, and a report describes the study rather than any one finding's view of
        it. Under every other pooling the twelve are one vector expanded, so the mean is that vector and
        this line changes nothing for any member on disk.
        """
        h = self.proj(self.per_diagnosis(x)).mean(dim=2) + self.slot_emb
        present = (mask >= _PRESENT).float().unsqueeze(-1)
        pooled = (h * present).sum(dim=1) / present.sum(dim=1).clamp(min=1.0)
        return self.report(pooled)


class MeanSlotHead(SlotHead):
    """The same head with its per-diagnosis attention replaced by a uniform mean over present slots.

    THE DENOMINATOR, not a proposal. Every arm this project has run is measured against a recipe that
    already contains slot attention, so nothing says what the attention itself carries — and that number
    sets the bar for every future pooling idea. A new scheme beating mean-pooling by 0.01 is
    uninteresting if attention already beats it by 0.05, and it is the whole game if attention is worth
    0.002.

    IT IS THE MECHANISM'S LIMIT AND NOT A THIRD MODE, which is the trap this ablation was written to
    avoid: an ablation knob at zero that lands between two behaviours produces a number that reads as a
    finding and is an artefact. Only `attention` changes — the projection, the slot embeddings, the
    dropout, the readout and the parameter count are identical — so with constant attention logits the
    two heads compute the same function, and `tests/unit/kneemri/slot_head.py` asserts exactly that.
    """

    @override
    def attention(self, h: Float[Tensor, "b s o h"], mask: Float[Tensor, "b s"]) -> Float[Tensor, "b o s"]:
        """Uniform over the slots the study actually has, and zero on the ones it does not.

        Masked for the same reason the softmax is masked: an absent acquisition must not pull the
        average toward whatever an unfilled slot happens to hold. A study with no present slots at all
        would divide by zero, so the denominator is floored — it cannot occur in the corpus, and a
        silent NaN through a loss is not the way to discover that it has.
        """
        present = (mask >= _PRESENT).float()
        weights = present / present.sum(dim=1, keepdim=True).clamp(min=1.0)
        return weights.unsqueeze(1).expand(-1, self.query.shape[0], -1)


ATTENDED = "attend"
"""Per-diagnosis attention over the slots — what every member on disk was fitted with."""

AVERAGED = "mean"
"""The ablation: a uniform mean, so the attention has a denominator."""

SLOT_POOLINGS: dict[str, type[SlotHead]] = {ATTENDED: SlotHead, AVERAGED: MeanSlotHead}
"""The vocabulary AND the dispatch, in one object beside the heads it names.

A name a member records is a head this builds, and there is no third place for the two to disagree. The
dispatch was `MeanSlotHead if slot_pooling == "mean" else SlotHead`, which gave the attention head to a
typo and to every future pooling — the same shape as the slice-axis bug `StudyDesign.per_slice`
documents, where three sites branched on one word and adding a pooling left one of them refusing it.
An unknown name now raises here instead of quietly fitting a recipe nobody asked for."""
