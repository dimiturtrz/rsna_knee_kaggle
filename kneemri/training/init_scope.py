"""How much of a checkpoint a warm start takes: all of it, or the trunk without the head.

WHY THERE IS A CHOICE AT ALL. LP-FT's two stages are the same corpus twice, so stage one's head IS the
head stage two wants and taking anything less would throw away the fitted layer the method is named for.
A cross-corpus pretrain is the other case: `mrnet_acl_pretrain` fits a trunk on 1248 MRNet exams whose
three series map to three slots, and this corpus has six. `head.slot_emb` is (3, 256) there and (6, 256)
here, `head.prior` is (12, 3) against (12, 6), and a strict load raises on both — which is CORRECT, and
was the whole failure: the run died after stage one had already spent its card.

The head is not the part that transfers anyway. The published mechanism for a domain-matched pretrain
with partial label overlap is feature reuse in the early and middle blocks (Yosinski 2014; Raghu 2019;
Matsoukas CVPR 2022), and a slot embedding fitted over a different corpus's series layout is not a
representation of anything this corpus has. Dropping it is the honest reading, not a workaround for a
shape error — which is why it is a NAMED scope on the command line and not a silently lenient load.
"""

from abc import ABC, abstractmethod
from typing import override

from torch import Tensor

HEAD = "head."
"""The parameter-name prefix of everything downstream of the trunk — the part a foreign corpus fits."""


class InitScope(ABC):
    """Which of a checkpoint's tensors a run starts from."""

    @staticmethod
    def named(name: str) -> "InitScope":
        """The scope `--init-scope` asked for; raises on a name that does not exist, before the card."""
        return INIT_SCOPES[name]()

    @abstractmethod
    def carried(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """The tensors taken from the checkpoint — every one still loaded strictly by shape."""

    @abstractmethod
    def dropped(self, state: dict[str, Tensor]) -> set[str]:
        """The names deliberately left cold, so a load can tell them from a checkpoint that is missing them."""


class WholeModel(InitScope):
    """Everything the checkpoint holds: the two-stage same-corpus case, where the head is the point."""

    @override
    def carried(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        return state

    @override
    def dropped(self, state: dict[str, Tensor]) -> set[str]:
        return set()


class TrunkOnly(InitScope):
    """The encoder without the head: the cross-corpus case, where the head is fitted to a foreign layout."""

    @override
    def carried(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        return {name: tensor for name, tensor in state.items() if not name.startswith(HEAD)}

    @override
    def dropped(self, state: dict[str, Tensor]) -> set[str]:
        return {name for name in state if name.startswith(HEAD)}


INIT_SCOPES: dict[str, type[InitScope]] = {"all": WholeModel, "trunk": TrunkOnly}
"""The scopes `--init-scope` may name — dispatch, so an unknown name raises rather than defaulting."""
