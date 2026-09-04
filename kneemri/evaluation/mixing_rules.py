"""How one finding's column is rebuilt out of all twelve. The strategy family `column_mixing` measures.

Both rules answer the same question — given the twelve rank-normalised prediction columns, what row of
coefficients should produce finding `j`? — and both are written so the SHIPPED table sits at one end of
their own penalty path. That is not a stylistic preference: a comparison whose control cannot be selected
is not a comparison, and the first version of this work shipped exactly that mistake (see
`column_mixing`'s docstring, and the interpretation it points at).

`RidgeShrink` is the dense L2 form: every one of the 144 coefficients is free, and the penalty walks the
row back toward the identity. `SparseGraph` is the published form — Srivastava et al.'s Label-Graph
Refinement, `z' = z + alpha * sigma(z) * A` with `A` zero-diagonal and L1-regularised, evaluated on macro
AUC — reduced to its post-hoc case: the own column is kept at weight one and never penalised, and the
eleven cross terms are fitted under L1 so most of them can be exactly zero.

WHY BOTH, AND WHY THE SECOND ONE EXISTS. The dense L2 rule came back a clean null. That result bounds
DENSE mixing and nothing else: at 909 studies with 132 free cross terms, L2 spreads a little weight over
every pair, which is how a ridge behaves when almost all pairs are noise and one or two are real. L1 is
the estimator that can keep the real pairs and set the rest to zero, and it is the one the published
medical multi-label work actually uses. Reporting the L2 null as a verdict on cross-column structure
would have been a statement about the estimator dressed as a statement about the mechanism.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, override

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray
from sklearn.linear_model import Lasso


class MixingRule(ABC):
    """One finding's row of coefficients over the twelve columns, at a given penalty."""

    name: ClassVar[str]
    penalties: ClassVar[tuple[float, ...]]

    @abstractmethod
    def row(
        self,
        design: Float[NDArray[np.float64], "studies findings"],
        target: Int[NDArray[np.int64], " studies"],
        penalty: float,
        column: int,
    ) -> Float[NDArray[np.float64], " findings"]:
        """The coefficients that rebuild `column`, with the identity reachable as `penalty` grows."""


class RidgeShrink(MixingRule):
    """Dense L2: all twelve coefficients free, shrunk toward the identity rather than toward zero."""

    name = "ridge"
    penalties = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)

    @override
    def row(
        self,
        design: Float[NDArray[np.float64], "studies findings"],
        target: Int[NDArray[np.int64], " studies"],
        penalty: float,
        column: int,
    ) -> Float[NDArray[np.float64], " findings"]:
        """Regress what the own column still gets wrong on all twelve, then add the own column back.

        Shrinking toward zero instead would walk the row to a correlation-weighted average of all twelve
        columns — the target's own signal DILUTED — which is not the table we ship and is therefore not a
        control. The symptom of getting this wrong is that a penalty sweep gets monotonically worse
        instead of converging on the baseline.
        """
        residual = target.astype(np.float64) - design[:, column]
        gram = design.T @ design + penalty * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ residual).astype(np.float64)
        weights[column] += 1.0
        return weights


class SparseGraph(MixingRule):
    """L1 over the ELEVEN cross terms, own column pinned at one: the published label-graph form, post hoc.

    Zero diagonal is what makes the coefficient count 132 rather than 144, and it is also what makes the
    rule safe: the shipped column is never scaled, only added to, so at a penalty large enough to zero
    every cross term the output IS the shipped column. L1 rather than L2 because the prior is that a
    couple of finding pairs are genuinely coupled (Fracture and Contusion are the same injury described
    two ways) and the other hundred-odd are not — a prior L2 cannot express.
    """

    name = "sparse"
    penalties = (0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0)

    @override
    def row(
        self,
        design: Float[NDArray[np.float64], "studies findings"],
        target: Int[NDArray[np.int64], " studies"],
        penalty: float,
        column: int,
    ) -> Float[NDArray[np.float64], " findings"]:
        """Lasso the residual on the other eleven columns; the own column keeps weight one, unpenalised."""
        others = [index for index in range(design.shape[1]) if index != column]
        residual = target.astype(np.float64) - design[:, column]
        fitted = Lasso(alpha=penalty, fit_intercept=True, max_iter=5000).fit(design[:, others], residual)
        weights = np.zeros(design.shape[1])
        weights[others] = fitted.coef_
        weights[column] = 1.0
        return weights


RULES: dict[str, MixingRule] = {rule.name: rule for rule in (RidgeShrink(), SparseGraph())}
"""The rules by name, for the CLI. Declaration-only dispatch, not a bucket module."""
