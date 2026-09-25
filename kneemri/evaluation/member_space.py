"""The members as POINTS — the feature space every member-distance ruler is taken in.

Its own module because it is its own subject: the rulers argue about how to measure a distance, and
this argues about what the coordinates ARE. Keeping it separate is also what makes the seam real — an
embedding dump (knee-iyad's stated ideal, a forward pass per member over the holdout) replaces this
class and touches no ruler.
"""

import numpy as np
from jaxtyping import Float

from kneemri.evaluation.pool import Pool
from kneemri.inference.vote import Vote


class MemberSpace:
    """The members as points, one row each, in the space they are actually combined in.

    Per-label rank normalisation rather than raw probabilities, for the reason the pool's vote rule
    rank-averages: a member that is well ordered and badly scaled is a good member, and a Euclidean or
    Mahalanobis distance over raw probabilities would price its calibration instead of its opinion.
    """

    def __init__(self, pool: Pool) -> None:
        self.pool = pool

    def matrix(self) -> Float[np.ndarray, "members features"]:
        """Every member as one flattened rank vector, centred on the pool mean.

        Centring is what makes the covariance downstream the covariance OF THE POOL rather than of the
        origin, which is the whole claim: distance is measured against how this set of members varies.
        """
        rows = np.stack([Vote.ranks(member).ravel() for member in self.pool.predictions])
        return rows - rows.mean(axis=0, keepdims=True)
