"""The local selection objective: fitted log likelihood plus a coding regularizer.

regularizer(panel, f) = -log C(2^L, free) - claimed_code - log K - log(K + 1)
                        + (1/K) sum_a log(K f_a)
The dictionary term codes the ``free`` rows as new arbitrary binary haplotypes; rows claimed by
the assembled context's founders are coded by reference to their founder plus their edits
(``ContextFounders``). The frequency term is a shrinkage penalty centred at uniform frequencies
(zero there). This is a regularised best-panel objective, not a Bayesian posterior over K.
"""
from __future__ import annotations

import math

import numpy as np

from ..discovery.panel import log_haplotype_set_count


def regularizer(panel, frequencies, free=None, claimed_code=0.0) -> float:
    k, length = panel.shape
    free = k if free is None else int(free)
    dictionary = float(log_haplotype_set_count(length, free)) if free else 0.
    value = -dictionary - float(claimed_code) - math.log(k) - math.log(k + 1)
    # Shrinkage centred at uniform; its MAP extra counts total one.
    value += float(np.log(k * frequencies).sum() / k)
    return value


class ContextFounders:
    """The assembled context's founders in one block, for the dictionary code.

    ``founders`` holds the distinct local sequences of the context founders (this round's
    context proposals; -1 = no call). A founder claims its nearest panel row on its called sites
    (ties: the lowest row); a claimed row is coded by reference to its founder plus its edits,
    log C(n_f, d) for d differing sites among the founder's n_f called sites (the smallest over
    the row's claimers, and never more than coding it as one more free row), so a claimed row
    equal to a proposal costs nothing. Rows no founder claims are free. Founders without a
    called site claim no row. Distances are cached per distinct row.
    """

    def __init__(self, founders):
        founders = np.asarray(founders, dtype=np.int64)
        called = founders >= 0
        informative = called.any(axis=1)
        self.founders = np.ascontiguousarray(founders[informative])
        self.called = np.ascontiguousarray(called[informative])
        self.n_called = self.called.sum(axis=1)
        self.distances: dict[bytes, np.ndarray] = {}

    def __len__(self):
        return len(self.founders)

    def _columns(self, panel):
        columns = []
        for row in np.asarray(panel, dtype=np.int64):
            key = row.tobytes()
            value = self.distances.get(key)
            if value is None:
                value = self.distances[key] = ((self.founders != row[None, :]) & self.called).sum(axis=1)
            columns.append(value)
        return np.stack(columns, axis=1)

    def claims(self, panel):
        """(claimed mask, free-row count, claimed rows' edit code in nats)."""
        k, length = np.shape(panel)
        claimed = np.zeros(k, dtype=bool)
        if not len(self.founders):
            return claimed, k, 0.
        distance = self._columns(panel)
        owner = np.argmin(distance, axis=1)
        claimed[owner] = True
        free = k - int(claimed.sum())
        # One more free row would cost log C(2^L, free + 1) - log C(2^L, free).
        free_row = float(log_haplotype_set_count(length, free + 1)) - (
            float(log_haplotype_set_count(length, free)) if free else 0.)
        code = 0.
        for row in np.flatnonzero(claimed):
            claimers = np.flatnonzero(owner == row)
            edits = min(math.lgamma(self.n_called[f] + 1) - math.lgamma(distance[f, row] + 1)
                        - math.lgamma(self.n_called[f] - distance[f, row] + 1) for f in claimers)
            code += min(edits, free_row)
        return claimed, free, code

    def free(self, panel) -> int:
        return self.claims(panel)[1]


def coded_regularizer(panel, frequencies, context: ContextFounders | None) -> float:
    """The regularizer with context coding (every row free without a context)."""
    if context is None:
        return regularizer(panel, frequencies)
    _, free, code = context.claims(panel)
    return regularizer(panel, frequencies, free, code)


def selection_objective(fit, context: ContextFounders | None) -> float:
    """A fit's selection objective: its own objective without a context, else its log
    likelihood plus the context-coded regularizer."""
    if context is None:
        return fit.objective
    return fit.log_likelihood + coded_regularizer(fit.panel, fit.frequencies, context)
