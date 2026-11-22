"""Pure resident components for certified M1 queries; stages own scheduling.
Method: docs/methods/pedigree.md#candidate-panel
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ...contracts.pedigree import ParentEligibility, PreparedChromosome
from . import forward, read_score, resident
from .parameters import PedigreeParameters
from .selective_bounds import Geometry


def _claim() -> None:
    pass


# EXACT_CERTIFICATE: with the read state the screen's certificate of a pair is its exact M1. The class-carrying
# group bound cost a third to a half of an exact pass over every eligible pair, and the screen queried 85-100 % of
# the eligible chromosome pairs anyway on every simulated validation cohort: scoring each
# eligible pair once and answering the queries from those scores is cheaper. The screen's record does not depend on
# which valid certificates it reads (the best, cutoff, retained pairs, local maxima and floors are exact-score
# quantities), so the choice changes no value.
@dataclass
class Component:
    resident: resident.ResidentComponent
    geometry: Geometry | None
    zero: np.ndarray
    cache_range: tuple[int, int] | None = None
    products: np.ndarray | None = None
    powered: np.ndarray | None = None
    columns: tuple[np.ndarray, np.ndarray] | None = None
    classes: np.ndarray | None = None
    classes_range: tuple[int, int] | None = None
    scored: tuple[tuple[int, int], np.ndarray, np.ndarray] | None = None
    claim: Callable[[], None] = field(default=_claim, repr=False)

    def bound(self, children: np.ndarray, span: tuple[int, int] | None = None,
              eligible: np.ndarray | None = None) -> np.ndarray:
        """Upper bounds of ``children`` [len(children), N]; ``span`` the batch's child range [first, stop) (default:
        the children's), ``eligible`` bool [len(children), N] the pairs the screen may query (default: all; the others
        are +inf). With the read state the bound is the exact M1 of every eligible pair (EXACT_CERTIFICATE), which the
        batch's queries then read."""
        self.cache_range = self.products = self.powered = self.columns = self.classes = self.scored = None
        p, t = self.resident.projection, self.resident.terms
        if t.n_classes > 1:
            first, stop = span if span is not None else (
                (int(np.min(children)), int(np.max(children)) + 1) if len(children) else (0, 0))
            mask = (np.ones((len(children), len(p.available)), dtype=bool) if eligible is None
                    else np.asarray(eligible, dtype=bool))
            rows, candidates = np.nonzero(mask)
            values = np.full(mask.shape, np.inf)
            if len(rows):
                pairs = np.column_stack((np.asarray(children, dtype=np.int64)[rows], candidates)).astype(np.int64)
                values[rows, candidates] = self.exact(pairs, first, stop)
            position = np.full(stop - first, -1, dtype=np.int64)
            position[np.asarray(children, dtype=np.int64) - first] = np.arange(len(children))
            self.scored = ((first, stop), position, values)
            return values.copy()
        return self.geometry.upper(p, t, children, self.zero, self.claim)

    def exact(self, pairs: np.ndarray, first: int, stop: int) -> np.ndarray:
        p, t = self.resident.projection, self.resident.terms
        if self.scored is not None and self.scored[0] == (first, stop):
            _, position, values = self.scored
            rows = position[pairs[:, 0] - first]
            if np.all(rows >= 0) and np.all(np.isfinite(values[rows, pairs[:, 1]])):
                return values[rows, pairs[:, 1]]
        if self.cache_range != (first, stop):
            self.claim()
            self.products, self.powered = t.common_null_second(p, first, stop)
            self.columns = (
                forward.named_column_powers(self.products, t.exponent, first)
                if self.products is not None and self.powered is not None
                else None
            )
            if self.classes is None or self.classes_range != (first, stop):
                self.classes = t.class_cache(p, first, stop)
                self.classes_range = (first, stop)
            self.cache_range = (first, stop)
        return read_score.component_m1(
            p, t, pairs, self.zero, first, self.products, self.powered, self.claim, self.columns, self.classes
        )


def prepare(
    prepared: PreparedChromosome,
    eligibility: ParentEligibility,
    config: PedigreeParameters,
    claim: Callable[[], None] = _claim,
    *,
    reverse: bool | None = None,
) -> list[Component]:
    result = []
    for data in prepared.components:
        state = resident.resident(data, config, claim, reverse=reverse, include_pair_masses=False)
        p, t = state.projection, state.terms
        zero = read_score.component_m0(p, t, eligibility.children, claim)
        # the group certificates serve the scalar chains; with the read state the certificate is the exact M1
        geometry = Geometry(data, p, eligibility.parents.any(0), claim) if t.n_classes == 1 else None
        result.append(Component(state, geometry, zero, claim=claim))
    return result
