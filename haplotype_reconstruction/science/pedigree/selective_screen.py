"""Pure resident components for certified M1 queries; stages own scheduling.
Method: docs/methods/pedigree.md#candidate-panel
PE1–PE3 preserve this 3.5 method.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ...contracts.pedigree import ParentEligibility, PreparedChromosome
from . import forward, resident
from .parameters import PedigreeConfig
from .selective_bounds import Geometry


def _claim() -> None:
    pass


@dataclass
class Component:
    resident: resident.ResidentComponent
    geometry: Geometry
    zero: np.ndarray
    cache_range: tuple[int, int] | None = None
    products: np.ndarray | None = None
    powered: np.ndarray | None = None
    columns: tuple[np.ndarray, np.ndarray] | None = None
    claim: Callable[[], None] = field(default=_claim, repr=False)

    def bound(self, children: np.ndarray) -> np.ndarray:
        self.cache_range = self.products = self.powered = self.columns = None
        return self.geometry.upper(self.resident.projection, self.resident.terms, children, self.zero, self.claim)

    def exact(self, pairs: np.ndarray, first: int, stop: int) -> np.ndarray:
        p, t = self.resident.projection, self.resident.terms
        if self.cache_range != (first, stop):
            self.claim()
            self.products, self.powered = t.common_null_second(p, first, stop)
            self.columns = (
                forward.named_column_powers(self.products, t.exponent, first)
                if self.products is not None and self.powered is not None
                else None
            )
            self.cache_range = (first, stop)
        return forward.component_m1(
            p, t, pairs, self.zero, first, self.products, self.powered, self.claim, self.columns
        )


def prepare(
    prepared: PreparedChromosome,
    eligibility: ParentEligibility,
    config: PedigreeConfig,
    claim: Callable[[], None] = _claim,
    *,
    reverse: bool | None = None,
) -> list[Component]:
    result = []
    for data in prepared.components:
        state = resident.resident(data, config, claim, reverse=reverse, include_pair_masses=False)
        p, t = state.projection, state.terms
        zero = forward.component_m0(p, t, eligibility.children, claim)
        geometry = Geometry(data, p, eligibility.parents.any(0), claim)
        result.append(Component(state, geometry, zero, claim=claim))
    return result
