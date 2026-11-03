"""Release full-marker parent counts after the sibling guard has withdrawn unsafe edges.

Owns P0 PD22 count release, never scores genotype data or adds edges.
Method: docs/methods/pedigree.md#full-marker-mendelian-exclusion-of-m0-and-m1-counts.
"""

from __future__ import annotations

import numpy as np

from ...contracts.pedigree import (
    NO_PARENT,
    UNRESOLVED,
    CountDiagnostic,
    CountExclusion,
    CountRelease,
    CountReleaseBasis,
)
from ...core.errors import InputError
from . import exclusion
from .eligibility import ParentEligibility
from .parameters import PedigreeConfig
from .release import PartialEdges, TierCalls

__all__ = ["target_dyads", "release_counts"]


def target_dyads(tier_b: TierCalls, rows: np.ndarray, eligibility: ParentEligibility) -> np.ndarray:
    """int64 [D, 2] (child, candidate): every eligible candidate of each child whose Tier-B count
    is unresolved (no state, or fewer released parents than the called count)."""
    dyads: list[tuple[int, int]] = []
    for child in np.flatnonzero(eligibility.children):
        row = int(tier_b.row[child])
        named = 0 if row < 0 else int(np.count_nonzero(rows[row, 1:] >= 0))
        state = int(tier_b.state[child])
        if state == UNRESOLVED or named != state:
            dyads.extend((int(child), int(parent)) for parent in np.flatnonzero(eligibility.parents[child]))
    return np.asarray(dyads, dtype=np.int64).reshape(-1, 2)


def release_counts(
    basis: CountReleaseBasis,
    bounds: CountExclusion,
    config: PedigreeConfig,
    *,
    withheld: frozenset = frozenset(),
) -> CountRelease:
    """Release M0/M1 counts that the exclusion bounds and the supported directions settle.

    A target child is released when every eligible candidate was scored, it is safe (eligible,
    informative chromosomes holding enough map units: ``informative`` float64 [N]; no graph conflict) and the
    candidates that survive exclusion
    (log E below log(2 N / alpha)) minus those it is the supported parent of form at most one
    sample, equal to both its supported marginal parents and its complete graph parents.

    ``withheld``: (parent, child) edges the sibling guard withholds (pedigree.guard). The release uses the final
    supported directions: a withheld edge neither removes its child as a reverse candidate of its parent nor counts
    as a supported parent, and a child with a withheld edge is not released (the guard leaves its count unresolved).
    """
    tier_b, partial_b, complete = basis.tier_b, basis.partial_b, basis.complete
    graph_conflict, informative, rows = basis.graph_conflict, basis.informative_units, basis.rows
    eligibility = basis.eligibility
    sample_ids = eligibility.sample_ids
    dyads, log_e, observed_sites, log_e_upper = bounds.dyads, bounds.log_e, bounds.observed_sites, bounds.log_e_upper
    n = len(sample_ids)
    evidence = {(int(c), int(p)): float(e) for (c, p), e in zip(dyads, log_e, strict=False)}
    outgoing: list[set[int]] = [set() for _ in range(n)]
    for child in range(n):
        for slot in range(2):
            if partial_b.supported[child, slot] and (int(partial_b.parents[child, slot]), child) not in withheld:
                outgoing[int(partial_b.parents[child, slot])].add(child)
    threshold = exclusion.log_threshold(n, config.exclusion.alpha)
    if log_e_upper is not None:
        upper_evidence = {(int(c), int(p)): float(e) for (c, p), e in zip(dyads, log_e_upper, strict=False)}
        if np.any((log_e < threshold) & (log_e_upper >= threshold) & (log_e != log_e_upper)):
            raise InputError("unresolved exclusion interval requires exact refinement")
    state = tier_b.state.copy()
    row = tier_b.row.copy()
    status = list(tier_b.status)
    parents = partial_b.parents.copy()
    supported = partial_b.supported.copy()
    exact = partial_b.exact.copy()
    count_minimum = partial_b.count_minimum.copy()
    count_maximum = partial_b.count_maximum.copy()
    partial_state = partial_b.state.copy()
    partial_status = list(partial_b.status)
    release_evidence = list(partial_b.release_evidence)
    released = np.zeros(n, dtype=np.bool_)
    kept_by_child: list[tuple[int, ...]] = [()] * n
    records = []
    name = sample_ids.__getitem__
    for child in sorted({int(c) for c, _ in dyads}):
        candidates = [int(p) for p in np.flatnonzero(eligibility.parents[child])]
        covered = all((child, p) in evidence for p in candidates)
        compatible = {p for p in candidates if not covered or evidence[child, p] < threshold}
        reverse = compatible & outgoing[child]
        survivors = compatible - reverse
        named = {int(p) for p in partial_b.parents[child][partial_b.supported[child]]}
        supported_parents = {p for p in named if (p, child) not in withheld}
        guard_withheld = len(supported_parents) != len(named)
        complete_row = int(complete.row[child])
        selected = set() if complete_row < 0 else {int(p) for p in rows[complete_row, 1:] if p >= 0}
        complete_count = int(complete.state[child])
        safe = bool(
            eligibility.children[child]
            and informative[child] >= config.release.minimum_informative_units
            and not graph_conflict[child]
        )
        concordant = (
            survivors == supported_parents == selected
            and complete.state[child] != UNRESOLVED
            and complete_count == len(survivors)
        )
        is_released = bool(covered and safe and len(survivors) <= 1 and concordant and not guard_withheld)
        excluded = [evidence[child, p] for p in candidates if covered and p not in compatible]
        weakest = min(excluded, default=np.nan)
        lower_min = upper_min = exact_min = None
        if log_e_upper is not None:
            excluded_upper = [upper_evidence[child, p] for p in candidates if covered and p not in compatible]
            exact_min = all(
                evidence[child, p] == upper_evidence[child, p] for p in candidates if covered and p not in compatible
            )
            lower_min, upper_min = weakest, min(excluded_upper, default=np.nan)
            if not exact_min:
                weakest = np.nan
        records.append(
            CountDiagnostic(
                child,
                len(candidates),
                tuple(sorted(compatible, key=name)),
                tuple(sorted(reverse, key=name)),
                tuple(sorted(survivors, key=name)),
                weakest,
                threshold,
                covered,
                is_released,
                bool(reverse),
                lower_min,
                upper_min,
                exact_min,
            )
        )
        if not is_released:
            continue
        kept = tuple(sorted(survivors, key=name))
        released[child] = True
        kept_by_child[child] = kept
        state[child] = len(kept)
        row[child] = complete_row
        status[child] = "tier_b_fixed_genome_directed_exclusion"
        partial_state[child] = len(kept)
        partial_status[child] = status[child]
        release_evidence[child] = "mendelian_exclusion_and_supported_direction"
        parents[child] = (kept[0] if kept else NO_PARENT, NO_PARENT)
        supported[child] = parents[child] >= 0
        exact[child] = True
        count_minimum[child] = count_maximum[child] = len(kept)
    final = TierCalls(
        state,
        row,
        tuple(status),
        tier_b.state_pass,
        tier_b.identity_pass,
        tier_b.orientation_withheld,
        tier_b.parent_flags,
    )
    final_partial = PartialEdges(
        parents,
        supported,
        exact,
        count_minimum,
        count_maximum,
        partial_state,
        tuple(partial_status),
        tuple(release_evidence),
    )
    return CountRelease(
        final, final_partial, released, tuple(kept_by_child), tuple(records), dyads, log_e, observed_sites, log_e_upper
    )
