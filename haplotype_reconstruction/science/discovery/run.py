"""Build one discovery block from its founder-view evidence in memory."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.likelihoods import LikelihoodSlice, gather_likelihoods, slice_likelihoods
from ...contracts.observations import PairArray, dense_observed, slice_observed
from ...contracts.roles import record_field
from ...core.concurrency import ThreadSource
from ...core.errors import InvariantError
from ...core.ties import inverse_permutation, run_sample_order
from .diagnostics import BlockDiagnostics
from .panel import FixedKSolution, restricted
from .parameters import DiscoveryParameters
from .release import DiscoveredBlock, materialise
from .search import BlockSearch, _search


@dataclass(frozen=True)
class BlockEvidence:
    """The founder-view evidence of one discovery block.

    block, site_start: block index and its first contig site; positions: int64[L];
    kept: bool[L] discovery sites; likelihoods: float64[N, L, 3] normalised GLs (zero depth
    exactly 1/3); observed: bool[N, L] positive depth (False for excluded samples).
    sample_ids: the samples' names: the search runs in the run's canonical
    sample order (core.ties.run_sample_order), so its result does not depend on the input order.
    None means the evidence is already in canonical order.
    """

    block: int = record_field("obs", axes=(), units="block", dtype="int")
    site_start: int = record_field("obs", axes=(), units="site", dtype="int")
    positions: np.ndarray = record_field("obs", axes=("L",), units="bp", dtype="int64")
    kept: np.ndarray = record_field("obs", axes=("L",), units="flag", dtype="bool")
    likelihoods: np.ndarray | LikelihoodSlice = record_field(
        "evidence", axes=("N", "L", "3"), units="probability", dtype="float64"
    )
    observed: np.ndarray | PairArray = record_field("obs", axes=("N", "L"), units="flag", dtype="bool")
    sample_ids: tuple[str, ...] | None = record_field(
        "obs", axes=("N",), units="identity", dtype="tuple[str]|None", default=None
    )

    @property
    def informative(self) -> bool:
        """At least one discovery site is observed in at least one sample."""
        return bool(self.kept.any() and self.observed[:, self.kept].any())


def block_evidence(evidence: ContigEvidence, block: int) -> BlockEvidence:
    """Views (no copies) of ``evidence`` (a founder-view ``ContigEvidence``) on one block."""
    start, stop = (int(value) for value in evidence.block_spans[block])
    return BlockEvidence(
        block=int(block),
        site_start=start,
        positions=evidence.positions[start:stop],
        kept=evidence.discovery_sites[start:stop],
        likelihoods=slice_likelihoods(evidence.likelihoods, start, stop),
        observed=slice_observed(evidence.observed, start, stop),
        sample_ids=tuple(evidence.sample_ids),
    )


def _sample_subset(solution: FixedKSolution, index: np.ndarray) -> FixedKSolution:
    """Permute per-sample state back to input order while preserving the fitted NLL."""
    return replace(restricted(solution, index), nll=solution.nll)


def search_block(
    likelihoods: np.ndarray,
    observed: np.ndarray,
    config: DiscoveryParameters,
    *,
    sample_ids: tuple[str, ...] | None = None,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
    diagnostics: BlockDiagnostics | None = None,
    threads: ThreadSource = 1,
    map_work=map,
) -> BlockSearch:
    """Search neutralised float64[N,L,3] evidence in the explicit canonical sample order.

    Input arrays are unchanged. Per-sample solution fields return in input order;
    NLL retains the canonical-order calculation. Mismatched sample names are an
    invariant failure. Convention changes content ordering, never usage schedules.
    """
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    likelihood = np.array(likelihoods, dtype=np.float64, order="C", copy=True)
    likelihood[~observed] = 1.0 / 3.0
    order = None
    if sample_ids is not None:
        if len(sample_ids) != likelihood.shape[0]:
            raise InvariantError("sample_ids must name every sample of the block")
        order = run_sample_order(sample_ids, reverse=reverse)
        if np.array_equal(order, np.arange(len(order))):
            order = None
    if order is not None:
        likelihood = np.ascontiguousarray(likelihood[order])
        observed = np.ascontiguousarray(observed[order])
    result = _search(
        likelihood,
        observed,
        config,
        reverse=reverse,
        keep=keep,
        diagnostics=diagnostics,
        threads=threads,
        map_work=map_work,
    )
    if order is None or result.selected is None:
        return result
    inverse = inverse_permutation(order)
    return BlockSearch(
        selected=_sample_subset(result.selected, inverse),
        summary=result.summary,
        alternatives=tuple(_sample_subset(panel, inverse) for panel in result.alternatives),
    )


def discover_block(
    evidence: BlockEvidence,
    config: DiscoveryParameters,
    *,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
    diagnostics: BlockDiagnostics | None = None,
    threads: ThreadSource = 1,
    map_work=map,
) -> DiscoveredBlock | None:
    """Search one founder-view block and release alleles; None means uninformative.

    The input is unchanged. Returned arrays belong to this result. Configuration,
    thread team and convention are explicit; no resources or clocks are consulted.
    ``map_work`` is ordered, with one native thread per concurrent cavity candidate.
    """
    if not evidence.informative:
        return None
    likelihoods = (
        evidence.likelihoods
        if isinstance(evidence.likelihoods, np.ndarray)
        else gather_likelihoods(evidence.likelihoods, slice(None))
    )
    kept = evidence.kept
    observed = dense_observed(evidence.observed)
    result = search_block(
        np.ascontiguousarray(likelihoods[:, kept]),
        np.ascontiguousarray(observed[:, kept]),
        config,
        sample_ids=evidence.sample_ids,
        reverse=reverse,
        keep=keep,
        diagnostics=diagnostics,
        threads=threads,
        map_work=map_work,
    )
    return materialise(
        evidence.block,
        evidence.site_start,
        evidence.positions,
        kept,
        likelihoods,
        observed,
        result,
        config,
        diagnostics,
    )


__all__ = ["BlockEvidence", "block_evidence", "discover_block", "search_block"]
