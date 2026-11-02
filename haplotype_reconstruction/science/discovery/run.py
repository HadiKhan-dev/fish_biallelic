"""Build one discovery block from its founder-view evidence in memory."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...contracts.evidence import ContigEvidence
from ...contracts.roles import record_field
from ...core.errors import InvariantError
from ...core.ties import inverse_permutation, run_sample_order
from .diagnostics import BlockDiagnostics
from .panel import FixedKSolution, restricted
from .parameters import DiscoveryConfig
from .release import DiscoveredBlock, materialise
from .search import BlockSearch, _search


@dataclass(frozen=True)
class BlockEvidence:
    """The founder-view evidence of one discovery block.

    block, site_start: block index and its first contig site; positions: int64[L];
    kept: bool[L] discovery sites; likelihoods: float64[N, L, 3] normalised GLs (zero depth
    exactly 1/3); observed: bool[N, L] positive depth (False for excluded samples).
    sample_ids: the samples' names (release 2.1; optional): the search runs in the run's canonical
    sample order (maths.ties.run_sample_order), so its result does not depend on the input order.
    None means the evidence is already in canonical order.
    """

    block: int = record_field("obs", axes=(), units="block", dtype="int")
    site_start: int = record_field("obs", axes=(), units="site", dtype="int")
    positions: np.ndarray = record_field("obs", axes=("L",), units="bp", dtype="int64")
    kept: np.ndarray = record_field("obs", axes=("L",), units="flag", dtype="bool")
    likelihoods: np.ndarray = record_field("evidence", axes=("N", "L", "3"), units="probability", dtype="float64")
    observed: np.ndarray = record_field("obs", axes=("N", "L"), units="flag", dtype="bool")
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
        likelihoods=evidence.likelihoods[:, start:stop],
        observed=evidence.observed[:, start:stop],
        sample_ids=tuple(evidence.sample_ids),
    )


def _sample_subset(solution: FixedKSolution, index: np.ndarray) -> FixedKSolution:
    """Permute per-sample state back to input order while preserving the fitted NLL."""
    return replace(restricted(solution, index), nll=solution.nll)


def search_block(
    likelihoods: np.ndarray,
    observed: np.ndarray,
    config: DiscoveryConfig,
    *,
    sample_ids: tuple[str, ...] | None = None,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
    diagnostics: BlockDiagnostics | None = None,
    threads: int = 1,
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
    result = _search(likelihood, observed, config, reverse=reverse, keep=keep, diagnostics=diagnostics, threads=threads)
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
    config: DiscoveryConfig,
    *,
    reverse: bool = False,
    keep: tuple[str, ...] = (),
    diagnostics: BlockDiagnostics | None = None,
    threads: int = 1,
) -> DiscoveredBlock | None:
    """Search one founder-view block and release alleles; None means uninformative.

    The input is unchanged. Returned arrays belong to this result. Configuration,
    thread team and convention are explicit; no resources or clocks are consulted.
    """
    if not evidence.informative:
        return None
    kept = evidence.kept
    result = search_block(
        np.ascontiguousarray(evidence.likelihoods[:, kept]),
        np.ascontiguousarray(evidence.observed[:, kept]),
        config,
        sample_ids=evidence.sample_ids,
        reverse=reverse,
        keep=keep,
        diagnostics=diagnostics,
        threads=threads,
    )
    return materialise(
        evidence.block,
        evidence.site_start,
        evidence.positions,
        kept,
        evidence.likelihoods,
        evidence.observed,
        result,
        config,
        diagnostics,
    )


__all__ = ["BlockEvidence", "block_evidence", "discover_block", "search_block"]
