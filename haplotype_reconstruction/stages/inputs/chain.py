"""Retain identity-bound selection and assembly inputs on their contig worker."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...core.errors import SettingsError

_CHAIN_INPUTS = "chain-inputs"


def _chain_input(key: str, load: Callable[[], Any]) -> Any:
    from ...engine.workers import resident_or_load

    return resident_or_load(f"{_CHAIN_INPUTS}/{key}", load)


def _chain_evidence(reader, store, contig: str) -> tuple[Any, Any, Any, Any]:
    """Cache count-free evidence and depth keys; defer assembly arrays to their consumer."""
    import numpy as np

    from ...contracts.evidence import ContigEvidence
    from ...science.assembly.evidence import DEPTH_ROWS, AssemblyEvidenceSource, depth_row_keys
    from ..evidence import records as evidence_records

    request = evidence_records.result_reader(store, reader.manifest, product="evidence", contig=contig)
    if request is None:
        raise SettingsError(f"{contig}: selection and assembly need complete evidence")

    def load() -> tuple[Any, Any, Any]:
        evidence = reader.read(request, contig)
        if not isinstance(evidence, ContigEvidence):
            raise SettingsError(f"{contig}: selection and assembly need complete evidence")
        founder = evidence.founder_view()
        # The keys consume counts once. Decode DEPTH_ROWS sample rows at a time rather than retaining an
        # N x L x 2 count buffer for the rest of the chain. Hash independent rows on spare threads.
        from ...engine.cpu import CpuBudget, spare_threads

        columns = np.arange(evidence.n_sites, dtype=np.int64)
        keys = []
        map_work = CpuBudget.fixed(spare_threads(DEPTH_ROWS)).thread_map
        with evidence_records.open_arrays(
            store.contig(contig), ("counts",), expected_identity=request.identity
        ) as arrays:
            if arrays is None:
                raise SettingsError(f"{contig}: assembly depth keys need evidence counts")
            for first in range(0, evidence.n_samples, DEPTH_ROWS):
                stop = min(evidence.n_samples, first + DEPTH_ROWS)
                rows = arrays["counts"].gather(slice(first, stop), columns)
                observed = np.asarray(founder.observed[first:stop])
                keys.extend(depth_row_keys(rows, observed, founder.discovery_sites, map_work=map_work))
                del rows
        return evidence, founder, tuple(keys)

    items = ",".join(getattr(request, "items", None) or ())
    evidence, founder, depth_keys = _chain_input(f"evidence/{request.identity}/{items}", load)
    source = AssemblyEvidenceSource(
        founder.positions,
        founder.likelihoods,
        founder.observed,
        founder.discovery_sites,
        request.identity,
        depth_keys,
        bool(evidence.discovery_sites.all() and not evidence.founder_excluded.any()),
    )
    return request, evidence, founder, source
