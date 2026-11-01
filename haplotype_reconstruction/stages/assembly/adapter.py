"""Normalize a refinement fixture into explicit contracts and back into its comparison schema.

Owns the numeric seam adapter, not the search or source checkpoint decoding. Every
input key is consumed explicitly. Array ownership transfers from the fixture loader
for the lifetime of the task; science must copy before editing founder rows.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from haplotype_reconstruction.contracts.assembly import RefinementContext, RefinementProblem, RefinementSweep
from haplotype_reconstruction.contracts.roles import require_schema
from haplotype_reconstruction.core.errors import InputError

__all__ = ["problem_from_payload", "sweep_payload"]


def problem_from_payload(payload: dict[str, Any]) -> RefinementProblem:
    """Build a schema-1 sweep problem from all numeric fixture inputs.

    Scientific frontier/work settings are consumed by the task, rather than this
    problem record. Unknown fields raise InputError. Arrays are not mutated.
    """
    expected = {
        "rows",
        "evidence",
        "leaves",
        "offsets",
        "kept",
        "partial",
        "hazards",
        "emissions",
        "normalisers",
        "context",
        "penalty",
        "source_start",
        "bin_size",
        "frontier_bytes",
        "window_work_coefficient",
    }
    if set(payload) != expected:
        raise InputError(
            f"refinement input fields differ: missing={expected - set(payload)}, extra={set(payload) - expected}"
        )
    context = payload["context"]
    coarse = None if context is None else RefinementContext(np.asarray(context[0], np.int64), tuple(context[1]))
    result = RefinementProblem(
        rows=payload["rows"],
        evidence=payload["evidence"],
        leaves=tuple(payload["leaves"]),
        offsets=payload["offsets"],
        kept=payload["kept"],
        partial=payload["partial"],
        hazards=payload["hazards"],
        emissions=tuple(payload["emissions"]),
        normalisers=payload["normalisers"],
        context=coarse,
        penalty=float(payload["penalty"]),
        source_start=int(payload["source_start"]),
        bin_size=int(payload["bin_size"]),
    )
    require_schema(result, 1)
    samples, sites, genotypes = result.evidence.shape
    founders, blocks = result.rows.shape
    if genotypes != 3 or result.offsets.shape != (blocks + 1,) or result.offsets[-1] != sites:
        raise InputError(
            f"refinement axes: evidence={result.evidence.shape}, rows={result.rows.shape}, "
            f"offsets={result.offsets.shape}"
        )
    if result.normalisers.shape != (samples,) or len(result.leaves) != blocks or len(result.emissions) != blocks:
        raise InputError("refinement sample/block axes disagree")
    if founders < 2 or blocks < 2:
        raise InputError(f"a refinement sweep requires K>=2 and B>=2, got K={founders}, B={blocks}")
    required = (
        (result.rows, np.int64, 2, "rows"),
        (result.evidence, np.float32, 3, "evidence"),
        (result.offsets, np.int64, 1, "offsets"),
        (result.kept, np.bool_, 1, "kept"),
        (result.partial, np.bool_, 1, "partial"),
        (result.hazards, np.float64, 1, "hazards"),
        (result.normalisers, np.float64, 1, "normalisers"),
    )
    for values, dtype, ndim, name in required:
        if values.dtype != np.dtype(dtype) or values.ndim != ndim:
            raise InputError(
                f"refinement {name}: expected {np.dtype(dtype)} with {ndim} axes, got {values.dtype} {values.shape}"
            )
    if result.offsets[0] != 0 or np.any(np.diff(result.offsets) <= 0):
        raise InputError("refinement offsets must increase from local site zero")
    for values in (result.kept, result.partial, result.hazards):
        if values.shape != (sites,):
            raise InputError(f"refinement site array has shape {values.shape}; expected {(sites,)}")
    for block, (leaf, emission) in enumerate(zip(result.leaves, result.emissions, strict=True)):
        width = int(result.offsets[block + 1] - result.offsets[block])
        if leaf.dtype != np.int16 or leaf.ndim != 2 or leaf.shape[1] != width or leaf.shape[0] == 0:
            raise InputError(f"refinement leaf {block}: {leaf.dtype} {leaf.shape}; site width {width}")
        row_count = leaf.shape[0]
        if (
            emission.dtype != np.float64
            or emission.ndim != 3
            or emission.shape[:2] != (samples, row_count * (row_count + 1) // 2)
        ):
            raise InputError(f"refinement emissions {block}: {emission.dtype} {emission.shape}")
        if np.any(result.rows[:, block] < 0) or np.any(result.rows[:, block] >= row_count):
            raise InputError(f"refinement founder row outside block {block}'s {row_count} prepared rows")
    if coarse is not None:
        groups = coarse.groups
        if (
            groups.ndim != 2
            or groups.shape[1] != 2
            or len(groups) == 0
            or np.any(groups[:, 0] < 0)
            or np.any(groups[:, 1] > blocks)
            or np.any(groups[:, 1] <= groups[:, 0])
            or any(tuple(a) >= tuple(b) for a, b in zip(groups[:-1], groups[1:], strict=True))
            or len(coarse.paths) != len(groups)
        ):
            raise InputError(
                "refinement coarse spans must be distinct, sorted, nonempty and contained in the component"
            )
        for (start, stop), paths in zip(groups, coarse.paths, strict=True):
            if paths.dtype != np.int64 or paths.ndim != 2 or paths.shape[1] != stop - start:
                raise InputError(f"refinement coarse paths over [{start},{stop}): {paths.dtype} {paths.shape}")
    return result


def sweep_payload(result: RefinementSweep) -> dict[str, Any]:
    """Return every scientific result field in the frozen 3.5-side seam schema.

    Schema is checked separately; proposal records map by name to the original
    ordered tuples. No field is dropped and no timing is hidden in the result.
    """
    require_schema(result, 1)
    trace = np.empty((len(result.decision_trace.indices), 10), np.float64)
    trace[:, [0, 1, 2, 3, 4, 5, 6, 9]] = result.decision_trace.indices
    trace[:, 7] = result.decision_trace.operand
    trace[:, 8] = result.decision_trace.cutoff
    return dict(
        rows=result.rows,
        scores=result.scores,
        F0=result.F0,
        F=result.F,
        gain=result.gain,
        accounted=result.accounted,
        deviation=result.deviation,
        accepted=result.accepted,
        converged=result.converged,
        lex_next=result.lex_next,
        counters=asdict(result.counters),
        proposals=tuple((p.kind, p.start, p.stop, p.founder, p.rows) for p in result.proposals),
        proposal_statistics=asdict(result.proposal_statistics),
        window_work=asdict(result.window_work),
        window_domains=tuple(asdict(domain) for domain in result.window_domains),
        junction_gain=result.junction_gain,
        junction_pair=result.junction_pair,
        weak_junctions=tuple((r.first, r.last, r.pair, r.gain) for r in result.weak_junctions),
        decision_trace=trace,
    )
