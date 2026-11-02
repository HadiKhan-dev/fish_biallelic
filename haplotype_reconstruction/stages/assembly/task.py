"""Carry one fixed-K sweep through the common task skeleton and a typed record adapter.

Owns task resources, explicit storage selection, wall timing and record encoding.
Must not choose founder counts or alter proposal/acceptance order. The task imports
only 4.0; fixture extraction and 3.5 checkpoint decoding remain outside the package.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from haplotype_reconstruction.contracts.assembly import (
    AcceptanceCounters,
    ProposalCounters,
    RefinementDecisionTrace,
    RefinementProblem,
    RefinementProposal,
    RefinementSweep,
    RefinementWeakJunction,
    WindowAccount,
    WindowDomain,
    WindowPathAlphabet,
    WindowRangeAlphabet,
    WindowWorkCounters,
    WindowWorkShape,
)
from haplotype_reconstruction.contracts.roles import require_schema
from haplotype_reconstruction.core.errors import InputError, SettingsError
from haplotype_reconstruction.engine.scheduler import current_budget
from haplotype_reconstruction.science.assembly.refinement.parameters import SweepControl, SweepParameters
from haplotype_reconstruction.science.assembly.refinement.search import sweep
from haplotype_reconstruction.stages.framework import ResourceRequest, Task, TaskKey

from .adapter import sweep_payload
from .storage import StoragePlan, make_workspace
from .team import RefinementTeam

__all__ = ["SweepExecution", "RefinementAdapter", "run_sweep", "sweep_task"]

_DEFAULT_STORAGE = StoragePlan()


@dataclass(frozen=True)
class SweepExecution:
    """Scientific product plus timing/route diagnostics, kept outside the product schema."""

    product: RefinementSweep
    wall_seconds: float
    resident: bool
    capacity: int
    message_step: int
    cores: int


class RefinementAdapter:
    """Translate schema-1 RefinementSweep records to plain codec payloads by field name."""

    schema = 1

    def encode(self, product: RefinementSweep) -> Mapping[str, Any]:
        """Encode all product fields; arrays are borrowed for immediate publication."""
        require_schema(product, self.schema)
        return dict(schema=self.schema, **sweep_payload(product))

    def decode(self, payload: Mapping[str, Any]) -> RefinementSweep:
        """Rebuild the frozen record; reject unknown/missing fields and schema changes."""
        require_schema(payload, self.schema)
        expected = {f.name for f in fields(RefinementSweep)}
        if set(payload) != expected:
            raise InputError(f"refinement product fields differ: {set(payload) ^ expected}")
        values = dict(payload)
        values["proposals"] = tuple(RefinementProposal(*p) for p in payload["proposals"])
        values["counters"] = AcceptanceCounters(**payload["counters"])
        values["proposal_statistics"] = ProposalCounters(**payload["proposal_statistics"])
        values["window_work"] = WindowWorkCounters(**payload["window_work"])
        values["weak_junctions"] = tuple(RefinementWeakJunction(*entry) for entry in payload["weak_junctions"])
        trace = np.asarray(payload["decision_trace"])
        if trace.ndim != 2 or trace.shape[1] != 10:
            raise InputError("refinement trace must have ten columns")
        integer = trace[:, [0, 1, 2, 3, 4, 5, 6, 9]]
        if not np.array_equal(integer, integer.astype(np.int64)):
            raise InputError("refinement trace index columns are not integers")
        values["decision_trace"] = RefinementDecisionTrace(integer.astype(np.int64), trace[:, 7], trace[:, 8])
        domains = []
        for item in payload["window_domains"]:
            domain = dict(item)
            alphabet = domain["alphabet"]
            if alphabet["encoding"] == "prepared_row_range":
                domain["alphabet"] = WindowRangeAlphabet(**alphabet)
            elif alphabet["encoding"] == "prepared_row_paths":
                domain["alphabet"] = WindowPathAlphabet(**alphabet)
            else:
                raise InputError(f"unknown refinement window alphabet {alphabet['encoding']!r}")
            account = dict(domain["account"])
            account["work_shape"] = WindowWorkShape(**account["work_shape"])
            domain["account"] = WindowAccount(**account)
            domains.append(WindowDomain(**domain))
        values["window_domains"] = tuple(domains)
        return RefinementSweep(**values)


def run_sweep(
    problem: RefinementProblem,
    parameters: SweepParameters,
    control: SweepControl,
    storage: StoragePlan = _DEFAULT_STORAGE,
    *,
    cores: int = 1,
) -> SweepExecution:
    """Run one sweep with an explicit team and storage plan, returning scientific state and diagnostics.

    The enclosing engine initializes scalar-libm/OpenMP numerical policy before
    imports. Input arrays are borrowed, founder rows are copied by science, and all
    scratch lives until this call returns. Exceptions propagate; no result publishes
    after a failed work account or exactness check.
    """
    require_schema(problem, 1)
    if cores < 1:
        raise SettingsError(f"refinement cores must be positive, got {cores}")
    if parameters.window_work_coefficient <= 0 or parameters.frontier_bytes <= 0:
        raise SettingsError("refinement frontier and work coefficient must be positive")
    if control.direction not in ("fwd", "rev"):
        raise SettingsError(f"unknown refinement direction {control.direction!r}")
    if control.convention.kind != "refine_F":
        raise SettingsError(f"refinement requires the refine_F policy, got {control.convention.kind!r}")
    if storage.capacity is not None and storage.capacity < 1:
        raise SettingsError(f"refinement window capacity must be positive, got {storage.capacity}")
    if storage.message_step is not None and storage.message_step < 1:
        raise SettingsError(f"refinement message step must be positive, got {storage.message_step}")
    team = RefinementTeam(cores, current_budget())
    started = time.perf_counter()
    with team.scope():
        workspace = make_workspace(problem, storage)
        product = sweep(problem, parameters, control, workspace, team)
    return SweepExecution(
        product,
        time.perf_counter() - started,
        workspace.windows.resident,
        workspace.windows.capacity,
        workspace.message_step,
        cores,
    )


def sweep_task(
    problem: RefinementProblem,
    parameters: SweepParameters,
    control: SweepControl,
    storage: StoragePlan = _DEFAULT_STORAGE,
    *,
    scope: str = "pilot",
    max_cores: int = 1,
) -> Task:
    """Declare an importable sweep task and a conservative resident footprint estimate.

    The fixed scientific frontier cap is a search policy, not a memory admission
    amount. The minimal pilot executor records this estimate; live admission is a
    later engine responsibility. Result order is independent of the assigned team.
    """
    samples, sites = problem.evidence.shape[:2]
    founders, blocks = problem.rows.shape
    arrays: tuple[NDArray[Any], ...] = (
        problem.rows,
        problem.evidence,
        problem.offsets,
        problem.kept,
        problem.partial,
        problem.hazards,
        problem.normalisers,
        *problem.leaves,
        *problem.emissions,
    )
    immutable = sum(array.nbytes for array in arrays)
    messages = 4 * 8 * (blocks + 1) * samples * founders * (founders + 1) // 2
    windows = samples * (48 * sites + 32 * int(problem.partial.sum()))
    resources = ResourceRequest(
        min_cores=1, max_cores=max_cores, working_set_bytes=int(immutable + messages + windows), scratch_bytes=64 << 20
    )
    return Task(
        TaskKey("assembly", "refinement_sweep", scope),
        run_sweep,
        args=(problem, parameters, control, storage),
        resources=resources,
        thread_argument="cores",
    )
