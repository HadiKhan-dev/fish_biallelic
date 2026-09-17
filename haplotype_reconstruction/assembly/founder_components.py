"""Independent complete founder refinements with bounded shared-memory threads.

All passes for one component stay on the same worker with a private workspace.
No component can change another's objective or candidates. Results are restored
to genomic order; completion order affects only the available CPU budget.
"""
import copy
from functools import partial

from .checkpoints import AssemblyCheckpointStore
from .founder_candidates import completed_candidates
from .founder_workspace import resolve_threads
from ..core import haplotypes, parallel


class ScopedCheckpoints:
    def __init__(self, store, prefix):
        self.store, self.prefix = store, prefix

    def load(self, phase):
        return self.store.load(f"{self.prefix}.{phase}")

    def save(self, phase, payload):
        self.store.save(f"{self.prefix}.{phase}", payload)


def scoped_checkpoints(store, prefix):
    """Give concurrent components distinct files and single-threaded small I/O."""
    if store is None:
        return None
    if isinstance(store, AssemblyCheckpointStore):
        store = copy.copy(store)
        store.checkpoint_store = copy.copy(store.checkpoint_store)
        store.checkpoint_store.nthreads = 1
    return ScopedCheckpoints(store, prefix)


def _renumber(value, number):
    # Component-local passes number their singleton 0. Preserve the chromosome
    # index in every diagnostic without changing founder/deletion indices.
    if isinstance(value, dict):
        return {key: number if key == "component" else _renumber(item, number)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_renumber(item, number) for item in value]
    return value


def _refine_one(batch, component, neutral, sites, options, budget):
    from .founder_refinement import _refine_serial_components
    return _refine_serial_components(batch, haplotypes.BlockResults([component]),
        neutral, sites, **dict(options, num_threads=budget or options["num_threads"]))


def refine_independent_components(prepared, components, neutral, sites, *,
                                  config, num_threads, checkpoints, l1_blocks,
                                  cc_scale):
    prepared = list(prepared)
    starts = {int(b.positions[0]): i for i, b in enumerate(prepared)}
    ends = {int(b.positions[-1]): i + 1 for i, b in enumerate(prepared)}
    from .founder_checkpoints import FounderCheckpointStore
    completed_store = (None if checkpoints is None else FounderCheckpointStore(
        scoped_checkpoints(checkpoints, "completed"), prepared))
    functions, indices = [], []
    answers = [None] * len(components)
    largest_sites = 0
    for number, block in enumerate(components):
        batch = prepared[starts[int(block.positions[0])]:ends[int(block.positions[-1])]]
        context = (None if l1_blocks is None else [b for b in l1_blocks
            if b.positions[0] >= block.positions[0] and b.positions[-1] <= block.positions[-1]])
        cached = None if completed_store is None else completed_store.load(f"component{number}")
        if cached is not None:
            answers[number] = (haplotypes.BlockResults([cached["block"]]), cached["diagnostic"])
            continue
        # Small components finish quickly. A compact completion checkpoint is
        # enough; thousands of per-proposal files would dominate L1 I/O. Long
        # components retain their ordinary resumable passes and iterations.
        detail = (None if len(batch) <= config.window_blocks else
                  scoped_checkpoints(checkpoints, f"component{number}"))
        options = dict(config=config, num_threads=num_threads, cc_scale=cc_scale,
            checkpoints=detail,
            l1_blocks=context)
        indices.append(number)
        functions.append(partial(_refine_one, batch, block, neutral, sites, options))
        largest_sites = max(largest_sites, len(block.positions))
    # Immutable chromosome evidence is shared; private models/logs/traceback
    # scale with the component. Include ample Python/native workspace headroom.
    per_component = 512 * 1024**2 + 128 * len(neutral) * largest_sites
    with parallel.numba_thread_scope(resolve_threads(num_threads)):
        for index, result in completed_candidates(functions, num_threads, per_component):
            number = indices[index]
            answers[number] = result
            if completed_store is not None:
                completed_store.save(f"component{number}", {
                    "block": result[0][0], "diagnostic": result[1]})
    output = haplotypes.BlockResults([result[0][0] for result in answers])
    diagnostics = dict(answers[0][1], components=[
        _renumber(result[1]["components"][0], number)
        for number, result in enumerate(answers)])
    return output, diagnostics
