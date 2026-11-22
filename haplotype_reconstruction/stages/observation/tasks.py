"""The run's observation pilot (records module docstring): per analysed contig, independent strict fits of its pilot
blocks and a gather that publishes the contig's record; then the run-level product (the outlier rate).

Every fit is its own task: a discovery block search at the strict outlier rate that grows into the CPUs its finished
siblings leave, as discovery's blocks do.
"""

from __future__ import annotations

import math
from math import fsum
from time import perf_counter

import numpy as np

from ...contracts.contigs import AnalysisContigs
from ...science.discovery.run import block_evidence
from ...science.observation.estimate import rate_grid, refined_maximum, two_level_rate
from ...science.observation.pilot import pilot_block_indices
from ...science.observation.run import pilot_fit
from ...store.protocol import claim, publish, read_items, resume
from ..framework import Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from ..inputs.products import required_product, task_inputs
from . import records

STAGES = ("observation_fits", "observation")


def pilot_counts(store, manifest, observation) -> dict[str, int]:
    """The pilot's block count of every analysed contig: its share of the contig's blocks, at least the contig
    minimum, with the run's minimum and maximum spread over contigs in proportion to their blocks (every block when
    a contig has fewer)."""
    from ..evidence import records as evidence_records

    contigs = tuple(AnalysisContigs.from_manifest(manifest).included)
    blocks = {}
    for c in contigs:
        request = evidence_records.expected(store, manifest, product="evidence", contig=c)
        saved = read_items(store.contig(c), evidence_records.EVIDENCE, request.identity, ("block_spans",))
        blocks[c] = 0 if saved is None else int(np.asarray(saved.payload["block_spans"]).shape[0])
    total = max(sum(blocks.values()), 1)
    counts = {}
    for c, n in blocks.items():
        share = n / total
        wanted = max(
            int(observation.minimum_contig_blocks),
            math.ceil(observation.pilot_share * n),
            math.ceil(observation.minimum_run_blocks * share),
        )
        ceiling = max(int(observation.minimum_contig_blocks), math.ceil(observation.maximum_run_blocks * share))
        counts[c] = min(n, wanted, ceiling)
    return counts


def pilot_task(root, contig, block, *, threads=1):
    """One strict pilot fit (science.observation.run.pilot_fit) of one block: (fit, seconds)."""
    from ...core.concurrency import threaded_each
    from ...engine.cpu import share_reader

    began = perf_counter()
    reader, _store = task_inputs(root)
    discovery, observation = records.parameters(reader.manifest)
    fit = pilot_fit(block, discovery, observation, threads=share_reader(), map_work=threaded_each)
    return fit, perf_counter() - began


def publish_fits(root, contig, fits):
    """Publish a contig's strict fits: the summed rate-grid log likelihood of their known-state cells, the cell count,
    the units of the two-level diagnostic and the fitted blocks (ascending)."""
    reader, store = task_inputs(root)
    request = records.expected(store, reader.manifest, product="fits", contig=contig)
    file = store.contig(contig)
    with claim(file, request.key, request.identity, blocking=False) as needed:
        if needed:
            _, observation = records.parameters(reader.manifest)
            grid = rate_grid(observation.grid_low, observation.grid_high, observation.grid_points)
            done = sorted((fit for fit, _ in fits), key=lambda fit: int(fit.block))
            total = np.zeros(len(grid))
            for fit in done:
                total += fit.grid
            payload = dict(
                schema=records.SCHEMA, contig=contig, blocks=tuple(int(fit.block) for fit in done),
                grid_total=total, cells=int(sum(int(fit.cells) for fit in done)),
                units=np.ascontiguousarray(np.concatenate([np.zeros((0, len(grid)))] + [fit.units for fit in done])),
                seconds=fsum(s for _, s in fits),
            )
            publish(file, request.key, request.identity, payload)
    return request.identity


def _pilots(store, manifest):
    out = []
    for c in AnalysisContigs.from_manifest(manifest).included:
        q = records.expected(store, manifest, product="fits", contig=c)
        saved = resume(store.contig(c), q.key, q.identity)
        if saved is None:
            raise RuntimeError(f"observation prerequisite missing: fits/{c}")
        out.append(saved.payload)
    return out


def publish_observation(root):
    """The run's observation product (docs/methods/observation.md, ob01): the one-rate maximum likelihood over every
    known-state cell of the strict fits (the grid maximum refined between its neighbours, ADVICE2 14.1 item 12), at
    least 0.5 / cells (a numerical regulariser); the strict rate when the fits decode no cell. Published with it: the
    raw grid maximum, whether it sits on the grid's lower bound (a legitimate near-zero) or its upper bound (censored:
    the rate is at least that), and the two-level fit as a diagnostic (ADVICE2 section 8)."""
    began = perf_counter()
    reader, store = task_inputs(root)
    manifest = reader.manifest
    request = records.expected(store, manifest)
    with claim(store.run, request.key, request.identity, blocking=False) as needed:
        if needed:
            _, observation = records.parameters(manifest)
            grid = rate_grid(observation.grid_low, observation.grid_high, observation.grid_points)
            strict = float(observation.strict_outlier_rate)
            pilots = _pilots(store, manifest)
            cells = int(sum(int(p["cells"]) for p in pilots))
            payload = dict(
                schema=records.SCHEMA, cells=cells, grid=tuple(map(float, grid)),
                blocks={p["contig"]: tuple(p["blocks"]) for p in pilots},
            )
            if cells == 0:
                payload.update(outlier_rate_blocks=strict, estimated=False, raw_rate=strict, rate_at_floor=False,
                               rate_censored=False, two_level=None)
            else:
                total = np.zeros(len(grid))
                for p in pilots:
                    total += np.asarray(p["grid_total"])
                best = int(np.argmax(total))
                units = np.concatenate([np.asarray(p["units"]).reshape(-1, len(grid)) for p in pilots])
                two = two_level_rate(units, cells, grid, fallback=float(grid[best]))
                payload.update(
                    outlier_rate_blocks=max(refined_maximum(grid, total, best), 0.5 / cells), estimated=True,
                    raw_rate=float(grid[best]), rate_at_floor=bool(best == 0),
                    rate_censored=bool(best == len(grid) - 1),
                    two_level=dict(
                        rate=two.outlier_rate, unexplained_rate=two.unexplained_rate,
                        unexplained_share=two.unexplained_share, gain=two.gain, units=two.units,
                    ),
                )
            payload["seconds"] = perf_counter() - began
            publish(store.run, request.key, request.identity, payload)
    return request.identity


def run(root, product, contig, resolved, maps, ties, *, team=1, cores=1):
    from ..evidence import records as evidence_records

    if product == "observation":
        return publish_observation(root)
    reader, store = task_inputs(root)
    manifest = reader.manifest
    request = records.expected(store, manifest, product="fits", contig=contig)
    if resume(store.contig(contig), request.key, request.identity) is not None:
        return request.identity
    _, observation = records.parameters(manifest)
    count = pilot_counts(store, manifest, observation)[contig]
    founder = required_product(reader, evidence_records, "evidence", contig).founder_view()
    chosen = pilot_block_indices(contig, len(founder.block_spans), count)
    tasks = tuple(
        Task(
            TaskKey("observation_fits", "fit", contig, f"{int(index):06d}"),
            pilot_task,
            (root, contig, block_evidence(founder, int(index))),
            resources=ResourceRequest(1, max(1, cores)),
            thread_argument="threads",
        )
        for index in chosen
    )
    gather = Task(
        TaskKey("observation_fits", "gather", contig),
        publish_fits,
        (root, contig, tuple(ResultRef(task.key) for task in tasks)),
        depends_on=tuple(task.key for task in tasks),
        resources=ResourceRequest(1, 1),
    )
    return Expansion(tasks, gather)


def task(key, context):
    product = {"observation_fits": "fits", "observation": "observation"}[key.stage]
    contig = None if key.scope == "run" else key.scope
    return Task(
        key,
        run,
        (context.root, product, contig, context.resolved, context.maps, context.ties),
        kwargs=dict(cores=context.cores),
        outputs=(RecordKey(key.scope, records.key(product)),),
        resources=ResourceRequest(1, 1),
        thread_argument="team",
    )
