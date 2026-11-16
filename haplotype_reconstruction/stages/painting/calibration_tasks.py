"""CPU-budgeted per-contig fits and two durable calibration reductions.

A contig's calibration is a set of independent fits: one per candidate outlier rate (robustness), then one per fold
and candidate divisor (divisor). Each fit is its own one-CPU task (outlier_fit, divisor_fit) and a gather publishes
the contig's record (fits_publish). Run as one team per contig, the fits ran in sequence and the team idled through
every one-thread section of every fit: measured on the divisor step of one contig of 40 samples and 330,000 sites,
71 s of one-thread work and 830 CPU-seconds of kernel work, about half of 19 threads busy.
"""

from functools import partial
from math import fsum
from time import perf_counter
from ...contracts.contigs import AnalysisContigs
from ...contracts.panels import materialise
from ...science.painting import calibration as science
from ...store.protocol import claim, publish, resume
from ..framework import Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from ..inputs.products import task_inputs, required_product
from . import calibration_records as records

PRODUCTS = dict(
    calibration_robustness="robustness",
    calibration_outlier="outlier",
    calibration_divisor="divisor",
    calibration="calibration",
)


def _fit(root, contig, resolved, maps, cores, function, **options):
    """One calibration fit of a contig on its final assembly and evidence windows: (value, seconds). ``cores`` is the
    run's CPU count: it sets the sample batch as in every other painting task (the scores do not depend on it), not
    the fit's CPUs (one)."""
    from ..assembly import records as assembly_records
    from ..evidence import records as evidence_records
    from ..relatedness.working_map import working_maps
    from .tasks import read_evidence, windowed_emissions, _at_batch_boundary

    began = perf_counter()
    reader, store = task_inputs(root)
    manifest = reader.manifest
    assembled = required_product(reader, assembly_records, "assembly", contig)
    ev = evidence_records.expected(store, manifest, product="evidence", contig=contig)
    batch = max(1, cores)
    builder = partial(_at_batch_boundary, partial(windowed_emissions, sample_tile=batch), batch)
    with read_evidence(store.contig(contig), ev.identity) as source:
        value = function(
            materialise(assembled.components, assembled.prepared),
            source,
            resolved.parameters.painting,
            working_maps(store, manifest, maps).for_contig(contig),
            batch_samples=batch,
            emission_builder=builder,
            **options,
        )
    return value, perf_counter() - began


def outlier_fit(root, contig, resolved, maps, epsilon, *, cores=1):
    """One candidate outlier rate of a contig: (row, seconds), row = (epsilon, likelihood sum)."""
    score, seconds = _fit(root, contig, resolved, maps, cores, science.outlier_unit, epsilon=epsilon)
    return (float(epsilon), score), seconds


def divisor_fit(root, contig, resolved, maps, ties, epsilon, fold, divisor, *, cores=1):
    """One fold and candidate divisor of a contig: (row, seconds), row = (divisor, fold, predictive log density,
    observed cells, path score)."""
    (predictive, cells, path), seconds = _fit(
        root, contig, resolved, maps, cores, science.divisor_unit,
        epsilon=epsilon, fold=fold, divisor=divisor, reverse=ties.reverse,
    )
    return (float(divisor), int(fold), predictive, cells, path), seconds


def fits_publish(root, product, contig, fits):
    """Publish a contig's record from its fits' rows in ascending order (the order of calibration.outlier_scores
    and calibration.divisor_scores); ``seconds`` is the fits' summed time."""
    reader, store = task_inputs(root)
    request = records.expected(store, reader.manifest, product=product, contig=contig)
    file = store.contig(contig)
    with claim(file, request.key, request.identity, blocking=False) as needed:
        if needed:
            rows = tuple(sorted(tuple(row) for row, _ in fits))
            seconds = fsum(seconds for _, seconds in fits)
            payload = dict(schema=1, contig=contig, rows=rows, seconds=seconds, skipped=False)
            publish(file, request.key, request.identity, payload)
    return request.identity


def _expansion(root, product, contig, fits):
    """Independent one-CPU fits and the gather that publishes the contig's record."""
    stage = "calibration_" + product
    width = len(str(len(fits)))
    tasks = tuple(
        Task(
            TaskKey(stage, "fit", contig, f"{index:0{width}d}"),
            function,
            arguments,
            kwargs=options,
            resources=ResourceRequest(1, 1),
        )
        for index, (function, arguments, options) in enumerate(fits)
    )
    gather = Task(
        TaskKey(stage, "gather", contig),
        fits_publish,
        (root, product, contig, tuple(ResultRef(task.key) for task in tasks)),
        depends_on=tuple(task.key for task in tasks),
        resources=ResourceRequest(1, 1),
    )
    return Expansion(tasks, gather)


def run(root, product, contig, resolved, maps, ties, *, team=1, cores=1):
    reader, store = task_inputs(root)
    manifest = reader.manifest
    request = records.expected(store, manifest, product=product, contig=contig)
    file = store.run if contig is None else store.contig(contig)
    config = resolved.parameters.painting
    grid = sorted(set(config.robustness_grid if product == "robustness" else config.divisor_grid))
    if product in ("robustness", "divisor") and len(grid) > 1:
        # several candidate values: the contig's fits are independent one-CPU tasks (a single value fits nothing)
        if resume(file, request.key, request.identity) is not None:
            return request.identity
        options = dict(cores=cores)
        if product == "robustness":
            return _expansion(
                root, product, contig, [(outlier_fit, (root, contig, resolved, maps, value), options) for value in grid]
            )
        outlier = records.expected(store, manifest, product="outlier")
        saved = resume(store.run, outlier.key, outlier.identity)
        if saved is None:
            raise RuntimeError("calibration prerequisite missing: outlier/None")
        epsilon = saved.payload["robustness_epsilon"]
        return _expansion(
            root,
            product,
            contig,
            [
                (divisor_fit, (root, contig, resolved, maps, ties, epsilon, fold, value), options)
                for value in grid
                for fold in range(config.calibration_folds)
            ],
        )
    with claim(file, request.key, request.identity, blocking=False) as needed:
        if needed:
            began = perf_counter()
            analysed = tuple(AnalysisContigs.from_manifest(manifest).included)

            def get(which, c=None):
                q = records.expected(store, manifest, product=which, contig=c)
                saved = resume(store.run if c is None else store.contig(c), q.key, q.identity)
                if saved is None:
                    raise RuntimeError(f"calibration prerequisite missing: {which}/{c}")
                return saved.payload

            if product in ("robustness", "divisor"):
                # a single-value grid fixes the quantity: nothing is fitted
                payload = dict(schema=1, contig=contig, rows=(), seconds=perf_counter() - began, skipped=True)
            elif product == "outlier":
                contigs = [get("robustness", c) for c in analysed]
                grid = sorted(set(config.robustness_grid))
                pooled = (
                    tuple((float(e), fsum(dict(v["rows"])[e] for v in contigs)) for e in grid) if len(grid) > 1 else ()
                )
                epsilon = min(pooled, key=lambda x: (-x[1], x[0]))[0] if pooled else float(grid[0])
                payload = dict(
                    schema=1, robustness_epsilon=epsilon, pooled=pooled, contigs=contigs, seconds=perf_counter() - began
                )
            else:
                outlier = get("outlier")
                contigs = [get("divisor", c) for c in analysed]
                grid = sorted(set(config.divisor_grid))
                pooled = (
                    tuple(
                        (
                            float(d),
                            fsum(row[2] for v in contigs for row in v["rows"] if row[0] == d),
                            sum(row[3] for v in contigs for row in v["rows"] if row[0] == d),
                            fsum(row[4] for v in contigs for row in v["rows"] if row[0] == d),
                        )
                        for d in grid
                    )
                    if len(grid) > 1
                    else ()
                )
                # Two strengths from one pass: the confidence divisor maximises the pooled predictive density, the path
                # divisor the pooled path score; ties to the smaller divisor in both.
                divisor = min(pooled, key=lambda x: (-x[1], x[0]))[0] if pooled else float(grid[0])
                path = min(pooled, key=lambda x: (-x[3], x[0]))[0] if pooled else float(grid[0])
                payload = dict(
                    schema=1,
                    robustness_epsilon=outlier["robustness_epsilon"],
                    confidence_divisor=divisor,
                    path_divisor=path,
                    confidence_correlation=science.correlation(divisor, config.reference_snps_per_bin),
                    path_correlation=science.correlation(path, config.reference_snps_per_bin),
                    robustness_pooled=outlier["pooled"],
                    robustness_contigs=outlier["contigs"],
                    divisor_pooled=pooled,
                    divisor_contigs=contigs,
                    seconds=perf_counter() - began,
                )
            publish(file, request.key, request.identity, payload)
    return request.identity


def fit_count(product, config):
    """The number of independent fits of one contig's ``product`` record (0: a single-value grid fits nothing)."""
    grid = set(config.robustness_grid if product == "robustness" else config.divisor_grid)
    if product not in ("robustness", "divisor") or len(grid) == 1:
        return 0
    return len(grid) * (config.calibration_folds if product == "divisor" else 1)


def task(key, context):
    product = PRODUCTS[key.stage]
    contig = None if key.scope == "run" else key.scope
    # A contig's task claims the CPUs of its fits and holds the claim until its gather: its fits run inside it. A
    # claim wider than its fits keeps CPUs from the other contigs (the outlier-rate step has 6 fits a contig: with
    # the run's whole CPU count as the maximum, 4 contigs claimed 19 CPUs each on 76, ran 24 fits, and 52 CPUs idled).
    fits = fit_count(product, context.resolved.parameters.painting)
    return Task(
        key,
        run,
        (context.root, product, contig, context.resolved, context.maps, context.ties),
        kwargs=dict(cores=context.cores),
        outputs=(RecordKey(key.scope, records.KEYS[product]),),
        resources=ResourceRequest(1, max(1, min(context.cores, fits))),
        thread_argument="team",
    )
