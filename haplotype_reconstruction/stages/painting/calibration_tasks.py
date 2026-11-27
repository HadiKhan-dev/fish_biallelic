"""CPU-budgeted per-contig fits and durable calibration reductions."""

from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from math import fsum
from time import perf_counter

import numpy as np

from ...contracts.contigs import AnalysisContigs
from ...contracts.panels import materialise
from ...core.concurrency import threaded_each
from ...core.errors import InvariantError, SettingsError
from ...engine.cpu import TaskBudget, budget_context, current_budget, thread_team
from ...engine.records import record_diagnostic
from ...science.painting import calibration as science
from ...science.painting import read_calibration
from ...store.protocol import claim, publish, read_items, resume
from ..framework import Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from ..inputs.products import required_product, task_inputs
from . import calibration_records as records

PRODUCTS = dict(
    calibration_robustness="robustness",
    calibration_refine="refine",
    calibration_outlier="outlier",
    calibration_share="share",
    calibration_divisor="divisor",
    calibration_divisor_fit="divisor_fit",
    calibration="calibration",
)
# A contig's per-bin fields of its rate fits (the outlier profile's material units): kept in the contig records only.
BIN_FIELDS = ("material_coordinates", "bin_rows")
# The zero_stress read-state units used 80.6% of an eight-thread team.
READ_DIVISOR_CORES = 8
# Keep each candidate's original workspace so it can retain the fit emissions
# across EM iterations. Wider custom grids run in groups to bound private scratch.
READ_DIVISOR_CONCURRENT_FITS = 5
# Window workers each use one CPU. The saved chr8 shard took 27 s with either
# four or 19 workers; four used 85 CPU s against 99 with 19.
READ_SHARE_CORES = 4
READ_SHARE_SAMPLE_BATCH = 19
# A record identity's upstream binding maps over the analysed contigs, mostly holding the GIL: the 4.23 TroMau robustness
# root used 7.6 CPU-s in 5.6 s with ten threads.
BINDING_THREADS = 4


def _fit(root, contig, resolved, maps, cores, function, *, read_state=None, foreign_read_share=0.0, **options):
    """Return a fit's value and elapsed seconds on the final assembly and evidence windows.

    ``cores`` sets the requested sample batch. The scheduler controls the kernel team. When read-state
    detection is present, the builder computes its matrices at each candidate's confidence strength.
    """
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
    painting = resolved.parameters.painting
    file = store.contig(contig)
    with read_evidence(file, ev.identity) as source:
        builder = partial(
            _at_batch_boundary,
            partial(windowed_emissions, sample_tile=batch, cohort=source.cohort_frequencies),
            batch,
        )
        if read_state is not None:
            from .read_state import stage_builder, with_detection

            painting = replace(painting, foreign_read_share=foreign_read_share)
            # The window arrays share one seek-based reader. Load the compact GL and observation sources
            # once so concurrent fits and their later scoring passes can share in-memory evidence.
            likelihoods = evidence_records.load_likelihoods(file, expected_identity=ev.identity)
            observed = evidence_records.load_observed(file, expected_identity=ev.identity, likelihoods=likelihoods)
            source = replace(source, likelihoods=likelihoods, observed=observed)
            request = records.expected(store, manifest, product="robustness", contig=contig)
            robust = read_items(store.contig(contig), request.key, request.identity, ("read_setup",))
            if robust is None:
                raise InvariantError(f"{contig}: read-aware calibration requires the robustness receipt")
            builder = stage_builder(
                builder, store.contig(contig), ev.identity, source.cohort_frequencies, None,
                with_detection(painting, read_state), unit_batches=True, read_setup=robust.payload["read_setup"],
            )
        value = function(
            materialise(assembled.components, assembled.prepared),
            source,
            painting,
            working_maps(store, manifest, maps).for_contig(contig),
            batch_samples=batch,
            emission_builder=builder,
            **options,
        )
    return value, perf_counter() - began


def _fit_candidates(function, items):
    """Ordered candidate fits with up to eight kernel threads apiece from the task's live budget."""
    if len(items) == 1:
        return [function(items[0])]

    def fit(candidate, budget):
        child = TaskBudget(budget, 1, READ_DIVISOR_CORES)
        with budget_context(child), thread_team(child.threads()):
            return function(candidate)

    budget = current_budget()
    values = [None] * len(items)
    for start in range(0, len(items), READ_DIVISOR_CONCURRENT_FITS):
        group = items[start:start + READ_DIVISOR_CONCURRENT_FITS]
        for index, value in budget.concurrent(
            (partial(fit, item) for item in group), label="read calibration candidates",
        ):
            values[start + index] = value
    return values


def fit_read_divisors(root, contig, resolved, maps, team, **options):
    """Fit a fold with shared evidence and label-independent parts."""
    return _fit(
        root, contig, resolved, maps, min(READ_DIVISOR_CORES, team), read_calibration.confidence_unit,
        fit_candidates=_fit_candidates, **options,
    )


def fit_share(root, contig, resolved, maps, **options):
    from ...science.painting import share_estimator

    # A window worker batches read-table setup independently of its one-thread kernel team.
    return _fit(
        root, contig, resolved, maps, READ_SHARE_SAMPLE_BATCH, share_estimator.score_contig,
        map_windows=threaded_each, window_workers=READ_SHARE_CORES, **options,
    )


def _pooled_rows(store, manifest, product, grid):
    """{rate: the analysed contigs' summed values} of one per-contig calibration product (math.fsum: the same sum in
    any contig order). The contigs' records are read concurrently."""

    def rows(c):
        q = records.expected(store, manifest, product=product, contig=c)
        saved = resume(store.contig(c), q.key, q.identity)
        if saved is None:
            raise RuntimeError(f"calibration prerequisite missing: {product}/{c}")
        return saved.payload["rows"]

    values = {float(e): [] for e in grid}
    for found in threaded_each(rows, AnalysisContigs.from_manifest(manifest).included):
        for e, v in found:
            values[float(e)].append(v)
    return {e: fsum(v) for e, v in values.items()}


def outlier_fit(root, contig, resolved, maps, epsilon, *, cores=1):
    """One candidate outlier rate of a contig: (row, seconds), row = (epsilon, likelihood sum, per-sample sums, regime
    counts of the returned fits: label changes, anchored faithful-state bins, anchored error-state bins, and per
    informative component its bins' working-map coordinates and per-bin likelihood terms). The coordinates follow
    the pedigree's material axis, clipped as the run's pedigree clips it (MaterialParameters.switch_clip)."""
    (score, samples, regime, curves), seconds = _fit(
        root, contig, resolved, maps, cores, science.outlier_unit, epsilon=epsilon, per_sample=True,
        switch_clip=resolved.parameters.pedigree.material.switch_clip,
    )
    return (float(epsilon), score, tuple(float(v) for v in samples), tuple(regime), tuple(curves)), seconds


def divisor_fit(root, contig, resolved, maps, ties, epsilon, fold, divisor, *, cores=1):
    """One fold and candidate divisor of a contig: (row, seconds), row = (divisor, fold, predictive log density,
    observed cells, path score)."""
    (predictive, cells, path), seconds = _fit(
        root, contig, resolved, maps, cores, science.divisor_unit,
        epsilon=epsilon, fold=fold, divisor=divisor, reverse=ties.reverse,
    )
    return (float(divisor), int(fold), predictive, cells, path), seconds


def contig_morgans(store, manifest, maps, contig):
    """A contig's working-map Morgans between its first and last evidence position (0 without evidence): its length
    in the calibration material (read_calibration.material_prefix). Reads the evidence positions only."""
    from ..evidence import records as evidence_records
    from ..relatedness.working_map import working_maps

    ev = evidence_records.expected(store, manifest, product="evidence", contig=contig)
    positions = evidence_records.load_arrays(store.contig(contig), ("positions",), expected_identity=ev.identity)[
        "positions"
    ]
    if not len(positions):
        return 0.0
    genetic_map = working_maps(store, manifest, maps).for_contig(contig)
    return float(genetic_map.interval_morgans(float(np.min(positions)), float(np.max(positions))))


def calibration_material(store, manifest, maps, config, included):
    """((contig, Morgans), ...): the analysed contigs the read-aware confidence fits run on, in configured order
    (read_calibration.material_prefix at PaintingParameters.calibration_morgans; contigs past the material are never
    read). Every contig's divisor task and the run reduction derive it alike from the bound evidence and map."""
    return read_calibration.material_prefix(
        ((c, contig_morgans(store, manifest, maps, c)) for c in included), config.calibration_morgans
    )


def divisor_plan(root, resolved, maps):
    """The active divisor grid and contigs, in configured order, from the outlier reduction."""
    from .read_state import READ_STATE_ON

    reader, store = task_inputs(root)
    manifest = reader.manifest
    config = resolved.parameters.painting
    request = records.expected(store, manifest, product="outlier")
    saved = resume(store.run, request.key, request.identity)
    if saved is None:
        raise SettingsError("calibration prerequisite missing: outlier")
    outlier = saved.payload
    read = config.read_classes > 1 and outlier.get("read_state", READ_STATE_ON) == READ_STATE_ON
    share = config.foreign_read_share
    if resolved.parameters.estimate_foreign_read_share and config.read_classes > 1:
        request = records.expected(store, manifest, product="share")
        saved = resume(store.run, request.key, request.identity)
        if saved is None:
            raise SettingsError("calibration prerequisite missing: share")
        share = saved.payload["estimate"]
    outlier = dict(outlier, foreign_read_share=share)
    included = resolved.analysis.included
    if read:
        material = dict(calibration_material(store, manifest, maps, config, included))
        grid = tuple(sorted(set(config.confidence_grid)))
    else:
        material = dict.fromkeys(included)
        grid = tuple(sorted(set(config.divisor_grid)))
        if len(grid) == 1:
            grid = ()
    return outlier, read, material, grid


def _divisor_cores(grid, read):
    return READ_DIVISOR_CORES * min(len(grid), READ_DIVISOR_CONCURRENT_FITS) if read else 1


def divisor_fit_plan(plan, resolved, *, default_by_fold):
    """The scatter units and useful CPU widths after the outlier barrier."""
    _, read, material, grid = plan
    folds = range(resolved.parameters.painting.calibration_folds)
    return dict(
        by_fold=read and default_by_fold,
        read_state=read,
        fold_fits=[
            dict(contig=contig, fold=fold, cores=_divisor_cores(grid, read))
            for contig in material
            for fold in folds
        ] if read else [],
        fits=[
            dict(contig=contig, fold=fold, divisor=divisor, cores=READ_DIVISOR_CORES if read else 1)
            for contig in material
            for fold in folds
            for divisor in grid
        ],
    )


def publish_divisor_fits(root, contig, resolved, maps, ties, epsilon, detection, share, fold, grid, *, team=1):
    """Publish one receipt per candidate, with one read-state builder shared within this fit task."""
    reader, store = task_inputs(root)
    file = store.contig(contig)
    requests = {
        value: records.expected(store, reader.manifest, product="divisor_fit", contig=contig, fold=fold, divisor=value)
        for value in grid
    }
    with ExitStack() as locks:
        pending = tuple(value for value, request in requests.items()
                        if locks.enter_context(claim(file, request.key, request.identity, blocking=False)))
        if pending:
            if detection is not None:
                scores, seconds = fit_read_divisors(
                    root, contig, resolved, maps, team,
                    read_state=detection, foreign_read_share=share, epsilon=epsilon, reverse=ties.reverse,
                    grid=pending, folds=(fold,),
                )
                fits = [(value, dict(samples=samples, cells=cells)) for value, _, samples, cells in scores]
            elif len(pending) == 1:
                value = pending[0]
                (_, _, density, cells, path), seconds = divisor_fit(
                    root, contig, resolved, maps, ties, epsilon, fold, value,
                )
                fits = [(value, dict(density=density, cells=cells, path=path))]
            else:
                scores, seconds = _fit(
                    root, contig, resolved, maps, team, science.divisor_scores,
                    epsilon=epsilon, reverse=ties.reverse, grid=pending, folds=(fold,),
                )
                fits = [(value, dict(density=density, cells=cells, path=path))
                        for value, _, density, cells, path in scores]
            for value, payload in fits:
                request = requests[value]
                publish(file, request.key, request.identity, dict(schema=1, **payload, seconds=seconds / len(fits)))
    return {RecordKey(contig, request.key): request.identity for request in requests.values()}


def gather_divisor(root, contig, grid, folds, read, morgans):
    """Read the checked fold/candidate receipts and publish the existing per-contig divisor payload."""
    reader, store = task_inputs(root)
    manifest = reader.manifest
    file = store.contig(contig)
    request = records.expected(store, manifest, product="divisor", contig=contig)
    with claim(file, request.key, request.identity, blocking=False) as needed:
        if needed:
            began = perf_counter()
            rows, sample_rows, times = [], [], []
            for value in grid:
                for fold in folds:
                    fit = records.expected(
                        store, manifest, product="divisor_fit", contig=contig, fold=fold, divisor=value,
                    )
                    saved = resume(file, fit.key, fit.identity)
                    if saved is None:
                        raise SettingsError(f"{contig}: divisor fit missing for --fold {fold} --divisor {value}")
                    payload = saved.payload
                    if read:
                        samples = payload["samples"]
                        sample_rows.append((value, fold, samples))
                        rows.append((value, fold, fsum(samples), payload["cells"], None))
                    else:
                        rows.append((value, fold, payload["density"], payload["cells"], payload["path"]))
                    times.append(payload["seconds"])
            payload = dict(schema=1, contig=contig, rows=tuple(rows), skipped=not grid,
                           seconds=fsum(times) if grid else perf_counter() - began)
            if read:
                payload["sample_rows"] = tuple(sample_rows)
                if morgans is not None:
                    payload["morgans"] = morgans
            publish(file, request.key, request.identity, payload)
    return request.identity


def _divisor_task(key, root, resolved, maps, ties, outlier, read, fold, grid, cores):
    width = min(cores, _divisor_cores(grid, read))
    # A fold starts on one candidate's team (its sample batch) and takes its fair share of free CPUs up to its width,
    # so a node runs several folds at once. A minimum of half the node (4.23) let one fold widen to 40 of 76 CPUs and
    # keep every other fold waiting for 38: a single-node calibration ran its folds one at a time.
    minimum = min(width, READ_DIVISOR_CORES)
    return Task(
        key, publish_divisor_fits,
        (root, key.scope, resolved, maps, ties, outlier["robustness_epsilon"],
         outlier["read_detection"] if read else None, outlier["foreign_read_share"], fold, grid),
        outputs=tuple(RecordKey(key.scope, records.fit_key(fold, value)) for value in grid),
        resources=ResourceRequest(minimum, width),
        thread_argument="team",
    )


def run_divisor(root, contig, resolved, maps, ties, *, cores=1, gather_only=False, by_fold=False):
    reader, store = task_inputs(root)
    request = records.expected(store, reader.manifest, product="divisor", contig=contig)
    if resume(store.contig(contig), request.key, request.identity) is not None:
        return request.identity
    outlier, read, material, grid = divisor_plan(root, resolved, maps)
    folds = range(resolved.parameters.painting.calibration_folds)
    grid = grid if contig in material else ()
    gather_args = (root, contig, grid, folds, read, material.get(contig))
    if gather_only or not grid:
        return gather_divisor(*gather_args)
    # The outlier barrier determines the route and material before the engine admits these independent fits.
    tasks = tuple(
        _divisor_task(
            TaskKey("calibration_divisor_fit", "contig", contig, records.fit_round(fold, value)),
            root, resolved, maps, ties, outlier, read, fold, grid if value is None else (value,), cores,
        )
        for fold in folds
        for value in ((None,) if read and by_fold else grid)
    )
    gather = Task(TaskKey("calibration_divisor", "gather", contig), gather_divisor, gather_args)
    return Expansion(tasks, gather)


def read_calibration_payload(config, contigs, material):
    """The read-aware confidence choice from the divisor payloads ``contigs`` of the calibration material
    ``material`` ((contig, Morgans), ... from calibration_material): each sample's totals per candidate over the folds
    and those contigs (math.fsum, contigs in configured order), the paired one-standard-error choice
    (read_calibration.paired_one_se) and the fixed path divisor (read_calibration.path_divisor). Returns (confidence
    divisor, path divisor, pooled rows (divisor, score, cells, None), the read_confidence record: the selection, the
    material's contigs and Morgans, and whether the best candidate is the grid's largest)."""
    if tuple(v["contig"] for v in contigs) != tuple(c for c, _ in material):
        raise InvariantError("the read-aware confidence payloads are not the calibration material's contigs")
    grid = sorted({float(c) for c in config.confidence_grid})
    contributions = {c: [] for c in grid}
    for v in contigs:
        for c, _, values in v["sample_rows"]:
            contributions[float(c)].append(np.asarray(values, dtype=np.float64))
    totals = np.array([[fsum(column) for column in zip(*contributions[c], strict=True)] for c in grid])
    selection = read_calibration.paired_one_se(grid, totals)
    pooled = tuple(
        (c, fsum(row[2] for v in contigs for row in v["rows"] if row[0] == c),
         sum(row[3] for v in contigs for row in v["rows"] if row[0] == c), None)
        for c in grid
    )
    record = dict(grid=tuple(grid), samples=int(totals.shape[1]), sample_totals=tuple(tuple(row) for row in totals),
                  **selection, best_at_top=selection["best"] == grid[-1],
                  budget_morgans=float(config.calibration_morgans), contigs=tuple(c for c, _ in material),
                  contig_morgans=tuple(m for _, m in material), material_morgans=fsum(m for _, m in material))
    return selection["choice"], read_calibration.path_divisor(config), pooled, record


def _read_sharing(store, manifest, contig):
    """The contig's libraries and candidate library splits with their read-sharing decisions at the run's
    family-wise z* (stages.painting.read_state.contig_read_libraries: dict(libraries, splits)), None without the
    read state (read_classes = 1): the contig's
    part of the run's read-state route, kept on its robustness record (the run's first per-contig calibration step,
    before any read-aware fit)."""
    from ...contracts.payload import from_payload
    from ...science.painting.parameters import PaintingParameters
    from ..evidence import records as evidence_records
    from .read_state import contig_read_libraries, run_detection, with_detection

    config = from_payload(PaintingParameters, manifest["parameters"]["painting"])
    if config.read_classes == 1:
        return dict(read_libraries=None, read_sharing=None, read_setup=None)
    ev = evidence_records.expected(store, manifest, product="evidence", contig=contig)
    detection = run_detection(store, manifest, config.read_state)
    libraries, splits, setup = contig_read_libraries(store.contig(contig), ev.identity, with_detection(config, detection))
    return dict(read_libraries=dict(libraries=libraries, splits=splits),
                read_sharing=any(row["status"] == "fitted" for row in libraries), read_setup=setup)


def read_sharing(root, contig):
    """Graph task: the contig's read-sharing fields of its robustness record (_read_sharing). They read only the
    evidence, so the robustness expansion runs this beside the rate fits instead of after them in its gather."""
    reader, store = task_inputs(root)
    return _read_sharing(store, reader.manifest, contig)


def record_identity(root, product, contig):
    """Graph task: the identity of a contig's calibration record (calibration_records.expected). It binds every
    analysed contig's upstream chain (5.5 s in a fresh process of a TroMau step command), so a robustness expansion
    with no record to resume derives it beside its fits."""
    reader, store = task_inputs(root)
    return records.expected(store, reader.manifest, product=product, contig=contig).identity


def fits_publish(root, product, contig, fits, sharing=None, identity=None):
    """Publish a contig's record from its fits' rows in ascending order (the order of calibration.outlier_scores
    and calibration.divisor_scores); ``seconds`` is the fits' summed time. A robustness record also carries the
    contig's read-sharing fields: ``sharing`` (the read_sharing task's value), computed here when not given.
    ``identity``: the record's identity (the record_identity task's value), derived here when not given."""
    reader, store = task_inputs(root)
    if identity is None:
        request = records.expected(store, reader.manifest, product=product, contig=contig)
    else:
        request = records.Reader(identity, records.KEYS[product])
    file = store.contig(contig)
    with claim(file, request.key, request.identity, blocking=False) as needed:
        if needed:
            seconds = fsum(seconds for _, seconds in fits)
            # (rate, likelihood sum) rows, and each rate's per-sample sums apart (A11, ADVICE2 14.1 item 9)
            rows = tuple(sorted((row[0], row[1]) for row, _ in fits))
            sample_rows = tuple(sorted((row[0], tuple(row[2])) for row, _ in fits))
            # each rate's regime at its returned fit (ADVICE2 11.2): (rate, label changes, anchored faithful-state
            # bins, anchored error-state bins)
            regime_rows = tuple(sorted((row[0], *row[3]) for row, _ in fits))
            # the informative components' bin coordinates (the same at every rate) and each rate's per-bin
            # likelihood terms: the outlier profile's material units (ADVICE2 11.2)
            coordinates = tuple(x for x, _ in fits[0][0][4]) if fits else ()
            for row, _ in fits:
                if len(row[4]) != len(coordinates) or not all(
                    np.array_equal(x, y) for (x, _), y in zip(row[4], coordinates, strict=True)
                ):
                    raise InvariantError(f"{contig}: the calibration fits' component bins differ between rates")
            bin_rows = tuple(sorted(((row[0], tuple(t for _, t in row[4])) for row, _ in fits), key=lambda r: r[0]))
            payload = dict(schema=1, contig=contig, rows=rows, sample_rows=sample_rows, regime_rows=regime_rows,
                           material_coordinates=coordinates, bin_rows=bin_rows, seconds=seconds, skipped=False)
            if product == "robustness":
                payload.update(_read_sharing(store, reader.manifest, contig) if sharing is None else sharing)
            publish(file, request.key, request.identity, payload)
    return request.identity


def material_curves(records, grid):
    """(coordinates, curves float64 [G, B]) of every informative component of the analysed contigs, contigs in
    configured order and components in map order (the order of the fits): ``records`` holds each contig's
    (robustness, refine) payloads, whose bin rows together cover every rate of ``grid``."""
    components = []
    for coarse, refine in records:
        coordinates = tuple(coarse["material_coordinates"])
        later = tuple(refine["material_coordinates"])
        if len(later) != len(coordinates) or not all(np.array_equal(x, y) for x, y in zip(coordinates, later, strict=True)):
            raise InvariantError(f"{coarse['contig']}: the calibration fits' component bins differ between rates")
        terms = {float(e): values for e, values in tuple(coarse["bin_rows"]) + tuple(refine["bin_rows"])}
        missing = [e for e in grid if e not in terms]
        if missing:
            raise InvariantError(f"{coarse['contig']}: per-bin calibration terms missing at rates {missing}")
        for k, x in enumerate(coordinates):
            components.append((x, np.stack([np.asarray(terms[e][k], dtype=np.float64) for e in grid])))
    return components


def _expansion(root, product, contig, fits, *, derive_identity=False):
    """Independent rate fits and the gather that publishes the contig's record; a robustness expansion also runs the
    contig's read-sharing task (read_sharing) beside its fits, and with ``derive_identity`` the record's identity
    (record_identity). A fit's kernels are parallel over the samples of its batch (``cores``): it starts on one CPU
    and takes its fair share of free CPUs at each emission-batch boundary (stages.painting.tasks._at_batch_boundary),
    so one contig's few fits use a whole node while many contigs' fits run one CPU each. On TroMau chr2 a fit took
    30.0 s on one thread, 5.0 s on eight and 1.9 s on 76, with the same value (thread counts never change a
    result)."""
    stage = "calibration_" + product
    width = len(str(len(fits)))
    extra, kwargs = (), {}
    if product == "robustness":
        sharing = Task(TaskKey(stage, "read_sharing", contig), read_sharing, (root, contig),
                       resources=ResourceRequest(1, 1))
        extra, kwargs = (sharing,), dict(sharing=ResultRef(sharing.key))
    if derive_identity:
        identity = Task(TaskKey(stage, "identity", contig), record_identity, (root, product, contig),
                        resources=ResourceRequest(1, BINDING_THREADS))
        extra, kwargs = (identity, *extra), dict(kwargs, identity=ResultRef(identity.key))
    tasks = tuple(
        Task(
            TaskKey(stage, "fit", contig, f"{index:0{width}d}"),
            function,
            arguments,
            kwargs=options,
            resources=ResourceRequest(1, max(1, int(options.get("cores", 1)))),
        )
        for index, (function, arguments, options) in enumerate(fits)
    )
    gather = Task(
        TaskKey(stage, "gather", contig),
        fits_publish,
        (root, product, contig, tuple(ResultRef(task.key) for task in tasks)),
        kwargs=kwargs,
        depends_on=tuple(task.key for task in extra + tasks),
        resources=ResourceRequest(1, 1),
    )
    return Expansion(extra + tasks, gather)


def run(root, product, contig, resolved, maps, ties, *, cores=1):
    reader, store = task_inputs(root)
    manifest = reader.manifest
    file = store.run if contig is None else store.contig(contig)
    config = resolved.parameters.painting
    grid = sorted(set(config.robustness_grid))

    def fits(values):
        return [(outlier_fit, (root, contig, resolved, maps, value), dict(cores=cores)) for value in values]

    if product == "robustness" and len(grid) > 1 and not file.has(records.KEYS[product]):
        # No record to resume: the fits start at once and the record's identity is derived beside them.
        return _expansion(root, product, contig, fits(grid), derive_identity=True)
    request = records.expected(store, manifest, product=product, contig=contig)
    if product in ("robustness", "refine") and len(grid) > 1:
        if resume(file, request.key, request.identity) is not None:
            return request.identity
        # Refinement fits the geometric midpoints around the pooled grid maximum.
        if product == "refine":
            values = _pooled_rows(store, manifest, "robustness", grid)
            grid = science.midpoints(grid, [values[e] for e in grid])
        return _expansion(root, product, contig, fits(grid))
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

            if product in ("robustness", "refine"):
                # a single-value grid fixes the quantity: nothing is fitted
                payload = dict(schema=1, contig=contig, rows=(), seconds=perf_counter() - began, skipped=True)
                if product == "robustness":
                    payload.update(_read_sharing(store, manifest, contig))
            elif product == "outlier":
                # every contig's robustness and refine records, each read once and the contigs concurrently (their
                # decodes and identity checks dominated this serial round: up to six reads of each record)
                pairs = [(which, c) for c in analysed for which in ("robustness", "refine")]
                saved = dict(zip(pairs, threaded_each(lambda pair: get(*pair), pairs), strict=True))
                # each contig's rows: the literal grid and the two midpoints around the pooled grid maximum (the
                # per-bin fields stay in the contig records: the run record keeps the material units they make)
                contigs, fitted = [], []
                for c in analysed:
                    coarse, refine = saved["robustness", c], saved["refine", c]
                    fitted.append((coarse, refine))
                    contigs.append(dict(
                        {k: v for k, v in coarse.items() if k not in BIN_FIELDS and k != "read_setup"},
                        rows=tuple(sorted(tuple(coarse["rows"]) + tuple(refine["rows"]))),
                        sample_rows=tuple(sorted(tuple(coarse.get("sample_rows", ())) + tuple(refine.get("sample_rows", ())))),
                        regime_rows=tuple(sorted(tuple(coarse.get("regime_rows", ())) + tuple(refine.get("regime_rows", ())))),
                    ))
                # the fitted rates (none when a single-value grid fixes the rate: nothing was fitted)
                grid = sorted({float(row[0]) for v in contigs for row in v["rows"]}) or sorted(set(config.robustness_grid))
                pooled = (
                    tuple((float(e), fsum(dict(v["rows"])[e] for v in contigs)) for e in grid) if len(grid) > 1 else ()
                )
                epsilon = min(pooled, key=lambda x: (-x[1], x[0]))[0] if pooled else float(grid[0])
                # The profile (ADVICE2 11.2): the grid maximum, its boundary state, the plain support set and, where
                # it can be estimated, the variance inflation over the pedigree's equal-material units (the run's
                # MaterialParameters: unit_morgans units of the genome-ordered working map, assessed from
                # assessable_morgans of informative material; owner, 16 Nov) and the adjusted set; a flat profile is
                # "not identified", and the run then uses the pilot's rate.
                profile, units = None, None
                if pooled:
                    material = resolved.parameters.pedigree.material
                    values, lengths, morgans = science.material_units(
                        material_curves(fitted, grid), material.unit_morgans, material.tail_morgans
                    )
                    # each occupied unit's working Morgans and its curve over the grid (plain floats: exported as JSON)
                    units = dict(working_morgans=morgans, unit_morgans=material.unit_morgans,
                                 tail_morgans=material.tail_morgans, lengths=lengths,
                                 values=tuple(tuple(float(v) for v in row) for row in values))
                    profile = science.outlier_profile(
                        grid, [[dict(v["rows"])[e] for e in grid] for v in contigs], values, morgans,
                        material.assessable_morgans,
                    )
                    if not profile["identified"]:
                        from ..observation import records as observation_records

                        epsilon = observation_records.outlier_rate_blocks(reader, store)
                # each rate's pooled regime at its returned fits (ADVICE2 11.2): (rate, expected label changes,
                # error-track occupancy = error-state bins / anchored track-bins)
                regime = ()
                if pooled and all(len(v.get("regime_rows", ())) == len(grid) for v in contigs):
                    by_rate = {e: [row for v in contigs for row in v["regime_rows"] if row[0] == e] for e in grid}
                    regime = tuple(
                        (e, fsum(r[1] for r in rows), fsum(r[3] for r in rows) / max(fsum(r[2] + r[3] for r in rows), 1e-300))
                        for e, rows in by_rate.items()
                    )
                # A11 (ADVICE2 14.1 item 9): each sample's own rate from its per-sample sums over the contigs
                rates, at_top = (), ()
                if pooled:
                    per_sample = {}
                    for c in analysed:
                        for which in ("robustness", "refine"):
                            for e, values in saved[which, c].get("sample_rows", ()):
                                total = per_sample.setdefault(float(e), np.zeros(len(values)))
                                total += np.asarray(values, dtype=np.float64)
                    if per_sample and all(e in per_sample for e in grid):
                        found, top = science.sample_rates(grid, np.stack([per_sample[e] for e in grid], axis=1))
                        rates, at_top = tuple(float(x) for x in found), tuple(bool(x) for x in top)
                # the run's read-state route (owner, 18 Nov): with no library of the run sharing reads, the read
                # model's exact limit is one read class, and the run takes the route without read state throughout
                from .read_state import read_state_route

                read_state = read_state_route(config, [saved["robustness", c].get("read_sharing") for c in analysed])
                from .read_state import run_detection

                # the family-wise detection the contigs' read_sharing used (alpha, tests M, z*): recorded for the run
                detection = run_detection(store, manifest, config.read_state) if config.read_classes > 1 else None
                libraries, splits, warnings = [], [], []
                if detection is not None:
                    from .read_state import detection_margins

                    # each library's largest excess / SE and each candidate split's contrast against z* (signed
                    # margins); a warning line for any decision within a factor of two of z*
                    found = [(c, saved["robustness", c].get("read_libraries") or {}) for c in analysed]
                    libraries, splits, warnings = detection_margins(
                        [(c, v.get("libraries"), v.get("splits")) for c, v in found], detection, read_state,
                        config.read_state.library_survival_tolerance)
                    for line in warnings:
                        record_diagnostic("WARNING " + line)
                payload = dict(
                    schema=1, robustness_epsilon=float(epsilon), pooled=pooled, contigs=contigs, profile=profile,
                    read_state=read_state, read_detection=detection, read_libraries=tuple(libraries),
                    read_splits=tuple(splits), read_warnings=tuple(warnings),
                    units=units, regime=regime,
                    identified=True if profile is None else profile["identified"],
                    boundary="none" if profile is None else profile["boundary"],
                    likelihood_set=() if profile is None else profile["plain_set"],
                    sample_rates=rates, sample_rate_at_top=at_top, seconds=perf_counter() - began,
                )
            elif product == "share":
                from ...science.painting import share_estimator

                outlier = get("outlier")
                read = outlier["read_state"] == "on"
                n_samples = len(manifest["sample_ids"])
                if contig is not None:
                    if read:
                        scored, _ = fit_share(
                            root, contig, resolved, maps,
                            read_state=outlier["read_detection"], epsilon=outlier["robustness_epsilon"],
                        )
                    else:
                        scored = dict(windows=0, rows=(), supported=0)
                    payload = dict(schema=1, contig=contig, **scored, seconds=perf_counter() - began)
                else:
                    shards = [get("share", c) for c in analysed]
                    active = [v["rows"] for v in shards if v["supported"]]
                    profile = share_estimator.pool_profiles(active, n_samples) if read else dict(estimate=0.0)
                    identified = bool(active) and max(profile["pooled"]) > min(profile["pooled"])
                    payload = dict(
                        schema=1, **profile, mode="estimated" if identified else "unavailable",
                        contigs=tuple(dict(contig=v["contig"], windows=v["windows"], supported=v["supported"],
                                           seconds=v["seconds"])
                                      for v in shards),
                        seconds=perf_counter() - began,
                    )
            else:
                outlier = get("outlier")
                auto = resolved.parameters.estimate_foreign_read_share
                share = get("share") if auto and config.read_classes > 1 else dict(
                    estimate=config.foreign_read_share, mode="unavailable" if auto else "declared",
                )
                contigs = [get("divisor", c) for c in analysed]
                grid = sorted(set(config.divisor_grid))
                read_confidence = None
                if outlier.get("read_state", "on") == "on" and config.read_classes > 1:
                    # the material's contigs only; the others published empty records (outside the map budget)
                    material = calibration_material(store, manifest, maps, config, analysed)
                    names = {c for c, _ in material}
                    divisor, path, pooled, read_confidence = read_calibration_payload(
                        config, [v for v in contigs if v["contig"] in names], material
                    )
                pooled = pooled if read_confidence is not None else (
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
                # divisor the pooled path score; ties to the smaller divisor in both. With the read state: the paired
                # one-standard-error confidence choice and the fixed path divisor (read_calibration_payload).
                if read_confidence is None:
                    divisor = min(pooled, key=lambda x: (-x[1], x[0]))[0] if pooled else float(grid[0])
                    path = min(pooled, key=lambda x: (-x[3], x[0]))[0] if pooled else float(grid[0])
                payload = dict(
                    schema=1,
                    robustness_epsilon=outlier["robustness_epsilon"],
                    foreign_read_share=share["estimate"],
                    foreign_read_share_profile={k: v for k, v in share.items() if k != "schema"},
                    read_state=outlier.get("read_state", "on"),
                    read_detection=outlier.get("read_detection"),
                    read_libraries=tuple(outlier.get("read_libraries", ())),
                    read_splits=tuple(outlier.get("read_splits", ())),
                    read_warnings=tuple(outlier.get("read_warnings", ())),
                    robustness_profile=outlier.get("profile"),
                    robustness_units=outlier.get("units"),
                    robustness_regime=tuple(outlier.get("regime", ())),
                    sample_ids=tuple(manifest["sample_ids"]),
                    sample_rates=tuple(outlier.get("sample_rates", ())),
                    sample_rate_at_top=tuple(outlier.get("sample_rate_at_top", ())),
                    confidence_divisor=divisor,
                    path_divisor=path,
                    confidence_correlation=science.correlation(divisor, config.reference_snps_per_bin),
                    path_correlation=science.correlation(path, config.reference_snps_per_bin),
                    robustness_pooled=outlier["pooled"],
                    robustness_contigs=outlier["contigs"],
                    divisor_pooled=pooled,
                    divisor_contigs=contigs,
                    **({} if read_confidence is None else dict(read_confidence=read_confidence)),
                    seconds=perf_counter() - began,
                )
            publish(file, request.key, request.identity, payload)
    return request.identity


def fit_count(product, config):
    """Number of independent rate fits (zero when the grid fixes the rate)."""
    if product not in ("robustness", "refine") or len(set(config.robustness_grid)) == 1:
        return 0
    return 2 if product == "refine" else len(set(config.robustness_grid))


def task(key, context, *, gather_only=False, by_fold=False):
    product = PRODUCTS[key.stage]
    contig = None if key.scope == "run" else key.scope
    config = context.resolved.parameters.painting
    if product == "divisor_fit":
        outlier, read, material, grid = divisor_plan(context.root, context.resolved, context.maps)
        fold_text, _, value = key.round.partition("-")
        fold = int(fold_text)
        if not 0 <= fold < config.calibration_folds:
            raise SettingsError(f"--fold must be between 0 and {config.calibration_folds - 1}")
        if contig not in material or not grid:
            raise SettingsError(f"{contig}: no divisor fits on the recorded calibration route")
        if value:
            divisor = float(value)
            if divisor not in grid:
                raise SettingsError(f"--divisor {divisor} is outside the active grid {grid}")
            grid = (divisor,)
        return _divisor_task(
            key, context.root, context.resolved, context.maps, context.ties, outlier, read, fold, grid, context.cores,
        )
    outputs = (RecordKey(key.scope, records.KEYS[product]),)
    if product == "divisor":
        return Task(
            key, run_divisor, (context.root, contig, context.resolved, context.maps, context.ties),
            kwargs=dict(cores=context.cores, gather_only=gather_only, by_fold=by_fold), outputs=outputs,
        )
    threads = READ_SHARE_CORES if product == "share" and contig is not None else fit_count(product, config)
    return Task(
        key,
        run,
        (context.root, product, contig, context.resolved, context.maps, context.ties),
        kwargs=dict(cores=context.cores),
        outputs=outputs,
        resources=ResourceRequest(1, max(1, min(context.cores, threads))),
    )
