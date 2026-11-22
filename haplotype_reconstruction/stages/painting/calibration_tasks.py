"""CPU-budgeted per-contig fits and two durable calibration reductions.

A contig's calibration is a set of independent fits: one per candidate outlier rate (robustness), then one per fold
and candidate divisor (divisor). Each fit is its own one-CPU task (outlier_fit, divisor_fit) and a gather publishes
the contig's record (fits_publish). Run as one team per contig, the fits ran in sequence and the team idled through
every one-thread section of every fit: measured on the divisor step of one contig of 40 samples and 330,000 sites,
71 s of one-thread work and 830 CPU-seconds of kernel work, about half of 19 threads busy.
"""

from functools import partial
from math import fsum

import numpy as np
from time import perf_counter
from ...contracts.contigs import AnalysisContigs
from ...contracts.panels import materialise
from ...core.errors import InvariantError
from ...engine.records import record_diagnostic
from ...science.painting import calibration as science
from ...science.painting import read_calibration
from ...store.protocol import claim, publish, resume
from ..framework import Expansion, RecordKey, ResourceRequest, ResultRef, Task, TaskKey
from ..inputs.products import task_inputs, required_product
from . import calibration_records as records

PRODUCTS = dict(
    calibration_robustness="robustness",
    calibration_refine="refine",
    calibration_outlier="outlier",
    calibration_divisor="divisor",
    calibration="calibration",
)
# A contig's per-bin fields of its rate fits (the outlier profile's material units): kept in the contig records only.
BIN_FIELDS = ("material_coordinates", "bin_rows")


def _fit(root, contig, resolved, maps, cores, function, *, read_state=None, **options):
    """One calibration fit of a contig on its final assembly and evidence windows: (value, seconds). ``cores`` is the
    run's CPU count: it sets the sample batch as in every other painting task (the scores do not depend on it), not
    the fit's CPUs (one). ``read_state``: the run's read-sharing detection (run_detection's dict) when the emission
    builder carries the painting's read state (stages.painting.read_state; no confidence strength: the read-aware
    confidence fits build at each candidate), else None."""
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
    with read_evidence(store.contig(contig), ev.identity) as source:
        builder = partial(
            _at_batch_boundary,
            partial(windowed_emissions, sample_tile=batch, cohort=source.cohort_frequencies),
            batch,
        )
        if read_state is not None:
            from .read_state import stage_builder, with_detection

            builder = stage_builder(builder, store.contig(contig), ev.identity, source.cohort_frequencies, None,
                                    with_detection(resolved.parameters.painting, read_state), unit_batches=True)
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


def _pooled_rows(store, manifest, product, grid):
    """{rate: the analysed contigs' summed values} of one per-contig calibration product (math.fsum: the same sum in
    any contig order)."""
    values = {float(e): [] for e in grid}
    for c in AnalysisContigs.from_manifest(manifest).included:
        q = records.expected(store, manifest, product=product, contig=c)
        saved = resume(store.contig(c), q.key, q.identity)
        if saved is None:
            raise RuntimeError(f"calibration prerequisite missing: {product}/{c}")
        for e, v in saved.payload["rows"]:
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


def calibration_material(store, manifest, maps, config):
    """((contig, Morgans), ...): the analysed contigs the read-aware confidence fits run on, in configured order
    (read_calibration.material_prefix at PaintingParameters.calibration_morgans; contigs past the material are never
    read). Every contig's divisor task and the run reduction derive it alike from the bound evidence and map."""
    analysed = AnalysisContigs.from_manifest(manifest).included
    return read_calibration.material_prefix(
        ((c, contig_morgans(store, manifest, maps, c)) for c in analysed), config.calibration_morgans
    )


def read_divisor_record(root, contig, resolved, maps, ties, epsilon, detection, *, cores=1):
    """The read-aware confidence fits of a contig (read_calibration.confidence_unit: every fold and candidate in this
    task, one read-state builder whose cached label-independent parts serve them all): the contig's divisor payload,
    rows (divisor, fold, score, hidden observed cells, None: no path score) and sample_rows (divisor, fold, per-sample
    scores)."""
    rows, seconds = _fit(
        root, contig, resolved, maps, cores, read_calibration.confidence_unit, read_state=detection, epsilon=epsilon,
        reverse=ties.reverse,
    )
    return dict(
        schema=1, contig=contig, rows=tuple((c, f, fsum(s), cells, None) for c, f, s, cells in rows),
        sample_rows=tuple((c, f, s) for c, f, s, _ in rows), seconds=seconds, skipped=False,
    )


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
        return None
    ev = evidence_records.expected(store, manifest, product="evidence", contig=contig)
    detection = run_detection(store, manifest, config.read_state)
    libraries, splits = contig_read_libraries(store.contig(contig), ev.identity, with_detection(config, detection))
    return dict(libraries=libraries, splits=splits)


def fits_publish(root, product, contig, fits):
    """Publish a contig's record from its fits' rows in ascending order (the order of calibration.outlier_scores
    and calibration.divisor_scores); ``seconds`` is the fits' summed time."""
    reader, store = task_inputs(root)
    request = records.expected(store, reader.manifest, product=product, contig=contig)
    file = store.contig(contig)
    with claim(file, request.key, request.identity, blocking=False) as needed:
        if needed:
            seconds = fsum(seconds for _, seconds in fits)
            if product in ("robustness", "refine"):
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
                    payload["read_libraries"] = _read_sharing(store, reader.manifest, contig)
                    payload["read_sharing"] = (
                        None if payload["read_libraries"] is None else
                        any(row["status"] == "fitted" for row in payload["read_libraries"]["libraries"]))
            else:
                rows = tuple(sorted(tuple(row) for row, _ in fits))
                payload = dict(schema=1, contig=contig, rows=rows, seconds=seconds, skipped=False)
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
    grid = sorted(set(config.robustness_grid if product in ("robustness", "refine") else config.divisor_grid))
    if product == "divisor" and _read_route(store, manifest, config):
        # the read-aware confidence fits: every fold and candidate in this task, sharing one read-state builder, on the
        # contigs of the calibration material (PaintingParameters.calibration_morgans); any other contig fits nothing
        with claim(file, request.key, request.identity, blocking=False) as needed:
            if needed:
                began = perf_counter()
                material = dict(calibration_material(store, manifest, maps, config))
                if contig in material:
                    outlier = records.expected(store, manifest, product="outlier")
                    saved = resume(store.run, outlier.key, outlier.identity)
                    if saved is None:
                        raise RuntimeError("calibration prerequisite missing: outlier/None")
                    payload = read_divisor_record(
                        root, contig, resolved, maps, ties, saved.payload["robustness_epsilon"],
                        saved.payload["read_detection"], cores=cores,
                    )
                    payload = dict(payload, morgans=material[contig])
                else:
                    payload = dict(schema=1, contig=contig, rows=(), sample_rows=(), seconds=perf_counter() - began,
                                   skipped=True)
                publish(file, request.key, request.identity, payload)
        return request.identity
    if product in ("robustness", "refine", "divisor") and len(grid) > 1:
        # several candidate values: the contig's fits are independent one-CPU tasks (a single value fits nothing)
        if resume(file, request.key, request.identity) is not None:
            return request.identity
        options = dict(cores=cores)
        if product == "robustness":
            return _expansion(
                root, product, contig, [(outlier_fit, (root, contig, resolved, maps, value), options) for value in grid]
            )
        if product == "refine":
            # the geometric midpoints around the pooled grid maximum of every analysed contig's grid fits
            values = _pooled_rows(store, manifest, "robustness", grid)
            return _expansion(
                root, product, contig,
                [(outlier_fit, (root, contig, resolved, maps, value), options)
                 for value in science.midpoints(grid, [values[e] for e in grid])],
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

            if product in ("robustness", "refine", "divisor"):
                # a single-value grid fixes the quantity: nothing is fitted
                payload = dict(schema=1, contig=contig, rows=(), seconds=perf_counter() - began, skipped=True)
                if product == "robustness":
                    payload["read_libraries"] = _read_sharing(store, manifest, contig)
                    payload["read_sharing"] = (
                        None if payload["read_libraries"] is None else
                        any(row["status"] == "fitted" for row in payload["read_libraries"]["libraries"]))
            elif product == "outlier":
                # each contig's rows: the literal grid and the two midpoints around the pooled grid maximum (the
                # per-bin fields stay in the contig records: the run record keeps the material units they make)
                contigs, fitted = [], []
                for c in analysed:
                    coarse, refine = get("robustness", c), get("refine", c)
                    fitted.append((coarse, refine))
                    contigs.append(dict(
                        {k: v for k, v in coarse.items() if k not in BIN_FIELDS},
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
                            for e, values in get(which, c).get("sample_rows", ()):
                                total = per_sample.setdefault(float(e), np.zeros(len(values)))
                                total += np.asarray(values, dtype=np.float64)
                    if per_sample and all(e in per_sample for e in grid):
                        found, top = science.sample_rates(grid, np.stack([per_sample[e] for e in grid], axis=1))
                        rates, at_top = tuple(float(x) for x in found), tuple(bool(x) for x in top)
                # the run's read-state route (owner, 18 Nov): with no library of the run sharing reads, the read
                # model's exact limit is one read class, and the run takes the route without read state throughout
                from .read_state import read_state_route

                read_state = read_state_route(config, [get("robustness", c).get("read_sharing") for c in analysed])
                from .read_state import run_detection

                # the family-wise detection the contigs' read_sharing used (alpha, tests M, z*): recorded for the run
                detection = run_detection(store, manifest, config.read_state) if config.read_classes > 1 else None
                libraries, splits, warnings = [], [], []
                if detection is not None:
                    from .read_state import detection_margins

                    # each library's largest excess / SE and each candidate split's contrast against z* (signed
                    # margins); a warning line for any decision within a factor of two of z*
                    found = [(c, get("robustness", c).get("read_libraries") or {}) for c in analysed]
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
            else:
                outlier = get("outlier")
                contigs = [get("divisor", c) for c in analysed]
                grid = sorted(set(config.divisor_grid))
                read_confidence = None
                if outlier.get("read_state", "on") == "on" and config.read_classes > 1:
                    # the material's contigs only; the others published empty records (outside the map budget)
                    material = calibration_material(store, manifest, maps, config)
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


def _read_route(store, manifest, config):
    """Whether the run calibrates with the read state: read_classes > 1 and the outlier record's read_state 'on'
    (some library of the run shares reads; stages.painting.read_state.read_state_route)."""
    if config.read_classes == 1:
        return False
    from .read_state import READ_STATE_ON

    outlier = records.expected(store, manifest, product="outlier")
    saved = resume(store.run, outlier.key, outlier.identity)
    if saved is None:
        raise RuntimeError("calibration prerequisite missing: outlier/None")
    return saved.payload.get("read_state", READ_STATE_ON) == READ_STATE_ON


def fit_count(product, config):
    """The number of independent fits of one contig's ``product`` record (0: a single-value grid fits nothing; the
    refinement fits the two midpoints around the pooled grid maximum, one at a boundary maximum)."""
    grid = set(config.robustness_grid if product in ("robustness", "refine") else config.divisor_grid)
    if product == "divisor" and config.read_classes > 1:
        # the read-aware confidence fits run in the contig's own task with its team (read_divisor_record); a contig
        # outside the calibration material fits nothing, which is known only once the evidence exists
        return len(set(config.confidence_grid)) * config.calibration_folds
    if product not in ("robustness", "refine", "divisor") or len(grid) == 1:
        return 0
    if product == "refine":
        return 2
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
