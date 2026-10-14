"""Block-cavity refitting of assembled context into local founder proposals.

The current block's emission is excluded from its carrier weights. Original
reads then refit its alleles once; assembly proposals are not extra reads.
The shared founder-allele site posterior (``assembly.site_posterior``) is exact
for at most ten context founders and polynomial above, so every context
component is refitted. Partially covered contexts retain their input panels.
"""
import atexit
import copy
import time
import numpy as np
from numba import njit, prange, set_num_threads
from haplotype_reconstruction.core import parallel
from haplotype_reconstruction.core.haplotypes import BlockResult, BlockResults
from haplotype_reconstruction.assembly import site_posterior

# Per site, a copy state's allele probability q is 0 or 1 (a called founder) or the
# site background (an uncalled founder or the background state): codes 0, 1, 2. The
# emission term of a state pair depends only on its unordered code pair, terms 0-5
# for (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), or, for a founder paired with
# itself (both copies one founder), on its code: terms 6, 7, 8.
_EMISSION_TERMS = 9
_TERM_LOW = (0, 0, 0, 1, 1, 2)
_TERM_HIGH = (0, 1, 2, 1, 2, 2)
_CODE_PAIR_TERM = (0, 1, 2, 1, 3, 4, 2, 4, 5)


@njit(cache=True, inline='always')
def _code_probability(code, background):
    return background if code == 2 else float(code)


@njit(cache=True, parallel=True)
def emission_grid(h, gl, observed, starts, stops, mix):
    """Per sample and bin, the summed log emission of every copy-state pair.

    A pair's term at an observed site with a called founder is the log of the
    mix-robust genotype likelihood of its allele probabilities. Each sample
    evaluates the _EMISSION_TERMS distinct terms of a site once and every
    pair adds its term site by site in ascending order.
    """
    n, length, _ = gl.shape
    k = len(h)
    s = k + 1
    emission = np.zeros((n, len(starts), s, s))
    background = np.empty(length)
    informative = np.empty(length, dtype=np.bool_)
    code = np.empty((s, length), dtype=np.int64)
    for site in range(length):
        called = 0
        alt = 0
        for i in range(k):
            if h[i, site] >= 0:
                called += 1
                alt += h[i, site]
                code[i, site] = h[i, site]
            else:
                code[i, site] = 2
        code[k, site] = 2
        background[site] = (1.0 + alt) / (2.0 + called)
        informative[site] = called > 0
    width = 0
    for b in range(len(starts)):
        width = max(width, stops[b] - starts[b])
    pair_term = np.array(_CODE_PAIR_TERM)
    for sample in prange(n):
        terms = np.empty((width, _EMISSION_TERMS))
        for b in range(len(starts)):
            start, stop = starts[b], stops[b]
            for site in range(start, stop):
                if not observed[sample, site] or not informative[site]:
                    continue
                e = gl[sample, site]
                total = e.sum()
                for term in range(_EMISSION_TERMS):
                    if term < 6:
                        # Symmetric in the two copies (IEEE products commute).
                        qi = _code_probability(_TERM_LOW[term], background[site])
                        qj = _code_probability(_TERM_HIGH[term], background[site])
                        p0 = (1 - qi) * (1 - qj)
                        p2 = qi * qj
                        p1 = 1 - p0 - p2
                    else:
                        qi = _code_probability(term - 6, background[site])
                        p0 = 1 - qi
                        p1 = 0.0
                        p2 = qi
                    likelihood = (e[0] * p0 + e[1] * p1 + e[2] * p2) / total if total > 0 else 1 / 3
                    terms[site - start, term] = np.log((1 - mix) * likelihood + mix / 3)
            for i in range(s):
                for j in range(i, s):
                    value = 0.0
                    for site in range(start, stop):
                        if not observed[sample, site] or not informative[site]:
                            continue
                        term = (6 + code[i, site] if i == j and i < k
                                else pair_term[3 * code[i, site] + code[j, site]])
                        value += terms[site - start, term]
                    emission[sample, b, i, j] = value
                    emission[sample, b, j, i] = value
    return emission, background


@njit(cache=True)
def refresh_forward(x, pi, r):
    s = len(pi)
    out = np.empty_like(x)
    row = x.sum(axis=1)
    col = x.sum(axis=0)
    total = x.sum()
    for i in range(s):
        for j in range(s):
            out[i, j] = ((1 - r) ** 2 * x[i, j] + r * (1 - r) * (row[i] * pi[j] + pi[i] * col[j])
                        +r * r * pi[i] * pi[j] * total)
    return out

@njit(cache=True)
def refresh_backward(x, pi, r):
    s = len(pi)
    out = np.empty_like(x)
    row = x @ pi
    col = pi @ x
    total = pi @ row
    for i in range(s):
        for j in range(s):
            out[i, j] = (1 - r) ** 2 * x[i, j] + r * (1 - r) * (row[i] + col[j]) + r * r * total
    return out

@njit(cache=True, parallel=True)
def carrier_weights(log_emission, centers, rate, background_mass):
    n, bins, s, _ = log_emission.shape
    pi = np.full(s, (1 - background_mass) / (s - 1))
    pi[-1] = background_mass
    cavity = np.empty_like(log_emission)
    posterior = np.empty_like(log_emission)
    for sample in prange(n):
        predicted = np.empty((bins, s, s))
        emit = np.empty((bins, s, s))
        x = np.outer(pi, pi)
        for b in range(bins):
            if b:
                x = refresh_forward(x, pi, -np.expm1(-rate * (centers[b] - centers[b - 1])))
            predicted[b] = x
            emit[b] = np.exp(log_emission[sample, b] - np.max(log_emission[sample, b]))
            x = x * emit[b]
            x /= x.sum()
        beta = np.ones((s, s))
        for b in range(bins - 1, -1, -1):
            w = predicted[b] * beta
            w /= w.sum()
            cavity[sample, b] = w
            w = w * emit[b]
            w /= w.sum()
            posterior[sample, b] = w
            if b:
                beta = refresh_backward(beta * emit[b], pi, -np.expm1(-rate * (centers[b] - centers[b - 1])))
                beta /= beta.sum()
    return cavity, posterior

@njit(cache=True, parallel=True)
def refit_sites(result, probabilities, supporters, gl, observed, cavity, background, start, stop, threshold,
                site_info, site_boundary, site_withheld, settings):
    for site in prange(start, stop):
        q, support, shell, info, boundary = site_posterior.site_marginals(
            gl[:, site], observed[:, site], cavity, background[site], settings)
        probabilities[:, site] = q
        supporters[:, site] = support
        site_info[:, site] = info
        site_boundary[site] = boundary
        if info[site_posterior.INFO_MULTIMODAL]:
            # Two comparable, unconnected posterior basins: release nothing here.
            site_withheld[site] = 0
            continue
        withheld = 0
        for i in range(len(result)):
            # Carrier support (site_posterior): two samples each more likely than
            # not carrying founder i, or an expected carrier mass of two.
            if support[i] >= 2:
                if q[i] >= threshold or 1 - q[i] >= threshold:
                    # Shell guard (ball regime only; shell is zero otherwise).
                    if site_posterior.guard_withholds(q[i], shell[i], threshold, settings):
                        withheld += 1
                    elif q[i] >= threshold:
                        result[i, site] = 1
                    else:
                        result[i, site] = 0
        site_withheld[site] = withheld

def refit_component(
    h,
    gl,
    observed,
    starts,
    stops,
    centers,
    threshold,
    background_mass,
    rate
):
    parallel.apply_dynamic_threads()
    emission, background = emission_grid(h, gl, observed, starts, stops, .01)
    parallel.apply_dynamic_threads()
    cavity, posterior = carrier_weights(emission, centers, rate, background_mass)
    result = h.copy()
    probabilities = np.full(h.shape, .5)
    supporters = np.zeros(h.shape)
    site_info = np.zeros((site_posterior.INFO_FIELDS, h.shape[1]), np.int64)
    site_boundary = np.zeros(h.shape[1])
    site_withheld = np.zeros(h.shape[1], np.int64)
    settings = site_posterior.settings()
    for b in range(len(starts)):
        parallel.apply_dynamic_threads()
        refit_sites(result, probabilities, supporters, gl, observed, cavity[:, b], background,
                    starts[b], stops[b], threshold, site_info, site_boundary, site_withheld, settings)
    return (result, probabilities, supporters, posterior,
            site_posterior.summarize_sites(site_info, site_boundary, site_withheld))

_ARRAYS = None
_HANDLES = []
def initialize(shared, cpus, active, extra, startup):
    global _ARRAYS, _HANDLES
    set_num_threads(1)
    parallel.set_dynamic_thread_state(cpus, active, extra, **startup)
    values = [parallel.attach_shared_array(m) for m in shared]
    _HANDLES = [x[0] for x in values]
    _ARRAYS = [x[1] for x in values]
    atexit.register(parallel.close_shared_memory, _HANDLES)

def _worker(task):
    index, h, global_indices, starts, stops, centers, options = task
    gl, observed = _ARRAYS
    local_gl = np.ascontiguousarray(gl[:, global_indices], dtype=np.float64)
    local_observed = np.ascontiguousarray(observed[:, global_indices])
    started = time.monotonic()
    result, q, support, post, posterior_report = refit_component(
        h, local_gl, local_observed, starts, stops, centers,
        options["threshold"], options["background_mass"], options["rate"])
    wildcard = np.empty((len(starts), len(gl)), np.int8)
    for b in range(len(starts)):
        pairs = post[:, b].reshape(len(gl), -1).argmax(axis=1)
        k = len(h)
        wildcard[b] = (pairs // (k + 1) == k).astype(np.int8) + (pairs % (k + 1) == k)
    return index, result, q, support, wildcard, time.monotonic() - started, posterior_report

def worker(task):
    parallel.increment_active()
    try:
        return _worker(task)
    finally:
        set_num_threads(1)
        parallel.release_dynamic_extra()
        parallel.decrement_active()

def refine(raw_blocks, super_blocks, gl, sites, observed, cpus, options, chromosome_map=None,
           n_generations=3):
    """Replace covered panels, exact-dedup; preserve unsupported template alleles.

    ``cpus``: an int, or a callable live budget (a contig lane's growing lease):
    the pool's process count follows its value at the start and the workers'
    shared thread total its growth (core.parallel.live_total).
    """
    budget, cpus = cpus, parallel.budget_value(cpus)
    originals = list(raw_blocks)
    output = BlockResults(list(originals))
    observed = observed.copy()
    for block in originals:
        if block.keep_flags is not None:
            indices = np.searchsorted(sites, block.positions)
            observed[:, indices] &= np.asarray(block.keep_flags) > 0
    raw_first = np.array([b.positions[0] for b in originals])
    raw_last = np.array([b.positions[-1] for b in originals])
    specs = {}
    tasks = []
    skipped = []
    for super_index, super_block in enumerate(super_blocks):
        positions = np.asarray(super_block.positions)
        block_ids = np.flatnonzero((raw_first >= positions[0]) & (raw_last <= positions[-1]))
        if not len(block_ids):
            continue
        expected = np.concatenate([originals[b].positions for b in block_ids])
        if not np.array_equal(expected, positions):
            skipped.append((super_index, "incomplete_original_blocks"))
            continue
        # Exclude completion fills from carrier evidence, as the canonical painter does.
        inference = np.asarray(
            getattr(super_block, "missing_aware_inference_discrete_haps", super_block.discrete_haps),
            np.int8
        )
        _, representatives = np.unique(inference, axis=0, return_index=True)
        representatives = np.sort(representatives)
        h = np.ascontiguousarray(inference[representatives])
        template = np.ascontiguousarray(super_block.discrete_haps[representatives])
        starts = np.r_[0, np.cumsum([len(originals[b].positions) for b in block_ids])[:-1]].astype(np.int64)
        stops = np.r_[starts[1:], len(positions)].astype(np.int64)
        centers = np.array([np.mean(originals[b].positions) for b in block_ids])
        local_options = dict(options)
        if chromosome_map is not None:
            if chromosome_map.has_map:
                # Rate is now per Morgan, not per bp. The map is integrated
                # between block centres, including its configured tails.
                centers = chromosome_map.cumulative_morgans(centers)
                local_options["rate"] = float(n_generations)
            else:
                local_options["rate"] = chromosome_map.fallback_rate_per_bp * n_generations
        indices = np.searchsorted(sites, positions)
        assert np.array_equal(sites[indices], positions)
        specs[super_index] = (block_ids, starts, stops, template)
        tasks.append((super_index, h, indices, starts, stops, centers, local_options))
    handles = []
    metadata = []
    try:
        for array in (gl, observed):
            handle, meta = parallel.create_shared_array(array, copy_threads=cpus)
            handles.append(handle)
            metadata.append(meta)
        diagnostics = []
        completed = 0
        workers = min(cpus, len(tasks))
        if not workers:
            return output, dict(
                components=[],
                skipped=skipped,
                options=options,
                changed_alleles=0,
                filled_alleles=0
            )
        context = parallel.forkserver_context
        active = context.Value("i", 0)
        extra = context.Value("i", 0)
        startup = {name: context.Value("i", value) for name, value in dict(
            started_counter=0, participant_counter=0, batch_generation=0,
            batch_task_count=len(tasks), startup_target=workers, startup_ready=0).items()}
        with parallel.live_total(budget) as live, parallel.ForkserverPool(
                workers, initializer=initialize,
                initargs=(metadata, cpus, active, extra, dict(startup, live_total=live))) as pool:
            for index, result, q, support, wildcard, seconds, posterior_report in pool.imap_unordered(
                worker,
                tasks,
                chunksize=1
            ):
                block_ids, starts, stops, template = specs[index]
                # If the frozen inference template was unknown but completion had filled it,
                # retain that existing release unless the new refit supports a concrete call.
                fallback = (result < 0) & (template >= 0)
                result[fallback] = template[fallback]
                changes = int(((result >= 0) & (template >= 0) & (result != template)).sum())
                fills = int(((result >= 0) & (template < 0)).sum())
                diagnostics.append(dict(component=index, founders=len(result), seconds=seconds,
                                         allele_changes=changes, filled_alleles=fills,
                                         site_posterior=posterior_report))
                for b, lo, hi, slots in zip(block_ids, starts, stops, wildcard):
                    panel = result[:, lo:hi]
                    _, unique = np.unique(panel, axis=0, return_index=True)
                    unique = np.sort(unique)
                    panel = np.ascontiguousarray(panel[unique])
                    old = originals[b]
                    block = BlockResult(old.positions.copy(),
                        {i: np.stack((np.where(row < 0, .5, 1 - row), np.where(row < 0, .5, row)), axis=-1)
                         for i, row in enumerate(panel)}, keep_flags=copy.deepcopy(old.keep_flags),
                         genotype_evidence_mode=old.genotype_evidence_mode)
                    block.discrete_haps = panel
                    # The distinct local sequences of this context's founders:
                    # path selection codes rows they claim as founder parameters.
                    block.context_founder_rows = panel.copy()
                    block.n_directional_site_supporters = support[unique, lo:hi].copy()
                    block.feedback_posterior_alt = q[unique, lo:hi].copy()
                    block.founder_alt_pseudo_probability = q[unique, lo:hi].copy()
                    start = np.searchsorted(sites, old.positions[0])
                    stop = start + len(old.positions)
                    depth = np.any(
                        observed[:, start:stop] & np.asarray(old.keep_flags if old.keep_flags is not None else np.ones(stop - start), bool),
                        axis=1
                    )
                    block.sample_has_observed_kept_depth = depth
                    block.wildcard_slots = np.where(depth, slots, 2).astype(np.int8)
                    block.wildcard_mass = float(slots[depth].sum() / max(2 * depth.sum(), 1))
                    for name in ("missing_aware_break_before", "missing_aware_break_after"):
                        if hasattr(old, name):
                            setattr(block, name, getattr(old, name))
                    # Do not inherit old selected-path or immutable input identities.
                    output.blocks[b] = block
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    print(
                        "FEEDBACK",
                        completed,
                        len(tasks),
                        "changes",
                        sum(x["allele_changes"] for x in diagnostics),
                        flush=True
                    )
                if completed == len(tasks):
                    totals = {key: sum(x["site_posterior"][key] for x in diagnostics) for key in (
                        "cube_sites", "core_cube_sites", "core_ball_sites", "bp_nonconverged",
                        "ascent_backstop_hits", "recentre_backstop_hits", "complement_mode_sites",
                        "multimodal_abstained", "guard_withheld")}
                    print("FEEDBACK site posterior", " ".join(f"{key}={value}" for key, value in totals.items()),
                          "max_founders", max(x["founders"] for x in diagnostics), flush=True)
        # Pool results arrive in completion order; report them by component.
        diagnostics.sort(key=lambda item: item["component"])
        return output, dict(components=diagnostics, skipped=skipped, options=options,
                           changed_alleles=sum(x["allele_changes"] for x in diagnostics),
                           filled_alleles=sum(x["filled_alleles"] for x in diagnostics))
    finally:
        parallel.close_shared_memory(handles, unlink=True)
