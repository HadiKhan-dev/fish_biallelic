"""Label-continuous founder rows: relabel row suffixes before component painting.

Assembly builds each founder row of a phase component as a path through the
local rows of consecutive atomic (prepared) blocks. Next to stretches where
rows are locally interchangeable (identical copies inside a founder's run of
homozygosity, haplotypes shared between founders), linking and refinement can
let two rows exchange founder identity at a link point, the boundary between
two atomic blocks. The alleles at every site stay right as a set, but painting
reads label continuity as haplotype continuity: every sample that carries
either haplotype across the point pays a track switch there, or keeps its
labels by exchanging its two tracks, which flips a parent's phase frame and
puts a false crossover into every child.

Model. Exchanging the labels of rows a and b at every site from link point j
onwards (a suffix transposition) leaves the set of row alleles at every site
unchanged. The painting state space (whole-component trajectory classes plus
BACKGROUND) is renamed, not changed, so each emission of the painting HMM is
unchanged except in the one bin that straddles j, whose sites left of j keep
their labels. Only which ordered state sequences count as continuous changes.
A transposition t at j is scored with the painting model's own Viterbi
objective, for every sample s,

    V_s(t) = max_x [ G_s(x) + L_s(x) + Q_s(t(x)) ],   Q_s = P_s + B_s,

where G_s is the max-product forward message into the bin holding j (every bin
to its left, with the current labels), L_s and P_s are that bin's emissions on
its sites left and right of j, and B_s is the backward message from that bin
(every bin to its right). Backward messages of the relabelled panel are the
current ones renamed by t, and the Hamming transition kernel is invariant when
both of its arguments are renamed, so V_s(t) is the exact best-path score of
the relabelled panel: paths re-optimise on both sides, and a switch that sat
anywhere inside a stretch of identical rows vanishes when the relabelling makes
the haplotype continuous. One painted track switch costs |log P(one track
change)| (about 109 nats at the default painting settings), which dominates
V_s, so a gain is a fall in the painted switches that
``painting.model.minimum_unordered_switch_counts`` counts.

Acceptance uses no pedigree, generation label or truth. Candidates are ranked
by their summed score gain, and a candidate is considered only when that gain
is at least ``minimum_switch_gain`` painted switches (u, the cost of one track
change, median over the component's bin edges). It is then decided on painted
switches exactly as painting reports them: every sample is re-decoded with the
painting Viterbi under the candidate labels, and its minimum-unordered switch
count is compared with the current one. The candidate is accepted when the
total falls by at least ``minimum_switch_gain`` and the samples whose count
rises (losers) are at most ``maximum_loser_ratio`` times the samples whose count
falls (winners). Under a label exchange the only losers are carriers with a real
change of haplotype inside the link interval. Under a real founder-to-child
crossover inherited by one lineage, every other carrier of the two haplotypes
loses a whole switch, so the ratio rule keeps such a crossover in the rows even
when the recombinant lineage is the majority (one unsequenced F1 can parent half
of an F2 cohort). Score changes are not used to classify samples: a sample whose
path re-routes at an emission cost, with no added switch, is not a loser. The
first acceptable candidate is applied, every message is recomputed, and the
search repeats until no candidate is acceptable; each accepted move lowers the
painted switch total, so the search terminates. Only singleton, anchored
trajectory classes move; pooled duplicate rows and BACKGROUND keep their labels.
The relabelled component is kept only when repainting it with the painting
functions also lowers the total.

Cost. A transposition t of rows a and b fixes every state that involves
neither, so the swapped maximum at a link is the larger of the best unaffected
identity term (the first entry outside rows and columns a and b of the
identity terms, sorted once per link and sample) and the 4S - 4 affected states
evaluated under t: O(links N S^3) per iteration, with the same maxima and the
same summation order as a scan over all S^2 states, hence identical gains.
A candidate check re-decodes only the samples for which the best unaffected
state does not strictly beat every affected state under both the identity and
t. For any other sample every optimal relabelled path crosses the link in a
state that t fixes; renaming such a path's suffix by t (t applied to both
arguments preserves the Hamming kernel, the emissions and the
minimum-unordered switch count) maps the optimal relabelled paths one to one
onto the optimal current paths, which cross the link in the same unaffected
states, with equal scores and switch counts. Its painted switches therefore
cannot change (barring exact score ties between optimal paths with different
switch counts), and each sample's Viterbi decode depends only on its own
emissions, so the re-decoded samples are bitwise the same as in a full decode.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from numba import njit, prange

from . import evidence as painting_evidence
from . import model as painting_model

LABEL_CONTINUITY_MODEL = "suffix-transposition-painting-viterbi-switch-count-v1"

# Founder-axis arrays of an assembled component (founders first). Each is
# relabelled with the same per-segment row map; positions, flags and atomic
# span geometry carry no founder axis and are shared unchanged.
_SITE_ROW_ATTRIBUTES = (
    "discrete_haps",
    "missing_aware_inference_discrete_haps",
    "founder_alt_pseudo_probability",
    "n_directional_site_supporters",
)
_SPAN_ROW_ATTRIBUTE = "missing_aware_atomic_source_row_paths"


@njit(cache=True)
def _max_transition(values, cost_zero, cost_one, cost_two, n_states, out,
                    float_work, index_work):
    """out[z] = max_y values[y] + T(y, z) over ordered founder-pair states.

    T charges cost_zero, cost_one or cost_two for zero, one or two changed
    tracks, exactly as painting.model.viterbi_label_grid. O(S^2) values only.
    """
    row_first = float_work[0]
    row_second = float_work[1]
    column_first = float_work[2]
    column_second = float_work[3]
    row_first_column = index_work[0]
    column_first_row = index_work[1]
    for index in range(n_states):
        row_first[index] = -np.inf
        row_second[index] = -np.inf
        column_first[index] = -np.inf
        column_second[index] = -np.inf
        row_first_column[index] = -1
        column_first_row[index] = -1
    for row in range(n_states):
        for column in range(n_states):
            value = values[row * n_states + column]
            if value > row_first[row]:
                row_second[row] = row_first[row]
                row_first[row] = value
                row_first_column[row] = column
            elif value > row_second[row]:
                row_second[row] = value
            if value > column_first[column]:
                column_second[column] = column_first[column]
                column_first[column] = value
                column_first_row[column] = row
            elif value > column_second[column]:
                column_second[column] = value
    for row in range(n_states):
        # Best entry outside this row, per column, then outside one column.
        complement_first = -np.inf
        complement_second = -np.inf
        complement_first_column = -1
        for column in range(n_states):
            value = (column_first[column] if column_first_row[column] != row
                     else column_second[column])
            if value > complement_first:
                complement_second = complement_first
                complement_first = value
                complement_first_column = column
            elif value > complement_second:
                complement_second = value
        for column in range(n_states):
            best = values[row * n_states + column] + cost_zero
            same_row = (row_first[row] if row_first_column[row] != column
                        else row_second[row])
            same_column = (column_first[column] if column_first_row[column] != row
                           else column_second[column])
            candidate = max(same_row, same_column) + cost_one
            if candidate > best:
                best = candidate
            candidate = (complement_first if complement_first_column != column
                         else complement_second) + cost_two
            if candidate > best:
                best = candidate
            out[row * n_states + column] = best


@njit(cache=True)
def _centre(values):
    largest = -np.inf
    for value in values:
        if value > largest:
            largest = value
    for index in range(len(values)):
        values[index] -= largest


@njit(cache=True, parallel=True)
def _link_messages(emissions, cost_zero, cost_one, cost_two, n_states, slot_of_bin, n_slots):
    """Max-product forward messages into, and backward messages from, chosen bins.

    forward[s, k] = max over paths through bins < b of their score plus the
    transition into each state of b (zero for the first bin); backward[s, k] =
    max over continuations after b of their transitions and emissions. Both
    exclude bin b's own emission and are centred per bin; the centring
    constants cancel in every gain because one link combines one forward and
    one backward message for the identity and every relabelling alike.
    """
    n_samples, n_diplotypes, n_bins = emissions.shape
    forward = np.zeros((n_samples, n_slots, n_diplotypes))
    backward = np.zeros((n_samples, n_slots, n_diplotypes))
    for sample in prange(n_samples):
        alpha = np.empty(n_diplotypes)
        message = np.empty(n_diplotypes)
        float_work = np.empty((4, n_states))
        index_work = np.empty((2, n_states), dtype=np.int64)
        for state in range(n_diplotypes):
            alpha[state] = emissions[sample, state, 0]
        _centre(alpha)
        for bin_index in range(1, n_bins):
            _max_transition(alpha, cost_zero[bin_index - 1], cost_one[bin_index - 1],
                            cost_two[bin_index - 1], n_states, message,
                            float_work, index_work)
            _centre(message)
            slot = slot_of_bin[bin_index]
            if slot >= 0:
                for state in range(n_diplotypes):
                    forward[sample, slot, state] = message[state]
            for state in range(n_diplotypes):
                alpha[state] = message[state] + emissions[sample, state, bin_index]
            _centre(alpha)
        beta = np.zeros(n_diplotypes)
        for bin_index in range(n_bins - 1, -1, -1):
            slot = slot_of_bin[bin_index]
            if slot >= 0:
                for state in range(n_diplotypes):
                    backward[sample, slot, state] = beta[state]
            if bin_index > 0:
                for state in range(n_diplotypes):
                    message[state] = emissions[sample, state, bin_index] + beta[state]
                _max_transition(message, cost_zero[bin_index - 1], cost_one[bin_index - 1],
                                cost_two[bin_index - 1], n_states, beta,
                                float_work, index_work)
                _centre(beta)
    return forward, backward


@njit(cache=True, parallel=True)
def _relabelled_emissions(part_emissions, samples, part_bin, part_segment, state_maps, n_bins):
    """Bin emissions of the current labelling, for ``samples``, from emissions on bin parts.

    state_maps[segment, x] is the original state whose content the current
    state x carries in that segment; a bin straddling a link sums its parts.
    Each row is computed from its own sample alone, in the same order.
    """
    _, n_diplotypes, n_parts = part_emissions.shape
    result = np.zeros((len(samples), n_diplotypes, n_bins))
    for row in prange(len(samples)):
        sample = samples[row]
        for part in range(n_parts):
            bin_index = part_bin[part]
            segment = part_segment[part]
            for state in range(n_diplotypes):
                result[row, state, bin_index] += part_emissions[
                    sample, state_maps[segment, state], part]
    return result


@njit(cache=True, parallel=True)
def _link_scores(part_emissions, part_segment, state_maps, link_slot, link_segment,
                 link_first_part, link_stop_part, forward, backward):
    """Per link: left = forward + bin parts left of the link; right = parts at or after it + backward."""
    n_samples, n_diplotypes, _ = part_emissions.shape
    n_links = len(link_slot)
    left = np.empty((n_links, n_samples, n_diplotypes))
    right = np.empty((n_links, n_samples, n_diplotypes))
    for task in prange(n_links * n_samples):
        link = task // n_samples
        sample = task % n_samples
        slot = link_slot[link]
        for state in range(n_diplotypes):
            left[link, sample, state] = forward[sample, slot, state]
            right[link, sample, state] = backward[sample, slot, state]
        for part in range(link_first_part[link], link_stop_part[link]):
            segment = part_segment[part]
            target = left if segment < link_segment[link] else right
            for state in range(n_diplotypes):
                target[link, sample, state] += part_emissions[
                    sample, state_maps[segment, state], part]
    return left, right


@njit(cache=True, parallel=True)
def _transposition_gains(left, right, eligible, pairs, n_states):
    """Summed exact Viterbi-score gain of each (link, transposition) over eligible samples.

    Unaffected states keep their identity terms, so per (link, sample) the
    terms are sorted once and each pair takes the first sorted entry outside
    its rows and columns, then evaluates only its 4S - 4 affected states.
    """
    n_links, n_samples, n_diplotypes = left.shape
    n_pairs = pairs.shape[0]
    total = np.zeros((n_links, n_pairs))
    for link in prange(n_links):
        terms = np.empty(n_diplotypes)
        relabel = np.arange(n_states)
        for sample in range(n_samples):
            if not eligible[sample]:
                continue
            identity = -np.inf
            for state in range(n_diplotypes):
                value = left[link, sample, state] + right[link, sample, state]
                terms[state] = value
                if value > identity:
                    identity = value
            order = np.argsort(terms)
            for pair in range(n_pairs):
                first_row = pairs[pair, 0]
                second_row = pairs[pair, 1]
                swapped = -np.inf
                for position in range(n_diplotypes - 1, -1, -1):
                    state = order[position]
                    row = state // n_states
                    column = state % n_states
                    if (row != first_row and row != second_row
                            and column != first_row and column != second_row):
                        swapped = terms[state]
                        break
                relabel[first_row] = second_row
                relabel[second_row] = first_row
                for row in (first_row, second_row):
                    for column in range(n_states):
                        value = (left[link, sample, row * n_states + column]
                                 + right[link, sample,
                                         relabel[row] * n_states + relabel[column]])
                        if value > swapped:
                            swapped = value
                for column in (first_row, second_row):
                    for row in range(n_states):
                        if row == first_row or row == second_row:
                            continue
                        value = (left[link, sample, row * n_states + column]
                                 + right[link, sample, row * n_states + relabel[column]])
                        if value > swapped:
                            swapped = value
                relabel[first_row] = first_row
                relabel[second_row] = second_row
                total[link, pair] += swapped - identity
    return total


@njit(cache=True, parallel=True)
def _candidate_samples(left, right, eligible, candidate_links, candidate_pairs, pairs,
                       n_states):
    """Per candidate, the eligible samples whose painted switches it can change.

    False only when the best state involving neither transposed row strictly
    beats every affected state, under the identity and under the transposition.
    """
    n_candidates = len(candidate_links)
    n_samples = left.shape[1]
    result = np.zeros((n_candidates, n_samples), dtype=np.bool_)
    for task in prange(n_candidates * n_samples):
        candidate = task // n_samples
        sample = task % n_samples
        if not eligible[sample]:
            continue
        link = candidate_links[candidate]
        first_row = pairs[candidate_pairs[candidate], 0]
        second_row = pairs[candidate_pairs[candidate], 1]
        unaffected = -np.inf
        affected = -np.inf
        for row in range(n_states):
            moved_row = row
            if row == first_row:
                moved_row = second_row
            elif row == second_row:
                moved_row = first_row
            for column in range(n_states):
                state = row * n_states + column
                value = left[link, sample, state] + right[link, sample, state]
                if moved_row == row and column != first_row and column != second_row:
                    if value > unaffected:
                        unaffected = value
                    continue
                if value > affected:
                    affected = value
                moved_column = column
                if column == first_row:
                    moved_column = second_row
                elif column == second_row:
                    moved_column = first_row
                value = (left[link, sample, state]
                         + right[link, sample, moved_row * n_states + moved_column])
                if value > affected:
                    affected = value
        result[candidate, sample] = not unaffected > affected
    return result


def _transition_costs(centers, n_states, *, recomb_rate, switch_penalty_per_snp,
                      snps_per_bin, double_recomb_factor, chromosome_map=None):
    """Per bin-edge log transition costs of painting.model.viterbi_label_grid."""
    genetic = None
    if chromosome_map is not None:
        recomb_rate = chromosome_map.fallback_rate_per_bp
        if chromosome_map.has_map:
            genetic = np.asarray(chromosome_map.interval_morgans(centers[:-1], centers[1:]),
                                 dtype=np.float64)
    distance = np.maximum(np.diff(np.asarray(centers, dtype=np.float64)), 1.0)
    theta = np.minimum(0.5, np.maximum(1e-15, distance * recomb_rate))
    if genetic is not None:
        theta = np.minimum(0.5, np.maximum(1e-15, genetic))
    log_switch = np.log(theta) - switch_penalty_per_snp * snps_per_bin
    log_stay = np.log(1.0 - theta)
    log_n_minus_one = math.log(float(n_states - 1)) if n_states > 1 else 0.0
    cost_zero = 2.0 * log_stay
    cost_one = log_switch + log_stay - log_n_minus_one
    cost_two = double_recomb_factor * log_switch - 2.0 * log_n_minus_one
    return (np.ascontiguousarray(cost_zero), np.ascontiguousarray(cost_one),
            np.ascontiguousarray(cost_two), genetic)


def _state_maps(perms, n_states):
    return np.ascontiguousarray(
        (perms[:, :, None] * n_states + perms[:, None, :]).reshape(len(perms), -1))


def _painted_switches(emissions, binning, n_states, eligible, painting):
    """Per-sample painted switches: painting's Viterbi, minimum unordered count."""
    grid = painting_model.viterbi_label_grid(
        np.ascontiguousarray(emissions), binning.centers, n_states, painting["recomb_rate"],
        painting["switch_penalty_per_snp"], painting["snps_per_bin"],
        painting["double_recomb_factor"], painting["genetic_distances"])
    switches = painting_model.minimum_unordered_switch_counts(grid)
    switches[~eligible] = 0
    return switches


def _relabel_block(block, row_maps, position_bounds, span_bounds):
    """Copy ``block`` with each founder-axis array relabelled per segment."""
    keys = tuple(sorted(block.haplotypes))
    n_rows = len(keys)
    identity = np.arange(n_rows)
    shape = (n_rows, len(block.positions))
    known = set(_SITE_ROW_ATTRIBUTES) | {_SPAN_ROW_ATTRIBUTE}
    for name, value in vars(block).items():
        if (name not in known and isinstance(value, np.ndarray)
                and value.ndim == 2 and value.shape == shape):
            raise ValueError(f"unrelabelled founder-axis array on component: {name}")

    def relabel(array, bounds):
        source = np.asarray(array)
        result = source.copy()
        for (start, stop), row_map in zip(bounds, row_maps):
            if not np.array_equal(row_map, identity):
                result[:, start:stop] = source[row_map, start:stop]
        return result

    relabelled = copy.copy(block)
    for name in _SITE_ROW_ATTRIBUTES:
        value = getattr(block, name, None)
        if value is not None:
            setattr(relabelled, name, relabel(value, position_bounds))
    relabelled.missing_aware_inference_discrete_haps = np.ascontiguousarray(
        relabelled.missing_aware_inference_discrete_haps, dtype=np.int8)
    relabelled.missing_aware_inference_discrete_haps.setflags(write=False)
    if getattr(block, _SPAN_ROW_ATTRIBUTE, None) is not None:
        setattr(relabelled, _SPAN_ROW_ATTRIBUTE,
                relabel(getattr(block, _SPAN_ROW_ATTRIBUTE), span_bounds))
    haplotypes = {}
    for row, key in enumerate(keys):
        value = np.array(block.haplotypes[key], copy=True)
        for (start, stop), row_map in zip(position_bounds, row_maps):
            if row_map[row] != row:
                value[start:stop] = np.asarray(block.haplotypes[keys[row_map[row]]])[start:stop]
        haplotypes[key] = value
    relabelled.haplotypes = haplotypes
    return relabelled


def relabel_component(block, evidence, observed, kept, *, recomb_rate,
                      switch_penalty_per_snp, robustness_epsilon,
                      double_recomb_factor, snps_per_bin, chromosome_map=None,
                      minimum_switch_gain=2.0, maximum_loser_ratio=0.1):
    """Return ``(component, diagnostic)``, relabelled when that lowers painted switches.

    ``evidence``/``observed`` are the component's raw sample evidence and
    observation mask on its own positions and ``kept`` its retained-site mask,
    exactly as ComponentPainter passes them to the painting model.
    """
    from ..assembly import observations as assembly_observations, paths as assembly_paths

    diagnostic = {"model": LABEL_CONTINUITY_MODEL, "moves": [], "accepted": False}
    panel = assembly_observations.founder_inference_panel_from_block_result(block)
    space = painting_model.build_ragged_state_space(panel, kept)
    classes = space.equivalence_classes
    anchored = np.any(space.called, axis=1)
    movable = [index for index, members in enumerate(classes)
               if len(members) == 1 and anchored[index]]
    _, offsets, _, _ = assembly_paths.missing_aware_atomic_source_provenance(block)
    site_indices = np.asarray(space.site_indices)
    n_selected = len(site_indices)
    links = []
    seen = set()
    for span in range(1, len(offsets)):
        first = int(np.searchsorted(site_indices, offsets[span]))
        if 0 < first < n_selected and first not in seen:
            seen.add(first)
            links.append((first, span))
    binning = painting_model.build_ragged_bins(space.positions, space.active, snps_per_bin)
    n_bins = len(binning.indices)
    diagnostic.update(link_points=len(links), movable_rows=len(movable))
    if len(movable) < 2 or not links or n_bins < 2:
        diagnostic["reason"] = "nothing_to_relabel"
        return block, diagnostic
    selected_evidence, selected_observed = painting_evidence.select_site_evidence(
        evidence, observed, site_indices)
    eligible = np.asarray(painting_model.samples_with_ragged_evidence(
        selected_evidence, selected_observed), dtype=np.bool_)
    if not np.any(eligible):
        diagnostic["reason"] = "no_evidence"
        return block, diagnostic

    # Split painting bins at link points; part emissions use the painting's own kernel.
    link_sites = np.asarray([first for first, _ in links], dtype=np.int64)
    parts, part_bin = [], []
    for bin_index, indices in enumerate(binning.indices):
        start, stop = int(indices[0]), int(indices[-1]) + 1
        cuts = link_sites[(link_sites > start) & (link_sites < stop)]
        for left, right in zip(np.r_[start, cuts], np.r_[cuts, stop]):
            parts.append(np.arange(left, right))
            part_bin.append(bin_index)
    part_bin = np.asarray(part_bin, dtype=np.int64)
    part_start = np.asarray([part[0] for part in parts], dtype=np.int64)
    part_segment = np.searchsorted(link_sites, part_start, side="right").astype(np.int64)
    part_binning = painting_model.RaggedBinning(
        tuple(parts), np.zeros(len(parts)), np.zeros(len(parts) + 1, dtype=np.int64),
        np.asarray([len(part) for part in parts], dtype=np.int64))
    part_emissions, _ = painting_model.calculate_ragged_binned_emissions(
        selected_evidence, selected_observed, space, part_binning,
        robustness_epsilon=robustness_epsilon)
    part_emissions = np.ascontiguousarray(part_emissions)

    n_states = space.background_index + 1
    cost_zero, cost_one, cost_two, genetic = _transition_costs(
        binning.centers, n_states, recomb_rate=recomb_rate,
        switch_penalty_per_snp=switch_penalty_per_snp, snps_per_bin=snps_per_bin,
        double_recomb_factor=double_recomb_factor, chromosome_map=chromosome_map)
    unit = float(np.median(np.abs(cost_one)))
    painting = dict(recomb_rate=(recomb_rate if chromosome_map is None
                                 else chromosome_map.fallback_rate_per_bp),
                    switch_penalty_per_snp=switch_penalty_per_snp,
                    snps_per_bin=snps_per_bin, double_recomb_factor=double_recomb_factor,
                    genetic_distances=genetic)

    # A link's messages live at the bin holding its first right-hand site.
    bin_of_site = np.empty(n_selected, dtype=np.int64)
    for bin_index, indices in enumerate(binning.indices):
        bin_of_site[indices] = bin_index
    link_bin = bin_of_site[link_sites]
    slot_bins = np.unique(link_bin)
    slot_of_bin = np.full(n_bins, -1, dtype=np.int64)
    slot_of_bin[slot_bins] = np.arange(len(slot_bins))
    link_slot = slot_of_bin[link_bin]
    link_segment = np.arange(1, len(links) + 1, dtype=np.int64)
    first_part_of_bin = np.searchsorted(part_bin, np.arange(n_bins), side="left")
    stop_part_of_bin = np.searchsorted(part_bin, np.arange(n_bins), side="right")
    link_first_part = first_part_of_bin[link_bin].astype(np.int64)
    link_stop_part = stop_part_of_bin[link_bin].astype(np.int64)
    pairs = np.asarray([(first, second) for index, first in enumerate(movable)
                        for second in movable[index + 1:]], dtype=np.int64)

    all_samples = np.arange(len(eligible), dtype=np.int64)
    perms = np.tile(np.arange(n_states, dtype=np.int64), (len(links) + 1, 1))
    maps = _state_maps(perms, n_states)
    emissions = _relabelled_emissions(part_emissions, all_samples, part_bin, part_segment,
                                      maps, n_bins)
    current = _painted_switches(emissions, binning, n_states, eligible, painting)
    switches_before = int(np.sum(current))
    checked = 0
    while True:
        forward, backward = _link_messages(emissions, cost_zero, cost_one, cost_two,
                                           n_states, slot_of_bin, len(slot_bins))
        left, right = _link_scores(part_emissions, part_segment, maps, link_slot,
                                   link_segment, link_first_part, link_stop_part,
                                   forward, backward)
        del forward, backward
        total = _transposition_gains(left, right, eligible, pairs, n_states)
        # Rank by the exact score gain; accept on exact per-sample painted switches.
        candidates = []
        for flat in np.argsort(-total, axis=None, kind="stable"):
            if total.flat[flat] < minimum_switch_gain * unit:
                break
            candidates.append(divmod(int(flat), len(pairs)))
        if candidates:
            candidate_links, candidate_pairs = np.asarray(candidates, dtype=np.int64).T
            redecode = _candidate_samples(
                left, right, eligible, np.ascontiguousarray(candidate_links),
                np.ascontiguousarray(candidate_pairs), pairs, n_states)
        del left, right
        accepted = None
        for index, (link, pair) in enumerate(candidates):
            first, second = (int(value) for value in pairs[pair])
            trial = perms.copy()
            trial[link + 1:, [first, second]] = trial[link + 1:, [second, first]]
            trial_maps = _state_maps(trial, n_states)
            # The other samples' painted switches cannot change (module docstring).
            samples = np.flatnonzero(redecode[index])
            change = np.zeros_like(current)
            if len(samples):
                change[samples] = _painted_switches(
                    _relabelled_emissions(part_emissions, samples, part_bin, part_segment,
                                          trial_maps, n_bins),
                    binning, n_states, eligible[samples], painting) - current[samples]
            checked += 1
            winners, losers = np.flatnonzero(change < 0), np.flatnonzero(change > 0)
            if (-int(np.sum(change)) >= minimum_switch_gain
                    and len(losers) <= maximum_loser_ratio * len(winners)):
                trial_emissions = _relabelled_emissions(part_emissions, all_samples, part_bin,
                                                        part_segment, trial_maps, n_bins)
                accepted = (link, pair, trial, trial_maps, trial_emissions,
                            change, winners, losers)
                break
        if accepted is None:
            break
        link, pair, perms, maps, emissions, change, winners, losers = accepted
        first, second = (int(value) for value in pairs[pair])
        current = current + change
        span = links[link][1]
        diagnostic["moves"].append({
            "atomic_span": int(span), "position": int(block.positions[offsets[span]]),
            "rows": (int(classes[first][0]), int(classes[second][0])),
            "score_gain_switches": float(total[link, pair] / unit),
            "painted_switch_change": int(np.sum(change)),
            "winner_samples": winners.tolist(), "loser_samples": losers.tolist(),
        })

    diagnostic.update(switch_unit_nats=unit, painted_switches_before=switches_before,
                      candidate_checks=checked)
    if not diagnostic["moves"]:
        diagnostic["painted_switches_after"] = switches_before
        return block, diagnostic
    diagnostic["painted_switches_search"] = int(np.sum(current))

    # Movable classes are singleton rows, so the class maps are row maps.
    row_of_class = np.asarray([members[0] for members in classes], dtype=np.int64)
    row_maps = []
    for perm in perms:
        row_map = np.arange(len(block.haplotypes))
        row_map[row_of_class[movable]] = row_of_class[perm[movable]]
        row_maps.append(row_map)
    spans = np.r_[0, [span for _, span in links], len(offsets)]
    ends = np.r_[offsets, len(block.positions)]
    span_bounds = list(zip(spans[:-1], spans[1:]))
    position_bounds = [(int(ends[start]), int(ends[stop])) for start, stop in span_bounds]
    relabelled = _relabel_block(block, row_maps, position_bounds, span_bounds)

    # Verify with the painting functions on the relabelled panel itself.
    new_space = painting_model.build_ragged_state_space(
        assembly_observations.founder_inference_panel_from_block_result(relabelled), kept)
    if (new_space.equivalence_classes != classes
            or not np.array_equal(new_space.site_indices, site_indices)):
        diagnostic.update(painted_switches_after=switches_before,
                          reason="trajectory_classes_changed")
        return block, diagnostic
    new_emissions, _ = painting_model.calculate_ragged_binned_emissions(
        selected_evidence, selected_observed, new_space, binning,
        robustness_epsilon=robustness_epsilon)
    switches_after = int(np.sum(
        _painted_switches(new_emissions, binning, n_states, eligible, painting)))
    diagnostic["painted_switches_after"] = switches_after
    if switches_after >= switches_before:
        diagnostic["painted_switches_after"] = switches_before
        diagnostic["reason"] = "painted_switches_not_lowered"
        diagnostic["rejected_painted_switches"] = switches_after
        return block, diagnostic
    diagnostic["accepted"] = True
    return relabelled, diagnostic


def relabel_components(components, probabilities, sites, observed, *, painting_kwargs,
                       minimum_switch_gain, maximum_loser_ratio):
    """Relabel every component before painting; return the components and diagnostics.

    Each returned component is a shallow copy carrying its diagnostic in
    ``founder_label_continuity``; unchanged components share their arrays.
    """
    sites = np.asarray(sites)
    settings = {name: painting_kwargs[name] for name in (
        "recomb_rate", "switch_penalty_per_snp", "robustness_epsilon",
        "double_recomb_factor", "snps_per_bin")}
    settings["chromosome_map"] = painting_kwargs.get("chromosome_map")
    results, diagnostics = [], []
    for index, block in enumerate(components):
        positions = np.asarray(block.positions)
        evidence, flags = painting_evidence.select_site_evidence(
            probabilities, observed, np.searchsorted(sites, positions))
        kept = (np.ones(len(positions), dtype=np.bool_)
                if getattr(block, "keep_flags", None) is None
                else np.asarray(block.keep_flags) > 0)
        result, diagnostic = relabel_component(
            block, evidence, flags, kept, minimum_switch_gain=minimum_switch_gain,
            maximum_loser_ratio=maximum_loser_ratio, **settings)
        if result is block:
            result = copy.copy(block)
        diagnostic = dict(diagnostic, component=index)
        result.founder_label_continuity = diagnostic
        results.append(result)
        diagnostics.append(diagnostic)
    return tuple(results), diagnostics
