"""Label-continuous founder rows: relabel row suffixes before component painting.

Assembly builds each founder row of a component as a path through the local
rows of consecutive atomic (prepared) blocks. Next to stretches where rows are
locally interchangeable (identical copies inside a founder's run of
homozygosity, haplotypes shared between founders), linking and refinement can
let two rows exchange founder identity at a link point, the boundary between
two atomic blocks. The alleles at every site stay right as a set, but painting
reads label continuity as haplotype continuity: every sample that carries
either haplotype across the point pays a track switch there, or keeps its
labels by exchanging its two tracks, which flips a parent's phase frame and
puts a false crossover into every child.

Model. Exchanging the labels of rows a and b at every site from link point j
onwards (a suffix transposition) leaves the set of row alleles at every site
unchanged. The painting state space (trajectory classes plus BACKGROUND) is
renamed, not changed, so each emission of the painting HMM is unchanged except
in the one bin that straddles j, whose sites left of j keep their labels. Only
which ordered state sequences count as continuous changes. The search uses the
component's copying model, estimated before it (painting.run.component_model)
and fixed throughout. A transposition t at j is screened with the Viterbi
objective of the model's label-level copying process (painting.transitions: the
estimated rate, tempered emissions, no panel-error state), for every sample s,

    V_s(t) = max_x [ G_s(x) + L_s(x) + Q_s(t(x)) ],   Q_s = P_s + B_s,

where G_s is the max-product forward message into the bin holding j (every bin
to its left, with the current labels), L_s and P_s are that bin's emissions on
its sites left and right of j, and B_s is the backward message from that bin
(every bin to its right). Backward messages of the relabelled panel are the
current ones renamed by t, and the Hamming transition is invariant when both
of its arguments are renamed, so V_s(t) is the exact best-path score of the
relabelled panel under that process: paths re-optimise on both sides, and a
switch that sat anywhere inside a stretch of identical rows vanishes when the
relabelling makes the haplotype continuous. Gains are measured in units u of
one track change (the median log-probability cost of a change against staying,
PaintingTransitions.switch_unit), so a gain approximates a fall in painted
switches; the screen only ranks and admits candidates, and every decision is made on the painted switches of the full
model.

Acceptance uses no pedigree, generation label or truth. Candidates are ranked
by their summed score gain and considered only when that gain is at least
``label_continuity_minimum_switch_gain`` units u. A candidate is
then decided on painted switches exactly as painting reports them: every
sample is re-decoded with the painting Viterbi (painting.fidelity, the class x
fidelity chain) under the candidate labels, and
its minimum-unordered switch count is compared with the current one. The
candidate is accepted when the total falls by at least the minimum gain and the
samples whose count rises (losers) are at most
``label_continuity_maximum_loser_ratio`` times the samples whose count falls
(winners). Under a label exchange the only losers are carriers with a real
change of haplotype inside the link interval. Under a real founder-to-child
crossover inherited by one lineage, every other carrier of the two haplotypes
loses a whole switch, so the ratio rule keeps such a crossover in the rows even
when the recombinant lineage is the majority (one unsequenced F1 can parent
half of an F2 cohort). Score changes are not used to classify samples. The
first acceptable candidate is applied, every message is recomputed, and the
search repeats until no candidate is acceptable; each accepted move lowers the
painted switch total, so the search terminates. When another acceptable candidate
of the same tie class would relabel differently (a competing repair the gains
cannot order), neither is applied and the search goes on with the next class. Only singleton, anchored
trajectory classes move; pooled duplicate rows and BACKGROUND keep their
labels, and a candidate that would change which of the two classes is called
anywhere (their panel-error states) is not accepted. The
relabelled component is kept only when repainting it with the painting
functions also lowers the total and leaves those states unchanged.

Cost. A transposition of rows a and b fixes states involving neither. At each link, compare the best
unaffected identity term (from one sorted list per sample) with the 4S - 4 affected states: O(links N S^3) per
iteration, with the maxima of a full S^2 scan. Every candidate check re-decodes all eligible samples under the
full chain, O(N S'^2 B). An unaffected optimal score alone cannot certify its traceback switch count when
paths tie. Equivalent placements of a transposition are checked once and retain per-sample count changes. The
workspace can reach tens of GB at K = 30, N = 290-320, so contiguous sample batches bound memory. Quantities
are per sample except summed gains, accumulated in sample order; batching preserves their arithmetic. Uncached
batches rebuild their part emissions in each pass.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import prange, types

from ...contracts.painting import LabelContinuity, RelabelMove
from ...contracts.panels import SITE_FOUNDER_FIELDS as _SITE_FOUNDER_FIELDS
from ...contracts.panels import ComponentPanel, relabel_rows
from ...core import ties
from ...core.kernels import kernel
from . import emissions as evidence_windows
from . import fidelity, hmm
from .parameters import PaintingParameters
from .states import build_bins, build_states
from .transitions import copying_transitions, edge_morgans

# Supported layouts for typed stage routes.
_OVERLOADS = {
    "_centre": ((types.Array(types.float64, 1, "C", readonly=False, aligned=True),),),
    "_link_messages": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=False, aligned=True),
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.Array(types.float64, 1, "C", readonly=True, aligned=True),
            types.int64,
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "_relabelled_emissions": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "_stacked_relabelled_emissions": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 3, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
    "_link_scores": (
        (
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 1, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
        ),
    ),
    "_add_transposition_gains": (
        (
            types.Array(types.float64, 2, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.float64, 3, "C", readonly=False, aligned=True),
            types.Array(types.boolean, 1, "C", readonly=False, aligned=True),
            types.Array(types.int64, 2, "C", readonly=False, aligned=True),
            types.int64,
        ),
    ),
}


@kernel(overloads=_OVERLOADS["_centre"], cache=True, nogil=True)
def _centre(values):
    largest = -np.inf
    for value in values:
        if value > largest:
            largest = value
    for index in range(len(values)):
        values[index] -= largest


@kernel(overloads=_OVERLOADS["_link_messages"], cache=True, parallel=True, nogil=True)
def _link_messages(emissions, log_same, log_one, log_two, n_states, slot_of_bin, n_slots):
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
        row_first = np.empty(n_states)
        row_second = np.empty(n_states)
        column_first = np.empty(n_states)
        column_second = np.empty(n_states)
        row_first_column = np.empty(n_states, dtype=np.int64)
        column_first_row = np.empty(n_states, dtype=np.int64)
        for state in range(n_diplotypes):
            alpha[state] = emissions[sample, state, 0]
        _centre(alpha)
        for bin_index in range(1, n_bins):
            hmm.hamming_max_values(
                alpha,
                log_same[bin_index - 1],
                log_one[bin_index - 1],
                log_two[bin_index - 1],
                n_states,
                message,
                row_first,
                row_second,
                column_first,
                column_second,
                row_first_column,
                column_first_row,
            )
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
                # The Hamming transition is symmetric, so the backward step uses
                # the same max-product contraction.
                hmm.hamming_max_values(
                    message,
                    log_same[bin_index - 1],
                    log_one[bin_index - 1],
                    log_two[bin_index - 1],
                    n_states,
                    beta,
                    row_first,
                    row_second,
                    column_first,
                    column_second,
                    row_first_column,
                    column_first_row,
                )
                _centre(beta)
    return forward, backward


@kernel(overloads=_OVERLOADS["_relabelled_emissions"], cache=True, parallel=True, nogil=True)
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
                result[row, state, bin_index] += part_emissions[sample, state_maps[segment, state], part]
    return result


@kernel(overloads=_OVERLOADS["_stacked_relabelled_emissions"], cache=True, parallel=True, nogil=True)
def _stacked_relabelled_emissions(part_emissions, samples, part_bin, part_segment, state_maps, n_bins):
    """_relabelled_emissions under each of several labellings at once: row g * len(samples) + r holds
    sample samples[r] under state_maps[g], each computed from its own sample alone, in the same order."""
    _, n_diplotypes, n_parts = part_emissions.shape
    n_rows = len(samples)
    result = np.zeros((len(state_maps) * n_rows, n_diplotypes, n_bins))
    for task in prange(len(state_maps) * n_rows):
        labelling, row = task // n_rows, task % n_rows
        sample = samples[row]
        for part in range(n_parts):
            bin_index = part_bin[part]
            segment = part_segment[part]
            for state in range(n_diplotypes):
                result[task, state, bin_index] += part_emissions[sample, state_maps[labelling, segment, state], part]
    return result


@kernel(overloads=_OVERLOADS["_link_scores"], cache=True, parallel=True, nogil=True)
def _link_scores(
    part_emissions,
    part_segment,
    state_maps,
    link_slot,
    link_segment,
    link_first_part,
    link_stop_part,
    forward,
    backward,
):
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
                target[link, sample, state] += part_emissions[sample, state_maps[segment, state], part]
    return left, right


@kernel(overloads=_OVERLOADS["_add_transposition_gains"], cache=True, parallel=True, nogil=True)
def _add_transposition_gains(total, left, right, eligible, pairs, n_states):
    """Add the exact Viterbi-score gain of each (link, transposition) of the eligible samples
    of ``left``/``right`` to total[link, pair], sample by sample in index order (so batches
    added in sample order give the same sums as all samples at once).

    Unaffected states keep their identity terms, so per (link, sample) the
    terms are sorted once and each pair takes the first sorted entry outside
    its rows and columns, then evaluates only its 4S - 4 affected states.
    """
    n_links, n_samples, n_diplotypes = left.shape
    n_pairs = pairs.shape[0]
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
                    if row != first_row and row != second_row and column != first_row and column != second_row:
                        swapped = terms[state]
                        break
                relabel[first_row] = second_row
                relabel[second_row] = first_row
                for row in (first_row, second_row):
                    for column in range(n_states):
                        value = (
                            left[link, sample, row * n_states + column]
                            + right[link, sample, relabel[row] * n_states + relabel[column]]
                        )
                        if value > swapped:
                            swapped = value
                for column in (first_row, second_row):
                    for row in range(n_states):
                        if row == first_row or row == second_row:
                            continue
                        value = (
                            left[link, sample, row * n_states + column]
                            + right[link, sample, row * n_states + relabel[column]]
                        )
                        if value > swapped:
                            swapped = value
                relabel[first_row] = first_row
                relabel[second_row] = second_row
                total[link, pair] += swapped - identity


def _state_maps(perms: np.ndarray, n_states: int) -> np.ndarray:
    return np.ascontiguousarray((perms[:, :, None] * n_states + perms[:, None, :]).reshape(len(perms), -1))


def _outcome(
    reason,
    *,
    link_points=0,
    movable_rows=0,
    unit=None,
    before=None,
    after=None,
    checks=0,
    moves=(),
    row_maps=None,
    accepted=False,
    tie_events=None,
    check_budget_reached=False,
) -> LabelContinuity:
    return LabelContinuity(
        accepted=accepted,
        reason=reason,
        link_points=link_points,
        movable_rows=movable_rows,
        switch_unit_nats=unit,
        switches_before=before,
        switches_after=after,
        candidate_checks=checks,
        moves=tuple(moves),
        row_maps=row_maps,
        tie_events=ties.TieCounts(()) if tie_events is None else tie_events,
        check_budget_reached=bool(check_budget_reached),
    )


def relabel_component(
    panel: ComponentPanel,
    likelihoods: evidence_windows.ArraySource,
    observed: evidence_windows.ArraySource,
    evidence_sites: np.ndarray,
    config: PaintingParameters,
    genetic_map,
    *,
    model,
    batch_samples: int = 1,
    reverse: bool,
    census_sink: Callable[[ties.TieCensus], None] | None = None,
    emission_builder: evidence_windows.EmissionBuilder = evidence_windows.tempered_emissions,
    emission_cache_bytes: int = 0,
) -> tuple[ComponentPanel, LabelContinuity]:
    """Return (painted panel, outcome): the relabelled panel when that lowers painted switches.

    ``reverse``: the caller's required TieSettings reversal; determines ranking inside tie classes.

    ``likelihoods``/``observed`` are the chromosome evidence (all samples) and
    ``evidence_sites`` int64[L] the evidence index of each panel position; ``model`` the component's copying model
    (painting.run.component_model; required for a nonempty linked search).
    """
    states = build_states(panel.positions, panel.inference, panel.kept)
    classes = states.classes
    anchored = states.anchored
    movable = [index for index, members in enumerate(classes) if len(members) == 1 and anchored[index]]
    offsets = np.asarray(panel.atomic_offsets, dtype=np.int64)
    site_indices = states.site_indices
    n_selected = len(site_indices)
    links = []
    seen = set()
    for span in range(1, len(offsets)):
        first = int(np.searchsorted(site_indices, offsets[span]))
        if 0 < first < n_selected and first not in seen:
            seen.add(first)
            links.append((first, span))
    bins = build_bins(states.positions, config.snps_per_bin)
    n_bins = bins.n_bins
    counts = dict(link_points=len(links), movable_rows=len(movable))
    if len(movable) < 2 or not links or n_bins < 2:
        return panel, _outcome("no_link_or_movable_row", **counts)
    return _relabel_linked(
        panel,
        likelihoods,
        observed,
        evidence_sites,
        config,
        genetic_map,
        states,
        movable,
        offsets,
        links,
        bins,
        counts,
        batch_samples,
        model,
        convention_reverse=reverse,
        census_sink=census_sink,
        emission_builder=emission_builder,
        emission_cache_bytes=emission_cache_bytes,
    )


# Bookkeeping of one search beyond its per-sample arrays.


class _SampleBatches:
    """Contiguous sample batches. A batch's part emissions and eligibility are kept for the whole search when they fit
    the caller's budget, else rebuilt when used: from the resident copy of the component's evidence where it covers
    the batch (in memory, in parallel), else from the stored evidence."""

    def __init__(
        self,
        likelihoods,
        observed,
        sites,
        states,
        part_start,
        part_stop,
        config,
        size,
        bin_starts,
        bin_stops,
        part_bin,
        emission_builder,
        keep_bytes=0,
        resident=None,
    ):
        self.emission_builder = emission_builder
        self.likelihoods, self.observed, self.sites = likelihoods, observed, sites
        # resident: (stop, likelihoods, observed, sites), the evidence of samples [0, stop) at the selected sites.
        self.resident = resident
        self.states, self.part_start, self.part_stop, self.config = states, part_start, part_stop, config
        self.bin_starts, self.bin_stops, self.part_bin = bin_starts, bin_stops, part_bin
        self.alt_probability = states.alt_probability()
        n_samples = likelihoods.shape[0]
        self.spans = [(start, min(start + size, n_samples)) for start in range(0, n_samples, size)]
        self._kept = {}
        # batches are kept while their part emissions (8 S^2 bytes per part and sample) fit
        # ``keep_bytes``, those outside the resident copy first (their rebuilds read the stored evidence), each group
        # in sample order. One batch holding every sample is always kept.
        per_sample = 8 * states.n_states**2 * len(part_start)
        covered = 0 if resident is None else resident[0]
        order = [span for span in self.spans if span[1] > covered] + [span for span in self.spans if span[1] <= covered]
        keep, used = set(), 0
        for start, stop in order:
            if used + (stop - start) * per_sample > keep_bytes and len(self.spans) > 1:
                break
            used += (stop - start) * per_sample
            keep.add(start)
        self._keep_starts = frozenset(keep)
        self.all_kept = len(keep) == len(self.spans)
        # The budget the kept emissions leave, for decoding more samples or candidates in one call.
        self.spare_bytes = max(0, int(keep_bytes) - used)
        # Every batch kept: their part emissions share one array over all samples, so that decoding
        # can take wider runs of samples than a batch.
        self.combined = None

    def source(self, start, stop):
        """(likelihoods, observed, sites) to build the batch [start, stop) from."""
        if self.resident is not None and stop <= self.resident[0]:
            return self.resident[1:]
        return self.likelihoods, self.observed, self.sites

    def _values(self, start, stop):
        if start in self._kept:
            return self._kept[start]
        raw, observed, informative = self.emission_builder(
            *self.source(start, stop),
            slice(start, stop),
            self.alt_probability,
            self.states.background_frequency,
            self.bin_starts,
            self.bin_stops,
            self.config,
            parts=(self.part_start, self.part_stop, self.part_bin),
        )
        value = raw, np.any(informative, axis=1)
        if start in self._keep_starts:
            if self.all_kept and len(self.spans) > 1:
                if self.combined is None:
                    self.combined = np.empty((self.spans[-1][1], *raw.shape[1:]), dtype=raw.dtype)
                self.combined[start:stop] = raw
                value = self.combined[start:stop], value[1]
                del raw
            self._kept[start] = value
        return value

    def units(self, width):
        """(start, stop, part emissions of samples start..stop) for each batch, built as iteration reaches it so
        only the current uncached batch's emissions are live. Fully cached batches provide runs of up to
        ``width`` samples from their shared array, one decoding call each. Callers iterate once."""
        if not (self.all_kept and len(self.spans) > 1) or width <= self.spans[0][1] - self.spans[0][0]:
            return ((start, stop, self.part_emissions(start, stop)) for start, stop in self.spans)
        for start, stop in self.spans:
            self._values(start, stop)
        n_samples = self.spans[-1][1]
        return [
            (start, min(start + width, n_samples), self.combined[start : min(start + width, n_samples)])
            for start in range(0, n_samples, width)
        ]

    def eligible(self, start, stop):
        return self._values(start, stop)[1]

    def part_emissions(self, start, stop):
        return self._values(start, stop)[0]


def _leftmost_equivalent_link(panel, offsets, links, perms, row_of_class, first, second, link, steps=None):
    """The leftmost link point at which transposing classes ``first`` and ``second`` gives the same relabelled panel
    as at ``link``: walk left while, under the current labelling ``perms``, the two rows carry identical founder-axis
    data (released and inference calls, ALT probabilities, support) at every panel site between consecutive link
    points (segment s lies between links s - 1 and s); at most ``steps`` points when given."""
    while link > 0 and (steps is None or steps > 0):
        steps = None if steps is None else steps - 1
        start, stop = int(offsets[links[link - 1][1]]), int(offsets[links[link][1]])
        left = int(row_of_class[perms[link, first]])
        right = int(row_of_class[perms[link, second]])
        if not all(
            np.array_equal(getattr(panel, name)[left, start:stop], getattr(panel, name)[right, start:stop])
            for name in _SITE_FOUNDER_FIELDS
        ):
            break
        link -= 1
    return link


# The look-ahead uses the same placement walk under a second name. It only chooses candidates to decode early;
# it does not decide a repair, so decision-placement diagnostics leave these speculative walks out.
_lookahead_equivalent_link = _leftmost_equivalent_link


def _relabel_linked(
    panel,
    likelihoods,
    observed,
    evidence_sites,
    config,
    genetic_map,
    states,
    movable,
    offsets,
    links,
    bins,
    counts,
    batch,
    model,
    *,
    convention_reverse: bool,
    census_sink: Callable[[ties.TieCensus], None] | None = None,
    emission_builder=evidence_windows.tempered_emissions,
    emission_cache_bytes: int = 0,
):
    """The relabelling search of a component with link points and movable rows (relabel_component), over
    contiguous batches of ``batch`` samples (every sample at once when ``batch`` covers them). The part emissions do
    not depend on the candidate relabelling (a candidate only remaps states), so every batch's part emissions are
    kept for the whole search when they fit ``emission_cache_bytes``, instead of rebuilt for each check."""
    classes = states.classes
    site_indices = states.site_indices
    n_bins = bins.n_bins
    n_samples = likelihoods.shape[0]

    # Split painting bins at link points; part emissions use the painting's own kernel.
    link_sites = np.asarray([first for first, _ in links], dtype=np.int64)
    part_starts, part_stops, part_bins = [], [], []
    for bin_index, (start, stop) in enumerate(zip(bins.starts, bins.stops, strict=True)):
        cuts = link_sites[(link_sites > start) & (link_sites < stop)]
        for left, right in zip(np.r_[start, cuts], np.r_[cuts, stop], strict=True):
            part_starts.append(left)
            part_stops.append(right)
            part_bins.append(bin_index)
    part_start = np.asarray(part_starts, dtype=np.int64)
    part_bin = np.asarray(part_bins, dtype=np.int64)
    part_segment = np.searchsorted(link_sites, part_start, side="right").astype(np.int64)
    size = max(1, int(batch))
    # when the budget cannot keep every batch's part emissions, the component's selected evidence is
    # read once into a compact resident copy (codes and dictionaries, far smaller than the emissions) charged to the
    # same budget: of every sample when it fits, else of the largest leading run of batches that fits. A rebuild of a
    # batch it covers is a parallel in-memory gather and emission kernel, never a serial read of the stored evidence.
    # The values gathered are the same.
    selected_sites = evidence_sites[site_indices]
    keep_bytes = emission_cache_bytes
    resident = None
    if (
        size < n_samples
        and n_samples * 8 * states.n_states**2 * len(part_start) > emission_cache_bytes
        and hasattr(likelihoods, "resident_columns")
        and hasattr(observed, "resident_columns")
    ):
        n_sites, covered, needed = len(selected_sites), 0, 0
        for stop in [min(start + size, n_samples) for start in range(0, n_samples, size)]:
            bytes_to_stop = likelihoods.resident_bytes(n_sites, slice(0, stop)) + observed.resident_bytes(
                n_sites, slice(0, stop)
            )
            if bytes_to_stop > emission_cache_bytes:
                break
            covered, needed = stop, bytes_to_stop
        if covered:
            copy = likelihoods.resident_columns(selected_sites, slice(0, covered))
            resident = (
                covered,
                copy,
                observed.resident_columns(selected_sites, copy, slice(0, covered)),
                np.arange(n_sites, dtype=np.int64),
            )
            keep_bytes = emission_cache_bytes - needed
    samples = _SampleBatches(
        likelihoods,
        observed,
        selected_sites,
        states,
        part_start,
        np.asarray(part_stops, dtype=np.int64),
        config,
        size,
        bins.starts,
        bins.stops,
        part_bin,
        emission_builder,
        keep_bytes=keep_bytes,
        resident=resident,
    )
    spans = samples.spans
    eligible = np.empty(n_samples, dtype=np.bool_)
    for start, stop in spans:
        eligible[start:stop] = samples.eligible(start, stop)
    if not np.any(eligible):
        return panel, _outcome("no_evidence", **counts)

    n_states = states.n_states
    # Samples per decoding call when every batch is kept: a batch, plus as many as the budget the kept emissions leave
    # holds (relabelled, expanded and message arrays per sample); the same per-sample values in fewer, wider calls.
    wide_bytes = 8 * n_bins * (3 * n_states**2 + (2 * n_states - 1) ** 2 + 2)
    decode_width = min(n_samples, size + samples.spare_bytes // max(1, wide_bytes))
    # The label-level copying process of the component's model (the screen) and the full chain (every decision).
    transitions = copying_transitions(edge_morgans(bins.centers, genetic_map), n_states, model.rate)
    capable = np.concatenate((np.asarray(states.anchored, dtype=np.int64), np.zeros(1, dtype=np.int64)))
    # u: the log-probability cost of one track change against staying, median over bin edges.
    unit = transitions.switch_unit()

    # A link's messages live at the bin holding its first right-hand site.
    bin_of_site = np.repeat(np.arange(n_bins, dtype=np.int64), bins.stops - bins.starts)
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
    pairs = np.asarray(
        [(first, second) for index, first in enumerate(movable) for second in movable[index + 1 :]], dtype=np.int64
    )

    def switches(start, stop, rows, maps, parts=None):
        """Painted switches of the batch's samples ``rows`` (batch positions) under ``maps`` (``parts``: the batch's
        part emissions when the caller holds them)."""
        emissions = _relabelled_emissions(
            samples.part_emissions(start, stop) if parts is None else parts, rows, part_bin, part_segment, maps, n_bins
        )
        return fidelity.painted_switches(emissions, model, capable, eligible[start + rows], tie=config.viterbi_tie)

    def winner_directions(winners, maps, first, second, link_bin):
        """(a -> b, b -> a): winners whose decoded path (the default convention, labels before the move) has a
        track in state ``first`` left of the link and ``second`` right of it, or the reverse, within two bins of it."""
        forward = reverse = 0
        low, high = max(0, link_bin - 2), min(n_bins - 1, link_bin + 2)
        for start, stop in spans:
            rows = winners[(winners >= start) & (winners < stop)] - start
            if not len(rows):
                continue
            emissions = _relabelled_emissions(
                samples.part_emissions(start, stop), rows, part_bin, part_segment, maps, n_bins
            )
            paths = fidelity.label_paths(
                np.ascontiguousarray(emissions), model, capable, reverse=False, tie=config.viterbi_tie
            )
            for row in range(len(rows)):
                left, right = paths[row, :, low], paths[row, :, high]
                found = 0
                for track in (0, 1):
                    for target in (0, 1):
                        if left[1 - track] != right[1 - target]:
                            continue
                        if left[track] == first and right[target] == second:
                            found = found or 1
                        elif left[track] == second and right[target] == first:
                            found = found or 2
                forward += found == 1
                reverse += found == 2
        return forward, reverse

    def link_scores(start, stop, maps, parts=None):
        """(left, right) link scores of the batch's samples under ``maps`` (``parts``: their part emissions when the
        caller holds them)."""
        parts = samples.part_emissions(start, stop) if parts is None else parts
        emissions = _relabelled_emissions(
            parts, np.arange(stop - start, dtype=np.int64), part_bin, part_segment, maps, n_bins
        )
        forward, backward = _link_messages(
            emissions,
            transitions.log_same,
            transitions.log_one,
            transitions.log_two,
            n_states,
            slot_of_bin,
            len(slot_bins),
        )
        del emissions
        return _link_scores(
            parts, part_segment, maps, link_slot, link_segment, link_first_part, link_stop_part, forward, backward
        )

    row_of_class = np.asarray([members[0] for members in classes], dtype=np.int64)
    # Error capabilities: a class has a panel-error state when it is called at a selected site. A suffix
    # exchange can move every call of a row into the other row (both classes keep their rows, the alphabet of error
    # states changes), which the model fixed before the search does not describe: such candidates are not accepted.
    # called_in[c, g] = the original class c is called at a selected site of segment g (between consecutive links).
    segment_of_site = np.searchsorted(link_sites, np.arange(len(site_indices)), side="right")
    called_in = np.zeros((n_states, len(links) + 1), dtype=np.bool_)
    for index in range(states.n_classes):
        called = np.flatnonzero(states.named_alleles[index] >= 0)
        called_in[index, np.unique(segment_of_site[called])] = True

    def anchored_under(trial_perms, labels):
        return tuple(bool(np.any(called_in[trial_perms[:, label], np.arange(len(links) + 1)])) for label in labels)

    def transposed(perms, key):
        """(trial labelling, its capability is unchanged) of the relabelling ``key`` = (pair, link), for the
        look-ahead (evaluate states the same test where a candidate is checked)."""
        pair, link = key
        first, second = (int(value) for value in pairs[pair])
        trial = perms.copy()
        trial[link + 1 :, [first, second]] = trial[link + 1 :, [second, first]]
        return trial, anchored_under(trial, (first, second)) == anchored_under(perms, (first, second))

    def decoded_changes(trials, current):
        """Painted-switch changes of every trial labelling against ``current``; each batch's part emissions are
        obtained once for all of them."""
        trial_maps = [_state_maps(trial, n_states) for trial in trials]
        changes = [np.zeros_like(current) for _ in trials]
        # Several candidates of a batch are decoded in one call (their relabelled emissions stacked along the sample
        # axis), as many as the budget the kept emissions leave can hold beyond the one-candidate working set
        # (relabelled and expanded emissions per stacked sample): more samples per parallel call, the same values.
        stacked_bytes = 8 * n_bins * (n_states**2 + (2 * n_states - 1) ** 2 + 2)
        for start, stop, parts in samples.units(decode_width):
            rows = np.arange(stop - start, dtype=np.int64)
            count = stop - start
            group = max(1, min(len(trials), 1 + samples.spare_bytes // max(1, count * stacked_bytes)))
            for first in range(0, len(trials), group):
                chunk = trial_maps[first : first + group]
                if len(chunk) == 1:
                    changes[first][start:stop] = switches(start, stop, rows, chunk[0], parts) - current[start:stop]
                    continue
                emissions = _stacked_relabelled_emissions(
                    parts, rows, part_bin, part_segment, np.ascontiguousarray(np.stack(chunk)), n_bins
                )
                stacked = fidelity.painted_switches(
                    emissions, model, capable, np.tile(eligible[start:stop], len(chunk)), tie=config.viterbi_tie
                )
                del emissions
                for offset, change in enumerate(changes[first : first + len(chunk)]):
                    change[start:stop] = stacked[offset * count : (offset + 1) * count] - current[start:stop]
            del parts
        return changes

    perms = np.tile(np.arange(n_states, dtype=np.int64), (len(links) + 1, 1))
    maps = _state_maps(perms, n_states)
    current = np.empty(n_samples, dtype=np.int64)
    for start, stop, parts in samples.units(decode_width):
        current[start:stop] = switches(start, stop, np.arange(stop - start, dtype=np.int64), maps, parts)
    switches_before = int(np.sum(current))
    minimum_gain = config.label_continuity_minimum_switch_gain
    maximum_loser_ratio = config.label_continuity_maximum_loser_ratio
    checked = 0
    budget = int(config.label_continuity_check_budget)
    budget_reached = False
    moves: list[RelabelMove] = []
    counter = ties.TieCounter()
    while True:
        # Summed gains: each batch adds its samples in sample order (the same sums as one pass).
        total = np.zeros((len(links), len(pairs)))
        for start, stop, parts in samples.units(decode_width):
            left, right = link_scores(start, stop, maps, parts)
            _add_transposition_gains(total, left, right, eligible[start:stop], pairs, n_states)
            del left, right
        # Rank score gains in tie classes, keeping the leftmost equivalent link and then pair order within a
        # class; rounding does not decide equivalent placements. Accept on exact per-sample painted switches.
        flats = np.flatnonzero(total.ravel() >= minimum_gain * unit)
        values = total.ravel()[flats]
        candidates = []
        candidate_class = []
        if len(flats):
            # Nats; descending gain. Flat link/pair indices are stable discrete content;
            # exact equal keys denote the same candidate. No secondary preference reverses.
            policy = ties.policy("painting.relabel_rank", kind="nll", reverse=convention_reverse)
            ranked = policy.classes(
                values, scale=float(np.max(np.abs(values))), content=[int(flat) for flat in flats], descending=True
            )
            counter.add("painting.relabel_rank", ranked)
            ties.census(
                "painting.relabel_rank",
                f"p{int(panel.positions[0])}/m{len(moves)}",
                np.sort(values)[::-1][:8],
                sink=census_sink,
            )
            candidates = [divmod(int(flats[index]), len(pairs)) for index in ranked.order]
            candidate_class = [int(ranked.classes[index]) for index in ranked.order]
        class_members: dict[int, list[int]] = {}
        for index, class_id in enumerate(candidate_class):
            class_members.setdefault(class_id, []).append(index)
        leftmost: dict[tuple[int, int], int] = {}
        results: dict[tuple[int, int], tuple[bool, np.ndarray, np.ndarray, np.ndarray]] = {}
        # Changes decoded ahead of their check (batches rebuilt per use only), and the next look-ahead width.
        ahead: dict[tuple[int, int], np.ndarray] = {}
        width = [1]

        def canonical(index, candidates=candidates, leftmost=leftmost, perms=perms):
            """(pair, link) of candidate ``index`` moved to its leftmost equivalent link (memoised along a stretch)."""
            link, pair = candidates[index]
            first, second = (int(value) for value in pairs[pair])
            walked = []
            while (pair, link) not in leftmost:
                left = _leftmost_equivalent_link(
                    panel, offsets, links, perms, row_of_class, first, second, link, steps=1
                )
                if left == link:
                    break
                walked.append(link)
                link = left
            target = leftmost.get((pair, link), link)
            for point in walked + [link]:
                leftmost[(pair, point)] = target
            return pair, target

        def ahead_key(index, candidates, perms=perms):
            """canonical(index) for the look-ahead: the same leftmost walk (single steps to their fixed point, reading
            but never writing canonical's memo), through _lookahead_equivalent_link, so that the look-ahead's walks,
            which only choose which candidates to decode early, stay out of the decision's own record."""
            link, pair = candidates[index]
            first, second = (int(value) for value in pairs[pair])
            while (pair, link) not in leftmost:
                left = _lookahead_equivalent_link(
                    panel, offsets, links, perms, row_of_class, first, second, link, steps=1
                )
                if left == link:
                    return pair, link
                link = left
            return pair, leftmost[(pair, link)]

        def evaluate(
            key, results=results, perms=perms, current=current, ahead=ahead, width=width, candidates=candidates
        ):
            """(acceptable, change, winners, losers) of ``key`` = (pair, link), computed once per iteration by
            re-decoding every eligible sample. For uncached batches, a miss also decodes following ranked
            candidates (1, then 2, 4, ... within the remaining main-loop budget) from the same rebuilt
            emissions. Their changes wait in ``ahead`` and count as checks when evaluated."""
            nonlocal checked
            if key not in results:
                pair, link = key
                first, second = (int(value) for value in pairs[pair])
                trial = perms.copy()
                trial[link + 1 :, [first, second]] = trial[link + 1 :, [second, first]]
                if anchored_under(trial, (first, second)) != anchored_under(perms, (first, second)):
                    results[key] = (
                        False,
                        np.zeros_like(current),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64),
                    )
                    counter.note("painting.relabel_capability_change", inside_tie=False)
                    return results[key]
                if key in ahead:
                    change = ahead.pop(key)
                else:
                    keys, trials = [key], [trial]
                    if not samples.all_kept:
                        more = min(width[0], budget - (checked - checked_before)) - 1
                        position = index + 1
                        while more > 0 and position < len(candidates):
                            other = ahead_key(position, candidates)
                            position += 1
                            if other in results or other in ahead or other in keys:
                                continue
                            other_trial, other_same = transposed(perms, other)
                            if other_same:
                                keys.append(other)
                                trials.append(other_trial)
                                more -= 1
                        width[0] *= 2
                    changes = decoded_changes(trials, current)
                    change = changes[0]
                    ahead.update(zip(keys[1:], changes[1:], strict=True))
                checked += 1
                winners, losers = np.flatnonzero(change < 0), np.flatnonzero(change > 0)
                acceptable = -int(np.sum(change)) >= minimum_gain and len(losers) <= maximum_loser_ratio * len(winners)
                results[key] = (acceptable, change, winners, losers)
            return results[key]

        # The first acceptable candidate in ranked order is applied, unless another acceptable candidate of its tie
        # class would relabel differently: competing repairs that this policy does not order leave the
        # assembled labels in place, and the search goes on with the next class. Equivalent placements of one
        # transposition (canonical) are evaluated once.
        accepted = None
        index = 0
        checked_before = checked
        while index < len(candidates):
            key = canonical(index)
            if key not in results and checked - checked_before >= budget:
                # The main ranked loop stops when checked candidates reach budget without an acceptable
                # repair, and flags the outcome. Certification against tied peers can perform additional
                # checks; it preserves whole-class ambiguity.
                budget_reached = True
                break
            if not evaluate(key)[0]:
                index += 1
                continue
            members = [other for other in class_members[candidate_class[index]] if other != index]
            if any(canonical(other) != key and evaluate(canonical(other))[0] for other in members):
                counter.note("painting.relabel_ambiguous", inside_tie=True)
                index = max(members) + 1
                continue
            accepted = key
            break
        if accepted is None:
            break
        pair, link = accepted
        acceptable, change, winners, losers = results[accepted]
        first, second = (int(value) for value in pairs[pair])
        # Tied suffix transpositions: where the two rows carry identical founder-axis data over the
        # stretch between two consecutive link points, transposing at either point gives the same per-site panel
        # data, painting and switches; the move is placed at the leftmost such point, a stated rule that the
        # convention reversal does not change.
        gain = float(total[link, pair] / unit)
        # Identity evidence: a label exchange of two founders' rows makes the carriers of either founder
        # switch at the link (both directions); one shared recombinant lineage switches one way only.
        forward, reverse = winner_directions(
            winners, maps, first, second, int(np.searchsorted(bins.starts, links[link][0], side="right") - 1)
        )
        perms = perms.copy()
        perms[link + 1 :, [first, second]] = perms[link + 1 :, [second, first]]
        maps = _state_maps(perms, n_states)
        current = current + change
        span = links[link][1]
        moves.append(
            RelabelMove(
                atomic_span=int(span),
                position=int(panel.positions[offsets[span]]),
                rows=(int(classes[first][0]), int(classes[second][0])),
                score_gain_switches=gain,
                painted_switch_change=int(np.sum(change)),
                winners=tuple(int(value) for value in winners),
                losers=tuple(int(value) for value in losers),
                winners_forward=int(forward),
                winners_reverse=int(reverse),
            )
        )

    counts.update(
        unit=unit,
        before=switches_before,
        checks=checked,
        moves=moves,
        tie_events=counter.record(),
        check_budget_reached=budget_reached,
    )
    if not moves:
        return panel, _outcome("no_move", after=switches_before, **counts)

    # Movable classes are singleton rows, so the class maps are row maps.
    n_rows = panel.founders
    segment_maps = []
    for perm in perms:
        row_map = np.arange(n_rows, dtype=np.int64)
        row_map[row_of_class[movable]] = row_of_class[perm[movable]]
        segment_maps.append(row_map)
    # Segment of each atomic span: links split the spans into len(links) + 1 runs.
    link_spans = np.asarray([span for _, span in links], dtype=np.int64)
    span_segment = np.searchsorted(link_spans, np.arange(len(offsets)), side="right")
    row_maps = np.ascontiguousarray(np.asarray(segment_maps, dtype=np.int64)[span_segment])
    relabelled = relabel_rows(panel, row_maps)

    # Verify with the painting functions on the relabelled panel itself.
    new_states = build_states(relabelled.positions, relabelled.inference, relabelled.kept)
    if new_states.classes != classes or not np.array_equal(new_states.site_indices, site_indices):
        return panel, _outcome("trajectory_classes_changed", after=switches_before, **counts)
    if not np.array_equal(new_states.anchored, states.anchored):
        return panel, _outcome("error_capabilities_changed", after=switches_before, **counts)
    switches_after = 0
    for start, stop in spans:
        new_emissions, _, _ = emission_builder(
            *samples.source(start, stop),
            slice(start, stop),
            new_states.alt_probability(),
            new_states.background_frequency,
            bins.starts,
            bins.stops,
            config,
        )
        switches_after += int(
            np.sum(
                fidelity.painted_switches(new_emissions, model, capable, eligible[start:stop], tie=config.viterbi_tie)
            )
        )
        del new_emissions
    if switches_after >= switches_before:
        return panel, _outcome("painted_switches_not_lowered", after=switches_before, **counts)
    return relabelled, _outcome("relabelled", after=switches_after, row_maps=row_maps, accepted=True, **counts)
