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
component's copying model, estimated before it (painting.stage.component_model)
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
switches; the screen only ranks and admits candidates (release 2.14: it leaves
out painting's panel-error state, whose mixture is not a sum over a straddling
bin's parts), and every decision is made on the painted switches of the full
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
cannot order), neither is applied and the search goes on with the next class
(release 2.14). Only singleton, anchored
trajectory classes move; pooled duplicate rows and BACKGROUND keep their
labels, and a candidate that would change which of the two classes is called
anywhere (their panel-error states) is not accepted (release 2.14). The
relabelled component is kept only when repainting it with the painting
functions also lowers the total and leaves those states unchanged.

Cost. A transposition t of rows a and b fixes every state that involves
neither, so the swapped maximum at a link is the larger of the best unaffected
identity term (the first entry outside rows and columns a and b of the
identity terms, sorted once per link and sample) and the 4S - 4 affected states
evaluated under t: O(links N S^3) per iteration, with the same maxima as a scan
over all S^2 states. A candidate check re-decodes every eligible sample under
the candidate labels with the chain, O(N S'^2 B) (release 2.14). Before, samples whose best unaffected state
beat every affected one were skipped: a relabelling maps the optimal paths one
to one, but it can change which of several exactly co-optimal paths a traceback
convention selects, and so the selected switch count, so the shortcut was not a
certificate. Equivalent placements of one transposition (the same per-site panel
data, see _leftmost_equivalent_link) are checked once, and a check keeps only its
per-sample count changes. Memory: the search holds about 20-28 N S^2 B bytes
plus a chain decode's 12 N S'^2 B (part and relabelled emissions, a Viterbi decode, messages and link scores;
relabel_bytes) when it runs on every sample at once (tens of GB for a long
chromosome at K = 30, N = 290-320). It therefore runs in contiguous sample
batches sized by a reservation from the run's memory governor (as much as the
measured headroom admits): every quantity is per sample except the summed
gains, which each batch adds in sample order, so
results are bitwise identical for any batch size; a batched search recomputes
the batch's part emissions in each pass instead of keeping them.
"""
from __future__ import annotations


import numpy as np
from numba import njit, prange

from .. import execution
from ..maths import ties
from ..model.components import _SITE_FOUNDER_FIELDS, ComponentPanel, relabel_rows
from . import fidelity, hmm
from .config import PaintingConfig
from .release import LabelContinuity, RelabelMove
from .states import build_bins, build_states
from .transitions import copying_transitions, edge_morgans


@njit(cache=True)
def _centre(values):
    largest = -np.inf
    for value in values:
        if value > largest:
            largest = value
    for index in range(len(values)):
        values[index] -= largest


@njit(cache=True, parallel=True)
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
            hmm.hamming_max_values(alpha, log_same[bin_index - 1], log_one[bin_index - 1],
                                   log_two[bin_index - 1], n_states, message, row_first,
                                   row_second, column_first, column_second, row_first_column,
                                   column_first_row)
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
                hmm.hamming_max_values(message, log_same[bin_index - 1], log_one[bin_index - 1],
                                       log_two[bin_index - 1], n_states, beta, row_first,
                                       row_second, column_first, column_second,
                                       row_first_column, column_first_row)
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


def _state_maps(perms: np.ndarray, n_states: int) -> np.ndarray:
    return np.ascontiguousarray(
        (perms[:, :, None] * n_states + perms[:, None, :]).reshape(len(perms), -1))


def _outcome(reason, *, link_points=0, movable_rows=0, unit=None, before=None, after=None,
             checks=0, moves=(), row_maps=None, accepted=False, tie_events=None) -> LabelContinuity:
    return LabelContinuity(accepted=accepted, reason=reason, link_points=link_points,
                           movable_rows=movable_rows, switch_unit_nats=unit,
                           switches_before=before, switches_after=after,
                           candidate_checks=checks, moves=tuple(moves), row_maps=row_maps,
                           tie_events=dict(tie_events or {}))


def relabel_component(panel: ComponentPanel, likelihoods: np.ndarray, observed: np.ndarray,
                      evidence_sites: np.ndarray, config: PaintingConfig, genetic_map, *, model=None
                      ) -> tuple[ComponentPanel, LabelContinuity]:
    """Return (painted panel, outcome): the relabelled panel when that lowers painted switches.

    ``likelihoods``/``observed`` are the chromosome evidence (all samples) and
    ``evidence_sites`` int64[L] the evidence index of each panel position; ``model`` the component's copying model
    (painting.stage.component_model; estimated here when None).
    """
    states = build_states(panel.positions, panel.inference, panel.kept)
    classes = states.classes
    anchored = states.anchored
    movable = [index for index, members in enumerate(classes)
               if len(members) == 1 and anchored[index]]
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
    # The search's arrays are admitted by the run's memory governor from measured pressure: every sample at once
    # when the headroom admits it, else contiguous sample batches of the granted size (identical results; a batched
    # search recomputes part emissions, ~3x slower).
    n_samples = len(likelihoods)
    # Selected sites covering the whole evidence axis are gathered as views (no copy) when every sample is searched
    # at once; a batched search may copy its re-decoded samples' rows, so a batch's size always counts them.
    views = (n_selected == likelihoods.shape[1] and likelihoods.dtype == np.float64
             and observed.dtype == np.bool_)
    n_track = states.n_states + int(np.sum(states.anchored))
    full = relabel_bytes(n_samples, states.n_states, n_bins, len(links), n_selected, gathered=not views,
                         n_track=n_track)
    single = relabel_bytes(1, states.n_states, n_bins, len(links), n_selected, n_track=n_track)
    with execution.reserve_memory(full, minimum=min(single, full), label="painting label continuity") as held:
        per_sample = max(1, single - LABEL_CONTINUITY_BASE_BYTES)
        batch = n_samples if held.nbytes >= full else max(1, (held.nbytes - LABEL_CONTINUITY_BASE_BYTES)
                                                             // per_sample)
        if model is None:
            from .stage import component_model
            model = component_model(panel, likelihoods, observed, evidence_sites, config, genetic_map)
        return _relabel_linked(panel, likelihoods, observed, evidence_sites, config, genetic_map, states,
                               movable, offsets, links, bins, counts, batch, model)


# Bookkeeping of one search beyond its per-sample arrays.
LABEL_CONTINUITY_BASE_BYTES = 64 << 20


def relabel_bytes(n_samples: int, n_states: int, n_bins: int, n_links: int, n_sites: int, *,
                  gathered: bool = True, n_track: int | None = None) -> int:
    """Peak bytes of a label-continuity search over ``n_samples`` samples at once (one batch), plus
    LABEL_CONTINUITY_BASE_BYTES. Per sample (D = S^2 ordered states, float64 unless noted): the gathered
    evidence (25 L, unless ``gathered`` is False: views of the evidence) and part emissions (8 D (B + links))
    held throughout, plus the largest of a painted-switch
    count (relabelled emissions 8 D B, their reversed copy 8 D B, the chain's expanded emissions 8 S'^2 B and int32
    Viterbi back-pointers 4 S'^2 B, S' = ``n_track`` track states; also the final verification), the
    link messages (relabelled emissions + forward/backward 16 D slots) and the link scores (messages + left/right
    16 D links); slots <= links. (The search before batching also kept the current and the accepted trial
    emissions: K30's 36 N S^2 B.)"""
    squares = n_states * n_states
    # A painted-switch count re-decodes with the class x fidelity chain (release 2.14): relabelled emissions 8 S^2 B,
    # their reversed copy 8 S^2 B (the mirrored convention), structured emissions 8 S'^2 B and int32 back-pointers
    # 4 S'^2 B, S' track states (at most 2 S - 1).
    tracks = (2 * n_states - 1) if n_track is None else n_track
    kept = 8 * squares * (n_bins + n_links) + (25 * n_sites if gathered else 0)
    working = max(16 * squares * n_bins + 12 * tracks * tracks * n_bins, 8 * squares * n_bins + 16 * squares * n_links,
                  32 * squares * n_links)
    # 1/8 margin: measured peaks reach 0.98-1.00 of the array sum (K30 synthetic 2,373 of 2,431 MiB; std_n80 chr8).
    return n_samples * (kept + working) * 9 // 8 + LABEL_CONTINUITY_BASE_BYTES


class _SampleBatches:
    """The component's samples in contiguous batches: their gathered evidence and part emissions, kept
    when one batch holds every sample, else recomputed for each batch when needed."""

    def __init__(self, likelihoods, observed, sites, states, part_start, part_stop, config, size, bin_starts,
                 part_bin):
        self.likelihoods, self.observed, self.sites = likelihoods, observed, sites
        self.states, self.part_start, self.part_stop, self.config = states, part_start, part_stop, config
        self.bin_starts, self.part_bin = bin_starts, part_bin
        n_samples = likelihoods.shape[0]
        self.spans = [(start, min(start + size, n_samples)) for start in range(0, n_samples, size)]
        self._kept = {}

    def gathered(self, start, stop):
        key = ("gathered", start)
        if key in self._kept:
            return self._kept[key]
        value = hmm.gather_sites(self.likelihoods, self.observed, self.sites, slice(start, stop))
        if len(self.spans) == 1:
            self._kept[key] = value
        return value

    def part_emissions(self, start, stop):
        key = ("parts", start)
        if key in self._kept:
            return self._kept[key]
        value = self._emissions(*self.gathered(start, stop))
        if len(self.spans) == 1:
            self._kept[key] = value
        return value

    def _emissions(self, likelihoods, observed):
        # Tempered as painting's: every part divided by its whole bin's design effect (painting.hmm.design_effect),
        # so the parts of a bin sum to the bin's tempered emission.
        design = hmm.design_effect(likelihoods, observed, self.bin_starts, self.config.emission_site_correlation)
        return np.ascontiguousarray(hmm.binned_emissions(
            likelihoods, observed, self.states.alt_probability(), self.states.background_frequency,
            self.part_start, self.part_stop, robustness_epsilon=self.config.robustness_epsilon,
            log_floor=self.config.emission_log_floor) / design[:, None, self.part_bin])


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
        if not all(np.array_equal(getattr(panel, name)[left, start:stop], getattr(panel, name)[right, start:stop])
                   for name in _SITE_FOUNDER_FIELDS):
            break
        link -= 1
    return link


def _relabel_linked(panel, likelihoods, observed, evidence_sites, config, genetic_map, states, movable,
                    offsets, links, bins, counts, batch, model):
    """The relabelling search of a component with link points and movable rows (relabel_component), over
    contiguous batches of ``batch`` samples (every sample at once when ``batch`` covers them)."""
    classes = states.classes
    site_indices = states.site_indices
    n_bins = bins.n_bins
    n_samples = likelihoods.shape[0]

    # Split painting bins at link points; part emissions use the painting's own kernel.
    link_sites = np.asarray([first for first, _ in links], dtype=np.int64)
    part_starts, part_stops, part_bin = [], [], []
    for bin_index, (start, stop) in enumerate(zip(bins.starts, bins.stops)):
        cuts = link_sites[(link_sites > start) & (link_sites < stop)]
        for left, right in zip(np.r_[start, cuts], np.r_[cuts, stop]):
            part_starts.append(left)
            part_stops.append(right)
            part_bin.append(bin_index)
    part_start = np.asarray(part_starts, dtype=np.int64)
    part_bin = np.asarray(part_bin, dtype=np.int64)
    part_segment = np.searchsorted(link_sites, part_start, side="right").astype(np.int64)
    size = max(1, int(batch))
    samples = _SampleBatches(likelihoods, observed, evidence_sites[site_indices], states, part_start,
                             np.asarray(part_stops, dtype=np.int64), config, size, bins.starts, part_bin)
    spans = samples.spans
    eligible = np.empty(n_samples, dtype=np.bool_)
    for start, stop in spans:
        eligible[start:stop] = np.any(hmm.informative_sites(*samples.gathered(start, stop)), axis=1)
    if not np.any(eligible):
        return panel, _outcome("no_evidence", **counts)

    n_states = states.n_states
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
    pairs = np.asarray([(first, second) for index, first in enumerate(movable)
                        for second in movable[index + 1:]], dtype=np.int64)

    def switches(start, stop, rows, maps):
        """Painted switches of the batch's samples ``rows`` (batch positions) under ``maps``."""
        emissions = _relabelled_emissions(samples.part_emissions(start, stop), rows, part_bin, part_segment,
                                          maps, n_bins)
        return fidelity.painted_switches(emissions, model, capable, eligible[start + rows])

    def winner_directions(winners, maps, first, second, link_bin):
        """(a -> b, b -> a): winners whose decoded path (the default convention, labels before the move) has a
        track in state ``first`` left of the link and ``second`` right of it, or the reverse, within two bins of it."""
        forward = reverse = 0
        low, high = max(0, link_bin - 2), min(n_bins - 1, link_bin + 2)
        for start, stop in spans:
            rows = winners[(winners >= start) & (winners < stop)] - start
            if not len(rows):
                continue
            emissions = _relabelled_emissions(samples.part_emissions(start, stop), rows, part_bin, part_segment,
                                              maps, n_bins)
            paths = fidelity.label_paths(np.ascontiguousarray(emissions), model, capable, reverse=False)
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

    def link_scores(start, stop, maps):
        """(left, right) link scores of the batch's samples under ``maps``."""
        parts = samples.part_emissions(start, stop)
        emissions = _relabelled_emissions(parts, np.arange(stop - start, dtype=np.int64), part_bin,
                                          part_segment, maps, n_bins)
        forward, backward = _link_messages(
            emissions, transitions.log_same, transitions.log_one, transitions.log_two,
            n_states, slot_of_bin, len(slot_bins))
        del emissions
        return _link_scores(parts, part_segment, maps, link_slot, link_segment, link_first_part,
                            link_stop_part, forward, backward)

    row_of_class = np.asarray([members[0] for members in classes], dtype=np.int64)
    # Error capabilities (release 2.14): a class has a panel-error state when it is called at a selected site. A suffix
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

    perms = np.tile(np.arange(n_states, dtype=np.int64), (len(links) + 1, 1))
    maps = _state_maps(perms, n_states)
    current = np.empty(n_samples, dtype=np.int64)
    for start, stop in spans:
        current[start:stop] = switches(start, stop, np.arange(stop - start, dtype=np.int64), maps)
    switches_before = int(np.sum(current))
    minimum_gain = config.label_continuity_minimum_switch_gain
    maximum_loser_ratio = config.label_continuity_maximum_loser_ratio
    checked = 0
    moves: list[RelabelMove] = []
    counter = ties.TieCounter()
    while True:
        # Summed gains: each batch adds its samples in sample order (the same sums as one pass).
        total = np.zeros((len(links), len(pairs)))
        for start, stop in spans:
            left, right = link_scores(start, stop, maps)
            _add_transposition_gains(total, left, right, eligible[start:stop], pairs, n_states)
            del left, right
        # Rank by the score gain in tie classes (release 2.1: gains equal up to rounding, e.g. one transposition at
        # two link points inside a stretch where both rows are identical, keep the leftmost link, then the pair
        # order, instead of the order their rounding gives); accept on exact per-sample painted switches.
        flats = np.flatnonzero(total.ravel() >= minimum_gain * unit)
        values = total.ravel()[flats]
        candidates = []
        if len(flats):
            quantum = ties.tie_quantum(float(np.max(np.abs(values))), "nll")
            ranked = ties.tie_classes(values, quantum, content=[int(flat) for flat in flats], descending=True,
                                      motion=ties.NOISE_MOTION["nll"])
            counter.add("painting.relabel_rank", ranked)
            ties.census("painting.relabel_rank", f"p{int(panel.positions[0])}/m{len(moves)}",
                        np.sort(values)[::-1][:8])
            candidates = [divmod(int(flats[index]), len(pairs)) for index in ranked.order]
            candidate_class = [int(ranked.classes[index]) for index in ranked.order]
        leftmost = {}
        results = {}

        def canonical(index):
            """(pair, link) of candidate ``index`` moved to its leftmost equivalent link (memoised along a stretch)."""
            link, pair = candidates[index]
            first, second = (int(value) for value in pairs[pair])
            walked = []
            while (pair, link) not in leftmost:
                left = _leftmost_equivalent_link(panel, offsets, links, perms, row_of_class, first, second, link,
                                                 steps=1)
                if left == link:
                    break
                walked.append(link)
                link = left
            target = leftmost.get((pair, link), link)
            for point in walked + [link]:
                leftmost[(pair, point)] = target
            return pair, target

        def evaluate(key):
            """(acceptable, change, winners, losers) of the relabelling ``key`` = (pair, link), computed once per
            iteration: every eligible sample is re-decoded under it (release 2.14)."""
            nonlocal checked
            if key not in results:
                pair, link = key
                first, second = (int(value) for value in pairs[pair])
                trial = perms.copy()
                trial[link + 1:, [first, second]] = trial[link + 1:, [second, first]]
                if anchored_under(trial, (first, second)) != anchored_under(perms, (first, second)):
                    results[key] = (False, np.zeros_like(current), np.zeros(0, dtype=np.int64),
                                    np.zeros(0, dtype=np.int64))
                    counter.note("painting.relabel_capability_change", inside_tie=False)
                    return results[key]
                trial_maps = _state_maps(trial, n_states)
                change = np.zeros_like(current)
                for start, stop in spans:
                    change[start:stop] = (switches(start, stop, np.arange(stop - start, dtype=np.int64), trial_maps)
                                          - current[start:stop])
                del trial_maps
                checked += 1
                winners, losers = np.flatnonzero(change < 0), np.flatnonzero(change > 0)
                acceptable = (-int(np.sum(change)) >= minimum_gain
                              and len(losers) <= maximum_loser_ratio * len(winners))
                results[key] = (acceptable, change, winners, losers)
            return results[key]

        # The first acceptable candidate in ranked order is applied, unless another acceptable candidate of its tie
        # class would relabel differently (release 2.14): competing repairs that this policy does not order leave the
        # assembled labels in place, and the search goes on with the next class. Equivalent placements of one
        # transposition (canonical) are evaluated once.
        accepted = None
        index = 0
        while index < len(candidates):
            key = canonical(index)
            if not evaluate(key)[0]:
                index += 1
                continue
            members = [other for other in range(len(candidates))
                       if candidate_class[other] == candidate_class[index] and other != index]
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
        # Tied suffix transpositions (release 2.14): where the two rows carry identical founder-axis data over the
        # stretch between two consecutive link points, transposing at either point gives the same per-site panel
        # data, painting and switches; the move is placed at the leftmost such point, a stated rule that the
        # convention reversal does not change.
        gain = float(total[link, pair] / unit)
        # Identity evidence (release 2.14): a label exchange of two founders' rows makes the carriers of either founder
        # switch at the link (both directions); one shared recombinant lineage switches one way only.
        forward, reverse = winner_directions(winners, maps, first, second,
                                             int(np.searchsorted(bins.starts, links[link][0], side="right") - 1))
        perms = perms.copy()
        perms[link + 1:, [first, second]] = perms[link + 1:, [second, first]]
        maps = _state_maps(perms, n_states)
        current = current + change
        span = links[link][1]
        moves.append(RelabelMove(
            atomic_span=int(span), position=int(panel.positions[offsets[span]]),
            rows=(int(classes[first][0]), int(classes[second][0])),
            score_gain_switches=gain,
            painted_switch_change=int(np.sum(change)),
            winners=tuple(int(value) for value in winners),
            losers=tuple(int(value) for value in losers),
            winners_forward=int(forward), winners_reverse=int(reverse)))

    counts.update(unit=unit, before=switches_before, checks=checked, moves=moves, tie_events=counter.record())
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
    if (new_states.classes != classes
            or not np.array_equal(new_states.site_indices, site_indices)):
        return panel, _outcome("trajectory_classes_changed", after=switches_before, **counts)
    if not np.array_equal(new_states.anchored, states.anchored):
        return panel, _outcome("error_capabilities_changed", after=switches_before, **counts)
    switches_after = 0
    for start, stop in spans:
        gathered = samples.gathered(start, stop)
        new_emissions = hmm.binned_emissions(
            *gathered, new_states.alt_probability(),
            new_states.background_frequency, bins.starts, bins.stops,
            robustness_epsilon=config.robustness_epsilon, log_floor=config.emission_log_floor
        ) / hmm.design_effect(*gathered, bins.starts, config.emission_site_correlation)[:, None, :]
        switches_after += int(np.sum(fidelity.painted_switches(new_emissions, model, capable, eligible[start:stop])))
        del new_emissions
    if switches_after >= switches_before:
        return panel, _outcome("painted_switches_not_lowered", after=switches_before, **counts)
    return relabelled, _outcome("relabelled", after=switches_after, row_maps=row_maps,
                                accepted=True, **counts)
