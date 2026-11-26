"""Read-state emission builder for the painting (PaintingParameters.read_classes > 1).

Wraps the stage's emission builder: for a batch of samples and a component's selected sites it returns, in place of
the tempered label emissions, a ReadEmissions (painting.read_decode) carrying them together with the read matrices of
every unordered label pair and bin (painting.read_operator), built from

- the samples' REF/ALT counts over the evidence grid spanning the selected sites (all called sites: the read state is
  carried through unselected sites by composing their operators; their alleles are not used);
- each sample's library's read survival and loss (science.shared.reads.fit_survival) and each sample's covering depth
  per homologue (its mean called depth / 2 / (1 - loss));
- each sample's per-read other-allele rate (the evidence read model's fitted error rate);
- the raw genotype likelihoods, the contig's cohort genotype law and outlier rate (the factor's GL anchor and outlier
  branch) and the bins' design-effect weights (the factor's tempering, R ** w, at the path strength and, when it
  differs, the confidence strength).

The path divisor is fixed. The confidence divisor is chosen by read_calibration on the read-state chain. The base
builder serves label continuity's scalar part emissions (``parts``). The stage (stages.painting.read_state) reads
the contig's inputs and builds this builder."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from ...core.concurrency import threaded_each
from ..shared import reads
from .emissions import contig_columns, gather_sites
from .parameters import READ_STATE_BATCH_BYTES, READ_STATE_CACHE_BYTES, READ_TABLE_CACHE_BYTES
from .read_decode import ReadEmissions
from .read_factor import deep_hidden_log
from .read_operator import (
    ALLELE_FACTOR_ROWS, compact_strength, compiled_strength, empty_tape, hidden_action,
    own_projection, site_tape, strength,
)

if TYPE_CHECKING:
    from .parameters import ReadStateParameters

__all__ = ["split_minor", "design_weights", "ReadEmissionBuilder", "UNIT_MINOR"]

# The split-class parameter of a batch carried with one class (2 minor + 3 = 1): read_factor.swap_classes(-1) is the
# identity on that class.
UNIT_MINOR = -1


def _unit_read_state(n_samples: int, n_pairs: int, n_bins: int):
    """(read matrices float32[n, U, B, 1, 1] = 1, log scales 0, class law [n, 1] = 1): the read state of one class."""
    return (np.ones((n_samples, n_pairs, n_bins, 1, 1), dtype=np.float32), np.zeros((n_samples, n_pairs, n_bins)),
            np.ones((n_samples, 1)))


def split_minor(read_classes: int) -> int:
    """The split-class parameter of ``read_classes`` > 1 classes (classes = 2 minor + 3; science.shared.reads.
    split_class_map): 3, 5, 7, 9 -> 0, 1, 2, 3."""
    return (int(read_classes) - 3) // 2


def design_weights(informative, bin_starts, correlation, group):
    """float64[n, B]: 1 / (1 + (n_w - 1) r) per sample and bin, n_w the informative sites of the bin's reference window
    (the stage's design effect; emissions.SharedEmissions)."""
    counts = np.add.reduceat(informative.astype(np.int64), np.asarray(bin_starts, dtype=np.int64), axis=1)
    window = np.add.reduceat(counts, np.arange(0, counts.shape[1], group), axis=1)
    extra = np.maximum(np.repeat(window, group, axis=1)[:, : counts.shape[1]] - 1, 0)
    return 1.0 / (1.0 + extra * float(correlation))


class ReadEmissionBuilder:
    """An EmissionBuilder (emissions.EmissionBuilder) adding the read state to ``base``'s emissions.

    counts int[N, L_evidence, 2] the contig's REF/ALT counts; positions int64[L_evidence]; survival the all-sample
    survival law; error_rate float64[N]; covering float64[N] covering depth per homologue; cohort float64[L_evidence,
    3]; minor the split-class parameter (2 minor + 3 classes); library int64[N] and survivals (one law per library,
    None for the all-sample law) the libraries; confidence_correlation the confidence strength's correlation (None
    when it equals the path's); label_correlation the label-error strength's (parameters.LABEL_ERROR_CORRELATION;
    None when no decode of this builder reads the label-error statistic).

    The label-independent parts of a sample batch (composed class operators, class laws and class likelihoods:
    functions of the counts, the batch's rows and the component's selected sites only) are cached per (rows, contig
    columns) within ``cache_bytes`` and reused by every later build of the same batch (the copying-model fits, label
    continuity, the decode and the reference painting): the same arrays, so the read matrices are identical whether a
    build hits or computes them. Inserts that do not fit evict the least recently used entries of OTHER site spans (an
    earlier component's); within one span the batches that fit stay (the builds sweep the batches in order, which
    would make plain least-recently-used eviction discard every batch before its reuse). The most recent batch's parts
    are also kept outside the budget, for its deferred confidence matrices. Nothing that depends on the observed
    flags or the tempering weights is cached: the read matrices are built at every call.

    A sample BYPASSES the read state (identity read state: one class, unit matrices, so its chain is the emission
    without read state exactly) when its every called depth over the span is at the deep bound (reads.deep_rows), or
    when its library shows no read sharing (its survival law's status is 'none': reads.fit_survival found no lag bin
    with sharing). Without sharing the read state carries nothing: with read_factor's bin-level correction its chain is
    then exactly the emission without read state, so the bypass is a shortcut to the same result (the tempered
    per-class factor R_s ** w alone, uncorrected, penalised heterozygous label pairs: sum_s pi_s R_s ** w < 1, about 8
    nats per bin measured, lab421 421b_lr_tempering_form); C = 1 is the exact limiting case. The rule acts per library
    when the samples form several libraries.

    ``corrected_rows``: the samples whose likelihood rows QC A replaced (stages.painting.contamination.
    corrected_evidence_with_rows). They carry no uniform foreign class (foreign_depth): their likelihoods already hold
    a fitted foreign-read model, so the class's GL-anchored donor ratios would explain the same reads a second time.
    Their foreign class forms no donor ratio either (foreign_counts), as an uncovered row's: with no entry and no
    arrival it carries no mass, but its ratios on contaminated reads would still dominate the unreachable classes'
    backward messages and underflow the own classes (long reads, sim_final2 chr13)."""

    def __init__(self, base, counts, positions, survival: reads.ReadSurvival, error_rate, covering, cohort, minor,
                 parameters: ReadStateParameters, confidence_correlation=None, library=None, survivals=None, *,
                 cache_bytes: int = READ_STATE_CACHE_BYTES, map_work=None, unit_batches: bool = False,
                 label_correlation=None, foreign_read_share: float = 0.0, action_bytes: int | None = None,
                 corrected_rows=()):
        self.base = base
        self.foreign_read_share = float(foreign_read_share)
        # memory for one strength's compiled actions of every sample (compiles); None: compile at every width
        self.action_bytes = None if action_bytes is None else int(action_bytes)
        self.counts = counts
        self.positions = np.asarray(positions, dtype=np.int64)
        self.survival = survival
        n = counts.shape[0]
        self.library = np.zeros(n, dtype=np.int64) if library is None else np.asarray(library, dtype=np.int64)
        self.survivals = [survival] if survivals is None else [survival if v is None else v for v in survivals]
        self.error_rate = np.asarray(error_rate, dtype=np.float64)
        self.covering = np.asarray(covering, dtype=np.float64)
        self.corrected = np.zeros(n, dtype=bool)
        self.corrected[np.asarray(corrected_rows, dtype=np.int64)] = True
        self.cohort = cohort
        self.minor = int(minor)
        self.parameters = parameters
        self._offset_widths = [
            reads.offset_pilot_width(self.covering[self.library == index], law.loss, parameters)
            if law.status == "fitted" and law.loss > 0.0 else parameters.offsets
            for index, law in enumerate(self.survivals)
        ]
        self._max_offsets = max(self._offset_widths, default=parameters.offsets)
        self.confidence_correlation = confidence_correlation
        self.label_correlation = label_correlation
        self.cache_bytes = int(cache_bytes)
        self._parts = OrderedDict()             # (rows, contig columns) -> (composed, laws, class_lik)
        self._parts_lock = nullcontext()
        self._parts_bytes = 0
        self._last = None                       # (key, parts) of the latest computed batch
        self._deep = {}                         # (rows, first, last) -> bool[n]
        # libraries without detected read sharing (survival status 'none'): their samples bypass the read state
        self.unshared = np.array([law.status != "fitted" for law in self.survivals], dtype=bool)
        self._statistics = dict(hits=0, misses=0, stored=0, evicted=0, uncached=0)
        # Independent table builds use the caller's ordered work map. Site kernels use its ambient Numba team.
        self.map_work = map_work
        # ``unit_batches``: a batch whose every sample bypasses the read state is returned with ONE class (C = 1: unit
        # read matrices, log scale 0, class law 1) instead of C classes of which only the first is ever reached: the
        # same chain and the same values (the other classes carry exact zeros), at the cost of the chain without read
        # state. The consumers read C from the arrays (2 minor + 3 = 1: minor = UNIT_MINOR)
        self.unit_batches = bool(unit_batches)
        self._tables = OrderedDict()            # reads.build_tables' cache, kept across the contig's batches

    def share_cached_parts(self):
        """Use one cache across the concurrent fits of a calibration fold."""
        self._parts_lock = Lock()

    def cache_statistics(self) -> dict:
        """Counters of the parts cache (diagnostic): hits, misses, stored, evicted, uncached inserts, bytes held."""
        return dict(self._statistics, bytes=self._parts_bytes, entries=len(self._parts))

    def compiles(self, n_labels: int, n_bins: int) -> bool:
        """Whether a positive-share build over ``n_bins`` bins with ``n_labels`` labels (BACKGROUND included) compiles its
        bin actions (dense binary64 per sample, unordered label pair and bin) or keeps the site walk; both routes are
        exact. Compiled actions are cheaper at every width measured as long as fits retain them and batches keep
        their concurrency, which holds while one strength's actions of every sample fit ``action_bytes``
        (parameters.READ_ACTION_MEMORY_FRACTION of the run's memory limit). The own-class route (no foreign reads)
        always compiles its fixed-width products."""
        if not self.foreign_read_share or self.action_bytes is None:
            return True
        width = 2 * self.minor + 3 + n_labels
        pairs = n_labels * (n_labels + 1) // 2
        return self.counts.shape[0] * pairs * n_bins * (8 * width * width + 16) <= self.action_bytes

    def sample_bytes(self, sites, n_labels: int, n_bins: int, confidence: bool | int, *, search: bool = True) -> int:
        """Working bytes per sample of one build over the selected contig columns ``sites`` with ``n_labels`` labels
        (BACKGROUND included) and ``n_bins`` bins: the read matrices and log scales at the path strength (and at
        ``confidence`` further strengths: the confidence and the label-error strength when each differs; a bool counts
        one), the label-independent parts at the selected sites, the genotype
        likelihoods, and the transient inputs of the parts over the evidence span (counts, offset weights and
        messages and posteriors, per-gap class operators). Path search carries two directional histories, and a fit
        carries one forward history per active sample (``search=False``)."""
        sites = np.asarray(sites)
        if not len(sites):
            return 1
        classes, offsets = 2 * self.minor + 3, self._max_offsets
        pairs = n_labels * (n_labels + 1) // 2
        part_bytes = len(sites) * ((classes * classes + 5 * classes) * 8 + 3 * 8 + 1)
        span = int(sites[-1]) - int(sites[0]) + 1
        # Peak depth-chain workspace includes both directional messages and its site marginal.
        transient = (span * (4 + 5 * offsets + classes * classes) + offsets * offsets) * 8
        tracks = 2 * n_labels - 1
        history_rows = 2 if search else 1
        if self.compiles(n_labels, n_bins):
            foreign = bool(self.foreign_read_share)
            width = classes + (n_labels if foreign else 0)
            # Strengths share the raw tape. One tempered tape and one set of D5P rows may be live
            # during a build, and the decode history is allocated afterwards.
            factor_width = classes + (3 if foreign else 0)
            tape_bytes = len(sites) * (ALLELE_FACTOR_ROWS * factor_width + (2 if foreign else 0)) * 8
            cell_bytes = 8 if foreign else 4
            matrices = (1 + int(confidence)) * pairs * n_bins * (cell_bytes * width * width + 16)
            correction = 8 * pairs * n_bins * (width + 2)
            history = 8 * history_rows * n_bins * tracks * tracks * (width + 2)
            return int(part_bytes + max(transient, matrices + tape_bytes + correction + max(tape_bytes, history)))
        # Site walk: own-block projections, log scales, D5P shifts and transient untempered row sums.
        matrices = pairs * n_bins * ((classes * classes * 4 + 16) * (1 + int(confidence)) + classes * 8 + 8)
        width = classes + n_labels
        factor_width = classes + 3
        matrices += len(sites) * ALLELE_FACTOR_ROWS * factor_width * (2 + int(confidence)) * 8
        matrices += len(sites) * 2 * 8 + pairs * n_bins * (width + 2 + int(confidence)) * 8
        history_bytes = 8 * history_rows * n_bins * tracks * tracks * (width + 2)
        return int(matrices + part_bytes + max(transient, history_bytes))

    def matrix_bytes(self, n_labels: int, n_bins: int, n_sites: int | None = None) -> int:
        """Bytes for an own-block consumer, or for a fit's retained action when its site count is given."""
        classes = 2 * self.minor + 3
        pairs = n_labels * (n_labels + 1) // 2
        own = pairs * n_bins * (classes * classes * 4 + 16)
        if not self.foreign_read_share or n_sites is None:
            return int(own)
        compiled = 8 * pairs * n_bins * ((classes + n_labels) ** 2 + 2) if self.compiles(n_labels, n_bins) else 0
        site_doubles = ALLELE_FACTOR_ROWS * (classes + 3) + classes * (classes + 1) + 2
        tape = 8 * (pairs * n_bins + n_sites * site_doubles)
        return int(max(compiled, tape))

    def fit_bytes(self, sites, n_labels: int, n_bins: int) -> int:
        """Per-sample fit input: the bin action and the ordered scalar emissions."""
        return self.matrix_bytes(n_labels, n_bins, len(sites)) + 8 * n_labels * n_labels * n_bins

    def batch_samples(self, sites, n_labels: int, n_bins: int, confidence: bool | int, requested: int,
                      budget: int = READ_STATE_BATCH_BYTES) -> int:
        """The sample batch for ``requested`` CPUs whose read-state working set (sample_bytes) fits ``budget``: the
        requested batch where it fits, else as many samples as fit (at least one)."""
        return max(1, min(int(requested), int(budget) // self.sample_bytes(sites, n_labels, n_bins, confidence)))

    def __call__(self, likelihoods, observed, sites, samples, alt_probability, background_frequency, bin_starts,
                 bin_stops, config, *, robustness_epsilon, site_correlation, parts=None, columns=None):
        # ``columns``: a resident copy's contig columns (emissions.contig_columns): ``sites`` then index the copy (the
        # GL gather), and the contig-wide counts, positions and cohort are read at contig_columns(sites, columns)
        raw, flags, informative = self.base(
            likelihoods, observed, sites, samples, alt_probability, background_frequency, bin_starts, bin_stops,
            config, robustness_epsilon=robustness_epsilon, site_correlation=site_correlation, parts=parts,
            **({} if columns is None else {"columns": columns}),
        )
        if parts is not None or len(sites) == 0:
            return raw, flags, informative
        sites = np.asarray(sites, dtype=np.int64)
        contig = contig_columns(sites, columns)
        start, stop, _ = samples.indices(self.counts.shape[0])
        rows = np.arange(start, stop)
        if self._unit_batch(rows, contig):
            n_pairs = (len(alt_probability) + 1) * (len(alt_probability) + 2) // 2
            read, read_log, initial = _unit_read_state(len(rows), n_pairs, len(bin_starts))
            return ReadEmissions(raw, read, read_log, initial, UNIT_MINOR), flags, informative
        group = max(1, int(config.reference_snps_per_bin) // int(config.snps_per_bin))
        weight = design_weights(informative, bin_starts, site_correlation, group)
        gl, seen = gather_sites(likelihoods, observed, sites, samples)
        seen = seen & flags
        alt_rows = np.vstack([np.asarray(alt_probability, dtype=np.float64),
                              np.asarray(background_frequency, dtype=np.float64)[None, :]])
        tape, extended, initial = site_tape(
            *self._operator_inputs(rows, contig, gl, seen, alt_rows, robustness_epsilon, bin_starts, bin_stops)
        )
        build = partial(self._structured, tape, extended, shared={},
                        compile=self.compiles(len(alt_rows), len(bin_starts)))
        matrices, log_scale, operator, shift = build(weight, with_shift=True)
        confidence = None
        if self.confidence_correlation is not None and float(self.confidence_correlation) != float(site_correlation):
            # deferred: built when the decode asks (ReadEmissions.at_confidence), from this batch's same inputs
            confidence = partial(build,
                                 design_weights(informative, bin_starts, self.confidence_correlation, group))
        label = None
        if self.label_correlation is not None and float(self.label_correlation) != float(site_correlation) and (
            self.confidence_correlation is None or float(self.label_correlation) != float(self.confidence_correlation)
        ):
            # deferred as the confidence matrices: built when the reference decode asks (ReadEmissions.at_label)
            label = partial(build,
                            design_weights(informative, bin_starts, self.label_correlation, group))
        emissions = ReadEmissions(
            raw, matrices, log_scale, initial, self.minor, confidence, label, shift, operator, bypass=tape[10],
        )
        return emissions, flags, informative

    def build_at(self, likelihoods, observed, sites, samples, alt_probability, background_frequency, bin_starts,
                 bin_stops, config, *, robustness_epsilon, site_correlation, bins, columns=None):
        """The batch of a build (__call__) at the bins ``bins`` only (int64 indices into ``bin_starts``, ascending,
        unique): (ReadEmissions whose raw, read and read_log hold those bins in that order (axis 2: position k is bin
        bins[k]), flags, informative of every bin). Each read matrix is built from its bin's sites and tempering
        weight. The log scale also uses the pair's class law carried through all previous bins. The builder
        computes the read matrices and scales through the last requested bin, then selects the requested bins.
        They match the full build bit for bit. The cached label-independent parts (the class operators and laws
        returned by site_parts) still cover every selected site. No deferred confidence matrices (a consumer at
        the path strength)."""
        raw, flags, informative = self.base(
            likelihoods, observed, sites, samples, alt_probability, background_frequency, bin_starts, bin_stops,
            config, robustness_epsilon=robustness_epsilon, site_correlation=site_correlation,
            **({} if columns is None else {"columns": columns}),
        )
        bins = np.asarray(bins, dtype=np.int64)
        if len(bins) > 1 and np.any(np.diff(bins) <= 0):
            raise ValueError("build_at needs ascending, unique bins")
        raw = np.ascontiguousarray(np.asarray(raw)[:, :, bins])
        sites = np.asarray(sites, dtype=np.int64)
        contig = contig_columns(sites, columns)
        start, stop, _ = samples.indices(self.counts.shape[0])
        rows = np.arange(start, stop)
        bin_starts = np.asarray(bin_starts, dtype=np.int64)
        bin_stops = np.asarray(bin_stops, dtype=np.int64)
        if len(sites) == 0 or self._unit_batch(rows, contig):
            n_pairs = (len(alt_probability) + 1) * (len(alt_probability) + 2) // 2
            read, read_log, initial = _unit_read_state(len(rows), n_pairs, len(bins))
            return ReadEmissions(raw, read, read_log, initial, UNIT_MINOR), flags, informative
        group = max(1, int(config.reference_snps_per_bin) // int(config.snps_per_bin))
        full_weight = design_weights(informative, bin_starts, site_correlation, group)
        gl, seen = gather_sites(likelihoods, observed, sites, samples)
        seen = seen & flags
        alt_rows = np.vstack([np.asarray(alt_probability, dtype=np.float64),
                              np.asarray(background_frequency, dtype=np.float64)[None, :]])
        # Resolve negative indices against the full grid before slicing. Keep the first bin for an empty request
        # so the correction can still initialise its carry.
        bins = np.where(bins < 0, bins + len(bin_starts), bins)
        prefix_stop = bins.max(initial=0) + 1
        weight = np.ascontiguousarray(full_weight[:, :prefix_stop])
        matrices, log_scale, initial = self.own_blocks(
            rows, contig, gl, seen, alt_rows, robustness_epsilon, bin_starts[:prefix_stop], bin_stops[:prefix_stop],
            weight, bins=bins,
        )
        bypass = self._bypassed(rows, int(contig[0]), int(contig[-1]) + 1)
        return ReadEmissions(raw, matrices, log_scale, initial, self.minor, bypass=bypass), flags, informative

    def own_blocks(self, rows, sites, gl, observed, alt_rows, epsilon, starts, stops, weight, *, bins=None):
        """Project the shared operator onto own classes for a consumer with gathered site inputs."""
        tape, extended, initial = site_tape(
            *self._operator_inputs(rows, sites, gl, observed, alt_rows, epsilon, starts, stops)
        )
        if tape[9].size:
            tempered, exponent, shift = strength(tape, extended, weight)
            matrices, log_scale = own_projection(tempered, exponent, shift, bins)
        else:
            matrices, log_scale, _ = compact_strength(tape, initial, weight, bins)
        return matrices, log_scale, initial

    def hidden_blocks(self, likelihoods, observed, sites, samples, alt_probability, background_frequency, bin_starts,
                      bin_stops, epsilon, weight, hidden, columns=None):
        """The visible-plus-hidden action (matrix, log gauge, optional tape) and scalar hidden log probabilities.

        The fold's hidden sites use the normalized count law at weight one. Observed holds the original flags.
        Samples bypassing the read state keep their scalar count law in the third result (zero for other samples).
        """
        sites = np.asarray(sites, dtype=np.int64)
        contig = contig_columns(sites, columns)
        start, stop, _ = samples.indices(self.counts.shape[0])
        rows = np.arange(start, stop)
        gl, seen = gather_sites(likelihoods, observed, sites, samples)
        alt_rows = np.vstack([np.asarray(alt_probability, dtype=np.float64),
                              np.asarray(background_frequency, dtype=np.float64)[None, :]])
        unit = self._unit_batch(rows, contig)
        n_pairs = alt_rows.shape[0] * (alt_rows.shape[0] + 1) // 2
        scalar = np.zeros((len(rows), n_pairs, len(bin_starts)))
        cohort = self.cohort[contig]
        bypassed = self._bypassed(rows, int(contig[0]), int(contig[-1]) + 1)
        if bypassed.any():
            counts = self.counts[np.ix_(rows[bypassed], contig)]
            scalar[bypassed] = deep_hidden_log(counts[..., 0], counts[..., 1], self.error_rate[rows[bypassed]],
                                                seen[bypassed], hidden, alt_rows, cohort, epsilon, bin_starts,
                                                bin_stops)
        if unit:
            matrices, log_scale, _ = _unit_read_state(len(rows), n_pairs, len(bin_starts))
            return matrices, log_scale, scalar, None
        matrices, log_scale, operator = hidden_action(
            *self._operator_inputs(rows, contig, gl, seen, alt_rows, epsilon, bin_starts, bin_stops),
            hidden, weight, np.zeros_like(scalar), compile=self.compiles(alt_rows.shape[0], len(bin_starts)),
        )
        return matrices, log_scale, scalar, operator

    def site_parts(self, sites, samples, columns=None):
        """(deep bool[n], composed operators float64[L, n, C, C], class laws float64[L, n, C], class likelihoods
        float64[n, L, 4, C]) of the sample batch ``samples`` at the selected contig columns ``sites``
        (contig_columns with ``columns``): the per-site parts of the read state for a consumer that needs its sites, not
        only the bins' products (the pedigree's within-bin class posteriors, science.pedigree.read_sources). A sample
        bypassing the read state (deep at every site, or of a library without read sharing; flagged in the first
        array) keeps identity operators, class 0's law and unit likelihoods: C = 1 for its consumers. The
        label-independent parts are the cached ones of this batch's builds."""
        sites = np.asarray(sites, dtype=np.int64)
        contig = contig_columns(sites, columns)
        start, stop, _ = samples.indices(self.counts.shape[0])
        rows = np.arange(start, stop)
        n_classes = 1 if self._unit_batch(rows, contig) else 2 * self.minor + 3
        if n_classes > 1 and len(sites) and len(np.unique(self.library[rows])) == 1:
            # one library and no bypassed sample: the cached parts themselves (callers only read them)
            library = int(self.library[rows[0]])
            if not self._bypassed(rows, int(contig[0]), int(contig[-1]) + 1).any():
                composed, laws, class_lik = self._label_independent(rows, contig, self.survivals[library])
                return np.zeros(len(rows), dtype=bool), composed, laws, class_lik
        deep = np.zeros(len(rows), dtype=bool)
        composed = np.zeros((len(sites), len(rows), n_classes, n_classes))
        composed[..., np.arange(n_classes), np.arange(n_classes)] = 1.0
        laws = np.zeros((len(sites), len(rows), n_classes))
        laws[..., 0] = 1.0
        class_lik = np.ones((len(rows), len(sites), 4, n_classes))
        if not len(sites):
            return deep, composed, laws, class_lik
        first, last = int(contig[0]), int(contig[-1]) + 1
        for library in np.unique(self.library[rows]):
            local = np.flatnonzero(self.library[rows] == library)           # positions within the batch
            bypassed = self._bypassed(rows[local], first, last)
            deep[local[bypassed]] = True
            shallow = local[~bypassed]
            if len(shallow):
                composed[:, shallow], laws[:, shallow], class_lik[shallow] = self._label_independent(
                    rows[shallow], contig, self.survivals[library])
        return deep, composed, laws, class_lik

    def _structured(self, tape, initial, weight, *, with_shift=False, shared=None, compile=True):
        if not tape[9].size:
            matrices, logs, shift = compact_strength(tape, initial, weight, shared=shared)
            operator = None
        else:
            compiled = compiled_strength(tape, initial, weight, shared=shared) if compile else None
            if compiled is None:
                tempered, exponent, shift = strength(tape, initial, weight, shared=shared)
                matrices = np.empty((0, 0, 0, 0, 0), dtype=np.float32)
                logs, operator = shift, (tempered, exponent, initial)
            else:
                matrices, logs, shift = compiled
                operator = (empty_tape(), np.empty((0, 0)), initial)
        return (matrices, logs, operator, shift) if with_shift else (matrices, logs, operator)

    def foreign_depth(self, rows) -> np.ndarray:
        """float64[n]: the Poisson mean foreign depth m = 2 share covering of the samples ``rows`` (an index array or a
        slice), zero for QC A's corrected rows (class docstring). The painting's operators, the pedigree's read gaps
        and its map-scale frames all take it from here; each zeroes the samples that bypass the read state itself."""
        depth = 2.0 * self.foreign_read_share * self.covering[rows]
        depth[self.corrected[rows]] = 0.0
        return depth

    def foreign_counts(self, rows, sites) -> np.ndarray:
        """float64[n, L, 2]: the REF/ALT counts the foreign class's donor ratios read for the samples ``rows`` (an
        index array or a slice) at contig columns ``sites``: zero for QC A's corrected rows (class docstring), whose
        ratios are then 1 at every site, as for an uncovered row."""
        # a slice of rows is a view and an index array is gathered with the sites: only the selected columns are copied
        picked = self.counts[rows][:, sites] if isinstance(rows, slice) else self.counts[np.ix_(rows, sites)]
        counts = np.ascontiguousarray(picked, dtype=np.float64)
        counts[self.corrected[rows]] = 0.0
        return counts

    def _operator_inputs(self, rows, sites, gl, observed, alt_rows, epsilon, starts, stops):
        bypass, transition, law, class_lik = self.site_parts(sites, slice(rows[0], rows[-1] + 1))
        depth = self.foreign_depth(rows)
        depth[bypass] = 0.0
        counts = np.empty((0, 0, 0))
        survival = np.empty((0, 0))
        loss = np.empty(0)
        if np.any(depth):
            counts = self.foreign_counts(rows, sites)
            gaps = np.diff(self.positions[sites], prepend=self.positions[sites[0]]).astype(np.float64)
            survival = np.empty((len(rows), len(sites)))
            loss = np.empty(len(rows))
            library_ids = self.library[rows]
            for library in np.unique(library_ids):
                use = library_ids == library
                model = self.survivals[library]
                survival[use] = model.survival(gaps)
                loss[use] = model.loss
            survival[:, 0] = 1.0
        return ((transition, law, class_lik), observed, alt_rows, gl, self.cohort[sites], epsilon, starts, stops,
                counts, self.error_rate[rows], loss, survival, depth, bypass)

    def _unit_batch(self, rows, contig):
        """Whether the batch ``rows`` over the contig columns ``contig`` is returned with one class (unit_batches and
        every sample bypassing the read state)."""
        return (self.unit_batches and len(rows) > 0 and len(contig) > 0
                and bool(self._bypassed(rows, int(contig[0]), int(contig[-1]) + 1).all()))

    def _bypassed(self, rows, first, last):
        """bool[n]: the samples ``rows`` that bypass the read state over contig columns [first, last): deep at every
        site, or of a library without read sharing (class docstring)."""
        return self._deep_rows(rows, first, last) | self.unshared[self.library[rows]]

    def _deep_rows(self, rows, first, last):
        key = (rows.tobytes(), first, last)
        deep = self._deep.get(key)
        if deep is None:
            deep = reads.deep_rows(np.asarray(self.counts[rows, first:last]).sum(axis=2), self.minor,
                                   self.parameters.deep_tail)
            self._deep[key] = deep
        return deep

    def _label_independent(self, rows, sites, survival):
        """(composed operators [L_kept, n, C, C], class laws [L_kept, n, C], class likelihoods [n, L_kept, 4, C]) of
        the samples ``rows`` (one library, none deep) at the selected contig columns ``sites``: cached (class
        docstring), else computed."""
        # A fold's concurrent candidates encounter the same sample batches. Share
        # the miss as well as the cached arrays and keep LRU/table updates ordered.
        with self._parts_lock:
            return self._cached_parts(rows, sites, survival)

    def _cached_parts(self, rows, sites, survival):
        key = (rows.tobytes(), sites.tobytes())
        hit = self._parts.get(key)
        if hit is not None:
            self._parts.move_to_end(key)
            self._statistics["hits"] += 1
            return hit
        if self._last is not None and self._last[0] == key:
            self._statistics["hits"] += 1
            return self._last[1]
        self._statistics["misses"] += 1
        pilot_width = self._offset_widths[self.library[rows[0]]]
        adaptive = pilot_width > self.parameters.offsets
        parameters = replace(self.parameters, offsets=pilot_width) if adaptive else self.parameters
        first, last = int(sites[0]), int(sites[-1]) + 1
        counts = np.asarray(self.counts[rows, first:last], dtype=np.int64)
        ref, alt = counts[..., 0], counts[..., 1]
        depth = ref + alt
        gaps = np.diff(self.positions[first:last], prepend=self.positions[first]).astype(np.float64)
        covering = self.covering[rows]
        nmax = reads.table_depth(depth, self.minor, parameters)
        tables = reads.build_tables(gaps, survival, covering, nmax, self.minor, parameters, cache=self._tables,
                                    cache_bytes=READ_TABLE_CACHE_BYTES, map_work=self.map_work or threaded_each)
        local_sites = np.ascontiguousarray(sites - first)
        if adaptive:
            posterior, operators, covering_depth = reads.adaptive_offset_operators(
                depth, covering, survival.loss, tables, self.parameters)
        else:
            posterior, operators, covering_depth = reads.offset_operators_fast(depth, covering, survival.loss, tables)
        posterior_kept = posterior
        if len(local_sites) == last - first:
            # The depth kernel already writes the identity at site zero.
            # Each kept interval has one edge, so the composed array is the original.
            composed = operators
        else:
            kept = np.zeros(last - first, dtype=bool)
            kept[local_sites] = True
            composed = reads.compose_operators_fast(operators, kept)
            ref, alt, posterior_kept = ref[:, kept], alt[:, kept], posterior[kept]
        del operators
        laws = reads.site_laws_fast(posterior, covering_depth, local_sites, tables)
        class_lik = reads.class_likelihoods_fast(ref, alt, self.error_rate[rows], posterior_kept, tables)
        parts = composed, laws, class_lik
        self._last = (key, parts)
        self._store(key, parts)
        return parts

    def _store(self, key, parts):
        size = sum(int(a.nbytes) for a in parts)
        if size <= self.cache_bytes:
            for old in [k for k in self._parts if k[1] != key[1]]:      # least recently used first
                if self._parts_bytes + size <= self.cache_bytes:
                    break
                self._parts_bytes -= sum(int(a.nbytes) for a in self._parts.pop(old))
                self._statistics["evicted"] += 1
        if self._parts_bytes + size <= self.cache_bytes:
            self._parts[key] = parts
            self._parts_bytes += size
            self._statistics["stored"] += 1
        else:
            self._statistics["uncached"] += 1
