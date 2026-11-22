"""Read-state emission builder for the painting (PaintingParameters.read_classes > 1).

Wraps the stage's emission builder: for a batch of samples and a component's selected sites it returns, in place of
the tempered label emissions, a ReadEmissions (painting.read_decode) carrying them together with the read matrices of
every unordered label pair and bin (painting.read_factor), built from

- the samples' REF/ALT counts over the evidence grid spanning the selected sites (all called sites: the read state is
  carried through unselected sites by composing their operators; their alleles are not used);
- each sample's library's read survival and loss (science.shared.reads.fit_survival) and each sample's covering depth
  per homologue (its mean called depth / 2 / (1 - loss));
- each sample's per-read other-allele rate (the evidence read model's fitted error rate);
- the raw genotype likelihoods, the contig's cohort genotype law and outlier rate (the factor's GL anchor and outlier
  branch) and the bins' design-effect weights (the factor's tempering, R ** w, at the path strength and, when it
  differs, the confidence strength).

Calibration parts (``parts``) are served by the base builder unchanged: the divisors are chosen on the model without
read state. The stage (stages.painting.read_state) reads the contig's inputs and builds this builder."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ...core.concurrency import threaded_each
from ..shared import reads
from .emissions import contig_columns, gather_sites
from .parameters import READ_STATE_BATCH_BYTES, READ_STATE_CACHE_BYTES, READ_TABLE_CACHE_BYTES
from .read_decode import ReadEmissions
from .read_factor import deep_hidden_log, hidden_read_matrices, read_matrices

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
    when it equals the path's).

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
    with sharing). Without sharing the read state carries nothing, while the tempered per-class factor R_s ** w at
    w < 1 is not neutral: sum_s pi_s R_s ** w < 1 (Jensen) penalises heterozygous label pairs, about 8 nats per bin
    measured (lab421, GPT 6.2 Astra 421b_lr_tempering_form; Tier 1); C = 1 is the exact limiting case. The rule acts
    per library when the samples form several libraries."""

    def __init__(self, base, counts, positions, survival: reads.ReadSurvival, error_rate, covering, cohort, minor,
                 parameters: ReadStateParameters, confidence_correlation=None, library=None, survivals=None, *,
                 cache_bytes: int = READ_STATE_CACHE_BYTES, map_work=None, unit_batches: bool = False):
        self.base = base
        self.counts = counts
        self.positions = np.asarray(positions, dtype=np.int64)
        self.survival = survival
        n = counts.shape[0]
        self.library = np.zeros(n, dtype=np.int64) if library is None else np.asarray(library, dtype=np.int64)
        self.survivals = [survival] if survivals is None else [survival if v is None else v for v in survivals]
        self.error_rate = np.asarray(error_rate, dtype=np.float64)
        self.covering = np.asarray(covering, dtype=np.float64)
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
        self.cache_bytes = int(cache_bytes)
        self._parts = OrderedDict()             # (rows, contig columns) -> (composed, laws, class_lik)
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

    def cache_statistics(self) -> dict:
        """Counters of the parts cache (diagnostic): hits, misses, stored, evicted, uncached inserts, bytes held."""
        return dict(self._statistics, bytes=self._parts_bytes, entries=len(self._parts))

    def sample_bytes(self, sites, n_labels: int, n_bins: int, confidence: bool) -> int:
        """Working bytes per sample of one build over the selected contig columns ``sites`` with ``n_labels`` labels
        (BACKGROUND included) and ``n_bins`` bins: the read matrices and log scales at the path strength (and the
        confidence strength when ``confidence``), the label-independent parts at the selected sites, the genotype
        likelihoods, and the transient inputs of the parts over the evidence span (counts, offset weights and
        messages and posteriors, per-gap class operators)."""
        sites = np.asarray(sites)
        if not len(sites):
            return 1
        classes, offsets = 2 * self.minor + 3, self._max_offsets
        pairs = n_labels * (n_labels + 1) // 2
        matrices = pairs * n_bins * (classes * classes * 4 + 8) * (2 if confidence else 1)
        parts = len(sites) * ((classes * classes + 5 * classes) * 8 + 3 * 8 + 1)
        span = int(sites[-1]) - int(sites[0]) + 1
        # Peak depth-chain workspace includes both directional messages and its site marginal. Joint edges use a
        # scratch matrix per worker. The site likelihood kernel contracts its weights in block-local scratch.
        transient = (span * (4 + 5 * offsets + classes * classes) + offsets * offsets) * 8
        return int(matrices + parts + transient)

    def matrix_bytes(self, n_labels: int, n_bins: int) -> int:
        """Bytes per sample of one strength's read matrices and log scales (what a copying-model fit holds)."""
        classes = 2 * self.minor + 3
        return int(n_labels * (n_labels + 1) // 2 * n_bins * (classes * classes * 4 + 8))

    def batch_samples(self, sites, n_labels: int, n_bins: int, confidence: bool, requested: int,
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
        inputs = (rows, contig, gl, seen, alt_rows, background_frequency, float(robustness_epsilon), bin_starts,
                  bin_stops)
        matrices, log_scale, initial = self._matrices(*inputs, weight)
        confidence = None
        if self.confidence_correlation is not None and float(self.confidence_correlation) != float(site_correlation):
            # deferred: built when the decode asks (ReadEmissions.at_confidence), from this batch's same inputs
            confidence = partial(self._confidence, inputs,
                                 design_weights(informative, bin_starts, self.confidence_correlation, group))
        return ReadEmissions(raw, matrices, log_scale, initial, self.minor, confidence), flags, informative

    def build_at(self, likelihoods, observed, sites, samples, alt_probability, background_frequency, bin_starts,
                 bin_stops, config, *, robustness_epsilon, site_correlation, bins, columns=None):
        """The batch of a build (__call__) at the bins ``bins`` only (int64 indices into ``bin_starts``, ascending,
        unique): (ReadEmissions whose raw, read and read_log hold those bins in that order (axis 2: position k is bin
        bins[k]), flags, informative of every bin). A bin's read matrix and log scale come from its own sites and its
        tempering weight alone (read_factor._matrices; the weights are the design weights of every bin, then
        selected), so they equal the full build's at those bins bit for bit, at the cost of those bins only; the
        label-independent parts are the batch's cached ones (computed for every selected site: the class operators
        and laws site_parts returns). No deferred confidence matrices (a consumer at the path strength)."""
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
        weight = np.ascontiguousarray(design_weights(informative, bin_starts, site_correlation, group)[:, bins])
        gl, seen = gather_sites(likelihoods, observed, sites, samples)
        seen = seen & flags
        alt_rows = np.vstack([np.asarray(alt_probability, dtype=np.float64),
                              np.asarray(background_frequency, dtype=np.float64)[None, :]])
        matrices, log_scale, initial = self._matrices(rows, contig, gl, seen, alt_rows, background_frequency,
                                                      float(robustness_epsilon), bin_starts[bins], bin_stops[bins],
                                                      weight)
        return ReadEmissions(raw, matrices, log_scale, initial, self.minor), flags, informative

    def hidden_matrices(self, likelihoods, observed, sites, samples, alt_probability, background_frequency, bin_starts,
                        bin_stops, epsilon, weight, hidden, columns=None):
        """(read matrices float32[n, U, B, C, C], log scales float64[n, U, B], scalar hidden log probabilities
        float64[n, U, B]) of the sample batch ``samples`` for calibration: the visible sites' factors at the tempering
        ``weight`` and the fold's hidden sites (``hidden`` bool over ``sites``) predicted under the normalised hidden
        law (read_factor.hidden_read_matrices). ``observed`` holds the ORIGINAL flags. A sample bypassing the read
        state (deep at every site, or of a library without read sharing: class docstring) keeps the identity read
        state and its hidden blocks' log probabilities come in the third array (read_factor.deep_hidden_log, the
        independent-reads count law; zero for the others). The label-independent parts are the
        cached ones of the visible builds."""
        sites = np.asarray(sites, dtype=np.int64)
        contig = contig_columns(sites, columns)
        start, stop, _ = samples.indices(self.counts.shape[0])
        rows = np.arange(start, stop)
        gl, seen = gather_sites(likelihoods, observed, sites, samples)
        alt_rows = np.vstack([np.asarray(alt_probability, dtype=np.float64),
                              np.asarray(background_frequency, dtype=np.float64)[None, :]])
        unit = self._unit_batch(rows, contig)
        n_classes = 1 if unit else 2 * self.minor + 3
        n_pairs = alt_rows.shape[0] * (alt_rows.shape[0] + 1) // 2
        matrices = np.zeros((len(rows), n_pairs, len(bin_starts), n_classes, n_classes), dtype=np.float32)
        matrices[..., 0, 0] = 1.0
        log_scale = np.zeros((len(rows), n_pairs, len(bin_starts)))
        scalar = np.zeros((len(rows), n_pairs, len(bin_starts)))
        cohort = self.cohort[contig]
        first, last = int(contig[0]), int(contig[-1]) + 1
        for library in np.unique(self.library[rows]):
            local = np.flatnonzero(self.library[rows] == library)
            deep = self._bypassed(rows[local], first, last)
            shallow, bypassed = local[~deep], local[deep]
            if len(shallow):
                composed, laws, class_lik = self._label_independent(rows[shallow], contig, self.survivals[library])
                matrices[shallow], log_scale[shallow] = hidden_read_matrices(
                    class_lik, laws, composed, seen[shallow], hidden, alt_rows, gl[shallow], cohort, epsilon,
                    bin_starts, bin_stops, weight[shallow])
            if len(bypassed):
                counts = np.asarray(self.counts[np.ix_(rows[bypassed], contig)], dtype=np.int64)
                scalar[bypassed] = deep_hidden_log(counts[..., 0], counts[..., 1], self.error_rate[rows[bypassed]],
                                                   seen[bypassed], hidden, alt_rows, cohort, epsilon, bin_starts,
                                                   bin_stops)
        return matrices, log_scale, scalar

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

    def _confidence(self, inputs, weight):
        return self._matrices(*inputs, weight)[:2]

    def _matrices(self, rows, contig, gl, observed, alt_rows, background_frequency, epsilon, bin_starts, bin_stops,
                  weight):
        """(read matrices, log scales, first-site class laws) of the batch's samples at the tempering ``weight``, each
        library with its own survival law."""
        n_classes = 2 * self.minor + 3
        n_pairs = alt_rows.shape[0] * (alt_rows.shape[0] + 1) // 2
        matrices = np.empty((len(rows), n_pairs, len(bin_starts), n_classes, n_classes), dtype=np.float32)
        log_scale = np.empty((len(rows), n_pairs, len(bin_starts)))
        initial = np.empty((len(rows), n_classes))
        for library in np.unique(self.library[rows]):
            local = np.flatnonzero(self.library[rows] == library)           # positions within the batch
            matrices[local], log_scale[local], initial[local] = self._library_part(
                rows[local], contig, gl[local], observed[local], alt_rows, background_frequency, epsilon, bin_starts,
                bin_stops, weight[local], self.survivals[library])
        return matrices, log_scale, initial

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

    def _library_part(self, rows, sites, gl, observed, alt_rows, background_frequency, epsilon, bin_starts, bin_stops,
                      weight, survival):
        """Read matrices of the batch's samples ``rows`` of one library (their own survival law and loss) at the
        tempering ``weight``. Samples that bypass the read state (deep at every site of the span, reads.deep_rows:
        then every site is 'balanced'; or of a library without read sharing) get the identity read state: the
        emission without read state exactly."""
        deep = self._bypassed(rows, int(sites[0]), int(sites[-1]) + 1)
        if deep.any():
            n_classes = 2 * self.minor + 3
            n_pairs = alt_rows.shape[0] * (alt_rows.shape[0] + 1) // 2
            m = np.zeros((len(rows), n_pairs, len(bin_starts), n_classes, n_classes), dtype=np.float32)
            m[..., 0, 0] = 1.0
            lg = np.zeros((len(rows), n_pairs, len(bin_starts)))
            init = np.zeros((len(rows), n_classes))
            init[:, 0] = 1.0
            shallow = np.flatnonzero(~deep)
            if len(shallow):
                m[shallow], lg[shallow], init[shallow] = self._library_part(
                    rows[shallow], sites, gl[shallow], observed[shallow], alt_rows, background_frequency, epsilon,
                    bin_starts, bin_stops, weight[shallow], survival)
            return m, lg, init
        composed, laws, class_lik = self._label_independent(rows, sites, survival)
        m, l_, _, _ = read_matrices(class_lik, laws, composed, observed, alt_rows, background_frequency, gl,
                                    self.cohort[sites], epsilon, bin_starts, bin_stops, weight)
        return m, l_, laws[0]

    def _label_independent(self, rows, sites, survival):
        """(composed operators [L_kept, n, C, C], class laws [L_kept, n, C], class likelihoods [n, L_kept, 4, C]) of
        the samples ``rows`` (one library, none deep) at the selected contig columns ``sites``: cached (class
        docstring), else computed."""
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
