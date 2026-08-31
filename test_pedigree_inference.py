"""Exact, enumerative tests for the raw-GL parent-state HMMs.

The probability- and log-space references below enumerate every hidden path.
They do not call private production helpers or reproduce the production
forward recursion.  Fixtures use founder rows that are locally distinct,
making the child-left-out external-source model explicit and independently
calculable.
"""

from itertools import product
import math
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import pedigree_inference as pedigree


_UNIFORM_GL = np.full(3, 1.0 / 3.0, dtype=np.float64)


def _information_groups(marker_counts, markers_per_information_block):
    groups = []
    group = 0
    markers_in_group = 0
    for block, marker_count in enumerate(marker_counts):
        if block > 0 and markers_in_group >= markers_per_information_block:
            group += 1
            markers_in_group = 0
        groups.append(group)
        markers_in_group += max(int(marker_count), 1)
    return np.asarray(groups, dtype=np.int64)


def _reference_context(
    gl,
    alleles,
    labels,
    founders,
    marker_counts,
    switch_probability,
    *,
    markers_per_information_block,
    effective_markers_per_information_block,
    external_state_pseudocount,
    external_transition_pseudocount,
):
    """Independently construct the child-left-out external-source model."""
    n_samples, n_bins, _, n_snps = alleles.shape
    n_states = founders.shape[0]

    # These tests intentionally avoid the separate issue of local IBS pooling.
    for block in range(n_bins):
        marker_count = int(marker_counts[block])
        rows = {
            tuple(founders[state, block, :marker_count])
            for state in range(n_states)
        }
        if len(rows) != n_states:
            raise AssertionError("reference fixture has duplicate founder rows")

    state_probability = np.zeros((n_samples, n_bins, n_states))
    background_alt = np.empty((n_samples, n_bins, n_snps))
    external_transition = np.zeros(
        (n_samples, n_bins, n_states, n_states), dtype=np.float64
    )

    for child in range(n_samples):
        for block in range(n_bins):
            counts = np.zeros(n_states, dtype=np.float64)
            for sample in range(n_samples):
                if sample == child:
                    continue
                for track in range(2):
                    state = int(labels[sample, block, track])
                    if state >= 0:
                        counts[state] += 1.0
            state_probability[child, block] = (
                counts + external_state_pseudocount / n_states
            ) / (counts.sum() + external_state_pseudocount)

            for snp in range(n_snps):
                observed = []
                for sample in range(n_samples):
                    if sample == child:
                        continue
                    for track in range(2):
                        value = int(alleles[sample, block, track, snp])
                        if value >= 0:
                            observed.append(value)
                background_alt[child, block, snp] = (
                    math.fsum(observed) + 0.5
                ) / (len(observed) + 1.0)

            if block == 0:
                continue
            for previous in range(n_states):
                transition_counts = np.zeros(n_states, dtype=np.float64)
                for sample in range(n_samples):
                    if sample == child:
                        continue
                    for track in range(2):
                        before = int(labels[sample, block - 1, track])
                        after = int(labels[sample, block, track])
                        if before == previous and after >= 0:
                            transition_counts[after] += 1.0
                linked = transition_counts.copy()
                # Founder identities in these fixtures stay locally distinct,
                # so their continuation bridge is the identity matrix.
                linked[previous] += external_transition_pseudocount
                linked /= (
                    transition_counts.sum()
                    + external_transition_pseudocount
                )
                theta = float(switch_probability[block])
                external_transition[child, block, previous] = (
                    (1.0 - theta) * linked
                    + theta * state_probability[child, block]
                )

    groups = _information_groups(
        marker_counts, markers_per_information_block
    )
    exponent = np.zeros((n_samples, n_bins), dtype=np.float64)
    for child in range(n_samples):
        for group in np.unique(groups):
            count = 0
            for block in np.flatnonzero(groups == group):
                for snp in range(int(marker_counts[block])):
                    if not np.array_equal(gl[child, block, snp], _UNIFORM_GL):
                        count += 1
            if count:
                value = min(
                    float(effective_markers_per_information_block),
                    float(count),
                ) / float(count)
                exponent[child, groups == group] = value
    return state_probability, background_alt, external_transition, exponent


def _transmitted_alt(hard_allele, background_alt, mismatch_probability):
    source_alt = (
        float(background_alt) if int(hard_allele) < 0 else float(hard_allele)
    )
    return (
        mismatch_probability
        + (1.0 - 2.0 * mismatch_probability) * source_alt
    )


def _gl_emission(gl_triplet, first_alt, second_alt):
    dosage_probability = np.asarray(
        (
            (1.0 - first_alt) * (1.0 - second_alt),
            first_alt * (1.0 - second_alt)
            + (1.0 - first_alt) * second_alt,
            first_alt * second_alt,
        ),
        dtype=np.float64,
    )
    return 3.0 * float(np.dot(gl_triplet, dosage_probability))


def _logsumexp(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in finite)
    )


def _two_track_log_reference(
    gl,
    first_sources,
    second_sources,
    marker_counts,
    mismatch_probability,
):
    """Enumerate four deterministic two-track paths in log space."""
    log_path_masses = []
    for first_track, second_track in product(range(2), repeat=2):
        log_mass = -math.log(4.0)
        for block, marker_count in enumerate(marker_counts):
            for snp in range(int(marker_count)):
                first_alt = _transmitted_alt(
                    first_sources[first_track, block, snp],
                    0.5,
                    mismatch_probability,
                )
                second_alt = _transmitted_alt(
                    second_sources[second_track, block, snp],
                    0.5,
                    mismatch_probability,
                )
                factor = _gl_emission(
                    gl[block, snp], first_alt, second_alt
                )
                log_mass += math.log(factor)
        log_path_masses.append(log_mass)
    return _logsumexp(log_path_masses)


def _track_transition(parent, before, after, block, hom, switch_probability):
    switch = (
        0.5
        if bool(hom[parent, block - 1]) or bool(hom[parent, block])
        else float(switch_probability[block])
    )
    return (1.0 - switch) if before == after else switch


def _enumerate_simple_model(
    model,
    child,
    parents,
    gl,
    alleles,
    hom,
    founders,
    switch_probability,
    state_probability,
    background_alt,
    external_transition,
    exponent,
    marker_counts,
    mismatch_probability,
):
    """Sum all hidden paths for M0, M1, or M2 in probability space."""
    n_bins = gl.shape[1]
    n_states = founders.shape[0]
    if model == 0:
        hidden_states = tuple(product(range(n_states), repeat=2))
    elif model == 1:
        hidden_states = tuple(product(range(2), range(n_states)))
    elif model == 2:
        hidden_states = tuple(product(range(2), repeat=2))
    else:
        raise AssertionError("unknown reference model")

    path_masses = []
    for path in product(hidden_states, repeat=n_bins):
        first = path[0]
        if model == 0:
            path_probability = (
                state_probability[child, 0, first[0]]
                * state_probability[child, 0, first[1]]
            )
        elif model == 1:
            path_probability = (
                0.5 * state_probability[child, 0, first[1]]
            )
        else:
            path_probability = 0.25

        for block in range(1, n_bins):
            before = path[block - 1]
            after = path[block]
            if model == 0:
                path_probability *= (
                    external_transition[
                        child, block, before[0], after[0]
                    ]
                    * external_transition[
                        child, block, before[1], after[1]
                    ]
                )
            elif model == 1:
                path_probability *= (
                    _track_transition(
                        parents[0], before[0], after[0], block,
                        hom, switch_probability,
                    )
                    * external_transition[
                        child, block, before[1], after[1]
                    ]
                )
            else:
                path_probability *= (
                    _track_transition(
                        parents[0], before[0], after[0], block,
                        hom, switch_probability,
                    )
                    * _track_transition(
                        parents[1], before[1], after[1], block,
                        hom, switch_probability,
                    )
                )

        emission = 1.0
        for block, state in enumerate(path):
            for snp in range(int(marker_counts[block])):
                if model == 0:
                    source1 = founders[state[0], block, snp]
                    source2 = founders[state[1], block, snp]
                elif model == 1:
                    source1 = alleles[parents[0], block, state[0], snp]
                    source2 = founders[state[1], block, snp]
                else:
                    source1 = alleles[parents[0], block, state[0], snp]
                    source2 = alleles[parents[1], block, state[1], snp]
                first_alt = _transmitted_alt(
                    source1,
                    background_alt[child, block, snp],
                    mismatch_probability,
                )
                second_alt = _transmitted_alt(
                    source2,
                    background_alt[child, block, snp],
                    mismatch_probability,
                )
                factor = _gl_emission(
                    gl[child, block, snp], first_alt, second_alt
                )
                emission *= factor ** exponent[child, block]
        path_masses.append(path_probability * emission)
    return math.log(math.fsum(path_masses))


def _enumerate_linked_model(
    model,
    child,
    parents,
    gl,
    alleles,
    hom,
    founders,
    switch_probability,
    state_probability,
    background_alt,
    external_transition,
    exponent,
    marker_counts,
    mismatch_probability,
    *,
    reset_fallback_at_missing=False,
):
    """Enumerate linked candidate-fallback paths in stable log space."""
    n_bins = gl.shape[1]
    n_states = founders.shape[0]
    if model == 1:
        hidden_states = tuple(
            product(range(2), range(n_states), range(n_states))
        )
    elif model == 2:
        hidden_states = tuple(
            product(
                range(2),
                range(n_states),
                range(2),
                range(n_states),
            )
        )
    else:
        raise AssertionError("linked fallback applies only to M1 and M2")

    log_path_masses = []
    for path in product(hidden_states, repeat=n_bins):
        first = path[0]
        if model == 1:
            initial_factors = (
                0.5,
                state_probability[child, 0, first[1]],
                state_probability[child, 0, first[2]],
            )
        else:
            initial_factors = (
                0.25,
                state_probability[child, 0, first[1]],
                state_probability[child, 0, first[3]],
            )
        if any(factor <= 0.0 for factor in initial_factors):
            continue
        log_mass = math.fsum(math.log(factor) for factor in initial_factors)

        reachable = True
        for block in range(1, n_bins):
            before = path[block - 1]
            after = path[block]
            if model == 1:
                fallback_transition = external_transition[
                    child, block, before[1], after[1]
                ]
                if (
                    reset_fallback_at_missing
                    and np.any(
                        alleles[
                            parents[0],
                            block,
                            after[0],
                            : int(marker_counts[block]),
                        ] < 0
                    )
                ):
                    fallback_transition = state_probability[
                        child, block, after[1]
                    ]
                transition_factors = (
                    _track_transition(
                        parents[0],
                        before[0],
                        after[0],
                        block,
                        hom,
                        switch_probability,
                    ),
                    fallback_transition,
                    external_transition[
                        child, block, before[2], after[2]
                    ],
                )
            else:
                transition_factors = (
                    _track_transition(
                        parents[0],
                        before[0],
                        after[0],
                        block,
                        hom,
                        switch_probability,
                    ),
                    external_transition[
                        child, block, before[1], after[1]
                    ],
                    _track_transition(
                        parents[1],
                        before[2],
                        after[2],
                        block,
                        hom,
                        switch_probability,
                    ),
                    external_transition[
                        child, block, before[3], after[3]
                    ],
                )
            if any(factor <= 0.0 for factor in transition_factors):
                reachable = False
                break
            log_mass += math.fsum(
                math.log(factor) for factor in transition_factors
            )
        if not reachable:
            continue

        for block, state in enumerate(path):
            for snp in range(int(marker_counts[block])):
                if model == 1:
                    source1 = alleles[
                        parents[0], block, state[0], snp
                    ]
                    if source1 < 0:
                        source1 = founders[state[1], block, snp]
                    source2 = founders[state[2], block, snp]
                else:
                    source1 = alleles[
                        parents[0], block, state[0], snp
                    ]
                    if source1 < 0:
                        source1 = founders[state[1], block, snp]
                    source2 = alleles[
                        parents[1], block, state[2], snp
                    ]
                    if source2 < 0:
                        source2 = founders[state[3], block, snp]
                first_alt = _transmitted_alt(
                    source1,
                    background_alt[child, block, snp],
                    mismatch_probability,
                )
                second_alt = _transmitted_alt(
                    source2,
                    background_alt[child, block, snp],
                    mismatch_probability,
                )
                factor = _gl_emission(
                    gl[child, block, snp], first_alt, second_alt
                )
                log_mass += exponent[child, block] * math.log(factor)
        log_path_masses.append(log_mass)
    return _logsumexp(log_path_masses)


def _reference_identity_information(
    gl, alleles, marker_counts, exponent
):
    n_samples, n_bins = gl.shape[:2]
    information = np.zeros((n_samples, n_samples), dtype=np.float64)
    fully_called = np.ones((n_samples, n_samples), dtype=np.bool_)
    for child in range(n_samples):
        for parent in range(n_samples):
            for block in range(n_bins):
                for snp in range(int(marker_counts[block])):
                    if np.array_equal(
                        gl[child, block, snp], _UNIFORM_GL
                    ):
                        continue
                    called = alleles[parent, block, :, snp] >= 0
                    if np.any(called):
                        information[child, parent] += exponent[child, block]
                    if not np.all(called):
                        fully_called[child, parent] = False
    return information, fully_called


def _reference_scores(
    gl,
    alleles,
    labels,
    hom,
    founders,
    marker_counts,
    switch_probability,
    trios,
    **kwargs,
):
    settings = {
        "mismatch_probability": 0.01,
        "markers_per_information_block": 100,
        "effective_markers_per_information_block": 1.0,
        "external_state_pseudocount": 1.0,
        "external_transition_pseudocount": 20.0,
    }
    settings.update(kwargs)
    context = _reference_context(
        gl,
        alleles,
        labels,
        founders,
        marker_counts,
        switch_probability,
        markers_per_information_block=settings[
            "markers_per_information_block"
        ],
        effective_markers_per_information_block=settings[
            "effective_markers_per_information_block"
        ],
        external_state_pseudocount=settings[
            "external_state_pseudocount"
        ],
        external_transition_pseudocount=settings[
            "external_transition_pseudocount"
        ],
    )
    state_probability, background, transition, exponent = context
    identity, fully_called = _reference_identity_information(
        gl, alleles, marker_counts, exponent
    )
    n_samples = len(alleles)
    zero = np.empty(n_samples, dtype=np.float64)
    one = np.full((n_samples, n_samples), -math.inf, dtype=np.float64)
    for child in range(n_samples):
        zero[child] = _enumerate_simple_model(
            0,
            child,
            (),
            gl,
            alleles,
            hom,
            founders,
            switch_probability,
            state_probability,
            background,
            transition,
            exponent,
            marker_counts,
            settings["mismatch_probability"],
        )
    for child in range(n_samples):
        for parent in range(n_samples):
            if child == parent:
                continue
            if identity[child, parent] <= 0.0:
                one[child, parent] = zero[child]
            elif fully_called[child, parent]:
                one[child, parent] = _enumerate_simple_model(
                    1,
                    child,
                    (parent,),
                    gl,
                    alleles,
                    hom,
                    founders,
                    switch_probability,
                    state_probability,
                    background,
                    transition,
                    exponent,
                    marker_counts,
                    settings["mismatch_probability"],
                )
            else:
                one[child, parent] = _enumerate_linked_model(
                    1,
                    child,
                    (parent,),
                    gl,
                    alleles,
                    hom,
                    founders,
                    switch_probability,
                    state_probability,
                    background,
                    transition,
                    exponent,
                    marker_counts,
                    settings["mismatch_probability"],
                )

    trio_array = np.asarray(trios, dtype=np.int64).reshape((-1, 3))
    two = np.empty(len(trio_array), dtype=np.float64)
    edge_information = np.empty((len(trio_array), 2), dtype=np.float64)
    for row, (child, parent1, parent2) in enumerate(trio_array):
        child = int(child)
        parent1 = int(parent1)
        parent2 = int(parent2)
        info1 = identity[child, parent1]
        info2 = identity[child, parent2]
        edge_information[row] = (info1, info2)
        if info1 <= 0.0 and info2 <= 0.0:
            two[row] = zero[child]
        elif info2 <= 0.0:
            two[row] = one[child, parent1]
        elif info1 <= 0.0:
            two[row] = one[child, parent2]
        elif (
            fully_called[child, parent1]
            and fully_called[child, parent2]
        ):
            two[row] = _enumerate_simple_model(
                2,
                child,
                (parent1, parent2),
                gl,
                alleles,
                hom,
                founders,
                switch_probability,
                state_probability,
                background,
                transition,
                exponent,
                marker_counts,
                settings["mismatch_probability"],
            )
        else:
            two[row] = _enumerate_linked_model(
                2,
                child,
                (parent1, parent2),
                gl,
                alleles,
                hom,
                founders,
                switch_probability,
                state_probability,
                background,
                transition,
                exponent,
                marker_counts,
                settings["mismatch_probability"],
            )
    return zero, one, two, identity, edge_information


def _base_inputs(n_bins=3, n_snps=1):
    founders = np.asarray([
        np.zeros((n_bins, n_snps), dtype=np.int8),
        np.ones((n_bins, n_snps), dtype=np.int8),
    ])
    alleles = np.empty((4, n_bins, 2, n_snps), dtype=np.int8)
    alleles[0, :, 0] = 0
    alleles[0, :, 1] = 1
    alleles[1, :, 0] = 0
    alleles[1, :, 1] = 1
    alleles[2, :, 0] = 1
    alleles[2, :, 1] = 0
    alleles[3, :, 0] = 0
    alleles[3, :, 1] = 1
    if n_bins >= 2:
        alleles[1, 1:, 0] = 1
        alleles[1, 1:, 1] = 0
    labels = alleles[..., 0].copy()
    hom = np.zeros((4, n_bins), dtype=np.bool_)
    if n_bins >= 2:
        hom[2, 1] = True
    gl = np.full((4, n_bins, n_snps, 3), 1.0 / 3.0)
    patterns = np.asarray(
        ([0.82, 0.15, 0.03], [0.08, 0.84, 0.08], [0.02, 0.18, 0.80])
    )
    for block in range(n_bins):
        for snp in range(n_snps):
            gl[0, block, snp] = patterns[(block + snp) % len(patterns)]
    marker_counts = np.full(n_bins, n_snps, dtype=np.int64)
    switch_probability = np.asarray(
        [0.0] + [0.12 + 0.09 * block for block in range(1, n_bins)],
        dtype=np.float64,
    )
    return (
        gl, alleles, labels, hom, founders,
        marker_counts, switch_probability,
    )


def _score(inputs, trios, **kwargs):
    return pedigree.score_parent_state_gl_hmms(
        *inputs, np.asarray(trios, dtype=np.int64), **kwargs
    )


class _StandardChunk:
    def __init__(self, start, end, hap1, hap2):
        self.start = start
        self.end = end
        self.hap1 = hap1
        self.hap2 = hap2


class _StandardPaintedSample:
    def __init__(self, chunks):
        self.chunks = chunks


class _StandardPainting(list):
    def __init__(self, samples, start_pos=0.0, end_pos=10000.0):
        super().__init__(samples)
        self.start_pos = start_pos
        self.end_pos = end_pos


class _StandardFounderBlock:
    def __init__(self, positions, haplotypes):
        self.positions = positions
        self.haplotypes = haplotypes


def _standard_raw_fixture(contig="ctg", dtype=np.float64):
    sample_ids = ("s0", "s1", "s2", "s3")
    positions = np.asarray(
        (50, 60, 150, 250, 260, 270, 9950), dtype=np.int64
    )
    haplotypes = {
        0: np.asarray((0, 1, 0, 1, 0, 1, 0), dtype=np.int8),
        1: np.asarray((1, 0, 1, 0, 1, 0, 1), dtype=np.int8),
    }
    painting = _StandardPainting([
        _StandardPaintedSample([_StandardChunk(0, 10000, 0, 1)]),
        _StandardPaintedSample([_StandardChunk(0, 10000, 0, 0)]),
        _StandardPaintedSample([_StandardChunk(0, 10000, 1, 1)]),
        _StandardPaintedSample([_StandardChunk(0, 10000, 1, 0)]),
    ])
    base = np.asarray(
        (
            (0.80, 0.15, 0.05),
            (0.10, 0.80, 0.10),
            (0.05, 0.20, 0.75),
            (0.60, 0.30, 0.10),
            (0.20, 0.60, 0.20),
            (0.15, 0.25, 0.60),
            (0.70, 0.20, 0.10),
        ),
        dtype=np.float64,
    )
    raw = np.empty((len(sample_ids), len(positions), 3), dtype=np.float64)
    for sample in range(len(sample_ids)):
        raw[sample] = np.roll(base, sample % 3, axis=1)
    raw = raw.astype(dtype)
    hard_item = {
        "contig": contig,
        "tolerance_painting": painting,
        "founder_block": _StandardFounderBlock(positions, haplotypes),
    }
    raw_item = dict(hard_item)
    raw_item.update(
        standard_state_evidence_mode="raw_likelihood",
        standard_raw_genotype_likelihoods=raw,
        standard_raw_positions=positions.copy(),
        standard_raw_sample_ids=sample_ids,
    )
    return sample_ids, hard_item, raw_item, raw


def _build_standard_test_cache(item, sample_ids):
    return pedigree._build_standard_contig_cache(
        item,
        0,
        len(sample_ids),
        100,
        5e-8,
        2,
        sample_ids=sample_ids,
    )


def _standard_test_trios(n_samples=4):
    rows = []
    for child in range(n_samples):
        parents = [parent for parent in range(n_samples) if parent != child]
        for first, second in product(parents, repeat=2):
            if first < second:
                rows.append((child, first, second))
    return np.asarray(rows, dtype=np.int64)


class StandardRawGLIntegrationTests(unittest.TestCase):
    def test_standard_cache_exactly_aligns_and_normalizes_raw_likelihoods(self):
        sample_ids, _, raw_item, raw = _standard_raw_fixture(
            dtype=np.float32
        )
        scales = np.arange(1, raw.shape[1] + 1, dtype=np.float32)
        scaled = raw * scales[None, :, None]
        raw_item["standard_raw_genotype_likelihoods"] = scaled
        cache = _build_standard_test_cache(raw_item, sample_ids)

        self.assertEqual(cache.state_evidence_mode, "raw_likelihood")
        self.assertEqual(cache.genotype_likelihoods.dtype, np.float64)
        selected = cache.selected_positions >= 0
        coordinates = cache.selected_positions[selected]
        self.assertEqual(len(coordinates), cache.informative_markers)
        raw_indices = np.searchsorted(
            raw_item["standard_raw_positions"], coordinates
        )
        bin_indices, slot_indices = np.nonzero(selected)
        expected = scaled[:, raw_indices].astype(np.float64)
        expected /= expected.sum(axis=2, keepdims=True)
        np.testing.assert_allclose(
            cache.genotype_likelihoods[:, bin_indices, slot_indices],
            expected,
            rtol=0.0,
            atol=2e-8,
        )
        np.testing.assert_array_equal(
            cache.genotype_likelihoods[:, ~selected],
            np.broadcast_to(
                _UNIFORM_GL,
                cache.genotype_likelihoods[:, ~selected].shape,
            ),
        )

    def test_standard_raw_dispatch_matches_direct_gl_scorer(self):
        sample_ids, hard_item, raw_item, _ = _standard_raw_fixture()
        hard_cache = _build_standard_test_cache(hard_item, sample_ids)
        raw_cache = _build_standard_test_cache(raw_item, sample_ids)
        trios = _standard_test_trios()
        config = pedigree.PedigreeConfig(bootstrap_replicates=1).validated()
        observed = pedigree._score_standard_contig_parent_states(
            raw_cache, trios, config
        )
        expected = pedigree.score_parent_state_gl_hmms(
            raw_cache.genotype_likelihoods,
            raw_cache.stacked_alleles,
            raw_cache.stacked_labels,
            raw_cache.stacked_hom_mask,
            raw_cache.founder_alleles,
            raw_cache.selected_markers_per_bin,
            raw_cache.switch_probabilities,
            trios,
            mismatch_probability=config.parent_state_mismatch_probability,
            phase_switch_probability=config.parent_state_phase_switch_probability,
            markers_per_information_block=config.markers_per_information_block,
            effective_markers_per_information_block=(
                config.parent_state_effective_markers_per_information_block
            ),
            external_state_pseudocount=(
                config.parent_state_external_state_pseudocount
            ),
            external_transition_pseudocount=(
                config.parent_state_external_transition_pseudocount
            ),
            candidate_source_mode=config.parent_state_candidate_source_mode,
            candidate_source_path_switch_probability=(
                config.parent_state_candidate_source_path_switch_probability
            ),
        )
        for field in (
            "zero_observed",
            "one_observed",
            "two_observed",
            "ancestry_junction_counts",
            "ancestry_callable_haplotype_bins",
            "one_parent_identity_information",
            "two_parent_edge_information",
        ):
            np.testing.assert_allclose(
                getattr(observed, field),
                getattr(expected, field),
                rtol=0.0,
                atol=1e-12,
            )
        np.testing.assert_allclose(
            pedigree._score_pair_hmm_contig(raw_cache, -3.0),
            pedigree._score_pair_hmm_contig(hard_cache, -3.0),
            rtol=0.0,
            atol=0.0,
        )
        hard_scores = pedigree._score_standard_contig_parent_states(
            hard_cache, trios, config
        )
        self.assertGreater(
            np.max(np.abs(
                observed.zero_observed - hard_scores.zero_observed
            )),
            1e-3,
        )

    def test_standard_end_to_end_preserves_screen_and_reports_raw_mode(self):
        sample_ids, hard_item, raw_item, _ = _standard_raw_fixture()
        config = pedigree.PedigreeConfig(
            bootstrap_replicates=4,
            parent_state_minimum_exposed_contigs=1,
        )
        kwargs = dict(
            top_k=3,
            snps_per_bin=100,
            recomb_rate=5e-8,
            mismatch_penalty=-3.0,
            max_snps_per_bin=2,
            n_workers=1,
            anchor_k=1,
            use_anchor_union=False,
        )
        hard_evidence = pedigree._standard_contig_evidence(
            [hard_item], sample_ids, config, **kwargs
        )
        raw_evidence = pedigree._standard_contig_evidence(
            [raw_item], sample_ids, config, **kwargs
        )
        np.testing.assert_array_equal(hard_evidence[1], raw_evidence[1])
        self.assertEqual(hard_evidence[-1], "hard_allele")
        self.assertEqual(raw_evidence[-1], "raw_likelihood")
        self.assertGreater(
            np.max(np.abs(
                hard_evidence[0][0].zero_parent_log_likelihoods
                - raw_evidence[0][0].zero_parent_log_likelihoods
            )),
            1e-3,
        )
        result = pedigree.infer_pedigree(
            [raw_item],
            sample_ids,
            config=config,
            scoring_kwargs=kwargs,
        )
        self.assertEqual(
            result.smart_standard_state_evidence_mode, "raw_likelihood"
        )
        self.assertIn(
            "exact-position-aligned raw linear genotype likelihoods",
            result.smart_evidence_source,
        )
        self.assertIn(
            "candidate panel remains selected by the unchanged hard-painted",
            result.smart_candidate_screening_scope,
        )

    def test_standard_raw_bundle_rejects_accidental_alignment_errors(self):
        sample_ids, hard_item, raw_item, _ = _standard_raw_fixture()
        partial = dict(raw_item)
        partial.pop("standard_raw_sample_ids")
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "requires all of"
        ):
            pedigree._standard_input_schema([partial])

        reversed_samples = dict(raw_item)
        reversed_samples["standard_raw_sample_ids"] = sample_ids[::-1]
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "exactly match ordered sample_ids"
        ):
            _build_standard_test_cache(reversed_samples, sample_ids)

        missing_coordinate = dict(raw_item)
        missing_coordinate["standard_raw_positions"] = (
            raw_item["standard_raw_positions"][:-1]
        )
        missing_coordinate["standard_raw_genotype_likelihoods"] = (
            raw_item["standard_raw_genotype_likelihoods"][:, :-1]
        )
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "missing selected founder coordinate"
        ):
            _build_standard_test_cache(missing_coordinate, sample_ids)

        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError,
            "raw likelihood evidence must be supplied for every contig",
        ):
            pedigree._standard_input_schema([raw_item, hard_item])



    def test_public_compact_transport_matches_full_raw_transport(self):
        sample_ids, _, raw_item, raw = _standard_raw_fixture()
        selection = dict(
            snps_per_bin=100,
            recombination_rate=5e-8,
            max_snps_per_bin=2,
        )
        bundle = pedigree.prepare_standard_compact_raw_gl(
            raw_item["tolerance_painting"],
            raw_item["founder_block"],
            sample_ids,
            raw,
            raw_item["standard_raw_positions"],
            contig=raw_item["contig"],
            **selection,
        )
        compact_item = {
            "contig": raw_item["contig"],
            "tolerance_painting": raw_item["tolerance_painting"],
            "founder_block": raw_item["founder_block"],
            "standard_compact_raw_gl": bundle,
        }
        full_cache = _build_standard_test_cache(raw_item, sample_ids)
        compact_cache = _build_standard_test_cache(compact_item, sample_ids)
        np.testing.assert_array_equal(
            bundle["selected_positions"], full_cache.selected_positions
        )
        np.testing.assert_array_equal(
            compact_cache.selected_positions, full_cache.selected_positions
        )
        np.testing.assert_allclose(
            compact_cache.genotype_likelihoods,
            full_cache.genotype_likelihoods,
            rtol=0.0,
            atol=2e-16,
        )

        trios = _standard_test_trios()
        config = pedigree.PedigreeConfig(bootstrap_replicates=1).validated()
        full_scores = pedigree._score_standard_contig_parent_states(
            full_cache, trios, config
        )
        compact_scores = pedigree._score_standard_contig_parent_states(
            compact_cache, trios, config
        )
        for field in (
            "zero_observed",
            "one_observed",
            "two_observed",
            "ancestry_junction_counts",
            "ancestry_callable_haplotype_bins",
            "one_parent_identity_information",
            "two_parent_edge_information",
        ):
            np.testing.assert_allclose(
                getattr(compact_scores, field),
                getattr(full_scores, field),
                rtol=0.0,
                atol=1e-12,
            )

        drift_bundle = dict(bundle)
        drift_bundle["selection_parameters"] = dict(
            bundle["selection_parameters"]
        )
        drift_bundle["selection_parameters"]["snps_per_bin"] += 1
        drift_item = dict(compact_item)
        drift_item["standard_compact_raw_gl"] = drift_bundle
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError,
            "compact selection parameter mismatch for snps_per_bin",
        ):
            _build_standard_test_cache(drift_item, sample_ids)

        both_item = dict(raw_item)
        both_item["standard_compact_raw_gl"] = bundle
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError,
            "cannot contain both full and compact transports",
        ):
            _build_standard_test_cache(both_item, sample_ids)

def _eligibility_test_evidence(
    *,
    excluded_child_shift=0.0,
    trios=None,
):
    sample_ids = ("s0", "s1", "s2", "s3")
    if trios is None:
        trios = _standard_test_trios(len(sample_ids))
    trios = np.asarray(trios, dtype=np.int64).reshape((-1, 3))
    evidence = []
    edge_matched = np.ones((4, 4), dtype=np.float64) * 100.0
    edge_exposed = edge_matched.copy()
    pair_explained = np.ones(len(trios), dtype=np.float64) * 100.0
    pair_exposed = pair_explained.copy()
    for contig_index in range(3):
        zero = np.asarray((0.0, 8.0, 8.0, 8.0), dtype=np.float64)
        one = np.full((4, 4), -8.0, dtype=np.float64)
        np.fill_diagonal(one, -np.inf)
        one[0, 1] = 30.0
        one[0, 2] = 20.0
        one[0, 3] = 50.0
        one[1, 2] = 100.0 + excluded_child_shift
        two = np.full(len(trios), -8.0, dtype=np.float64)
        for row, (child, first, second) in enumerate(trios):
            if child == 0 and (first, second) == (1, 2):
                two[row] = 25.0
            elif child == 0 and (first, second) == (1, 3):
                two[row] = 60.0
            elif child == 1:
                two[row] += excluded_child_shift
        evidence.append(pedigree.ParentStateEvidence(
            contig=f"ctg{contig_index}",
            trios=trios,
            zero_parent_log_likelihoods=zero,
            one_parent_log_likelihoods=one,
            two_parent_log_likelihoods=two,
            informative_markers=100,
            edge_matched_bins=edge_matched,
            edge_exposed_bins=edge_exposed,
            pair_explained_bins=pair_explained,
            pair_exposed_bins=pair_exposed,
            structure_total_bins=100.0,
        ))
    return sample_ids, evidence


def _ancestry_test_matrices(n_contigs, n_samples):
    junctions = np.tile(
        np.arange(n_samples, 0, -1, dtype=np.float64),
        (n_contigs, 1),
    )
    callable_bins = np.ones_like(junctions) * 100.0
    return junctions, callable_bins


def _eligibility_test_ancestry(sample_ids, evidence):
    junctions, callable_bins = _ancestry_test_matrices(
        len(evidence), len(sample_ids)
    )
    return {
        "ancestry_junction_counts": junctions,
        "ancestry_callable_haplotype_bins": callable_bins,
    }

def _without_structure_evidence(evidence):
    return [pedigree.ParentStateEvidence(
        contig=item.contig,
        trios=item.trios,
        zero_parent_log_likelihoods=item.zero_parent_log_likelihoods,
        one_parent_log_likelihoods=item.one_parent_log_likelihoods,
        two_parent_log_likelihoods=item.two_parent_log_likelihoods,
        informative_markers=item.informative_markers,
    ) for item in evidence]


def _eligibility_record(
    sample_ids,
    eligible_children,
    eligible_parents,
    eligible_pairs=None,
    *,
    policy_name="test_policy_v1",
):
    return pedigree.ParentEligibility(
        format_version=pedigree.PARENT_ELIGIBILITY_FORMAT_VERSION,
        sample_ids=tuple(sample_ids),
        eligible_children=np.asarray(eligible_children, dtype=np.bool_),
        eligible_parents=np.asarray(eligible_parents, dtype=np.bool_),
        eligible_parent_pairs=(
            None
            if eligible_pairs is None
            else np.asarray(eligible_pairs, dtype=np.bool_)
        ),
        policy_name=policy_name,
        source_fields=("test_field",),
        assumptions=("test assumption",),
        individual_parentage_ground_truth=False,
    )


class ParentEligibilityIntegrationTests(unittest.TestCase):
    @staticmethod
    def _config():
        return pedigree.PedigreeConfig(
            bootstrap_replicates=4,
            minimum_informative_contigs=1,
        )

    def test_explicit_all_eligible_matches_default_scientific_results(self):
        sample_ids, evidence = _eligibility_test_evidence()
        default = pedigree.infer_from_parent_state_evidence(
            evidence, sample_ids, config=self._config(), n_workers=1,
            **_eligibility_test_ancestry(sample_ids, evidence),
        )
        parents = np.ones((4, 4), dtype=np.bool_)
        np.fill_diagonal(parents, False)
        policy = _eligibility_record(
            sample_ids, np.ones(4, dtype=np.bool_), parents
        )
        explicit = pedigree.infer_from_parent_state_evidence(
            evidence,
            sample_ids,
            config=self._config(),
            parent_eligibility=policy,
            n_workers=1,
            **_eligibility_test_ancestry(sample_ids, evidence),
        )
        for field in (
            "complete_relationships",
            "tier_a_relationships",
            "tier_b_relationships",
            "smart_parent_state_calls",
        ):
            pd.testing.assert_frame_equal(
                getattr(default, field), getattr(explicit, field)
            )
        np.testing.assert_array_equal(
            default.smart_fitted_parent_state_prior_parameters,
            explicit.smart_fitted_parent_state_prior_parameters,
        )
        self.assertEqual(
            explicit.smart_parent_eligibility_policy_label, "test_policy_v1"
        )
        self.assertTrue(explicit.smart_parent_eligibility_supplied)

    def test_ineligible_candidates_are_not_screened_or_selected(self):
        sample_ids, evidence = _eligibility_test_evidence()
        children = np.asarray((True, False, False, False), dtype=np.bool_)
        parents = np.zeros((4, 4), dtype=np.bool_)
        parents[0, (1, 2)] = True
        pairs = np.zeros((4, 4, 4), dtype=np.bool_)
        pairs[0, 1, 2] = pairs[0, 2, 1] = True
        result = pedigree.infer_from_parent_state_evidence(
            evidence,
            sample_ids,
            config=self._config(),
            parent_eligibility=_eligibility_record(
                sample_ids, children, parents, pairs
            ),
            n_workers=1,
            **_eligibility_test_ancestry(sample_ids, evidence),
        )
        child = result.complete_relationships.iloc[0]
        self.assertEqual(child["Parent1"], "s1")
        self.assertIsNone(child["Parent2"])
        self.assertNotIn(
            "s3", {parent for parent in child[["Parent1", "Parent2"]]}
        )
        self.assertEqual(
            result.smart_diagnostics.loc[0, "EligibleParentCount"], 2
        )
        self.assertEqual(
            result.smart_diagnostics.loc[0, "EligibleParentPairCount"], 1
        )
        self.assertEqual(
            result.smart_diagnostics.loc[0, "ScoredCandidateCount2"], 1
        )
        for row in range(1, 4):
            self.assertEqual(
                result.complete_relationships.loc[row, "InferenceStatus"],
                "excluded_by_parent_eligibility",
            )
            self.assertEqual(
                result.complete_relationships.loc[row, "ParentState"],
                "unresolved",
            )


    def test_full_and_screened_counts_use_exact_eligible_universe(self):
        sample_ids, evidence = _eligibility_test_evidence(
            trios=np.asarray(((0, 1, 2),), dtype=np.int64)
        )
        children = np.asarray((True, False, False, False), dtype=np.bool_)
        parents = np.zeros((4, 4), dtype=np.bool_)
        parents[0, (1, 2, 3)] = True
        pairs = np.zeros((4, 4, 4), dtype=np.bool_)
        for first, second in ((1, 2), (1, 3)):
            pairs[0, first, second] = pairs[0, second, first] = True
        result = pedigree.infer_from_parent_state_evidence(
            evidence,
            sample_ids,
            config=self._config(),
            parent_eligibility=_eligibility_record(
                sample_ids, children, parents, pairs
            ),
            n_workers=1,
            **_eligibility_test_ancestry(sample_ids, evidence),
        )
        row = result.smart_diagnostics.iloc[0]
        self.assertEqual(row["ScoredCandidateCount1"], 3)
        self.assertEqual(row["FullCandidateCount1"], 3)
        self.assertEqual(row["ScoredCandidateCount2"], 1)
        self.assertEqual(row["FullCandidateCount2"], 2)
        self.assertTrue(row["M2StateEvidenceIsLowerBound"])

    def test_malformed_order_shapes_types_and_pair_symmetry_are_rejected(self):
        sample_ids = ("s0", "s1", "s2", "s3")
        children = np.ones(4, dtype=np.bool_)
        parents = np.ones((4, 4), dtype=np.bool_)
        np.fill_diagonal(parents, False)
        valid = {
            "format_version": 1,
            "sample_ids": sample_ids,
            "eligible_children": children,
            "eligible_parents": parents,
        }
        malformed = []
        malformed.append(({**valid, "format_version": 2}, "format_version"))
        malformed.append(({
            **valid, "sample_ids": sample_ids[::-1]
        }, "exactly match"))
        malformed.append(({
            **valid, "eligible_children": children.astype(np.int8)
        }, "eligible_children"))
        malformed.append(({
            **valid, "eligible_parents": parents[:-1]
        }, "eligible_parents"))
        asymmetric = np.zeros((4, 4, 4), dtype=np.bool_)
        asymmetric[0, 1, 2] = True
        malformed.append(({
            **valid, "eligible_parent_pairs": asymmetric
        }, "symmetric"))
        for record, message in malformed:
            with self.subTest(message=message), self.assertRaisesRegex(
                pedigree.PedigreeEvidenceError, message
            ):
                pedigree._resolve_parent_eligibility(record, sample_ids)

    def test_pipeline_wrapper_uses_first_item_policy_and_separate_adapter(self):
        sample_ids, hard_item, _, _ = _standard_raw_fixture()
        children = np.asarray((True, False, False, False), dtype=np.bool_)
        parents = np.zeros((4, 4), dtype=np.bool_)
        parents[0, (1, 2)] = True
        pairs = np.zeros((4, 4, 4), dtype=np.bool_)
        pairs[0, 1, 2] = pairs[0, 2, 1] = True
        hard_item = dict(hard_item)
        hard_item["smart_parent_eligibility"] = {
            "format_version": 1,
            "policy_name": "embedded_test_policy_v1",
            "sample_ids": tuple(sample_ids),
            "eligible_children": children,
            "eligible_parents": parents,
            "eligible_parent_pairs": pairs,
            "source_fields": ("generation",),
            "assumptions": ("integration test",),
            "individual_parentage_ground_truth": False,
        }
        hard_item["smart_config"] = pedigree.PedigreeConfig(
            bootstrap_replicates=2,
            minimum_informative_contigs=1,
            parent_state_minimum_exposed_contigs=1,
            parent_state_candidate_source_mode="hard_painted",
        )
        result = pedigree.infer_pedigree_for_pipeline(
            [hard_item],
            sample_ids,
            top_k=3,
            snps_per_bin=100,
            recomb_rate=5e-8,
            mismatch_penalty=-3.0,
            max_snps_per_bin=2,
            n_workers=1,
            anchor_k=1,
            use_anchor_union=False,
        )
        self.assertTrue(result.pipeline_control_adapter)
        self.assertEqual(result.smart_parent_state_algorithm_mode, "b1")
        self.assertEqual(
            result.smart_parent_state_candidate_source_mode,
            "hard_painted",
        )
        self.assertEqual(
            result.smart_parent_eligibility_policy_label,
            "embedded_test_policy_v1",
        )
        self.assertEqual(
            set(result.trio_candidate_scores["s0"][:1][0][:2]),
            {"s1", "s2"},
        )
        for row in range(1, 4):
            self.assertEqual(
                result.relationships.loc[row, "InferenceStatus"],
                "excluded_by_parent_eligibility",
            )
        self.assertNotIn("; no metadata", result.smart_evidence_source)
        self.assertEqual(
            result.smart_parent_eligibility_record["source_fields"],
            ("generation",),
        )

    def test_embedded_policy_is_rejected_outside_first_standard_item(self):
        sample_ids, hard_item, _, _ = _standard_raw_fixture()
        later = dict(hard_item)
        later["smart_parent_eligibility"] = {}
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "only on the first"
        ):
            pedigree._parent_eligibility_from_contig_inputs([hard_item, later])


    def test_exact_internal_bin_boundary_is_selected_once(self):
        sample_ids, hard_item, _, _ = _standard_raw_fixture()
        boundary_positions = np.asarray((100,), dtype=np.int64)
        founder_block = _StandardFounderBlock(
            boundary_positions,
            {
                0: np.asarray((0,), dtype=np.int8),
                1: np.asarray((1,), dtype=np.int8),
            },
        )
        raw = np.broadcast_to(
            np.asarray((0.8, 0.15, 0.05), dtype=np.float64),
            (len(sample_ids), 1, 3),
        ).copy()
        bundle = pedigree.prepare_standard_compact_raw_gl(
            hard_item["tolerance_painting"],
            founder_block,
            sample_ids,
            raw,
            boundary_positions,
            snps_per_bin=100,
            recombination_rate=5e-8,
            max_snps_per_bin=10,
            contig="ctg",
        )
        selected = np.argwhere(bundle["selected_positions"] == 100)
        np.testing.assert_array_equal(selected, np.asarray(((1, 0),)))

    def test_bcf_ad_loader_matches_independent_binomial_reference(self):
        sample_ids = ("s0", "s1")

        class Variant:
            def __init__(self, position, allele_depths):
                self.POS = position
                self.allele_depths = allele_depths

            def format(self, field):
                return self.allele_depths if field == "AD" else None

        class Reader:
            samples = sample_ids

            def __init__(self):
                self.requested_contig = None

            def __call__(self, contig):
                self.requested_contig = contig
                return iter((
                    Variant(100, np.asarray(((10, 0), (0, 0)))),
                    Variant(200, np.asarray(((2, 3), (-1, -1)))),
                ))

            def close(self):
                return None

        reader = Reader()
        with mock.patch("cyvcf2.VCF", return_value=reader) as constructor:
            observed, positions = pedigree.load_bcf_raw_genotype_likelihoods(
                "unused.bcf",
                "ctg",
                sample_ids,
                selected_positions=np.asarray((100, 200), dtype=np.int64),
                threads=4,
                read_error_probability=0.02,
            )
        constructor.assert_called_once_with("unused.bcf", threads=4)
        self.assertEqual(reader.requested_contig, "ctg")
        np.testing.assert_array_equal(positions, (100, 200))

        counts = np.asarray((
            ((10, 0), (2, 3)),
            ((0, 0), (0, 0)),
        ), dtype=np.float64)
        alt_probability = np.asarray((0.02, 0.5, 0.98))
        log_likelihood = (
            counts[:, :, 0, None] * np.log1p(-alt_probability)
            + counts[:, :, 1, None] * np.log(alt_probability)
        )
        log_likelihood -= np.max(log_likelihood, axis=2, keepdims=True)
        expected = np.exp(log_likelihood)
        expected /= np.sum(expected, axis=2, keepdims=True)
        expected[np.sum(counts, axis=2) == 0.0] = 1.0 / 3.0
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-15)

    def test_public_bcf_compact_uses_selected_ad_raw_likelihoods(self):
        sample_ids, hard_item, _, _ = _standard_raw_fixture()
        cache = _build_standard_test_cache(hard_item, sample_ids)
        requested = np.unique(
            cache.selected_positions[cache.selected_positions >= 0]
        )
        expected = np.broadcast_to(
            np.asarray((0.8, 0.15, 0.05), dtype=np.float64),
            (len(sample_ids), len(requested), 3),
        ).copy()
        expected[0, 0] = _UNIFORM_GL

        with mock.patch.object(
            pedigree,
            "load_bcf_raw_genotype_likelihoods",
            return_value=(expected, requested),
        ) as loader:
            bundle = pedigree.prepare_standard_compact_raw_gl_from_bcf(
                hard_item["tolerance_painting"],
                hard_item["founder_block"],
                sample_ids,
                "unused.bcf",
                bcf_contig="ctg",
                snps_per_bin=100,
                recombination_rate=5e-8,
                max_snps_per_bin=2,
                bcf_threads=3,
            )

        supplied_positions = loader.call_args.kwargs["selected_positions"]
        np.testing.assert_array_equal(supplied_positions, requested)
        self.assertEqual(loader.call_args.kwargs["threads"], 3)
        selected = bundle["selected_positions"] >= 0
        coordinates = bundle["selected_positions"][selected]
        raw_indices = np.searchsorted(requested, coordinates)
        bin_indices, slot_indices = np.nonzero(selected)
        np.testing.assert_allclose(
            bundle["genotype_likelihoods"][:, bin_indices, slot_indices],
            expected[:, raw_indices],
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(
            bundle["source_evidence_mode"],
            "bcf_ad_binomial_raw_likelihood_v1",
        )
        self.assertAlmostEqual(bundle["read_error_probability"], 0.02)


class RawGLParentStateExactTests(unittest.TestCase):
    def test_exact_enumeration_for_m0_m1_and_m2_known_sources(self):
        inputs = _base_inputs()
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        kwargs = dict(
            markers_per_information_block=1,
            effective_markers_per_information_block=1.0,
        )
        observed = _score(inputs, trios, **kwargs)
        expected = _reference_scores(*inputs, trios, **kwargs)
        self.assertAlmostEqual(observed.zero_observed[0], expected[0][0], places=11)
        self.assertAlmostEqual(observed.one_observed[0, 1], expected[1][0, 1], places=11)
        self.assertAlmostEqual(observed.two_observed[0], expected[2][0], places=11)

    def test_random_partial_masks_match_linked_path_enumeration(self):
        for n_bins in (2, 3):
            with self.subTest(n_bins=n_bins):
                rng = np.random.default_rng(9100 + n_bins)
                n_samples = 4
                n_snps = 2
                gl = np.full(
                    (n_samples, n_bins, n_snps, 3), 1.0 / 3.0
                )
                child_gl = rng.uniform(
                    0.05, 1.0, size=(n_bins, n_snps, 3)
                )
                gl[0] = child_gl / child_gl.sum(axis=2, keepdims=True)
                alleles = rng.integers(
                    0,
                    2,
                    size=(n_samples, n_bins, 2, n_snps),
                    dtype=np.int8,
                )
                labels = rng.integers(
                    0,
                    2,
                    size=(n_samples, n_bins, 2),
                    dtype=np.int16,
                )
                alleles[1, 0, 0, 0] = -1
                alleles[1, -1, 1, 1] = -1
                alleles[2, 0, 1, 1] = -1
                alleles[2, -1, 0, 0] = -1
                hom = np.zeros((n_samples, n_bins), dtype=np.bool_)
                hom[2, 1] = True
                founders = np.asarray(
                    (
                        np.zeros((n_bins, n_snps), dtype=np.int8),
                        np.ones((n_bins, n_snps), dtype=np.int8),
                    )
                )
                marker_counts = np.full(
                    n_bins, n_snps, dtype=np.int64
                )
                switch = np.asarray(
                    [0.0]
                    + [0.08 + 0.07 * block for block in range(1, n_bins)]
                )
                inputs = (
                    gl,
                    alleles,
                    labels,
                    hom,
                    founders,
                    marker_counts,
                    switch,
                )
                trios = np.asarray(((0, 1, 2),), dtype=np.int64)
                kwargs = dict(
                    markers_per_information_block=2,
                    effective_markers_per_information_block=2.0,
                )
                observed = _score(inputs, trios, **kwargs)
                expected = _reference_scores(
                    *inputs, trios, **kwargs
                )
                self.assertAlmostEqual(
                    observed.zero_observed[0], expected[0][0], places=11
                )
                self.assertAlmostEqual(
                    observed.one_observed[0, 1],
                    expected[1][0, 1],
                    places=11,
                )
                self.assertAlmostEqual(
                    observed.one_observed[0, 2],
                    expected[1][0, 2],
                    places=11,
                )
                self.assertAlmostEqual(
                    observed.two_observed[0], expected[2][0], places=11
                )
                np.testing.assert_allclose(
                    observed.one_parent_identity_information,
                    expected[3],
                    rtol=0.0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    observed.two_parent_edge_information,
                    expected[4],
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_fully_called_linked_paths_collapse_to_reduced_topology(self):
        inputs = _base_inputs(n_bins=2)
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        context = _reference_context(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[4],
            inputs[5],
            inputs[6],
            markers_per_information_block=1,
            effective_markers_per_information_block=1.0,
            external_state_pseudocount=1.0,
            external_transition_pseudocount=20.0,
        )
        state_probability, background, transition, exponent = context
        common = (
            inputs[0],
            inputs[1],
            inputs[3],
            inputs[4],
            inputs[6],
            state_probability,
            background,
            transition,
            exponent,
            inputs[5],
            0.01,
        )
        simple_one = _enumerate_simple_model(1, 0, (1,), *common)
        linked_one = _enumerate_linked_model(1, 0, (1,), *common)
        simple_two = _enumerate_simple_model(2, 0, (1, 2), *common)
        linked_two = _enumerate_linked_model(2, 0, (1, 2), *common)
        observed = _score(
            inputs, trios, markers_per_information_block=1
        )
        self.assertAlmostEqual(simple_one, linked_one, places=12)
        self.assertAlmostEqual(simple_two, linked_two, places=12)
        self.assertAlmostEqual(
            observed.one_observed[0, 1], simple_one, places=12
        )
        self.assertAlmostEqual(
            observed.two_observed[0], simple_two, places=12
        )

    def test_uniform_gl_is_exactly_neutral_for_every_model(self):
        inputs = list(_base_inputs(n_bins=2, n_snps=2))
        inputs[0][:] = 1.0 / 3.0
        observed = _score(tuple(inputs), ((0, 1, 2), (3, 1, 2)))
        np.testing.assert_array_equal(observed.zero_observed, 0.0)
        np.testing.assert_array_equal(
            observed.one_observed[~np.eye(4, dtype=bool)], 0.0
        )
        self.assertTrue(np.all(np.isneginf(np.diag(observed.one_observed))))
        np.testing.assert_array_equal(observed.two_observed, 0.0)
        np.testing.assert_array_equal(
            observed.one_parent_identity_information, 0.0
        )
        np.testing.assert_array_equal(
            observed.two_parent_edge_information, 0.0
        )

    def test_gl_triplets_must_be_normalized(self):
        inputs = list(_base_inputs(n_bins=1))
        inputs[0][0, 0, 0] = (2.0, 1.0, 1.0)
        with self.assertRaisesRegex(pedigree.PedigreeEvidenceError, "norm|sum"):
            _score(tuple(inputs), ((0, 1, 2),))

    def test_stacked_alleles_require_exactly_two_homologs(self):
        inputs = list(_base_inputs(n_bins=1))
        for homologs in (1, 3):
            with self.subTest(homologs=homologs):
                changed = list(inputs)
                if homologs == 1:
                    changed[1] = inputs[1][:, :, :1, :].copy()
                    changed[2] = inputs[2][:, :, :1].copy()
                else:
                    changed[1] = np.concatenate(
                        (inputs[1], inputs[1][:, :, :1, :]), axis=2
                    )
                    changed[2] = np.concatenate(
                        (inputs[2], inputs[2][:, :, :1]), axis=2
                    )
                with self.assertRaisesRegex(
                    pedigree.PedigreeEvidenceError, "homolog|shape.*2"
                ):
                    _score(tuple(changed), ((0, 1, 2),))

    def test_nonfinite_and_nonbinary_hom_masks_are_rejected(self):
        inputs = list(_base_inputs(n_bins=1))
        for invalid in (math.nan, math.inf, -1.0, 2.0):
            with self.subTest(invalid=invalid):
                changed = list(inputs)
                changed[3] = inputs[3].astype(np.float64)
                changed[3][0, 0] = invalid
                with self.assertRaisesRegex(
                    pedigree.PedigreeEvidenceError, "stacked_hom_mask"
                ):
                    _score(tuple(changed), ((0, 1, 2),))

    def test_missing_called_missing_uses_propagated_fallback_state(self):
        inputs = list(_base_inputs(n_bins=3))
        inputs[0][0, 0, 0] = (0.98, 0.015, 0.005)
        inputs[0][0, 1, 0] = (0.42, 0.53, 0.05)
        inputs[0][0, 2, 0] = (0.98, 0.015, 0.005)
        inputs[1][1, 0] = -1
        inputs[1][1, 1] = 0
        inputs[1][1, 2] = -1
        inputs[2][1:, :2] = 0
        inputs[2][1:, 2] = 1
        inputs[6] = np.zeros(3, dtype=np.float64)
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        kwargs = dict(
            markers_per_information_block=1,
            effective_markers_per_information_block=1.0,
            external_transition_pseudocount=100.0,
        )
        observed = _score(tuple(inputs), trios, **kwargs)
        expected = _reference_scores(*inputs, trios, **kwargs)
        context = _reference_context(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[4],
            inputs[5],
            inputs[6],
            markers_per_information_block=1,
            effective_markers_per_information_block=1.0,
            external_state_pseudocount=1.0,
            external_transition_pseudocount=100.0,
        )
        state_probability, background, transition, exponent = context
        common = (
            inputs[0],
            inputs[1],
            inputs[3],
            inputs[4],
            inputs[6],
            state_probability,
            background,
            transition,
            exponent,
            inputs[5],
            0.01,
        )
        correct = _enumerate_linked_model(1, 0, (1,), *common)
        reset_at_gap = _enumerate_linked_model(
            1,
            0,
            (1,),
            *common,
            reset_fallback_at_missing=True,
        )
        self.assertAlmostEqual(correct, expected[1][0, 1], places=12)
        self.assertAlmostEqual(
            observed.one_observed[0, 1], correct, places=11
        )
        self.assertGreater(abs(correct - reset_at_gap), 1e-3)

    def test_identity_information_and_zero_information_canonicalization(self):
        n_samples = 6
        n_bins = 2
        n_snps = 3
        gl = np.full((n_samples, n_bins, n_snps, 3), 1.0 / 3.0)
        gl[0, 0, 0] = (0.85, 0.12, 0.03)
        gl[0, 0, 1] = (0.10, 0.80, 0.10)
        gl[0, 1, 0] = (0.04, 0.16, 0.80)
        alleles = np.full(
            (n_samples, n_bins, 2, n_snps), -1, dtype=np.int8
        )
        alleles[1, 0, 0, 0] = 0
        alleles[1, 1, 1, 0] = 1
        alleles[3, 0, 0, 1] = 1
        alleles[4, :, 0] = 0
        alleles[4, :, 1] = 1
        labels = np.zeros((n_samples, n_bins, 2), dtype=np.int16)
        labels[:, :, 1] = 1
        hom = np.zeros((n_samples, n_bins), dtype=np.bool_)
        founders = np.asarray(
            (
                np.zeros((n_bins, n_snps), dtype=np.int8),
                np.ones((n_bins, n_snps), dtype=np.int8),
            )
        )
        marker_counts = np.asarray((2, 1), dtype=np.int64)
        switch = np.asarray((0.0, 0.17))
        inputs = (
            gl,
            alleles,
            labels,
            hom,
            founders,
            marker_counts,
            switch,
        )
        trios = np.asarray(
            (
                (0, 1, 2),
                (0, 2, 3),
                (0, 2, 5),
                (0, 2, 1),
                (0, 4, 2),
            ),
            dtype=np.int64,
        )
        kwargs = dict(
            markers_per_information_block=100,
            effective_markers_per_information_block=1.5,
        )
        observed = _score(inputs, trios, **kwargs)
        expected = _reference_scores(*inputs, trios, **kwargs)

        expected_child_information = np.asarray(
            (0.0, 1.0, 0.0, 0.5, 1.5, 0.0)
        )
        np.testing.assert_array_equal(
            observed.one_parent_identity_information[0],
            expected_child_information,
        )
        np.testing.assert_array_equal(
            observed.one_parent_identity_information, expected[3]
        )
        expected_edges = np.asarray(
            (
                (1.0, 0.0),
                (0.0, 0.5),
                (0.0, 0.0),
                (0.0, 1.0),
                (1.5, 0.0),
            )
        )
        np.testing.assert_array_equal(
            observed.two_parent_edge_information, expected_edges
        )
        np.testing.assert_array_equal(
            observed.two_parent_edge_information, expected[4]
        )
        np.testing.assert_allclose(
            observed.zero_observed, expected[0], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            observed.one_observed, expected[1], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            observed.two_observed, expected[2], rtol=0.0, atol=1e-12
        )

        self.assertEqual(
            observed.one_observed[0, 2], observed.zero_observed[0]
        )
        self.assertEqual(
            observed.one_observed[0, 5], observed.zero_observed[0]
        )
        self.assertEqual(
            observed.two_observed[0], observed.one_observed[0, 1]
        )
        self.assertEqual(
            observed.two_observed[1], observed.one_observed[0, 3]
        )
        self.assertEqual(
            observed.two_observed[2], observed.zero_observed[0]
        )
        self.assertEqual(
            observed.two_observed[3], observed.one_observed[0, 1]
        )
        self.assertEqual(
            observed.two_observed[4], observed.one_observed[0, 4]
        )
        self.assertEqual(
            observed.two_observed[0] - observed.one_observed[0, 1],
            0.0,
        )
        self.assertEqual(
            observed.two_observed[2] - observed.zero_observed[0],
            0.0,
        )

    def test_partial_missing_sources_use_linked_fallback_enumeration(self):
        inputs = list(_base_inputs(n_bins=1, n_snps=2))
        inputs[1][1, 0, :, 0] = -1
        inputs[1][3, 0, :, 1] = -1
        trios = np.asarray(((0, 1, 2), (0, 1, 3)), dtype=np.int64)
        observed = _score(tuple(inputs), trios)
        expected = _reference_scores(*inputs, trios)
        self.assertGreater(
            observed.one_parent_identity_information[0, 1], 0.0
        )
        self.assertGreater(
            observed.one_parent_identity_information[0, 3], 0.0
        )
        self.assertAlmostEqual(
            observed.one_observed[0, 1], expected[1][0, 1], places=12
        )
        np.testing.assert_allclose(
            observed.two_observed, expected[2], rtol=0.0, atol=1e-12
        )
    def test_both_linked_fallback_sources_can_end_missing(self):
        inputs = list(_base_inputs(n_bins=1, n_snps=2))
        inputs[1][1:3, 0, :, 0] = -1
        inputs[4][:, 0, 0] = -1
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        observed = _score(tuple(inputs), trios)
        expected = _reference_scores(*inputs, trios)
        self.assertGreater(
            observed.one_parent_identity_information[0, 1], 0.0
        )
        self.assertGreater(
            observed.one_parent_identity_information[0, 2], 0.0
        )
        self.assertAlmostEqual(
            observed.zero_observed[0], expected[0][0], places=12
        )
        self.assertAlmostEqual(
            observed.one_observed[0, 1], expected[1][0, 1], places=12
        )
        self.assertAlmostEqual(
            observed.two_observed[0], expected[2][0], places=12
        )
    def test_padded_snp_slots_are_ignored_by_oracle_and_scorer(self):
        baseline_inputs = list(_base_inputs(n_bins=2, n_snps=3))
        baseline_inputs[5] = np.asarray((1, 2), dtype=np.int64)
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        kwargs = dict(
            markers_per_information_block=10,
            effective_markers_per_information_block=3.0,
        )
        baseline = _score(tuple(baseline_inputs), trios, **kwargs)
        expected_baseline = _reference_scores(
            *baseline_inputs, trios, **kwargs
        )

        changed = list(baseline_inputs)
        changed[0] = baseline_inputs[0].copy()
        changed[1] = baseline_inputs[1].copy()
        changed[4] = baseline_inputs[4].copy()
        for block, marker_count in enumerate(baseline_inputs[5]):
            start = int(marker_count)
            changed[0][:, block, start:] = (0.0, 0.0, 1.0)
            changed[1][:, block, :, start:] = (
                1 - changed[1][:, block, :, start:]
            )
            changed[4][:, block, start:] = (
                1 - changed[4][:, block, start:]
            )
        observed = _score(tuple(changed), trios, **kwargs)
        expected_changed = _reference_scores(*changed, trios, **kwargs)

        np.testing.assert_allclose(
            expected_changed[0], expected_baseline[0], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            expected_changed[1], expected_baseline[1], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            expected_changed[2], expected_baseline[2], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            observed.zero_observed,
            baseline.zero_observed,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            observed.one_observed,
            baseline.one_observed,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            observed.two_observed,
            baseline.two_observed,
            rtol=0.0,
            atol=1e-12,
        )

    def test_parent_order_and_homolog_order_are_invariant(self):
        inputs = _base_inputs()
        trios = ((0, 1, 2), (0, 2, 1))
        baseline = _score(inputs, trios, markers_per_information_block=1)
        self.assertAlmostEqual(
            baseline.two_observed[0], baseline.two_observed[1], places=12
        )

        swapped = list(inputs)
        swapped[1] = swapped[1].copy()
        swapped[2] = swapped[2].copy()
        swapped[1][1] = swapped[1][1, :, ::-1, :]
        swapped[2][1] = swapped[2][1, :, ::-1]
        reordered = _score(
            tuple(swapped), trios, markers_per_information_block=1
        )
        self.assertAlmostEqual(
            baseline.one_observed[0, 1], reordered.one_observed[0, 1], places=12
        )
        np.testing.assert_allclose(
            baseline.two_observed, reordered.two_observed,
            rtol=0.0, atol=1e-12,
        )
        np.testing.assert_array_equal(
            baseline.two_parent_edge_information[0],
            baseline.two_parent_edge_information[1, ::-1],
        )
        np.testing.assert_array_equal(
            baseline.one_parent_identity_information,
            reordered.one_parent_identity_information,
        )
        np.testing.assert_array_equal(
            baseline.two_parent_edge_information,
            reordered.two_parent_edge_information,
        )

    def test_child_hard_painting_and_phase_setting_do_not_affect_gl_scores(self):
        inputs = _base_inputs()
        trios = ((0, 1, 2),)
        baseline = _score(inputs, trios, phase_switch_probability=0.0)
        changed = list(inputs)
        changed[1] = changed[1].copy()
        changed[2] = changed[2].copy()
        changed[3] = changed[3].copy()
        changed[1][0] = -1
        changed[2][0, :, 0] = -1
        changed[2][0, :, 1] = np.asarray([1, 0, 1], dtype=np.int8)
        changed[3][0] = ~changed[3][0]
        observed = _score(
            tuple(changed), trios, phase_switch_probability=0.49
        )
        self.assertAlmostEqual(
            baseline.zero_observed[0], observed.zero_observed[0], places=12
        )
        np.testing.assert_allclose(
            baseline.one_observed[0], observed.one_observed[0],
            rtol=0.0, atol=1e-12,
        )
        self.assertAlmostEqual(
            baseline.two_observed[0], observed.two_observed[0], places=12
        )

    def test_uniform_suffix_contributes_no_score_or_identity_evidence(self):
        short = _base_inputs(n_bins=1)
        long = []
        for index, value in enumerate(short):
            if index == 0:
                suffix = np.full((4, 2, 1, 3), 1.0 / 3.0)
                long.append(np.concatenate((value, suffix), axis=1))
            elif index == 1:
                suffix = np.repeat(value[:, -1:], 2, axis=1)
                long.append(np.concatenate((value, suffix), axis=1))
            elif index == 2:
                suffix = np.repeat(value[:, -1:], 2, axis=1)
                long.append(np.concatenate((value, suffix), axis=1))
            elif index == 3:
                suffix = np.zeros((4, 2), dtype=np.bool_)
                long.append(np.concatenate((value, suffix), axis=1))
            elif index == 4:
                suffix = np.repeat(value[:, -1:], 2, axis=1)
                long.append(np.concatenate((value, suffix), axis=1))
            elif index == 5:
                long.append(np.asarray((1, 1, 1), dtype=np.int64))
            else:
                long.append(np.asarray((0.0, 0.31, 0.27)))
        trios = ((0, 1, 2),)
        short_score = _score(short, trios)
        long_score = _score(tuple(long), trios)
        self.assertAlmostEqual(
            short_score.zero_observed[0], long_score.zero_observed[0], places=12
        )
        np.testing.assert_allclose(
            short_score.one_observed[0], long_score.one_observed[0],
            rtol=0.0, atol=1e-12,
        )
        self.assertAlmostEqual(
            short_score.two_observed[0], long_score.two_observed[0], places=12
        )
        np.testing.assert_array_equal(
            short_score.one_parent_identity_information,
            long_score.one_parent_identity_information,
        )
        np.testing.assert_array_equal(
            short_score.two_parent_edge_information,
            long_score.two_parent_edge_information,
        )

    def test_information_tempering_matches_exact_path_sum(self):
        inputs = _base_inputs(n_bins=1, n_snps=2)
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        tempered_kwargs = dict(
            markers_per_information_block=100,
            effective_markers_per_information_block=1.0,
        )
        full_kwargs = dict(
            markers_per_information_block=100,
            effective_markers_per_information_block=2.0,
        )
        tempered = _score(inputs, trios, **tempered_kwargs)
        full = _score(inputs, trios, **full_kwargs)
        expected_tempered = _reference_scores(
            *inputs, trios, **tempered_kwargs
        )
        expected_full = _reference_scores(*inputs, trios, **full_kwargs)
        self.assertAlmostEqual(
            tempered.zero_observed[0], expected_tempered[0][0], places=12
        )
        self.assertAlmostEqual(
            tempered.one_observed[0, 1], expected_tempered[1][0, 1], places=12
        )
        self.assertAlmostEqual(
            tempered.two_observed[0], expected_tempered[2][0], places=12
        )
        self.assertAlmostEqual(full.two_observed[0], expected_full[2][0], places=12)
        self.assertNotAlmostEqual(
            tempered.two_observed[0], full.two_observed[0], places=8
        )

    def test_long_deterministic_paths_remain_finite_for_m0_m1_and_m2(self):
        n_samples = 3
        n_bins = 1000
        mismatch_probability = 0.01
        gl = np.full((n_samples, n_bins, 1, 3), 1.0 / 3.0)
        gl[0, :, 0] = (0.0, 0.0, 1.0)
        alleles = np.zeros((n_samples, n_bins, 2, 1), dtype=np.int8)
        labels = np.zeros((n_samples, n_bins, 2), dtype=np.int16)
        hom = np.zeros((n_samples, n_bins), dtype=np.bool_)
        founders = np.zeros((1, n_bins, 1), dtype=np.int8)
        marker_counts = np.ones(n_bins, dtype=np.int64)
        switch = np.zeros(n_bins, dtype=np.float64)
        observed = pedigree.score_parent_state_gl_hmms(
            gl,
            alleles,
            labels,
            hom,
            founders,
            marker_counts,
            switch,
            np.asarray(((0, 1, 2),), dtype=np.int64),
            mismatch_probability=mismatch_probability,
            markers_per_information_block=1,
            effective_markers_per_information_block=1.0,
        )
        expected = n_bins * math.log(3.0 * mismatch_probability ** 2)
        self.assertTrue(np.isfinite(observed.zero_observed[0]))
        self.assertTrue(np.isfinite(observed.one_observed[0, 1]))
        self.assertTrue(np.isfinite(observed.two_observed[0]))
        self.assertAlmostEqual(
            observed.zero_observed[0], expected, places=9
        )
        self.assertAlmostEqual(
            observed.one_observed[0, 1], expected, places=9
        )
        self.assertAlmostEqual(
            observed.two_observed[0], expected, places=9
        )

    def test_tiny_paths_recover_against_bounded_logspace_enumeration(self):
        n_samples = 4
        n_bins = 2
        n_snps = 101
        mismatch_probability = 0.01
        gl = np.full((n_samples, n_bins, n_snps, 3), 1.0 / 3.0)
        gl[0, 0, :100] = (1.0, 0.0, 0.0)
        gl[0, 1] = (0.0, 0.0, 1.0)
        alleles = np.zeros((n_samples, n_bins, 2, n_snps), dtype=np.int8)
        alleles[:, :, 1] = 1
        labels = np.zeros((n_samples, n_bins, 2), dtype=np.int16)
        labels[:, :, 1] = 1
        hom = np.zeros((n_samples, n_bins), dtype=np.bool_)
        founders = np.asarray(
            (
                np.zeros((n_bins, n_snps), dtype=np.int8),
                np.ones((n_bins, n_snps), dtype=np.int8),
            )
        )
        marker_counts = np.asarray((100, 101), dtype=np.int64)
        switch = np.zeros(n_bins, dtype=np.float64)
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        observed = pedigree.score_parent_state_gl_hmms(
            gl,
            alleles,
            labels,
            hom,
            founders,
            marker_counts,
            switch,
            trios,
            mismatch_probability=mismatch_probability,
            markers_per_information_block=1000,
            effective_markers_per_information_block=201.0,
        )

        parent1_sources = np.swapaxes(alleles[1], 0, 1)
        parent2_sources = np.swapaxes(alleles[2], 0, 1)
        expected_zero = _two_track_log_reference(
            gl[0],
            founders,
            founders,
            marker_counts,
            mismatch_probability,
        )
        expected_one = _two_track_log_reference(
            gl[0],
            parent1_sources,
            founders,
            marker_counts,
            mismatch_probability,
        )
        expected_two = _two_track_log_reference(
            gl[0],
            parent1_sources,
            parent2_sources,
            marker_counts,
            mismatch_probability,
        )
        first_block_high = 100.0 * math.log(
            _gl_emission(gl[0, 0, 0], mismatch_probability, mismatch_probability)
        )
        first_block_low = 100.0 * math.log(
            _gl_emission(
                gl[0, 0, 0],
                1.0 - mismatch_probability,
                1.0 - mismatch_probability,
            )
        )
        self.assertGreater(
            first_block_high - first_block_low,
            -math.log(np.finfo(np.float64).tiny),
        )
        self.assertAlmostEqual(
            observed.zero_observed[0], expected_zero, places=10
        )
        self.assertAlmostEqual(
            observed.one_observed[0, 1], expected_one, places=10
        )
        self.assertAlmostEqual(
            observed.two_observed[0], expected_two, places=10
        )

    def test_unreachable_high_emission_local_states_have_zero_mass(self):
        n_samples = 6
        n_bins = 2
        n_snps = 220
        mismatch_probability = 0.01
        gl = np.full((n_samples, n_bins, n_snps, 3), 1.0 / 3.0)
        gl[0, 1] = (0.0, 0.0, 1.0)
        alleles = np.zeros((n_samples, n_bins, 2, n_snps), dtype=np.int8)
        alleles[3:] = 1
        labels = np.zeros((n_samples, n_bins, 2), dtype=np.int16)
        labels[:, :, 1] = 1
        founders = np.asarray(
            (
                np.zeros((n_bins, n_snps), dtype=np.int8),
                np.ones((n_bins, n_snps), dtype=np.int8),
            )
        )
        founders[1, 1] = 0
        hom = np.zeros((n_samples, n_bins), dtype=np.bool_)
        marker_counts = np.full(n_bins, n_snps, dtype=np.int64)
        switch = np.zeros(n_bins, dtype=np.float64)
        observed = pedigree.score_parent_state_gl_hmms(
            gl,
            alleles,
            labels,
            hom,
            founders,
            marker_counts,
            switch,
            np.asarray(((0, 1, 2),), dtype=np.int64),
            mismatch_probability=mismatch_probability,
            markers_per_information_block=n_snps,
            effective_markers_per_information_block=float(n_snps),
        )

        reachable_factor = 3.0 * mismatch_probability ** 2
        expected = n_snps * math.log(reachable_factor)
        background_alt = (6.0 + 0.5) / (10.0 + 1.0)
        inactive_alt = _transmitted_alt(
            -1, background_alt, mismatch_probability
        )
        inactive_m1_factor = (
            3.0 * mismatch_probability * inactive_alt
        )
        self.assertGreater(
            n_snps * math.log(inactive_m1_factor / reachable_factor),
            -math.log(np.finfo(np.float64).tiny),
        )
        self.assertAlmostEqual(
            observed.zero_observed[0], expected, places=9
        )
        self.assertAlmostEqual(
            observed.one_observed[0, 1], expected, places=9
        )
        self.assertAlmostEqual(
            observed.two_observed[0], expected, places=9
        )

    def test_founder_label_permutation_is_score_invariant(self):
        inputs = _base_inputs()
        trios = ((0, 1, 2),)
        baseline = _score(
            inputs, trios, markers_per_information_block=1
        )
        permuted = list(inputs)
        permuted[2] = 1 - inputs[2]
        permuted[4] = inputs[4][::-1].copy()
        observed = _score(
            tuple(permuted), trios, markers_per_information_block=1
        )
        np.testing.assert_allclose(
            observed.zero_observed,
            baseline.zero_observed,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            observed.one_observed,
            baseline.one_observed,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            observed.two_observed,
            baseline.two_observed,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            observed.one_parent_identity_information,
            baseline.one_parent_identity_information,
        )
        np.testing.assert_array_equal(
            observed.two_parent_edge_information,
            baseline.two_parent_edge_information,
        )

    def test_identical_source_model_collapses_m0_m1_and_m2(self):
        n_samples = 3
        gl = np.full((n_samples, 2, 1, 3), 1.0 / 3.0)
        gl[0, 0, 0] = (0.91, 0.08, 0.01)
        gl[0, 1, 0] = (0.74, 0.22, 0.04)
        alleles = np.zeros((n_samples, 2, 2, 1), dtype=np.int8)
        labels = np.zeros((n_samples, 2, 2), dtype=np.int16)
        hom = np.zeros((n_samples, 2), dtype=np.bool_)
        founders = np.zeros((1, 2, 1), dtype=np.int8)
        marker_counts = np.ones(2, dtype=np.int64)
        switch = np.asarray((0.0, 0.37))
        observed = pedigree.score_parent_state_gl_hmms(
            gl, alleles, labels, hom, founders, marker_counts, switch,
            np.asarray(((0, 1, 2),), dtype=np.int64),
            markers_per_information_block=1,
        )
        self.assertAlmostEqual(
            observed.zero_observed[0], observed.one_observed[0, 1], places=12
        )
        self.assertAlmostEqual(
            observed.one_observed[0, 1], observed.two_observed[0], places=12
        )

    def test_m1_is_observed_plus_external_not_duplicate_parent(self):
        inputs = list(_base_inputs(n_bins=1))
        inputs[0][0, 0, 0] = (0.01, 0.98, 0.01)
        inputs[1][1] = 0
        inputs[2][1] = 0
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        observed = _score(tuple(inputs), trios)
        expected = _reference_scores(*inputs, trios)
        self.assertAlmostEqual(
            observed.one_observed[0, 1], expected[1][0, 1], places=12
        )

        epsilon = 0.01
        known_alt = _transmitted_alt(0, 0.5, epsilon)
        duplicate_parent = math.log(
            _gl_emission(inputs[0][0, 0, 0], known_alt, known_alt)
        )
        self.assertGreater(observed.one_observed[0, 1], duplicate_parent)

    def test_all_missing_candidate_matches_m0_in_matched_external_model(self):
        inputs = list(_base_inputs(n_bins=1))
        inputs[1][1] = -1
        inputs[2][1] = -1
        inputs[1][2] = 0
        inputs[2][2] = 0
        inputs[1][3] = 1
        inputs[2][3] = 1
        observed = _score(tuple(inputs), ((0, 2, 3),))
        self.assertAlmostEqual(
            observed.one_observed[0, 1], observed.zero_observed[0], places=12
        )


class HardPaintingMissingAlleleExactTest(unittest.TestCase):
    def test_structured_missing_source_uses_background_marginalization(self):
        child = np.asarray([[0, 1], [1, 0]], dtype=np.int8)
        first = np.asarray([[-1, 0], [1, -1]], dtype=np.int8)
        second = np.asarray([[0, 1], [1, 0]], dtype=np.int8)
        alleles = np.asarray([[child], [first], [second]], dtype=np.int8)
        labels = np.asarray([[[0, 1]], [[-1, -1]], [[0, 1]]], dtype=np.int16)
        hom = np.zeros((3, 1), dtype=np.bool_)
        founders = np.asarray([[[0, 0]], [[1, 1]]], dtype=np.int8)
        trios = np.asarray(((0, 1, 2),), dtype=np.int64)
        epsilon = 0.01
        observed = pedigree.score_parent_state_hmms(
            alleles, labels, hom, founders, np.asarray((2,)),
            np.asarray((0.0,)), trios,
            mismatch_probability=epsilon,
            effective_markers_per_information_block=2.0,
        )

        background = []
        for snp in range(2):
            called = [
                int(alleles[sample, 0, track, snp])
                for sample in (1, 2)
                for track in range(2)
                if int(alleles[sample, 0, track, snp]) >= 0
            ]
            background.append((sum(called) + 0.5) / (len(called) + 1.0))

        state_likelihoods = []
        missing_as_zero_likelihoods = []
        for orientation, first_track, second_track in product(range(2), repeat=3):
            likelihood = 1.0
            missing_as_zero = 1.0
            for snp in range(2):
                observed_first = int(child[orientation, snp])
                observed_second = int(child[1 - orientation, snp])
                source_first = int(first[first_track, snp])
                source_second = int(second[second_track, snp])
                first_alt = _transmitted_alt(
                    source_first, background[snp], epsilon
                )
                second_alt = _transmitted_alt(
                    source_second, background[snp], epsilon
                )
                likelihood *= (
                    first_alt if observed_first else 1.0 - first_alt
                ) * (
                    second_alt if observed_second else 1.0 - second_alt
                )
                wrong_first_alt = _transmitted_alt(
                    max(source_first, 0), background[snp], epsilon
                )
                missing_as_zero *= (
                    wrong_first_alt if observed_first else 1.0 - wrong_first_alt
                ) * (
                    second_alt if observed_second else 1.0 - second_alt
                )
            state_likelihoods.append(likelihood)
            missing_as_zero_likelihoods.append(missing_as_zero)
        expected = math.log(math.fsum(state_likelihoods) / 8.0)
        wrong = math.log(math.fsum(missing_as_zero_likelihoods) / 8.0)
        self.assertAlmostEqual(observed.two_observed[0], expected, places=12)
        self.assertNotAlmostEqual(observed.two_observed[0], wrong, places=8)



class BootstrapParallelismTests(unittest.TestCase):
    @staticmethod
    def _run_counts(n_workers):
        n_samples = 14
        n_contigs = 4
        trios = np.asarray([
            (child, first, second)
            for child in range(n_samples)
            for first in range(n_samples)
            for second in range(first + 1, n_samples)
            if child != first and child != second
        ], dtype=np.int64)
        rng = np.random.default_rng(1907)
        zero = rng.normal(size=(n_contigs, n_samples))
        one = rng.normal(size=(n_contigs, n_samples, n_samples))
        for contig in range(n_contigs):
            np.fill_diagonal(one[contig], -np.inf)
        two = rng.normal(size=(n_contigs, len(trios)))
        (
            alternatives,
            states,
            contig_log_likelihoods,
            by_child,
            full_counts,
            _,
        ) = pedigree._parent_state_alternatives(
            trios, zero, one, two, contamination=0.0
        )
        settings = pedigree.PedigreeConfig(
            bootstrap_replicates=80,
            bootstrap_seed=2718,
            parent_state_contamination_probability=0.0,
        ).validated()
        junctions, callable_bins = _ancestry_test_matrices(
            n_contigs, n_samples
        )
        structure_pair_indices = pedigree._structure_pair_indices(
            alternatives, states, trios
        )
        edge_matched = np.ones(
            (n_contigs, n_samples, n_samples), dtype=np.float64
        ) * 100.0
        edge_exposed = edge_matched.copy()
        pair_explained = np.ones(
            (n_contigs, len(trios)), dtype=np.float64
        ) * 100.0
        pair_exposed = pair_explained.copy()
        structure_total_bins = np.ones(n_contigs) * 100.0
        count_arrays = (
            np.zeros(len(alternatives), dtype=np.int64),
            np.zeros(len(alternatives), dtype=np.int64),
            np.zeros((n_samples, 3), dtype=np.int64),
            np.zeros((n_samples, 3), dtype=np.int64),
            np.zeros((n_samples, n_samples), dtype=np.int64),
            np.zeros((n_samples, n_samples), dtype=np.int64),
        )
        worker_count, depth_refits = pedigree._run_parent_state_bootstraps(
            contig_log_likelihoods,
            alternatives,
            states,
            by_child,
            full_counts,
            junctions,
            callable_bins,
            settings,
            n_workers,
            *count_arrays,
            contig_information_weights=np.ones(n_contigs),
            structure_pair_indices=structure_pair_indices,
            edge_matched_by_contig=edge_matched,
            edge_exposed_by_contig=edge_exposed,
            pair_explained_by_contig=pair_explained,
            pair_exposed_by_contig=pair_exposed,
            structure_total_bins_by_contig=structure_total_bins,
        )
        return worker_count, depth_refits, count_arrays

    def test_real_panel_uses_complete_worker_budget(self):
        self.assertEqual(
            pedigree._smart_bootstrap_worker_count(90_596, 1_000, 112),
            112,
        )
        self.assertEqual(
            pedigree._smart_bootstrap_worker_count(90_596, 1_000, 7),
            7,
        )
        self.assertEqual(
            pedigree._smart_bootstrap_worker_count(90_596, 31, 112),
            1,
        )
        self.assertEqual(
            pedigree._smart_bootstrap_worker_count(10, 32, 112),
            1,
        )

    def test_parallel_counts_are_bit_identical_to_serial_counts(self):
        serial_workers, serial_depth_refits, serial_counts = (
            self._run_counts(1)
        )
        parallel_workers, parallel_depth_refits, parallel_counts = (
            self._run_counts(4)
        )
        self.assertEqual(serial_workers, 1)
        self.assertEqual(parallel_workers, 4)
        self.assertEqual(serial_depth_refits, 80)
        self.assertEqual(parallel_depth_refits, 80)
        for observed, expected in zip(parallel_counts, serial_counts):
            np.testing.assert_array_equal(observed, expected)

class ParentStateB1Tests(unittest.TestCase):
    @staticmethod
    def _config():
        return pedigree.PedigreeConfig(
            bootstrap_replicates=4,
            bootstrap_seed=20260725,
            minimum_informative_contigs=1,
        )

    def test_fixed_combined_b1_provenance_and_defaults(self):
        default = pedigree.PedigreeConfig().validated()
        self.assertEqual(default.parent_state_algorithm_mode, "b1")
        self.assertEqual(default.parent_state_structure_mode, "combined_v1")
        self.assertEqual(default.parent_state_candidate_source_mode, "hard_painted")
        self.assertEqual(default.parent_state_effective_markers_per_information_block, 3.0)
        self.assertEqual(default.parent_state_minimum_edge_coverage, 0.95)
        self.assertEqual(default.parent_state_minimum_pair_explainability, 0.95)
        self.assertEqual(default.parent_state_minimum_edge_exposed_bins, 1.0)
        self.assertEqual(default.parent_state_minimum_pair_exposed_bins, 1.0)
        self.assertEqual(default.parent_state_minimum_exposed_fraction, 0.10)
        self.assertEqual(default.parent_state_minimum_exposed_contigs, 3)
        self.assertEqual(default.parent_state_minimum_direction_probability, 0.01)
        self.assertEqual(default.primary_view, "tier_b")

    def test_retired_method_selectors_are_not_constructor_arguments(self):
        for field_name, value in (
            ("parent_state_structure_mode", "none"),
            ("parent_state_algorithm_mode", "b3"),
        ):
            with self.subTest(field_name=field_name), self.assertRaisesRegex(
                TypeError, "unexpected keyword argument"
            ):
                pedigree.PedigreeConfig(**{field_name: value})



    def test_b1_focal_call_is_invariant_to_other_child_evidence(self):
        sample_ids, baseline_evidence = _eligibility_test_evidence()
        _, shifted_evidence = _eligibility_test_evidence(
            excluded_child_shift=10000.0
        )
        baseline = pedigree.infer_from_parent_state_evidence(
            baseline_evidence, sample_ids,
            config=self._config(), n_workers=1,
            **_eligibility_test_ancestry(sample_ids, baseline_evidence),
        )
        shifted = pedigree.infer_from_parent_state_evidence(
            shifted_evidence, sample_ids,
            config=self._config(), n_workers=1,
            **_eligibility_test_ancestry(sample_ids, shifted_evidence),
        )
        pd.testing.assert_series_equal(
            baseline.complete_relationships.iloc[0],
            shifted.complete_relationships.iloc[0],
        )
        columns = [
            "StateLogEvidence0", "StateLogEvidence1", "StateLogEvidence2",
            "StateSupport0", "StateSupport1", "StateSupport2",
            "LOOStatePrior0", "LOOStatePrior1", "LOOStatePrior2",
        ]
        np.testing.assert_array_equal(
            baseline.smart_diagnostics.loc[0, columns].to_numpy(float),
            shifted.smart_diagnostics.loc[0, columns].to_numpy(float),
        )
        np.testing.assert_array_equal(
            baseline.smart_diagnostics.loc[0, columns[-3:]].to_numpy(float),
            np.full(3, 1.0 / 3.0),
        )





    def test_m2_side_sets_are_only_emitted_when_factorable(self):
        alternatives = np.asarray((
            (0, 1, 2), (0, 3, 4), (0, 1, 3), (0, 1, 4)
        ), dtype=np.int64)
        states = np.full(4, pedigree._TWO_OBSERVED, dtype=np.int8)
        scores = np.zeros(4)
        disjoint = pedigree._evidence_parent_support_sets(
            np.asarray((0, 1)), pedigree._TWO_OBSERVED, 0,
            alternatives, states, scores, 1.0,
        )
        self.assertEqual(disjoint[:2], ((), ()))
        self.assertEqual(disjoint[2], (0, 1))
        star = pedigree._evidence_parent_support_sets(
            np.asarray((0, 2, 3)), pedigree._TWO_OBSERVED, 0,
            alternatives, states, scores, 1.0,
        )
        self.assertEqual(star[0], (1,))
        self.assertEqual(star[1], (2, 3, 4))
        non_cartesian = pedigree._evidence_parent_support_sets(
            np.asarray((0, 1, 2)), pedigree._TWO_OBSERVED, 0,
            alternatives, states, scores, 1.0,
        )
        self.assertEqual(non_cartesian[:2], ((), ()))
        self.assertEqual(non_cartesian[2], (0, 1, 2))


class ExactCandidateSourceIntegrationTests(unittest.TestCase):
    @staticmethod
    def _exact_config():
        return pedigree.PedigreeConfig(
            bootstrap_replicates=1,
            parent_state_minimum_exposed_contigs=1,
            parent_state_candidate_source_mode="exact_raw_gl_v1",
        ).validated()

    def test_source_mode_validation_and_fixed_b1_provenance(self):
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "parent_state_candidate_source_mode"
        ):
            pedigree.PedigreeConfig(
                parent_state_candidate_source_mode="b4"
            ).validated()
        hard = pedigree.PedigreeConfig().validated()
        exact = pedigree.PedigreeConfig(
            parent_state_candidate_source_mode="exact_raw_gl_v1",
        ).validated()
        self.assertEqual(exact.parent_state_algorithm_mode, "b1")
        self.assertEqual(exact.parent_state_structure_mode, "combined_v1")
        self.assertNotEqual(hard, exact)
        self.assertIn("exact_raw_gl_v1", repr(exact))

    def test_direct_exact_scores_and_uniform_child_neutrality(self):
        inputs = _base_inputs(n_bins=3, n_snps=1)
        trios = _standard_test_trios()
        exact = _score(
            inputs, trios, candidate_source_mode="exact_raw_gl_v1"
        )
        self.assertEqual(
            exact.candidate_source_mode_applied, "exact_raw_gl_v1"
        )
        self.assertEqual(exact.complete_founder_marker_count, 3)
        self.assertEqual(exact.excluded_founder_marker_count, 0)
        self.assertGreater(exact.peak_streamed_tensor_bytes, 0)
        self.assertIsNotNone(exact.candidate_source_posterior)

        uniform = list(inputs)
        uniform[0] = np.full_like(uniform[0], 1.0 / 3.0)
        neutral = _score(
            tuple(uniform), trios,
            candidate_source_mode="exact_raw_gl_v1",
        )
        np.testing.assert_allclose(neutral.zero_observed, 0.0, atol=1e-14)
        off_diagonal = ~np.eye(4, dtype=np.bool_)
        np.testing.assert_allclose(
            neutral.one_observed[off_diagonal], 0.0, atol=1e-14
        )
        np.testing.assert_allclose(neutral.two_observed, 0.0, atol=1e-14)

    def test_complete_f8_panel_with_pooled_inactive_slots_stays_exact(self):
        n_founders, n_bins, n_slots, n_samples = 8, 24, 5, 10
        founders = np.zeros((n_founders, n_bins, n_slots), dtype=np.int8)
        for founder in range(n_founders):
            founders[founder, 0] = [
                (founder >> bit) & 1 for bit in range(n_slots)
            ]
            founders[founder, 1:] = founder % 2
        labels = np.empty((n_samples, n_bins, 2), dtype=np.int16)
        for sample in range(n_samples):
            labels[sample, :, 0] = sample % n_founders
            labels[sample, :, 1] = (sample + 3) % n_founders
        alleles = np.empty((n_samples, n_bins, 2, n_slots), dtype=np.int8)
        for sample in range(n_samples):
            for block in range(n_bins):
                for track in range(2):
                    alleles[sample, block, track] = founders[
                        labels[sample, block, track], block
                    ]
        rng = np.random.default_rng(809)
        gl = rng.random((n_samples, n_bins, n_slots, 3))
        gl /= gl.sum(axis=3, keepdims=True)
        marker_counts = np.full(n_bins, n_slots, dtype=np.int64)
        theta = np.asarray([0.0] + [0.01] * (n_bins - 1))
        trios = np.asarray(((0, 1, 2), (0, 3, 4)), dtype=np.int64)
        _, pooled_founders, _, _, _ = pedigree._pool_local_ibs_states(
            labels, founders
        )
        self.assertTrue(np.any(pooled_founders[:, 1:] < 0))

        exact = pedigree.score_parent_state_gl_hmms(
            gl, alleles, labels, np.zeros((n_samples, n_bins), dtype=np.bool_),
            founders, marker_counts, theta, trios,
            candidate_source_mode="exact_raw_gl_v1",
        )
        self.assertFalse(exact.candidate_source_fallback)
        self.assertEqual(exact.candidate_source_mode_applied, "exact_raw_gl_v1")
        self.assertEqual(exact.complete_founder_marker_count, n_bins * n_slots)
        self.assertEqual(exact.excluded_founder_marker_count, 0)

    def test_external_embedding_matches_direct_g2_and_f2g_references(self):
        founders = np.asarray((
            ((0,), (0,), (0,)),
            ((0,), (1,), (0,)),
            ((1,), (0,), (1,)),
            ((1,), (1,), (1,)),
        ), dtype=np.int8)
        mapping = pedigree._local_ibs_class_kernel(founders)[0]
        pooled_initial = np.asarray([[
            [0.35, 0.65], [0.5, 0.5], [0.6, 0.4]
        ]])
        pooled_transition = np.zeros((1, 3, 2, 2))
        pooled_transition[0, 1] = ((0.8, 0.2), (0.3, 0.7))
        pooled_transition[0, 2] = ((0.6, 0.4), (0.1, 0.9))
        lifted_initial, lifted_transition = (
            pedigree._lift_pooled_external_chains_to_physical_founders(
                pooled_initial, pooled_transition, founders
            )
        )

        rng = np.random.default_rng(991)
        g2_emission = rng.uniform(0.2, 1.0, size=(3, 2, 2))
        pooled = pooled_initial[0, 0, :, None] * pooled_initial[0, 0, None, :]
        lifted = lifted_initial[0, :, None] * lifted_initial[0, None, :]
        pooled_total = lifted_total = 0.0
        for block in range(3):
            if block:
                pooled = (
                    pooled_transition[0, block].T @ pooled
                    @ pooled_transition[0, block]
                )
                lifted = (
                    lifted_transition[0, block - 1].T @ lifted
                    @ lifted_transition[0, block - 1]
                )
            pooled *= g2_emission[block]
            lifted *= g2_emission[block][
                mapping[block, :, None], mapping[block, None, :]
            ]
            pooled_mass = pooled.sum()
            lifted_mass = lifted.sum()
            pooled_total += np.log(pooled_mass)
            lifted_total += np.log(lifted_mass)
            pooled /= pooled_mass
            lifted /= lifted_mass
        self.assertAlmostEqual(pooled_total, lifted_total, places=14)

        # The first axis is an arbitrary fixed candidate F^2 state. Applying
        # the same candidate transition proves the F^2 G external embedding.
        candidate_initial = np.asarray((0.2, 0.3, 0.5))
        candidate_transition = np.asarray(
            ((0.8, 0.1, 0.1), (0.2, 0.7, 0.1), (0.1, 0.2, 0.7))
        )
        f2g_emission = rng.uniform(0.2, 1.0, size=(3, 3, 2))
        pooled = candidate_initial[:, None] * pooled_initial[0, 0, None, :]
        lifted = candidate_initial[:, None] * lifted_initial[0, None, :]
        pooled_total = lifted_total = 0.0
        for block in range(3):
            if block:
                pooled = (
                    candidate_transition.T @ pooled
                    @ pooled_transition[0, block]
                )
                lifted = (
                    candidate_transition.T @ lifted
                    @ lifted_transition[0, block - 1]
                )
            pooled *= f2g_emission[block]
            lifted *= f2g_emission[block][:, mapping[block]]
            pooled_mass = pooled.sum()
            lifted_mass = lifted.sum()
            pooled_total += np.log(pooled_mass)
            lifted_total += np.log(lifted_mass)
            pooled /= pooled_mass
            lifted /= lifted_mass
        self.assertAlmostEqual(pooled_total, lifted_total, places=14)

    def test_global_duplicate_founder_label_is_exact_score_invariant(self):
        inputs = list(_base_inputs(n_bins=3, n_snps=1))
        trios = _standard_test_trios()
        baseline = _score(
            tuple(inputs), trios,
            candidate_source_mode="exact_raw_gl_v1",
        )
        duplicated = list(inputs)
        duplicated[4] = np.concatenate((inputs[4], inputs[4][0:1]), axis=0)
        duplicated[2] = inputs[2].copy()
        replace_mask = (
            (duplicated[2] == 0)
            & (np.indices(duplicated[2].shape)[0] % 2 == 1)
        )
        duplicated[2][replace_mask] = 2
        duplicate_score = _score(
            tuple(duplicated), trios,
            candidate_source_mode="exact_raw_gl_v1",
        )
        for field in ("zero_observed", "one_observed", "two_observed"):
            np.testing.assert_allclose(
                getattr(duplicate_score, field), getattr(baseline, field),
                rtol=0.0, atol=1e-12,
            )
        self.assertEqual(
            duplicate_score.candidate_source_posterior.
            lumped_initial_log_probability.shape[1],
            baseline.candidate_source_posterior.
            lumped_initial_log_probability.shape[1],
        )

    def test_unused_founder_padding_is_hard_and_exact_invariant(self):
        inputs = list(_base_inputs(n_bins=3, n_snps=2))
        inputs[5] = np.ones(3, dtype=np.int64)
        trios = _standard_test_trios()
        baseline_hard = _score(tuple(inputs), trios)
        baseline_exact = _score(
            tuple(inputs), trios,
            candidate_source_mode="exact_raw_gl_v1",
        )

        padded = list(inputs)
        duplicate = inputs[4][0:1].copy()
        duplicate[:, :, 1] = 1 - duplicate[:, :, 1]
        padded[4] = np.concatenate((inputs[4], duplicate), axis=0)
        padded[2] = inputs[2].copy()
        padded[1] = inputs[1].copy()
        relabel = (
            (padded[2] == 0)
            & (np.indices(padded[2].shape)[0] % 2 == 1)
        )
        padded[2][relabel] = 2
        for sample, block, track in zip(*np.nonzero(relabel)):
            padded[1][sample, block, track] = padded[4][2, block]

        padded_hard = _score(tuple(padded), trios)
        padded_exact = _score(
            tuple(padded), trios,
            candidate_source_mode="exact_raw_gl_v1",
        )
        for observed, expected in (
            (padded_hard, baseline_hard),
            (padded_exact, baseline_exact),
        ):
            for field in (
                "zero_observed", "one_observed", "two_observed",
                "one_parent_identity_information", "two_parent_edge_information",
            ):
                np.testing.assert_allclose(
                    getattr(observed, field), getattr(expected, field),
                    rtol=0.0, atol=1e-12,
                )

    def test_founder_missing_fallback_is_whole_contig_bit_identical(self):
        inputs = list(_base_inputs(n_bins=3, n_snps=1))
        inputs[4] = inputs[4].copy()
        inputs[4][0, 1, 0] = -1
        trios = _standard_test_trios()
        hard = _score(tuple(inputs), trios)
        fallback = _score(
            tuple(inputs), trios,
            candidate_source_mode="exact_raw_gl_v1",
        )
        self.assertTrue(fallback.candidate_source_fallback)
        self.assertEqual(
            fallback.candidate_source_fallback_reason,
            "founder_missing_selected_real_site_whole_contig_hard_fallback",
        )
        self.assertEqual(fallback.candidate_source_mode_applied, "hard_painted")
        for field in (
            "zero_observed", "one_observed", "two_observed",
            "one_parent_identity_information", "two_parent_edge_information",
        ):
            np.testing.assert_array_equal(
                getattr(fallback, field), getattr(hard, field)
            )

    def test_standard_exact_dispatch_reuses_source_and_skips_hard_screen(self):
        sample_ids, hard_item, raw_item, _ = _standard_raw_fixture()
        hard_cache = _build_standard_test_cache(hard_item, sample_ids)
        raw_cache = _build_standard_test_cache(raw_item, sample_ids)
        trios = _standard_test_trios()
        config = self._exact_config()
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "require raw genotype likelihoods"
        ):
            pedigree._score_standard_contig_parent_states(
                hard_cache, trios, config
            )
        observed = pedigree._score_standard_contig_parent_states(
            raw_cache, trios, config
        )
        self.assertEqual(
            observed.candidate_source_mode_applied, "exact_raw_gl_v1"
        )

        kwargs = dict(
            top_k=3, snps_per_bin=100, recomb_rate=5e-8,
            mismatch_penalty=-3.0, max_snps_per_bin=2, n_workers=1,
            anchor_k=1, use_anchor_union=False,
        )
        real_infer = pedigree.infer_candidate_source_posterior
        with mock.patch.object(
            pedigree, "infer_candidate_source_posterior", wraps=real_infer
        ) as source_infer, mock.patch.object(
            pedigree, "_score_pair_hmm_contig",
            side_effect=AssertionError("hard screen must not run"),
        ):
            result = pedigree.infer_pedigree(
                [raw_item], sample_ids, config=config,
                scoring_kwargs=kwargs,
            )
        self.assertEqual(source_infer.call_count, 1)
        self.assertEqual(
            result.smart_parent_state_candidate_source_mode,
            "exact_raw_gl_v1",
        )
        self.assertEqual(
            result.smart_evidence_summary.loc[
                0, "CandidateSourceModeApplied"
            ],
            "exact_raw_gl_v1",
        )
        self.assertGreater(
            result.smart_evidence_summary.loc[0, "PeakStreamedTensorBytes"], 0
        )
        self.assertIn(
            "hard-painted pair screen is not consulted",
            result.smart_candidate_screening_scope,
        )


class MatchedNullProductionIntegrationTests(unittest.TestCase):
    SOURCE_MODE = "matched_null_raw_gl_v2"
    SOURCE_RHO = 0.06

    @classmethod
    def _score_v2(cls, inputs, trios, **kwargs):
        return _score(
            inputs,
            trios,
            candidate_source_mode=cls.SOURCE_MODE,
            candidate_source_path_switch_probability=cls.SOURCE_RHO,
            **kwargs,
        )

    def test_v2_requires_explicit_bounded_source_rho(self):
        for value in (None, -0.01, 0.500001, np.nan, "invalid", False):
            with self.subTest(value=value), self.assertRaisesRegex(
                pedigree.PedigreeEvidenceError,
                "parent_state_candidate_source_path_switch_probability",
            ):
                pedigree.PedigreeConfig(
                    parent_state_candidate_source_mode=self.SOURCE_MODE,
                    parent_state_candidate_source_path_switch_probability=value,
                ).validated()


        inputs = _base_inputs(n_bins=3, n_snps=1)
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError,
            "candidate_source_path_switch_probability",
        ):
            _score(
                inputs,
                _standard_test_trios(),
                candidate_source_mode=self.SOURCE_MODE,
            )

    def test_hard_and_exact_v1_ignore_v2_rho_seam_bit_for_bit(self):
        inputs = _base_inputs(n_bins=3, n_snps=1)
        trios = _standard_test_trios()
        for mode in ("hard_painted", "exact_raw_gl_v1"):
            base = _score(inputs, trios, candidate_source_mode=mode)
            with_unused_rho = _score(
                inputs,
                trios,
                candidate_source_mode=mode,
                candidate_source_path_switch_probability=0.49,
            )
            for field in (
                "zero_observed",
                "one_observed",
                "two_observed",
                "one_parent_identity_information",
                "two_parent_edge_information",
            ):
                np.testing.assert_array_equal(
                    getattr(with_unused_rho, field), getattr(base, field)
                )

    def test_v2_matches_standalone_without_candidate_normalizer(self):
        inputs = _base_inputs(n_bins=3, n_snps=1)
        gl, _, _, _, founders, marker_counts, theta = inputs
        trios = _standard_test_trios()
        production = self._score_v2(inputs, trios)

        exponent = pedigree._gl_information_exponent_kernel(
            gl, marker_counts, 100, 1.0
        )
        posterior = pedigree.infer_candidate_source_posterior(
            gl,
            founders,
            marker_counts,
            self.SOURCE_RHO,
            eta=np.where(exponent > 0.0, exponent, 1.0),
            painted_track_labels=None,
            return_lumped_posterior=True,
            lumped_root_prior_mode="ordered_independent_uniform",
        )
        standalone = pedigree.score_candidate_source_batch_matched_null_exact(
            posterior,
            gl,
            founders,
            marker_counts,
            exponent,
            theta,
            trios,
            mismatch_probability=0.01,
        )
        np.testing.assert_array_equal(
            production.zero_observed, standalone.zero_observed
        )
        off_diagonal = ~np.eye(len(gl), dtype=np.bool_)
        np.testing.assert_array_equal(
            production.one_observed[off_diagonal],
            standalone.one_observed[off_diagonal],
        )
        np.testing.assert_array_equal(
            production.two_observed, standalone.two_observed
        )
        np.testing.assert_array_equal(
            production.one_parent_identity_information,
            standalone.one_parent_identity_information,
        )
        np.testing.assert_array_equal(
            production.two_parent_edge_information,
            standalone.two_parent_edge_information,
        )
        self.assertEqual(
            production.candidate_source_posterior.lumped_root_prior_mode,
            "ordered_independent_uniform",
        )

        # B5a uses theta as one biological selector for candidates and nulls;
        # the painted hard-homo mask cannot trigger candidate-only 1/2 resets.
        altered = list(inputs)
        altered[3] = ~inputs[3]
        no_homo_reset = self._score_v2(tuple(altered), trios)
        for field in ("zero_observed", "one_observed", "two_observed"):
            np.testing.assert_array_equal(
                getattr(no_homo_reset, field), getattr(production, field)
            )

    def test_v2_exact_nesting_and_epsilon_continuity_through_production(self):
        inputs = list(_base_inputs(n_bins=3, n_snps=1))
        trios = _standard_test_trios()
        nested = self._score_v2(tuple(inputs), trios)
        available = nested.candidate_source_available
        self.assertTrue(available[0])
        self.assertFalse(np.any(available[1:]))
        for child in range(len(inputs[0])):
            for parent in np.flatnonzero(~available):
                if child != parent:
                    self.assertEqual(
                        nested.one_observed[child, parent],
                        nested.zero_observed[child],
                    )
        for row, (child, first, second) in enumerate(trios):
            active = [parent for parent in (first, second) if available[parent]]
            if not active:
                expected = nested.zero_observed[child]
            elif len(active) == 1:
                expected = nested.one_observed[child, active[0]]
            else:
                continue
            self.assertEqual(nested.two_observed[row], expected)

        epsilon_inputs = list(inputs)
        epsilon_inputs[0] = inputs[0].copy()
        epsilon = 1e-9
        epsilon_inputs[0][1, 0, 0] = (
            _UNIFORM_GL + np.asarray((epsilon, -epsilon, 0.0))
        )
        perturbed = self._score_v2(tuple(epsilon_inputs), trios)
        self.assertTrue(perturbed.candidate_source_available[1])
        np.testing.assert_allclose(
            perturbed.one_observed[2, 1],
            nested.zero_observed[2],
            rtol=0.0,
            atol=1e-7,
        )
        row = np.flatnonzero(np.all(trios == (2, 0, 1), axis=1))[0]
        np.testing.assert_allclose(
            perturbed.two_observed[row],
            nested.one_observed[2, 0],
            rtol=0.0,
            atol=1e-7,
        )

    def test_v2_founder_missing_policy_is_explicit_whole_contig_fallback(self):
        inputs = list(_base_inputs(n_bins=3, n_snps=1))
        inputs[4] = inputs[4].copy()
        inputs[4][0, 1, 0] = -1
        trios = _standard_test_trios()
        hard = _score(tuple(inputs), trios)
        fallback = self._score_v2(tuple(inputs), trios)
        self.assertTrue(fallback.candidate_source_fallback)
        self.assertEqual(
            fallback.candidate_source_fallback_reason,
            "founder_missing_selected_real_site_whole_contig_hard_fallback",
        )
        self.assertEqual(
            fallback.candidate_source_mode_requested, self.SOURCE_MODE
        )
        self.assertEqual(fallback.candidate_source_mode_applied, "hard_painted")
        for field in (
            "zero_observed",
            "one_observed",
            "two_observed",
            "one_parent_identity_information",
            "two_parent_edge_information",
        ):
            np.testing.assert_array_equal(
                getattr(fallback, field), getattr(hard, field)
            )

    def test_v2_standard_dispatch_uses_fixed_b1_combined_method(self):
        sample_ids, _, raw_item, _ = _standard_raw_fixture()
        cache = _build_standard_test_cache(raw_item, sample_ids)
        trios = _standard_test_trios()
        common = dict(
            bootstrap_replicates=1,
            parent_state_minimum_exposed_contigs=1,
            parent_state_candidate_source_mode=self.SOURCE_MODE,
            parent_state_candidate_source_path_switch_probability=self.SOURCE_RHO,
        )
        scores = pedigree._score_standard_contig_parent_states(
            cache,
            trios,
            pedigree.PedigreeConfig(**common).validated(),
        )
        self.assertEqual(scores.candidate_source_mode_applied, self.SOURCE_MODE)

        result = pedigree.infer_pedigree(
            [raw_item],
            sample_ids,
            config=pedigree.PedigreeConfig(
                minimum_informative_contigs=1,
                **common,
            ),
            scoring_kwargs=dict(
                top_k=3,
                snps_per_bin=100,
                recomb_rate=5e-8,
                mismatch_penalty=-3.0,
                max_snps_per_bin=2,
                n_workers=1,
                anchor_k=1,
                use_anchor_union=False,
            ),
        )
        self.assertEqual(
            result.smart_parent_state_candidate_source_mode, self.SOURCE_MODE
        )
        self.assertEqual(
            result.smart_parent_state_candidate_source_path_switch_probability,
            self.SOURCE_RHO,
        )
        self.assertEqual(
            result.smart_evidence_summary.loc[
                0, "CandidateSourcePathSwitchProbability"
            ],
            self.SOURCE_RHO,
        )
        self.assertEqual(
            result.smart_evidence_summary.loc[
                0, "OffspringTransmissionSelector"
            ],
            "shared_biological_theta_no_hard_homo_reset",
        )
        self.assertIn(
            "two independent synthetic null-parent",
            result.smart_evidence_source,
        )
        self.assertIn("caller-specified", result.smart_limitations)


class CanonicalDispatchTests(unittest.TestCase):
    def test_unknown_schema_never_falls_back(self):
        for runner in (
            pedigree.infer_pedigree,
            pedigree.infer_pedigree_for_pipeline,
        ):
            with self.subTest(runner=runner.__name__), self.assertRaisesRegex(
                pedigree.PedigreeEvidenceError,
                "requires explicit parent-state evidence",
            ):
                runner([{"unsupported": True}], ("a", "b", "c"))

    def test_direct_parent_state_records_require_structure_and_ancestry(self):
        sample_ids, evidence = _eligibility_test_evidence()
        ancestry = _eligibility_test_ancestry(sample_ids, evidence)
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "parenthood structure"
        ):
            pedigree.infer_pedigree(
                _without_structure_evidence(evidence),
                sample_ids,
                config=pedigree.PedigreeConfig(bootstrap_replicates=2),
                **ancestry,
            )
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "ancestry junction"
        ):
            pedigree.infer_pedigree(
                evidence, sample_ids,
                config=pedigree.PedigreeConfig(bootstrap_replicates=2),
            )

        result = pedigree.infer_pedigree(
            evidence, sample_ids,
            config=pedigree.PedigreeConfig(bootstrap_replicates=2),
            **ancestry,
        )
        self.assertEqual(result.samples, list(sample_ids))
        self.assertEqual(
            result.smart_parent_state_structure_mode, "combined_v1"
        )

    def test_standard_default_supplies_and_reports_combined_structure(self):
        sample_ids, hard_item, _, _ = _standard_raw_fixture()
        contigs = []
        for index in range(3):
            item = dict(hard_item)
            item["contig"] = f"chr{index + 1}"
            contigs.append(item)
        result = pedigree.infer_pedigree(
            contigs,
            sample_ids,
            config=pedigree.PedigreeConfig(
                bootstrap_replicates=2,
                minimum_informative_contigs=1,
            ),
            scoring_kwargs=dict(
                top_k=3,
                snps_per_bin=100,
                recomb_rate=5e-8,
                mismatch_penalty=-3.0,
                max_snps_per_bin=2,
                n_workers=1,
                anchor_k=1,
                use_anchor_union=False,
            ),
        )
        self.assertEqual(
            result.smart_parent_state_structure_mode, "combined_v1"
        )
        self.assertEqual(len(result.smart_evidence_summary), 3)
        self.assertEqual(
            set(result.smart_diagnostics["ParentStateStructureMode"]),
            {"combined_v1"},
        )


    def test_pair_only_evidence_is_not_an_engine(self):
        item = pedigree.SmartContigEvidence(
            contig="chr1",
            trios=np.asarray(((0, 1, 2),), dtype=np.int64),
            linked_log_likelihoods=np.zeros(1, dtype=np.float64),
            genotype_log_likelihoods=np.zeros(1, dtype=np.float64),
            informative_markers=1,
        )
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "pair-only"
        ):
            pedigree.infer_pedigree([item], ("a", "b", "c"))



class CombinedStructureReleaseCandidateTests(unittest.TestCase):
    def _structure_fixture(self):
        alternatives = np.asarray((
            (0, -1, -1), (1, -1, -1), (2, -1, -1),
            (2, 0, -1), (2, 1, -1), (2, 0, 1),
        ), dtype=np.int64)
        states = np.asarray((0, 0, 0, 1, 1, 2), dtype=np.int8)
        pair_indices = np.asarray((-1, -1, -1, -1, -1, 0), dtype=np.int64)
        edge_matched = np.ones((3, 3, 3), dtype=np.float64) * 10.0
        edge_exposed = edge_matched.copy()
        pair_explained = np.ones((3, 1), dtype=np.float64) * 10.0
        pair_exposed = pair_explained.copy()
        total_bins = np.ones(3, dtype=np.float64) * 10.0
        depth = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 1.0)))
        config = pedigree.PedigreeConfig(
            parent_state_minimum_exposed_contigs=3,
        ).validated()
        return (
            alternatives, states, pair_indices, edge_matched, edge_exposed,
            pair_explained, pair_exposed, total_bins, depth, config,
        )

    def test_missing_bins_are_excluded_and_edge_counts_are_symmetric(self):
        labels = np.asarray((
            ((0, 1), (0, -1)),
            ((1, 2), (0, 1)),
            ((0, 2), (1, 0)),
        ), dtype=np.int16)
        trios = np.asarray(((2, 0, 1),), dtype=np.int64)
        matched, exposed, explained, pair_exposed = (
            pedigree._parenthood_structure_count_kernel(labels, trios)
        )
        np.testing.assert_array_equal(matched, matched.T)
        np.testing.assert_array_equal(exposed, exposed.T)
        self.assertEqual(exposed[0, 1], 1.0)
        self.assertEqual(explained[0], 1.0)
        self.assertEqual(pair_exposed[0], 1.0)

    def test_direction_is_per_edge_and_m2_requires_both_edges(self):
        args = self._structure_fixture()
        exposure, eligible, evaluable, _, pair_x, direction = (
            pedigree._parent_state_structure_mask(
                np.ones(3), *args[:-1], settings=args[-1]
            )
        )
        self.assertTrue(evaluable[2])
        self.assertTrue(eligible[3])
        self.assertFalse(eligible[4])
        self.assertFalse(eligible[5])
        self.assertEqual(pair_x[0], 1.0)
        self.assertEqual(direction[2, 0], 1.0)
        self.assertEqual(direction[2, 1], 0.0)

    def test_all_missing_child_is_unscorable_not_m0(self):
        args = list(self._structure_fixture())
        args[4][:, 2, :] = 0.0
        args[4][:, :, 2] = 0.0
        exposure, eligible, evaluable, *_ = pedigree._parent_state_structure_mask(
            np.ones(3), *args[:-1], settings=args[-1]
        )
        self.assertFalse(evaluable[2])
        self.assertTrue(eligible[2])
        self.assertFalse(exposure[3])
        self.assertFalse(np.any(eligible[3:]))

    def test_masked_identity_evidence_keeps_pre_screen_denominator(self):
        alternatives = np.asarray(((0, -1, -1), (0, 1, -1), (0, 2, -1)))
        states = np.asarray((0, 1, 1), dtype=np.int8)
        by_child = (np.arange(3, dtype=np.int64),)
        full_counts = np.asarray(((1, 2, 0),), dtype=np.int64)
        aggregate = np.asarray((0.0, 4.0, -np.inf))
        observed = pedigree._integrated_parent_state_log_evidence(
            aggregate, states, by_child, full_counts
        )
        self.assertAlmostEqual(observed[0, 1], 4.0 - math.log(2.0))

    def test_graph_fallback_demotes_without_promotion(self):
        alternatives = np.asarray((
            (0, -1, -1), (0, 2, -1), (0, 1, 2),
            (1, -1, -1), (1, 0, -1), (2, -1, -1),
        ), dtype=np.int64)
        states = np.asarray((0, 1, 2, 0, 1, 0), dtype=np.int8)
        by_child = (
            np.asarray((0, 1, 2)), np.asarray((3, 4)), np.asarray((5,)),
        )
        selected = pedigree._acyclic_parent_state_selection(
            alternatives,
            states,
            np.asarray((0.0, 5.0, 10.0, 0.0, 9.0, 0.0)),
            by_child,
            {0: 2, 1: 4, 2: 5},
            np.asarray((1.0, 10.0, 1.0)),
            np.asarray((1.0, 10.0, 1.0)),
            3,
            0,
            downward_fallback=True,
        )
        self.assertEqual(selected.rows[0], 1)
        self.assertEqual(states[selected.rows[0]], 1)
        self.assertEqual(states[selected.rows[2]], 0)

    def test_combined_explicit_evidence_requires_structure_counts(self):
        sample_ids, evidence = _eligibility_test_evidence()
        evidence = _without_structure_evidence(evidence)
        shape = (len(evidence), len(sample_ids))
        with self.assertRaisesRegex(
            pedigree.PedigreeEvidenceError, "parenthood structure"
        ):
            pedigree.infer_from_parent_state_evidence(
                evidence,
                sample_ids,
                config=pedigree.PedigreeConfig(
                    bootstrap_replicates=2,
                ),
                ancestry_junction_counts=np.ones(shape),
                ancestry_callable_haplotype_bins=np.ones(shape) * 10.0,
                n_workers=1,
            )


    def test_combined_bootstrap_serial_parallel_parity(self):
        sample_ids, legacy = _eligibility_test_evidence()
        n_samples = len(sample_ids)
        n_trios = len(legacy[0].trios)
        structured = [pedigree.ParentStateEvidence(
            contig=item.contig,
            trios=item.trios,
            zero_parent_log_likelihoods=item.zero_parent_log_likelihoods,
            one_parent_log_likelihoods=item.one_parent_log_likelihoods,
            two_parent_log_likelihoods=item.two_parent_log_likelihoods,
            informative_markers=item.informative_markers,
            edge_matched_bins=np.ones((n_samples, n_samples)) * 100.0,
            edge_exposed_bins=np.ones((n_samples, n_samples)) * 100.0,
            pair_explained_bins=np.ones(n_trios) * 100.0,
            pair_exposed_bins=np.ones(n_trios) * 100.0,
            structure_total_bins=100.0,
        ) for item in legacy]
        junctions = np.tile(np.asarray((0.0, 10.0, 20.0, 30.0)), (3, 1))
        callable_bins = np.ones_like(junctions) * 100.0
        config = pedigree.PedigreeConfig(
            bootstrap_replicates=32,
            minimum_informative_contigs=1,
        )
        serial = pedigree.infer_from_parent_state_evidence(
            structured, sample_ids, config=config,
            ancestry_junction_counts=junctions,
            ancestry_callable_haplotype_bins=callable_bins,
            n_workers=1,
        )
        with mock.patch.object(
            pedigree, "_SMART_BOOTSTRAP_MIN_WORK_ITEMS", 1
        ):
            parallel = pedigree.infer_from_parent_state_evidence(
                structured, sample_ids, config=config,
                ancestry_junction_counts=junctions,
                ancestry_callable_haplotype_bins=callable_bins,
                n_workers=2,
            )
        self.assertEqual(serial.smart_bootstrap_worker_count, 1)
        self.assertEqual(parallel.smart_bootstrap_worker_count, 2)
        columns = [
            "LocalWinnerParentState", "SelectedParentState",
            "LocalStateBootstrapFraction", "PairBootstrapFraction",
            "LocalStateLOCOFraction", "StructureChildEvaluable",
        ]
        pd.testing.assert_frame_equal(
            serial.smart_diagnostics[columns],
            parallel.smart_diagnostics[columns],
        )


    def test_neutral_completion_exact_formula_and_identity_mask(self):
        alternatives = np.asarray((
            (0, -1, -1), (0, 1, -1), (0, 2, -1), (0, 3, -1),
        ), dtype=np.int64)
        states = np.asarray((0, 1, 1, 1), dtype=np.int8)
        raw = np.asarray((0.0, 2.0, 1e6, -1e6))
        exposure = np.asarray((True, True, False, False))
        eligible = np.asarray((True, True, False, False))
        state_rows, identity_rows = (
            pedigree._structure_state_and_identity_aggregates(
                raw, alternatives, states, exposure, eligible,
                direction_available=True,
            )
        )
        expected = math.log((math.exp(2.0) + 2.0) / 3.0)
        observed = pedigree._integrated_parent_state_log_evidence(
            state_rows,
            states,
            (np.arange(4, dtype=np.int64),),
            np.asarray(((1, 3, 0),)),
        )
        self.assertAlmostEqual(observed[0, 1], expected)
        self.assertTrue(np.all(np.isneginf(identity_rows[2:])))
        permuted = raw.copy()
        permuted[2], permuted[3] = permuted[3], permuted[2]
        permuted_state, permuted_identity = (
            pedigree._structure_state_and_identity_aggregates(
                permuted, alternatives, states, exposure, eligible,
                direction_available=True,
            )
        )
        np.testing.assert_array_equal(state_rows, permuted_state)
        np.testing.assert_array_equal(identity_rows, permuted_identity)

    def test_m2_neutral_completion_uses_m0_and_masks_identity(self):
        alternatives = np.asarray((
            (0, -1, -1), (0, 1, 2), (0, 2, 3),
        ), dtype=np.int64)
        states = np.asarray((0, 2, 2), dtype=np.int8)
        raw = np.asarray((0.0, 3.0, 1e6))
        exposure = np.asarray((True, True, False))
        eligible = np.asarray((True, True, False))
        state_rows, identity_rows = (
            pedigree._structure_state_and_identity_aggregates(
                raw, alternatives, states, exposure, eligible,
                direction_available=True,
            )
        )
        expected = math.log((math.exp(3.0) + 1.0) / 2.0)
        observed = pedigree._integrated_parent_state_log_evidence(
            state_rows,
            states,
            (np.arange(3, dtype=np.int64),),
            np.asarray(((1, 0, 2),)),
        )
        self.assertAlmostEqual(observed[0, 2], expected)
        self.assertTrue(np.isneginf(identity_rows[2]))

    def test_all_underexposed_states_tie_and_remain_unresolved(self):
        alternatives = np.asarray((
            (0, -1, -1), (0, 1, -1), (0, 2, -1),
            (0, 1, 2), (0, 2, 3),
        ), dtype=np.int64)
        states = np.asarray((0, 1, 1, 2, 2), dtype=np.int8)
        raw = np.asarray((0.0, 1e6, -1e6, 5e5, -5e5))
        exposure = np.asarray((True, False, False, False, False))
        eligible = np.asarray((True, False, False, False, False))
        state_rows, identity_rows = (
            pedigree._structure_state_and_identity_aggregates(
                raw, alternatives, states, exposure, eligible,
                direction_available=True,
            )
        )
        by_child = (np.arange(5, dtype=np.int64),)
        full_counts = np.asarray(((1, 2, 2),))
        evidence = pedigree._integrated_parent_state_log_evidence(
            state_rows, states, by_child, full_counts
        )
        np.testing.assert_allclose(evidence[0], np.zeros(3))
        selected = pedigree._evaluate_parent_state_aggregate(
            state_rows,
            alternatives,
            states,
            by_child,
            full_counts,
            (1 / 3, 1 / 3, 1 / 3),
            3.0,
            10,
            1e-10,
            1,
            0,
            use_cohort_prior=False,
            algorithm_mode="b1",
            identity_log_likelihoods_override=identity_rows,
        )
        self.assertNotIn(0, selected.local_states)
        self.assertNotIn(0, selected.local_rows)


    def test_one_exposed_neutral_margin_is_vetoed_by_prior_sensitivity(self):
        sample_ids = ("child", "p1", "p2", "p3")
        trios = np.empty((0, 3), dtype=np.int64)
        exposed_score = math.log(3.0 * math.exp(-0.1) - 2.0) / 3.0
        evidence = []
        for contig in range(3):
            zero = np.zeros(4)
            one = np.full((4, 4), -5.0)
            np.fill_diagonal(one, -np.inf)
            one[0, 1] = exposed_score
            one[0, 2:] = 1e6
            edge_exposed = np.ones((4, 4)) * 100.0
            edge_matched = edge_exposed.copy()
            edge_exposed[0, 2:] = 0.0
            edge_exposed[2:, 0] = 0.0
            edge_matched[0, 2:] = 0.0
            edge_matched[2:, 0] = 0.0
            evidence.append(pedigree.ParentStateEvidence(
                contig=f"ctg{contig}",
                trios=trios,
                zero_parent_log_likelihoods=zero,
                one_parent_log_likelihoods=one,
                two_parent_log_likelihoods=np.empty(0),
                informative_markers=100,
                edge_matched_bins=edge_matched,
                edge_exposed_bins=edge_exposed,
                pair_explained_bins=np.empty(0),
                pair_exposed_bins=np.empty(0),
                structure_total_bins=100.0,
            ))
        junctions = np.tile(np.asarray((10.0, 0.0, 20.0, 30.0)), (3, 1))
        callable_bins = np.ones_like(junctions) * 100.0
        depth_model = pedigree._AncestryDepthModel(
            adjusted_junction_burden=np.asarray((10.0, 0.0, 20.0, 30.0)),
            callability_fraction=np.ones(4),
            posterior=np.asarray(((0.0, 1.0), (1.0, 0.0),
                                  (0.0, 1.0), (0.0, 1.0))),
            component_means=np.asarray((0.0, 20.0)),
            component_standard_deviations=np.ones(2),
            component_weights=np.asarray((0.25, 0.75)),
            selected_bic=0.0,
            tested_bics=(0.0, 1.0),
        )
        with mock.patch.object(
            pedigree, "_fit_ancestry_depth_model", return_value=depth_model
        ):
            result = pedigree.infer_from_parent_state_evidence(
                evidence,
                sample_ids,
                config=pedigree.PedigreeConfig(
                    parent_state_contamination_probability=0.0,
                    parent_state_minimum_exposed_contigs=1,
                    bootstrap_replicates=4,
                    minimum_informative_contigs=1,
                ),
                ancestry_junction_counts=junctions,
                ancestry_callable_haplotype_bins=callable_bins,
                n_workers=1,
            )
        row = result.smart_diagnostics.set_index("Sample").loc["child"]
        self.assertEqual(row["LocalWinnerParentState"], "zero_observed_parents")
        self.assertTrue(row["M0PriorSensitivityTierVeto"])
        self.assertFalse(row["TierAStateCall"])
        self.assertFalse(row["TierBStateCall"])
        tier_row = result.tier_b_relationships.set_index("Sample").loc["child"]
        self.assertIn("m0_prior_sensitivity", tier_row["InferenceStatus"])

    def test_all_underexposed_end_to_end_is_not_tier_released(self):
        sample_ids, legacy = _eligibility_test_evidence()
        n_samples = len(sample_ids)
        structured = []
        for item in legacy:
            edge_exposed = np.ones((n_samples, n_samples)) * 100.0
            edge_matched = edge_exposed.copy()
            edge_exposed[0, :] = edge_exposed[:, 0] = 0.0
            edge_matched[0, :] = edge_matched[:, 0] = 0.0
            pair_exposed = np.ones(len(item.trios)) * 100.0
            pair_explained = pair_exposed.copy()
            child_zero = item.trios[:, 0] == 0
            pair_exposed[child_zero] = 0.0
            pair_explained[child_zero] = 0.0
            structured.append(pedigree.ParentStateEvidence(
                contig=item.contig,
                trios=item.trios,
                zero_parent_log_likelihoods=item.zero_parent_log_likelihoods,
                one_parent_log_likelihoods=item.one_parent_log_likelihoods,
                two_parent_log_likelihoods=item.two_parent_log_likelihoods,
                informative_markers=item.informative_markers,
                edge_matched_bins=edge_matched,
                edge_exposed_bins=edge_exposed,
                pair_explained_bins=pair_explained,
                pair_exposed_bins=pair_exposed,
                structure_total_bins=100.0,
            ))
        junctions = np.tile(np.asarray((0.0, 10.0, 20.0, 30.0)), (3, 1))
        result = pedigree.infer_from_parent_state_evidence(
            structured,
            sample_ids,
            config=pedigree.PedigreeConfig(
                parent_state_minimum_exposed_contigs=1,
                bootstrap_replicates=4,
                minimum_informative_contigs=1,
            ),
            ancestry_junction_counts=junctions,
            ancestry_callable_haplotype_bins=np.ones_like(junctions) * 100.0,
            n_workers=1,
        )
        row = result.smart_diagnostics.set_index("Sample").loc["s0"]
        self.assertFalse(row["StructureChildEvaluable"])
        self.assertFalse(row["TierAStateCall"])
        self.assertFalse(row["TierBStateCall"])
        self.assertIsNone(row["LocalWinnerParentState"])


if __name__ == "__main__":
    unittest.main()
