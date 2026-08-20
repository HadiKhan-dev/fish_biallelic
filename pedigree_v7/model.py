"""Phase-invariant Tropheops V7 scoring and stability aggregation.

This module preserves the numerical and decision logic of the selected V7
implementation. Dataset paths, metadata parsing, and command-line policy live in
other package modules.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import checkpoint_io
import pipeline_runtime

from . import linked
from .design import CONTIGS
from .io import (
    atomic_npz,
    load_selected_likelihoods,
    local_founder_equivalence,
    painted_parent_tracks,
    smoothed_founder_probabilities,
)
from .multigeneration import (
    condition_tracks_on_genotype_likelihoods,
    reconstruct_inherited_tracks,
    reconstruct_parental_origin_tracks_unphased,
    score_binned_missing_parent_models,
)
from .ranking import descending_rank_votes as _descending_rank_votes


SCHEMA_VERSION = 7
MODEL_REVISION = (
    "phase_invariant_G0_to_F1_with_missing_parent_states__t09_h1_z1_atomic"
)
PRIMARY_VARIANT = 1
PARENT_STATE_NAMES = (
    "zero_parent", "father_only", "mother_only", "two_parent"
)
PARENT_STATE_PRIORS = {
    "equal_parent_count": (1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0),
    "missing_parent_conservative": (0.6, 0.15, 0.15, 0.1),
    "observed_parent_rich": (0.1, 0.1, 0.1, 0.7),
}
VARIANT_LABELS = (
    "error_0.005_block_100",
    "error_0.01_block_100_primary",
    "error_0.05_block_100",
    "error_0.01_block_50",
    "error_0.01_block_250",
    "error_0.01_block_500",
    "unsmoothed_error_0.01_block_100",
)
DEFAULT_BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20260723


def _variant_specifications():
    values = [
        (0.005, 100, False),
        (0.01, 100, False),
        (0.05, 100, False),
        (0.01, 50, False),
        (0.01, 250, False),
        (0.01, 500, False),
        (0.01, 100, True),
    ]
    if len(values) != len(VARIANT_LABELS):
        raise RuntimeError("Variant labels and specifications differ")
    return values


def _load_t09_scoring_inputs(
    checkpoint_dir, contig, expected_sample_ids, *, nthreads=4
):
    """Load sample-bound final-panel H1/Z1 inputs required for V7 scoring."""
    checkpoint_path = checkpoint_io.contig_path(
        checkpoint_dir, "T09_viterbi_painting", contig
    )
    payload = checkpoint_io.read(checkpoint_path, nthreads=nthreads)
    pipeline_runtime.validate_painting_bundle(
        payload,
        expected_sample_ids=expected_sample_ids,
        context=f"T09 checkpoint {checkpoint_path}",
    )
    painting = payload["tolerance_result"]
    founder_block = payload[pipeline_runtime.FOUNDER_BLOCK_KEY]
    del payload
    return painting, founder_block


def _score_contig(
    contig,
    output_dir,
    settings,
    candidates,
    selected_g0_pairs_local,
    resume,
):
    path = Path(output_dir) / f"v7_{contig}.npz"
    if resume and path.exists():
        with np.load(path) as cached:
            return {
                "contig": contig,
                "source": "cache",
                "markers": int(cached["markers"]),
                "seconds": 0.0,
            }
    started = time.monotonic()
    painting, founder_block = _load_t09_scoring_inputs(
        settings["checkpoint_dir"],
        contig,
        settings["ordered_bcf_sample_ids"],
    )
    likelihoods, _, positions, block_indices, hard = (
        load_selected_likelihoods(
            settings["bcf"],
            contig,
            founder_block,
            0,
            int(settings["bcf_threads"]),
        )
    )
    equivalence = local_founder_equivalence(
        np.asarray(founder_block.positions, dtype=np.int64),
        hard,
        positions,
        int(settings["equivalence_window_bp"]),
        float(settings["max_diff_fraction"]),
        int(settings["min_diff_sites"]),
    )
    raw_founders, smoothed_founders = smoothed_founder_probabilities(
        founder_block, block_indices, equivalence
    )
    raw_g0_tracks, raw_g0_coverage = painted_parent_tracks(
        painting,
        positions,
        raw_founders,
        candidates["g0_parents"],
    )
    smooth_g0_tracks, smooth_g0_coverage = painted_parent_tracks(
        painting,
        positions,
        smoothed_founders,
        candidates["g0_parents"],
    )
    g0_likelihoods = likelihoods[candidates["g0_parents"]]
    f1_likelihoods = likelihoods[candidates["f1_children"]]
    f2_likelihoods = likelihoods[candidates["f2_children"]]
    recombination_rate = float(settings["recombination_rate"])

    # This linked reconstruction is intentionally diagnostic-only.  The
    # ordered G0 paintings contain many more boundaries than a biological
    # meiosis, so their track order is not used in primary F2 scoring.
    linked_g0_tracks = condition_tracks_on_genotype_likelihoods(
        smooth_g0_tracks, g0_likelihoods, error_rate=0.01
    )
    linked_g0_diagnostic = reconstruct_inherited_tracks(
        f1_likelihoods,
        linked_g0_tracks,
        selected_g0_pairs_local,
        positions,
        parent1_recombination_rate=recombination_rate,
        parent2_recombination_rate=recombination_rate,
        error_rate=0.01,
        markers_per_block=100,
        effective_markers_per_block=1.0,
    )

    candidate_fathers_local = np.unique(candidates["f1_pairs_local"][:, 0])
    candidate_mothers_local = np.unique(candidates["f1_pairs_local"][:, 1])
    linked_scores = []
    zero_parent_scores = []
    father_only_scores = []
    mother_only_scores = []
    primary_diagnostic = None
    primary_reconstruction = None
    reconstruction_cache = {}
    for error_rate, block_size, unsmoothed in _variant_specifications():
        cache_key = (float(error_rate), bool(unsmoothed))
        if cache_key not in reconstruction_cache:
            g0_tracks = raw_g0_tracks if unsmoothed else smooth_g0_tracks
            reconstruction_cache[cache_key] = (
                reconstruct_parental_origin_tracks_unphased(
                    f1_likelihoods,
                    g0_tracks,
                    g0_likelihoods,
                    selected_g0_pairs_local,
                    error_rate=error_rate,
                )
            )
        reconstruction = reconstruction_cache[cache_key]
        diagnostic = linked.score_two_parent_binned_transmission_diagnostics(
            f2_likelihoods,
            reconstruction.tracks,
            candidates["f1_pairs_local"],
            positions,
            recombination_rate=recombination_rate,
            error_rate=error_rate,
            markers_per_block=block_size,
            effective_markers_per_block=1.0,
        )
        missing = score_binned_missing_parent_models(
            f2_likelihoods,
            reconstruction.tracks,
            candidate_fathers_local,
            candidate_mothers_local,
            positions,
            recombination_rate=recombination_rate,
            error_rate=error_rate,
            markers_per_block=block_size,
            effective_markers_per_block=1.0,
        )
        linked_scores.append(diagnostic.log_likelihood)
        zero_parent_scores.append(missing.zero_parent)
        father_only_scores.append(missing.father_only)
        mother_only_scores.append(missing.mother_only)
        if len(linked_scores) - 1 == PRIMARY_VARIANT:
            primary_diagnostic = diagnostic
            primary_reconstruction = reconstruction
    atomic_npz(
        path,
        linked_variants=np.asarray(linked_scores),
        zero_parent_variants=np.asarray(zero_parent_scores),
        father_only_variants=np.asarray(father_only_scores),
        mother_only_variants=np.asarray(mother_only_scores),
        expected_father_switches=primary_diagnostic.expected_parent1_switches,
        expected_mother_switches=primary_diagnostic.expected_parent2_switches,
        viterbi_father_switches=primary_diagnostic.viterbi_parent1_switches,
        viterbi_mother_switches=primary_diagnostic.viterbi_parent2_switches,
        phase_invariant_f1_parent1_mean_allele_entropy=(
            primary_reconstruction.mean_parent1_allele_entropy
        ),
        phase_invariant_f1_parent2_mean_allele_entropy=(
            primary_reconstruction.mean_parent2_allele_entropy
        ),
        g0_linked_diagnostic_parent1_expected_switches=(
            linked_g0_diagnostic.expected_parent1_switches
        ),
        g0_linked_diagnostic_parent2_expected_switches=(
            linked_g0_diagnostic.expected_parent2_switches
        ),
        g0_linked_diagnostic_mean_state_entropy=(
            linked_g0_diagnostic.mean_state_entropy
        ),
        g0_smoothed_coverage_fraction=np.mean(
            smooth_g0_coverage, axis=1
        ),
        g0_raw_coverage_fraction=np.mean(raw_g0_coverage, axis=1),
        markers=np.asarray(len(positions), dtype=np.int64),
    )
    return {
        "contig": contig,
        "source": "computed",
        "markers": len(positions),
        "seconds": time.monotonic() - started,
    }


def _load_contig_arrays(output_dir):
    keys = (
        "linked_variants",
        "zero_parent_variants",
        "father_only_variants",
        "mother_only_variants",
        "expected_father_switches",
        "expected_mother_switches",
        "viterbi_father_switches",
        "viterbi_mother_switches",
        "phase_invariant_f1_parent1_mean_allele_entropy",
        "phase_invariant_f1_parent2_mean_allele_entropy",
        "g0_linked_diagnostic_parent1_expected_switches",
        "g0_linked_diagnostic_parent2_expected_switches",
        "g0_linked_diagnostic_mean_state_entropy",
    )
    values = {key: [] for key in keys}
    for contig in CONTIGS:
        with np.load(Path(output_dir) / f"v7_{contig}.npz") as cached:
            for key in keys:
                values[key].append(np.asarray(cached[key]))
    return {key: np.asarray(value) for key, value in values.items()}


def _bootstrap_winners(utilities, replicates, seed):
    rng = np.random.default_rng(seed)
    n_contigs, n_children, n_pairs = utilities.shape
    counts = np.zeros((n_children, n_pairs), dtype=np.int64)
    for _ in range(replicates):
        selected = rng.integers(0, n_contigs, size=n_contigs)
        winners = np.argmax(np.mean(utilities[selected], axis=0), axis=1)
        counts[np.arange(n_children), winners] += 1
    return counts


def _leave_one_winners(utilities):
    total = np.sum(utilities, axis=0)
    return np.asarray([
        np.argmax((total - utilities[contig]) / (len(utilities) - 1), axis=1)
        for contig in range(len(utilities))
    ], dtype=np.int64)


def _support_set(values, labels, coverage=0.95):
    probabilities = np.asarray(values, dtype=np.float64)
    probabilities /= np.sum(probabilities)
    order = np.argsort(-probabilities, kind="stable")
    selected = []
    total = 0.0
    for index in order:
        if probabilities[index] <= 0.0:
            continue
        selected.append(f"{labels[index]}:{probabilities[index]:.3f}")
        total += probabilities[index]
        if total >= coverage:
            break
    return ";".join(selected)


def _parent_bootstrap(pair_counts, pairs, parent_column):
    parents = np.unique(pairs[:, parent_column])
    counts = np.zeros((len(pair_counts), len(parents)), dtype=np.int64)
    for index, parent in enumerate(parents):
        counts[:, index] = np.sum(
            pair_counts[:, pairs[:, parent_column] == parent], axis=1
        )
    return parents, counts


def _logsumexp_rows(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=1)
    return maximum + np.log(
        np.sum(np.exp(values - maximum[:, None]), axis=1)
    )


def _parent_state_result(zero, father_only, mother_only, two_parent, priors):
    zero = np.asarray(zero, dtype=np.float64)
    father_only = np.asarray(father_only, dtype=np.float64)
    mother_only = np.asarray(mother_only, dtype=np.float64)
    two_parent = np.asarray(two_parent, dtype=np.float64)
    n_children = len(zero)
    if father_only.shape[0] != n_children:
        raise ValueError("father-only scores do not match children")
    if mother_only.shape[0] != n_children:
        raise ValueError("mother-only scores do not match children")
    if two_parent.shape[0] != n_children:
        raise ValueError("two-parent scores do not match children")
    probabilities = np.asarray(priors, dtype=np.float64)
    if probabilities.shape != (4,) or np.any(probabilities <= 0.0):
        raise ValueError("parent-state priors must contain four positive values")
    probabilities /= np.sum(probabilities)
    state_log_evidence = np.column_stack([
        zero + np.log(probabilities[0]),
        _logsumexp_rows(father_only)
        - np.log(father_only.shape[1])
        + np.log(probabilities[1]),
        _logsumexp_rows(mother_only)
        - np.log(mother_only.shape[1])
        + np.log(probabilities[2]),
        _logsumexp_rows(two_parent)
        - np.log(two_parent.shape[1])
        + np.log(probabilities[3]),
    ])
    normalizer = _logsumexp_rows(state_log_evidence)
    state_weights = np.exp(state_log_evidence - normalizer[:, None])
    return {
        "state": np.argmax(state_log_evidence, axis=1),
        "state_weights": state_weights,
        "father": np.argmax(father_only, axis=1),
        "mother": np.argmax(mother_only, axis=1),
        "pair": np.argmax(two_parent, axis=1),
    }


def _build_parent_state_table(
    arrays,
    scoring,
    metadata,
    bootstrap_replicates,
):
    two_variants = arrays["linked_variants"]
    zero_variants = arrays["zero_parent_variants"]
    father_variants = arrays["father_only_variants"]
    mother_variants = arrays["mother_only_variants"]
    pairs = scoring["f1_pairs"]
    fathers = np.unique(pairs[:, 0])
    mothers = np.unique(pairs[:, 1])
    children = scoring["f2_children"]
    if father_variants.shape[3] != len(fathers):
        raise RuntimeError("Father-only score columns do not match candidates")
    if mother_variants.shape[3] != len(mothers):
        raise RuntimeError("Mother-only score columns do not match candidates")
    aliases = metadata.set_index("sample_index")["Alias"].to_dict()
    samples = metadata.set_index("sample_index")["Sample"].to_dict()
    primary_zero = zero_variants[:, PRIMARY_VARIANT]
    primary_father = father_variants[:, PRIMARY_VARIANT]
    primary_mother = mother_variants[:, PRIMARY_VARIANT]
    primary_two = two_variants[:, PRIMARY_VARIANT]

    prior_results = {
        name: _parent_state_result(
            np.sum(primary_zero, axis=0),
            np.sum(primary_father, axis=0),
            np.sum(primary_mother, axis=0),
            np.sum(primary_two, axis=0),
            values,
        )
        for name, values in PARENT_STATE_PRIORS.items()
    }
    reference = prior_results["equal_parent_count"]
    prior_states = np.asarray([
        result["state"] for result in prior_results.values()
    ])
    variant_results = [
        _parent_state_result(
            np.sum(zero_variants[:, variant], axis=0),
            np.sum(father_variants[:, variant], axis=0),
            np.sum(mother_variants[:, variant], axis=0),
            np.sum(two_variants[:, variant], axis=0),
            PARENT_STATE_PRIORS["equal_parent_count"],
        )
        for variant in range(len(VARIANT_LABELS))
    ]
    variant_states = np.asarray([
        result["state"] for result in variant_results
    ])
    variant_fathers = np.asarray([
        result["father"] for result in variant_results
    ])
    variant_mothers = np.asarray([
        result["mother"] for result in variant_results
    ])

    leave_results = []
    for omitted in range(len(CONTIGS)):
        retained = np.arange(len(CONTIGS)) != omitted
        leave_results.append(_parent_state_result(
            np.sum(primary_zero[retained], axis=0),
            np.sum(primary_father[retained], axis=0),
            np.sum(primary_mother[retained], axis=0),
            np.sum(primary_two[retained], axis=0),
            PARENT_STATE_PRIORS["equal_parent_count"],
        ))
    leave_states = np.asarray([result["state"] for result in leave_results])
    leave_fathers = np.asarray([result["father"] for result in leave_results])
    leave_mothers = np.asarray([result["mother"] for result in leave_results])

    state_counts = np.zeros(
        (len(children), len(PARENT_STATE_NAMES)), dtype=np.int64
    )
    father_counts = np.zeros((len(children), len(fathers)), dtype=np.int64)
    mother_counts = np.zeros((len(children), len(mothers)), dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1009)
    rows_index = np.arange(len(children))
    for _ in range(bootstrap_replicates):
        selected = rng.integers(0, len(CONTIGS), size=len(CONTIGS))
        result = _parent_state_result(
            np.sum(primary_zero[selected], axis=0),
            np.sum(primary_father[selected], axis=0),
            np.sum(primary_mother[selected], axis=0),
            np.sum(primary_two[selected], axis=0),
            PARENT_STATE_PRIORS["equal_parent_count"],
        )
        state_counts[rows_index, result["state"]] += 1
        father_counts[rows_index, result["father"]] += 1
        mother_counts[rows_index, result["mother"]] += 1

    rows = []
    state_labels = list(PARENT_STATE_NAMES)
    father_labels = [aliases[int(parent)] for parent in fathers]
    mother_labels = [aliases[int(parent)] for parent in mothers]
    for row_index, child in enumerate(children):
        state_index = int(reference["state"][row_index])
        father_index = int(reference["father"][row_index])
        mother_index = int(reference["mother"][row_index])
        state_fraction = (
            state_counts[row_index, state_index] / bootstrap_replicates
        )
        father_fraction = (
            father_counts[row_index, father_index] / bootstrap_replicates
        )
        mother_fraction = (
            mother_counts[row_index, mother_index] / bootstrap_replicates
        )
        state_variant_fraction = float(np.mean(
            variant_states[:, row_index] == state_index
        ))
        state_leave_fraction = float(np.mean(
            leave_states[:, row_index] == state_index
        ))
        state_prior_fraction = float(np.mean(
            prior_states[:, row_index] == state_index
        ))
        father_observed = state_index in (1, 3)
        mother_observed = state_index in (2, 3)
        father_observed_bootstrap_fraction = float(
            (
                state_counts[row_index, 1]
                + state_counts[row_index, 3]
            )
            / bootstrap_replicates
        )
        mother_observed_bootstrap_fraction = float(
            (
                state_counts[row_index, 2]
                + state_counts[row_index, 3]
            )
            / bootstrap_replicates
        )
        father_observed_variant_fraction = float(np.mean(
            np.isin(variant_states[:, row_index], (1, 3))
        ))
        mother_observed_variant_fraction = float(np.mean(
            np.isin(variant_states[:, row_index], (2, 3))
        ))
        father_observed_leave_fraction = float(np.mean(
            np.isin(leave_states[:, row_index], (1, 3))
        ))
        mother_observed_leave_fraction = float(np.mean(
            np.isin(leave_states[:, row_index], (2, 3))
        ))
        father_observed_prior_fraction = float(np.mean(
            np.isin(prior_states[:, row_index], (1, 3))
        ))
        mother_observed_prior_fraction = float(np.mean(
            np.isin(prior_states[:, row_index], (2, 3))
        ))
        father_presence_stable = all((
            father_observed,
            father_observed_bootstrap_fraction > 0.5,
            father_observed_variant_fraction == 1.0,
            father_observed_leave_fraction == 1.0,
            father_observed_prior_fraction == 1.0,
        ))
        mother_presence_stable = all((
            mother_observed,
            mother_observed_bootstrap_fraction > 0.5,
            mother_observed_variant_fraction == 1.0,
            mother_observed_leave_fraction == 1.0,
            mother_observed_prior_fraction == 1.0,
        ))
        father_absence_stable = all((
            not father_observed,
            father_observed_bootstrap_fraction < 0.5,
            father_observed_variant_fraction == 0.0,
            father_observed_leave_fraction == 0.0,
            father_observed_prior_fraction == 0.0,
        ))
        mother_absence_stable = all((
            not mother_observed,
            mother_observed_bootstrap_fraction < 0.5,
            mother_observed_variant_fraction == 0.0,
            mother_observed_leave_fraction == 0.0,
            mother_observed_prior_fraction == 0.0,
        ))
        father_variant_fraction = float(np.mean(
            variant_fathers[:, row_index] == father_index
        ))
        mother_variant_fraction = float(np.mean(
            variant_mothers[:, row_index] == mother_index
        ))
        father_leave_fraction = float(np.mean(
            leave_fathers[:, row_index] == father_index
        ))
        mother_leave_fraction = float(np.mean(
            leave_mothers[:, row_index] == mother_index
        ))
        state_stable = all((
            state_fraction > 0.5,
            state_variant_fraction == 1.0,
            state_leave_fraction == 1.0,
            state_prior_fraction == 1.0,
        ))
        father_identity_stable = all((
            father_fraction > 0.5,
            father_variant_fraction == 1.0,
            father_leave_fraction == 1.0,
        ))
        mother_identity_stable = all((
            mother_fraction > 0.5,
            mother_variant_fraction == 1.0,
            mother_leave_fraction == 1.0,
        ))
        father = int(fathers[father_index])
        mother = int(mothers[mother_index])
        row = {
            "child_index": int(child),
            "child_alias": aliases[int(child)],
            "parent_count_MAP_state": PARENT_STATE_NAMES[state_index],
            "parent_count_state_stable": state_stable,
            "parent_count_bootstrap_selection_fraction": state_fraction,
            "parent_count_variant_fraction": state_variant_fraction,
            "parent_count_leave_one_fraction": state_leave_fraction,
            "parent_count_prior_fraction": state_prior_fraction,
            "parent_count_resampling_95_set": _support_set(
                state_counts[row_index] / bootstrap_replicates,
                state_labels,
            ),
            "father_observed_MAP": father_observed,
            "father_presence_stable": father_presence_stable,
            "father_absence_stable": father_absence_stable,
            "father_observed_bootstrap_fraction": (
                father_observed_bootstrap_fraction
            ),
            "father_observed_variant_fraction": (
                father_observed_variant_fraction
            ),
            "father_observed_leave_one_fraction": (
                father_observed_leave_fraction
            ),
            "father_observed_prior_fraction": (
                father_observed_prior_fraction
            ),
            "mother_observed_MAP": mother_observed,
            "mother_presence_stable": mother_presence_stable,
            "mother_absence_stable": mother_absence_stable,
            "mother_observed_bootstrap_fraction": (
                mother_observed_bootstrap_fraction
            ),
            "mother_observed_variant_fraction": (
                mother_observed_variant_fraction
            ),
            "mother_observed_leave_one_fraction": (
                mother_observed_leave_fraction
            ),
            "mother_observed_prior_fraction": (
                mother_observed_prior_fraction
            ),
            "father_only_MAP_index": father,
            "father_only_MAP": samples[father],
            "father_only_MAP_alias": aliases[father],
            "father_only_identity_stable": father_identity_stable,
            "father_only_bootstrap_selection_fraction": father_fraction,
            "father_only_variant_fraction": father_variant_fraction,
            "father_only_leave_one_fraction": father_leave_fraction,
            "father_only_resampling_95_set": _support_set(
                father_counts[row_index] / bootstrap_replicates,
                father_labels,
            ),
            "mother_only_MAP_index": mother,
            "mother_only_MAP": samples[mother],
            "mother_only_MAP_alias": aliases[mother],
            "mother_only_identity_stable": mother_identity_stable,
            "mother_only_bootstrap_selection_fraction": mother_fraction,
            "mother_only_variant_fraction": mother_variant_fraction,
            "mother_only_leave_one_fraction": mother_leave_fraction,
            "mother_only_resampling_95_set": _support_set(
                mother_counts[row_index] / bootstrap_replicates,
                mother_labels,
            ),
        }
        for name, result in prior_results.items():
            row[f"{name}_MAP_state"] = PARENT_STATE_NAMES[
                int(result["state"][row_index])
            ]
            for index, state_name in enumerate(PARENT_STATE_NAMES):
                row[f"{name}_{state_name}_composite_weight"] = float(
                    result["state_weights"][row_index, index]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_assignment_tables(
    arrays,
    scoring,
    metadata,
    bootstrap_replicates,
    utility_transform=None,
):
    pairs = scoring["f1_pairs"]
    children = scoring["f2_children"]
    aliases = metadata.set_index("sample_index")["Alias"].to_dict()
    samples = metadata.set_index("sample_index")["Sample"].to_dict()
    if utility_transform is None:
        linked_votes = np.asarray([
            _descending_rank_votes(arrays["linked_variants"][:, variant])
            for variant in range(len(VARIANT_LABELS))
        ])
    else:
        linked_votes = np.asarray([
            utility_transform(
                arrays["linked_variants"][:, variant],
                variant,
            )
            for variant in range(len(VARIANT_LABELS))
        ])
    primary = linked_votes[PRIMARY_VARIANT]
    primary_winners = np.argmax(np.mean(primary, axis=0), axis=1)
    variant_winners = np.asarray([
        np.argmax(np.mean(variant, axis=0), axis=1)
        for variant in linked_votes
    ])
    leave_one = _leave_one_winners(primary)
    bootstrap_counts = _bootstrap_winners(
        primary, bootstrap_replicates, BOOTSTRAP_SEED
    )
    fathers, father_counts = _parent_bootstrap(bootstrap_counts, pairs, 0)
    mothers, mother_counts = _parent_bootstrap(bootstrap_counts, pairs, 1)
    father_bootstrap_winner = fathers[np.argmax(father_counts, axis=1)]
    mother_bootstrap_winner = mothers[np.argmax(mother_counts, axis=1)]
    variant_fathers = pairs[variant_winners, 0]
    variant_mothers = pairs[variant_winners, 1]
    leave_fathers = pairs[leave_one, 0]
    leave_mothers = pairs[leave_one, 1]
    pair_labels = [
        f"{aliases[int(father)]}+{aliases[int(mother)]}"
        for father, mother in pairs
    ]
    father_labels = [aliases[int(parent)] for parent in fathers]
    mother_labels = [aliases[int(parent)] for parent in mothers]
    expected_father = np.sum(arrays["expected_father_switches"], axis=0)
    expected_mother = np.sum(arrays["expected_mother_switches"], axis=0)
    viterbi_father = np.sum(arrays["viterbi_father_switches"], axis=0)
    viterbi_mother = np.sum(arrays["viterbi_mother_switches"], axis=0)
    rows = []
    candidate_rows = []
    for child_row, child in enumerate(children):
        winner = int(primary_winners[child_row])
        joint_father = int(pairs[winner, 0])
        joint_mother = int(pairs[winner, 1])
        father = int(father_bootstrap_winner[child_row])
        mother = int(mother_bootstrap_winner[child_row])
        father_index = int(np.flatnonzero(fathers == father)[0])
        mother_index = int(np.flatnonzero(mothers == mother)[0])
        pair_fraction = bootstrap_counts[child_row, winner] / bootstrap_replicates
        father_fraction = father_counts[child_row, father_index] / bootstrap_replicates
        mother_fraction = mother_counts[child_row, mother_index] / bootstrap_replicates
        pair_variant_fraction = float(np.mean(
            variant_winners[:, child_row] == winner
        ))
        father_variant_fraction = float(np.mean(
            variant_fathers[:, child_row] == father
        ))
        mother_variant_fraction = float(np.mean(
            variant_mothers[:, child_row] == mother
        ))
        pair_leave_fraction = float(np.mean(leave_one[:, child_row] == winner))
        father_leave_fraction = float(np.mean(
            leave_fathers[:, child_row] == father
        ))
        mother_leave_fraction = float(np.mean(
            leave_mothers[:, child_row] == mother
        ))
        father_stable = all((
            father == joint_father,
            father_fraction > 0.5,
            father_variant_fraction == 1.0,
            father_leave_fraction == 1.0,
        ))
        mother_stable = all((
            mother == joint_mother,
            mother_fraction > 0.5,
            mother_variant_fraction == 1.0,
            mother_leave_fraction == 1.0,
        ))
        exact_stable = all((
            father_stable,
            mother_stable,
            pair_fraction > 0.5,
            pair_variant_fraction == 1.0,
            pair_leave_fraction == 1.0,
        ))
        if exact_stable:
            evidence_class = "A_exact_multigeneration_stable"
        elif father_stable and mother_stable:
            evidence_class = "B_both_parents_stable_pair_ambiguous"
        elif father_stable:
            evidence_class = "C_father_stable_mother_ambiguous"
        elif mother_stable:
            evidence_class = "C_mother_stable_father_ambiguous"
        elif pair_variant_fraction == 1.0 and pair_leave_fraction == 1.0:
            evidence_class = "D_pair_method_stable_resampling_ambiguous"
        else:
            evidence_class = "E_competing_parental_hypotheses"
        pair_probabilities = bootstrap_counts[child_row] / bootstrap_replicates
        father_probabilities = father_counts[child_row] / bootstrap_replicates
        mother_probabilities = mother_counts[child_row] / bootstrap_replicates
        rows.append({
            "child_index": int(child),
            "child": samples[int(child)],
            "child_alias": aliases[int(child)],
            "joint_MAP_pair_index": winner,
            "joint_MAP_father_index": joint_father,
            "joint_MAP_father": samples[joint_father],
            "joint_MAP_father_alias": aliases[joint_father],
            "joint_MAP_mother_index": joint_mother,
            "joint_MAP_mother": samples[joint_mother],
            "joint_MAP_mother_alias": aliases[joint_mother],
            "marginal_resampling_father_index": father,
            "marginal_resampling_father": samples[father],
            "marginal_resampling_father_alias": aliases[father],
            "marginal_resampling_mother_index": mother,
            "marginal_resampling_mother": samples[mother],
            "marginal_resampling_mother_alias": aliases[mother],
            "joint_and_marginal_father_agree": father == joint_father,
            "joint_and_marginal_mother_agree": mother == joint_mother,
            "chromosome_bootstrap_pair_selection_fraction": pair_fraction,
            "chromosome_bootstrap_father_selection_fraction": father_fraction,
            "chromosome_bootstrap_mother_selection_fraction": mother_fraction,
            "pair_resampling_95_set": _support_set(
                pair_probabilities, pair_labels
            ),
            "father_resampling_95_set": _support_set(
                father_probabilities, father_labels
            ),
            "mother_resampling_95_set": _support_set(
                mother_probabilities, mother_labels
            ),
            "pair_variant_fraction": pair_variant_fraction,
            "father_variant_fraction": father_variant_fraction,
            "mother_variant_fraction": mother_variant_fraction,
            "pair_leave_one_fraction": pair_leave_fraction,
            "father_leave_one_fraction": father_leave_fraction,
            "mother_leave_one_fraction": mother_leave_fraction,
            "father_stable": father_stable,
            "mother_stable": mother_stable,
            "exact_pair_stable": exact_stable,
            "evidence_class": evidence_class,
            "expected_father_transmission_switches": expected_father[
                child_row, winner
            ],
            "expected_mother_transmission_switches": expected_mother[
                child_row, winner
            ],
            "viterbi_father_transmission_switches": int(
                viterbi_father[child_row, winner]
            ),
            "viterbi_mother_transmission_switches": int(
                viterbi_mother[child_row, winner]
            ),
        })
        mean_primary = np.mean(primary[:, child_row], axis=0)
        order = np.argsort(-mean_primary, kind="stable")
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(1, len(order) + 1)
        for pair_index, (candidate_father, candidate_mother) in enumerate(pairs):
            candidate_rows.append({
                "child_index": int(child),
                "child_alias": aliases[int(child)],
                "pair_index": pair_index,
                "father_index": int(candidate_father),
                "father_alias": aliases[int(candidate_father)],
                "mother_index": int(candidate_mother),
                "mother_alias": aliases[int(candidate_mother)],
                "primary_mean_rank_utility": mean_primary[pair_index],
                "primary_rank": int(ranks[pair_index]),
                "bootstrap_selection_fraction": pair_probabilities[pair_index],
                "variant_MAP_count": int(np.sum(
                    variant_winners[:, child_row] == pair_index
                )),
                "leave_one_MAP_count": int(np.sum(
                    leave_one[:, child_row] == pair_index
                )),
                "MAP": pair_index == winner,
            })
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def _apply_parent_state_support(pair_assignments, state_assignments):
    """Combine conditional pair identity with observed-parent state evidence."""
    result = pair_assignments.merge(
        state_assignments,
        on=["child_index", "child_alias"],
        how="left",
        validate="one_to_one",
    )
    if result["parent_count_MAP_state"].isna().any():
        raise RuntimeError("Missing parent-count result for an F2 child")
    result["two_parent_father_stable"] = result["father_stable"]
    result["two_parent_mother_stable"] = result["mother_stable"]
    result["two_parent_exact_pair_stable"] = result["exact_pair_stable"]
    result["two_parent_evidence_class"] = result["evidence_class"]

    for role in ("father", "mother"):
        result[f"reported_{role}_index"] = result[
            f"joint_MAP_{role}_index"
        ].astype(float)
        result[f"reported_{role}"] = result[
            f"joint_MAP_{role}"
        ].astype(object)
        result[f"reported_{role}_alias"] = result[
            f"joint_MAP_{role}_alias"
        ].astype(object)

    father_only = result["parent_count_MAP_state"].eq("father_only")
    mother_only = result["parent_count_MAP_state"].eq("mother_only")
    no_father = result["parent_count_MAP_state"].isin(
        ["zero_parent", "mother_only"]
    )
    no_mother = result["parent_count_MAP_state"].isin(
        ["zero_parent", "father_only"]
    )
    for suffix in ("index", "", "alias"):
        target = (
            "reported_father"
            if not suffix
            else f"reported_father_{suffix}"
        )
        source = (
            "father_only_MAP"
            if not suffix
            else f"father_only_MAP_{suffix}"
        )
        result.loc[father_only, target] = result.loc[father_only, source]
        target = (
            "reported_mother"
            if not suffix
            else f"reported_mother_{suffix}"
        )
        source = (
            "mother_only_MAP"
            if not suffix
            else f"mother_only_MAP_{suffix}"
        )
        result.loc[mother_only, target] = result.loc[mother_only, source]
    result.loc[no_father, [
        "reported_father_index", "reported_father", "reported_father_alias"
    ]] = np.nan
    result.loc[no_mother, [
        "reported_mother_index", "reported_mother", "reported_mother_alias"
    ]] = np.nan

    stable_state = result["parent_count_state_stable"].astype(bool)
    two_parent = result["parent_count_MAP_state"].eq("two_parent")
    stable_two = stable_state & two_parent
    stable_father_only = stable_state & father_only
    stable_mother_only = stable_state & mother_only
    cross_model_father = (
        ~stable_state
        & result["father_presence_stable"].astype(bool)
        & result["two_parent_father_stable"].astype(bool)
        & result["father_only_identity_stable"].astype(bool)
        & result["joint_MAP_father_alias"].eq(
            result["father_only_MAP_alias"]
        )
    )
    cross_model_mother = (
        ~stable_state
        & result["mother_presence_stable"].astype(bool)
        & result["two_parent_mother_stable"].astype(bool)
        & result["mother_only_identity_stable"].astype(bool)
        & result["joint_MAP_mother_alias"].eq(
            result["mother_only_MAP_alias"]
        )
    )
    result["father_stable"] = (
        (
            stable_two
            & result["two_parent_father_stable"].astype(bool)
        )
        | (
            stable_father_only
            & result["father_only_identity_stable"].astype(bool)
        )
        | cross_model_father
    )
    result["mother_stable"] = (
        (
            stable_two
            & result["two_parent_mother_stable"].astype(bool)
        )
        | (
            stable_mother_only
            & result["mother_only_identity_stable"].astype(bool)
        )
        | cross_model_mother
    )
    result["exact_pair_stable"] = (
        stable_two
        & result["two_parent_exact_pair_stable"].astype(bool)
    )

    evidence = []
    for row in result.itertuples(index=False):
        if row.parent_count_state_stable:
            if row.parent_count_MAP_state == "zero_parent":
                label = "A_zero_observed_parent_state_stable"
            elif row.parent_count_MAP_state == "father_only":
                label = (
                    "A_father_only_state_and_identity_stable"
                    if row.father_stable
                    else "B_father_only_state_identity_ambiguous"
                )
            elif row.parent_count_MAP_state == "mother_only":
                label = (
                    "A_mother_only_state_and_identity_stable"
                    if row.mother_stable
                    else "B_mother_only_state_identity_ambiguous"
                )
            else:
                label = f"two_parent__{row.two_parent_evidence_class}"
        elif row.father_stable or row.mother_stable:
            label = "B_parent_count_ambiguous_shared_parent_identity_stable"
        else:
            label = "U_parent_count_or_identity_unresolved"
        evidence.append(label)
    result["evidence_class"] = evidence
    return result
