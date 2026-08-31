"""Known-truth release validation for the current Smart parent-state engine.

This runner deliberately exercises the complete current path:

``score_parent_state_gl_hmms`` -> ``ParentStateEvidence`` ->
``infer_from_parent_state_evidence``.

It does not implement a surrogate release rule and does not compare against
legacy pedigree inference.  Each cohort has sixteen eligible candidate
parents (eight simulated female and eight simulated male candidates), all 64
opposite-sex pairs, and profile-controlled two-sibling families across six
known-truth observed-parent strata.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/pedigree-smart-mpl-{os.getuid()}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/pedigree-smart-cache-{os.getuid()}")

import thread_config  # numerical-library limits before NumPy/Numba

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numba
import numpy as np

import pedigree_inference as smart


SCHEMA_VERSION = 3
ATTEMPT_IDENTITY = "pedigree_smart_release_combined_v1_known_truth_v3"
BASE_READ_ERROR = 0.02
ELEVATED_READ_ERROR = 0.05
N_FOUNDERS = 8
N_CANDIDATES = 16
FEMALE_CANDIDATES = tuple(range(8))
MALE_CANDIDATES = tuple(range(8, 16))
N_SIBLINGS_PER_FAMILY = 2
BIN_BP = 1_000_000
RECOMBINATION_RATE = 2.0e-8
SOURCE_MIN_DEPTH = 2
DEFAULT_SCENARIOS = (
    "clean",
    "mcar_2x",
    "heterogeneous_sample_site",
    "contiguous_gaps",
    "mnar_dropout",
    "elevated_error_combined",
)
ALLOWED_SCENARIOS = (*DEFAULT_SCENARIOS, "low_child")
TRUTH_STRATA = (
    ("m0", 0, (), ()),
    ("m1_normal", 1, (0,), ()),
    ("m1_low_parent", 1, (4,), (4,)),
    ("m2_normal", 2, (0, 8), ()),
    ("m2_one_low_parent", 2, (5, 12), (12,)),
    ("m2_both_low_parents", 2, (6, 13), (6, 13)),
)
STATE_PROFILE_COUNTS = {
    "mixed": (2, 2, 2, 2, 2, 2),
    "tropheops_like": (0, 2, 2, 12, 6, 2),
    "m1_heavy": (2, 8, 8, 2, 2, 2),
}
RELATIONSHIP_LABELS = {
    0: "female_anchor",
    1: "female_full_sibling_decoy",
    2: "female_half_sibling_decoy",
    3: "female_near_duplicate_decoy",
    8: "male_anchor",
    9: "male_full_sibling_decoy",
    10: "male_half_sibling_decoy",
    11: "male_near_duplicate_decoy",
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--split", choices=("calibration", "holdout"), default="calibration"
    )
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--replicate-start", type=int, default=0)
    parser.add_argument("--replicate-step", type=int, default=1)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--state-profile", choices=tuple(STATE_PROFILE_COUNTS), default="mixed")
    parser.add_argument("--low-parent-depth", type=float, default=0.25)
    parser.add_argument("--effective-markers", default="1,5")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--contigs", type=int, default=22)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--snps-per-bin", type=int, default=5)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--threads-per-process", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _comma_values(text):
    return tuple(value.strip() for value in text.split(",") if value.strip())


def _validated_args(args):
    affinity = len(os.sched_getaffinity(0))
    scenarios = _comma_values(args.scenarios)
    unknown = sorted(set(scenarios) - set(ALLOWED_SCENARIOS))
    if not scenarios or len(scenarios) != len(set(scenarios)) or unknown:
        raise ValueError(f"invalid scenario list; unknown={unknown}")
    try:
        effective_markers = tuple(
            float(value) for value in _comma_values(args.effective_markers)
        )
    except ValueError as exc:
        raise ValueError("effective-markers must be comma-separated numbers") from exc
    if (
        not effective_markers
        or len(effective_markers) != len(set(effective_markers))
        or any(not np.isfinite(value) or value <= 0.0 for value in effective_markers)
    ):
        raise ValueError("effective markers must be unique, finite, and positive")
    if not 1 <= args.replicates <= 10_000:
        raise ValueError("replicates must lie in [1, 10000]")
    if args.replicate_start < 0 or not 1 <= args.replicate_step <= 10_000:
        raise ValueError("replicate-start/step are out of bounds")
    if not 3 <= args.contigs <= 22:
        raise ValueError("contigs must lie in [3, 22] for Tier LOCO release")
    if not 3 <= args.bins <= 100 or not 2 <= args.snps_per_bin <= 20:
        raise ValueError("bins must be in [3,100] and SNPs per bin in [2,20]")
    if not 1 <= args.bootstrap_replicates <= 100_000:
        raise ValueError("bootstrap-replicates must lie in [1,100000]")
    if not np.isfinite(args.low_parent_depth) or args.low_parent_depth < 0.0:
        raise ValueError("low-parent-depth must be finite and non-negative")
    if args.processes < 1 or args.threads_per_process < 1:
        raise ValueError("process and thread counts must be positive")
    aggregate_threads = args.processes * args.threads_per_process
    if aggregate_threads > affinity:
        raise ValueError(
            f"processes*threads-per-process={aggregate_threads} exceeds "
            f"the {affinity}-CPU affinity"
        )
    return scenarios, effective_markers, affinity


def _json_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return str(value)


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path, rows):
    rows = list(rows)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rng(seed, *identity):
    raw = "|".join(map(str, identity)).encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=16).digest()
    words = np.frombuffer(digest, dtype=np.uint32)
    entropy = [seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, *map(int, words)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _founders(rng, n_bins, n_snps):
    n_markers = n_bins * n_snps
    for _ in range(100):
        frequency = rng.beta(0.7, 0.7, size=n_markers)
        haplotypes = np.empty((N_FOUNDERS, n_markers), dtype=np.int8)
        haplotypes[:, 0] = rng.random(N_FOUNDERS) < frequency[0]
        for marker in range(1, n_markers):
            fresh = rng.random(N_FOUNDERS) < frequency[marker]
            retain = rng.random(N_FOUNDERS) < 0.72
            haplotypes[:, marker] = np.where(
                retain, haplotypes[:, marker - 1], fresh
            )
        if len(np.unique(haplotypes, axis=0)) == N_FOUNDERS:
            return np.ascontiguousarray(
                haplotypes.reshape(N_FOUNDERS, n_bins, n_snps)
            )
    # Very small smoke-test grids can repeatedly collapse under the linked
    # generator. Enumerate distinct binary haplotypes there; production-sized
    # grids virtually always return through the linked path above.
    if n_markers >= math.ceil(math.log2(N_FOUNDERS)):
        codes = rng.choice(1 << n_markers, size=N_FOUNDERS, replace=False)
        haplotypes = np.asarray([
            [(int(code) >> marker) & 1 for marker in range(n_markers)]
            for code in codes
        ], dtype=np.int8)
        return np.ascontiguousarray(
            haplotypes.reshape(N_FOUNDERS, n_bins, n_snps)
        )
    raise RuntimeError("marker grid cannot encode unique founder haplotypes")


def _mosaic(founders, rng, switch_probability=0.06):
    _, n_bins, n_snps = founders.shape
    alleles = np.empty((n_bins, 2, n_snps), dtype=np.int8)
    labels = np.empty((n_bins, 2), dtype=np.int16)
    for homolog in range(2):
        state = int(rng.integers(N_FOUNDERS))
        for block in range(n_bins):
            if block and rng.random() < switch_probability:
                state = (state + int(rng.integers(1, N_FOUNDERS))) % N_FOUNDERS
            alleles[block, homolog] = founders[state, block]
            labels[block, homolog] = state
    return alleles, labels


def _gamete(parent, labels, rng):
    n_bins, _, n_snps = parent.shape
    alleles = np.empty((n_bins, n_snps), dtype=np.int8)
    out_labels = np.empty(n_bins, dtype=np.int16)
    homolog = int(rng.integers(2))
    theta = 0.5 * (1.0 - math.exp(-2.0 * RECOMBINATION_RATE * BIN_BP))
    for block in range(n_bins):
        if block and rng.random() < theta:
            homolog = 1 - homolog
        alleles[block] = parent[block, homolog]
        out_labels[block] = labels[block, homolog]
    return alleles, out_labels


def _offspring(first, second, rng):
    first_gamete = _gamete(first[0], first[1], rng)
    second_gamete = _gamete(second[0], second[1], rng)
    return (
        np.stack((first_gamete[0], second_gamete[0]), axis=1),
        np.stack((first_gamete[1], second_gamete[1]), axis=1),
    )


def _near_duplicate(candidate, founders, rng):
    alleles = candidate[0].copy()
    labels = candidate[1].copy()
    original = alleles.copy()
    n_bins = alleles.shape[0]
    for _ in range(100):
        block = int(rng.integers(n_bins))
        homolog = int(rng.integers(2))
        old_state = int(labels[block, homolog])
        new_state = (old_state + int(rng.integers(1, N_FOUNDERS))) % N_FOUNDERS
        labels[block, homolog] = new_state
        alleles[block, homolog] = founders[new_state, block]
        if not np.array_equal(alleles, original):
            return alleles, labels
    raise RuntimeError("could not construct a genetically distinct near duplicate")


def _candidate_panel(founders, rng):
    candidates = [_mosaic(founders, rng) for _ in range(N_CANDIDATES)]
    for anchor in (0, 8):
        first_grandparent = _mosaic(founders, rng)
        second_grandparent = _mosaic(founders, rng)
        other_grandparent = _mosaic(founders, rng)
        candidates[anchor] = _offspring(
            first_grandparent, second_grandparent, rng
        )
        candidates[anchor + 1] = _offspring(
            first_grandparent, second_grandparent, rng
        )
        candidates[anchor + 2] = _offspring(
            first_grandparent, other_grandparent, rng
        )
        candidates[anchor + 3] = _near_duplicate(
            candidates[anchor], founders, rng
        )
    return candidates


def _state_profile_counts(profile):
    return {
        stratum[0]: count
        for stratum, count in zip(TRUTH_STRATA, STATE_PROFILE_COUNTS[profile])
    }


def _truth_rows(profile):
    rows = []
    counts = _state_profile_counts(profile)
    for stratum_index, (name, state, observed, low_depth) in enumerate(
        TRUTH_STRATA
    ):
        stratum_count = counts[name]
        if stratum_count % N_SIBLINGS_PER_FAMILY:
            raise ValueError(
                f"state profile {profile!r} has an incomplete sibling family "
                f"in stratum {name!r}"
            )
        for family_within_stratum in range(
            stratum_count // N_SIBLINGS_PER_FAMILY
        ):
            family = f"{profile}:{stratum_index:02d}:{family_within_stratum:02d}"
            for within_family in range(N_SIBLINGS_PER_FAMILY):
                rows.append({
                    "child": N_CANDIDATES + len(rows),
                    "sample": (
                        f"child_{profile}_{name}_family-"
                        f"{family_within_stratum:02d}_sibling-{within_family:02d}"
                    ),
                    "stratum": name,
                    "stratum_count": stratum_count,
                    "family": family,
                    "family_within_stratum": family_within_stratum,
                    "within_family": within_family,
                    "true_state": state,
                    "observed_parents": tuple(observed),
                    "low_depth_parents": tuple(low_depth),
                })
    return rows


def _genetics(seed, split, replicate, contig, n_bins, n_snps, truth_rows):
    rng = _rng(seed, split, "genetics", replicate, contig)
    founders = _founders(rng, n_bins, n_snps)
    candidates = _candidate_panel(founders, rng)
    external_by_family = {}
    for row in truth_rows:
        family = row["family"]
        if family not in external_by_family:
            external_by_family[family] = [
                _mosaic(founders, rng)
                for _ in range(2 - row["true_state"])
            ]

    child_alleles = []
    child_labels = []
    for row in truth_rows:
        parents = [candidates[index] for index in row["observed_parents"]]
        parents.extend(external_by_family[row["family"]])
        if len(parents) != 2:
            raise AssertionError("every simulated child must have two biological parents")
        child = _offspring(parents[0], parents[1], rng)
        child_alleles.append(child[0])
        child_labels.append(child[1])

    alleles = np.ascontiguousarray(np.stack(
        [candidate[0] for candidate in candidates] + child_alleles
    ))
    labels = np.ascontiguousarray(np.stack(
        [candidate[1] for candidate in candidates] + child_labels
    ))
    return founders, alleles, labels


def _observations(
    seed,
    split,
    replicate,
    scenario,
    contig,
    alleles,
    labels,
    low_depth_candidates,
    low_parent_depth,
):
    rng = _rng(seed, split, "observations", replicate, scenario, contig)
    dosage = np.sum(alleles, axis=2)
    n_samples, n_bins, n_snps = dosage.shape
    shape = dosage.shape
    mean = np.full(shape, 8.0, dtype=np.float64)
    forced_zero = np.zeros(shape, dtype=np.bool_)
    error = BASE_READ_ERROR

    if scenario == "clean":
        mean.fill(10.0)
    elif scenario == "mcar_2x":
        mean.fill(2.0)
        forced_zero |= rng.random(shape) < 0.15
    elif scenario == "heterogeneous_sample_site":
        sample_multiplier = rng.lognormal(-0.32, 0.8, (n_samples, 1, 1))
        site_multiplier = rng.lognormal(-0.32, 0.8, (1, n_bins, n_snps))
        mean = np.clip(2.0 * sample_multiplier * site_multiplier, 0.02, 30.0)
        forced_zero |= rng.random((1, n_bins, n_snps)) < 0.08
    elif scenario == "contiguous_gaps":
        mean.fill(6.0)
        length = max(1, n_bins // 4)
        shared_start = max(0, n_bins // 3)
        forced_zero[:, shared_start:shared_start + length] = True
        for sample in range(n_samples):
            start = int(rng.integers(max(1, n_bins - length + 1)))
            forced_zero[sample, start:start + length] = True
    elif scenario == "mnar_dropout":
        mean.fill(4.0)
        forced_zero |= rng.random(shape) < (0.05 + 0.35 * dosage / 2.0)
    elif scenario == "elevated_error_combined":
        error = ELEVATED_READ_ERROR
        sample_multiplier = rng.lognormal(-0.32, 0.8, (n_samples, 1, 1))
        site_multiplier = rng.lognormal(-0.32, 0.8, (1, n_bins, n_snps))
        mean = np.clip(2.0 * sample_multiplier * site_multiplier, 0.02, 20.0)
        forced_zero |= rng.random(shape) < (0.08 + 0.22 * dosage / 2.0)
        length = max(1, n_bins // 5)
        start = int(rng.integers(max(1, n_bins - length + 1)))
        forced_zero[:, start:start + length] = True
    elif scenario == "low_child":
        mean.fill(4.0)
        mean[N_CANDIDATES:] = 0.25
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    if low_depth_candidates:
        mean[np.asarray(low_depth_candidates, dtype=np.int64)] = low_parent_depth
    depth = rng.poisson(mean).astype(np.int32)
    depth[forced_zero] = 0
    alt_probability = np.where(
        dosage == 0, error, np.where(dosage == 1, 0.5, 1.0 - error)
    )
    alt = rng.binomial(depth, alt_probability).astype(np.int32)
    ref = depth - alt
    genotype_alt_probability = np.asarray((error, 0.5, 1.0 - error))
    log_gl = (
        ref[..., None] * np.log1p(-genotype_alt_probability)
        + alt[..., None] * np.log(genotype_alt_probability)
    )
    log_gl -= np.max(log_gl, axis=-1, keepdims=True)
    gl = np.exp(log_gl)
    gl /= np.sum(gl, axis=-1, keepdims=True)

    hard = alleles.copy()
    missing_marker = depth < SOURCE_MIN_DEPTH
    hard[np.repeat(missing_marker[:, :, None, :], 2, axis=2)] = -1
    wholly_missing_bin = np.all(hard < 0, axis=(2, 3))
    masked_labels = labels.copy()
    masked_labels[wholly_missing_bin] = -1
    # A fully missing or observed-homozygous bin cannot retain a phased
    # homolog orientation. Equality after masking therefore supplies the
    # same reset semantics used by the production cache representation.
    hom_reset_mask = np.all(
        hard[:, :, 0, :] == hard[:, :, 1, :], axis=2
    )
    return (
        np.ascontiguousarray(gl),
        np.ascontiguousarray(hard),
        np.ascontiguousarray(masked_labels),
        np.ascontiguousarray(hom_reset_mask),
        depth,
        wholly_missing_bin,
        error,
    )


def _sample_ids(truth_rows):
    return [f"candidate_{index:02d}" for index in range(N_CANDIDATES)] + [
        row["sample"] for row in truth_rows
    ]


def _trios_and_eligibility(sample_ids, truth_rows):
    n_samples = len(sample_ids)
    children = np.zeros(n_samples, dtype=np.bool_)
    parents = np.zeros((n_samples, n_samples), dtype=np.bool_)
    pairs = np.zeros((n_samples, n_samples, n_samples), dtype=np.bool_)
    trios = []
    for truth in truth_rows:
        child = truth["child"]
        children[child] = True
        parents[child, :N_CANDIDATES] = True
        for female in FEMALE_CANDIDATES:
            for male in MALE_CANDIDATES:
                pairs[child, female, male] = True
                pairs[child, male, female] = True
                trios.append((child, female, male))
    eligibility = smart.ParentEligibility(
        format_version=smart.PARENT_ELIGIBILITY_FORMAT_VERSION,
        sample_ids=tuple(sample_ids),
        eligible_children=children,
        eligible_parents=parents,
        eligible_parent_pairs=pairs,
        policy_name="simulated_16_parent_opposite_sex_panel_v1",
        source_fields=("simulated_candidate_role", "simulated_sex"),
        assumptions=(
            "all sixteen simulated candidates are eligible for M1",
            "M2 is restricted to all 64 simulated opposite-sex pairs",
            "eligibility is not individual parentage truth",
        ),
        individual_parentage_ground_truth=False,
    )
    return np.ascontiguousarray(trios, dtype=np.int64), eligibility


def _bootstrap_seed(seed, split, replicate, scenario):
    rng = _rng(seed, split, "bootstrap", replicate, scenario)
    return int(rng.integers(0, np.iinfo(np.int32).max))


def _value(frame, sample, column):
    return _json_scalar(frame.loc[sample, column])


def _parent_set(first, second):
    return frozenset(value for value in (first, second) if value is not None)


def _view_fields(frame, sample, prefix):
    return {
        f"{prefix}_state": _value(frame, sample, "ParentState"),
        f"{prefix}_observed_parent_count": _value(
            frame, sample, "ObservedParentCount"
        ),
        f"{prefix}_parent1": _value(frame, sample, "Parent1"),
        f"{prefix}_parent2": _value(frame, sample, "Parent2"),
        f"{prefix}_status": _value(frame, sample, "InferenceStatus"),
    }


def _unit_payload(spec):
    started = time.perf_counter()
    numba.set_num_threads(spec["threads_per_process"])
    truth_rows = _truth_rows(spec["state_profile"])
    sample_ids = _sample_ids(truth_rows)
    trios, eligibility = _trios_and_eligibility(sample_ids, truth_rows)
    low_depth_candidates = sorted({
        parent for row in truth_rows for parent in row["low_depth_parents"]
    })
    evidence = []
    junction_rows = []
    callable_rows = []
    sample_depth_sum = np.zeros(len(sample_ids), dtype=np.float64)
    sample_depth_sites = 0
    candidate_call_counts = np.zeros(N_CANDIDATES, dtype=np.int64)
    wholly_missing_counts = np.zeros(len(sample_ids), dtype=np.int64)
    identity_information_sum = np.zeros(
        (len(sample_ids), len(sample_ids)), dtype=np.float64
    )
    error_probabilities = []

    for contig in range(spec["contigs"]):
        founders, alleles, labels = _genetics(
            spec["seed"], spec["split"], spec["replicate"], contig,
            spec["bins"], spec["snps_per_bin"], truth_rows,
        )
        (
            gl,
            hard,
            masked_labels,
            hom_reset_mask,
            depth,
            wholly_missing_bin,
            error,
        ) = _observations(
            spec["seed"], spec["split"], spec["replicate"],
            spec["scenario"], contig, alleles, labels,
            low_depth_candidates, spec["low_parent_depth"],
        )
        theta = 0.5 * (
            1.0 - math.exp(-2.0 * RECOMBINATION_RATE * BIN_BP)
        )
        scores = smart.score_parent_state_gl_hmms(
            gl,
            hard,
            masked_labels,
            hom_reset_mask,
            founders,
            np.full(spec["bins"], spec["snps_per_bin"], dtype=np.int64),
            np.full(spec["bins"], theta, dtype=np.float64),
            trios,
            mismatch_probability=0.01,
            phase_switch_probability=0.01,
            markers_per_information_block=100,
            effective_markers_per_information_block=spec["effective_markers"],
            external_state_pseudocount=1.0,
            external_transition_pseudocount=20.0,
        )
        evidence.append(smart.ParentStateEvidence(
            contig=f"chr{contig + 1}",
            trios=trios,
            zero_parent_log_likelihoods=scores.zero_observed,
            one_parent_log_likelihoods=scores.one_observed,
            two_parent_log_likelihoods=scores.two_observed,
            informative_markers=spec["bins"] * spec["snps_per_bin"],
        ))
        identity_information_sum += scores.one_parent_identity_information
        junction_rows.append(scores.ancestry_junction_counts)
        callable_rows.append(scores.ancestry_callable_haplotype_bins)
        sample_depth_sum += np.sum(depth, axis=(1, 2))
        sample_depth_sites += spec["bins"] * spec["snps_per_bin"]
        candidate_call_counts += np.sum(
            hard[:N_CANDIDATES] >= 0, axis=(1, 2, 3)
        )
        wholly_missing_counts += np.sum(wholly_missing_bin, axis=1)
        error_probabilities.append(error)

    settings = smart.PedigreeConfig(
        bootstrap_replicates=spec["bootstrap_replicates"],
        bootstrap_seed=_bootstrap_seed(
            spec["seed"], spec["split"], spec["replicate"], spec["scenario"]
        ),
        minimum_informative_contigs=3,
        parent_state_effective_markers_per_information_block=(
            spec["effective_markers"]
        ),
    ).validated()
    result = smart.infer_from_parent_state_evidence(
        evidence,
        sample_ids,
        config=settings,
        parent_eligibility=eligibility,
        ancestry_junction_counts=np.stack(junction_rows),
        ancestry_callable_haplotype_bins=np.stack(callable_rows),
        n_workers=1,
    )

    frames = {
        "complete": result.complete_relationships.set_index("Sample"),
        "tier_a": result.tier_a_relationships.set_index("Sample"),
        "tier_b": result.tier_b_relationships.set_index("Sample"),
        "tier_a_partial": result.tier_a_partial_relationships.set_index("Sample"),
        "tier_b_partial": result.tier_b_partial_relationships.set_index("Sample"),
    }
    diagnostics = result.smart_diagnostics.set_index("Sample")
    records = []
    for truth in truth_rows:
        sample = truth["sample"]
        observed_parent_ids = tuple(
            sample_ids[parent] for parent in truth["observed_parents"]
        )
        observable_parent_indices = tuple(
            parent for parent in truth["observed_parents"]
            if identity_information_sum[truth["child"], parent] > 0.0
        )
        observable_parent_ids = tuple(
            sample_ids[parent] for parent in observable_parent_indices
        )
        low_parent_ids = tuple(
            sample_ids[parent] for parent in truth["low_depth_parents"]
        )
        record = {
            "split": spec["split"],
            "replicate": spec["replicate"],
            "scenario": spec["scenario"],
            "state_profile": spec["state_profile"],
            "state_profile_counts": json.dumps(
                spec["state_profile_counts"], sort_keys=True, separators=(",", ":")
            ),
            "state_profile_stratum_count": truth["stratum_count"],
            "low_parent_depth": spec["low_parent_depth"],
            "family_within_stratum": truth["family_within_stratum"],
            "effective_markers": spec["effective_markers"],
            "sample": sample,
            "stratum": truth["stratum"],
            "family": truth["family"],
            "within_family": truth["within_family"],
            "truth_state": truth["true_state"],
            "truth_parents": ";".join(observed_parent_ids),
            "truth_parent1": (
                observed_parent_ids[0] if len(observed_parent_ids) > 0 else None
            ),
            "truth_parent2": (
                observed_parent_ids[1] if len(observed_parent_ids) > 1 else None
            ),
            "observable_truth_state": len(observable_parent_ids),
            "observable_truth_parents": ";".join(observable_parent_ids),
            "observable_truth_parent1": (
                observable_parent_ids[0] if len(observable_parent_ids) > 0 else None
            ),
            "observable_truth_parent2": (
                observable_parent_ids[1] if len(observable_parent_ids) > 1 else None
            ),
            "low_depth_parents": ";".join(low_parent_ids),
            "mean_child_depth": float(
                sample_depth_sum[truth["child"]] / sample_depth_sites
            ),
            "wholly_missing_child_bins": int(
                wholly_missing_counts[truth["child"]]
            ),
            "read_error_probability": float(error_probabilities[0]),
            "candidate_count": N_CANDIDATES,
            "eligible_pair_count": len(FEMALE_CANDIDATES) * len(MALE_CANDIDATES),
        }
        for parent_number, parent in enumerate(truth["observed_parents"], start=1):
            record[f"truth_parent{parent_number}_mean_depth"] = float(
                sample_depth_sum[parent] / sample_depth_sites
            )
            record[f"truth_parent{parent_number}_called_alleles"] = int(
                candidate_call_counts[parent]
            )
            record[f"truth_parent{parent_number}_identity_information"] = float(
                identity_information_sum[truth["child"], parent]
            )
            record[f"truth_parent{parent_number}_relationship_label"] = (
                RELATIONSHIP_LABELS.get(parent, "independent_candidate")
            )
        for prefix, frame in frames.items():
            record.update(_view_fields(frame, sample, prefix))

        diagnostic = diagnostics.loc[sample]
        diagnostic_fields = (
            "LocalWinnerParentState",
            "LocalObservedParentCount",
            "SelectedParentState",
            "ObservedParentCount",
            "InformativeContigCount",
            "StateSupport0",
            "StateSupport1",
            "StateSupport2",
            "LOOStatePrior0",
            "LOOStatePrior1",
            "LOOStatePrior2",
            "StateWinnerMargin",
            "ConditionalIdentityMargin",
            "LocalStateBootstrapFraction",
            "LocalConfigurationBootstrapFraction",
            "GraphConfigurationBootstrapFraction",
            "Parent1BootstrapFraction",
            "Parent2BootstrapFraction",
            "LocalStateLOCOFraction",
            "LocalConfigurationLOCOFraction",
            "GraphConfigurationLOCOFraction",
            "Parent1LOCOFraction",
            "Parent2LOCOFraction",
            "TierAStateCall",
            "TierBStateCall",
            "TierAExactConfiguration",
            "TierBExactConfiguration",
            "TierAParent1",
            "TierAParent2",
            "TierBParent1",
            "TierBParent2",
            "DAGDisplaced",
            "GraphConflict",
            "GraphTieConflict",
            "PriorSensitivityLocalStateAgreementFraction",
            "PriorSensitivityLocalIdentityAgreementFraction",
        )
        for field in diagnostic_fields:
            record[f"diagnostic_{field}"] = _json_scalar(diagnostic[field])

        truth_set = frozenset(observed_parent_ids)
        observable_truth_set = frozenset(observable_parent_ids)
        raw_state = record["diagnostic_LocalObservedParentCount"]
        record["raw_state_correct"] = raw_state == truth["true_state"]
        record["raw_observable_state_correct"] = (
            raw_state == len(observable_truth_set)
        )
        for prefix in ("complete", "tier_a", "tier_b"):
            predicted = _parent_set(
                record[f"{prefix}_parent1"], record[f"{prefix}_parent2"]
            )
            state_correct = (
                record[f"{prefix}_observed_parent_count"] == truth["true_state"]
            )
            record[f"{prefix}_state_correct"] = state_correct
            record[f"{prefix}_exact_correct"] = (
                state_correct and predicted == truth_set
            )
            observable_state_correct = (
                record[f"{prefix}_observed_parent_count"]
                == len(observable_truth_set)
            )
            record[f"{prefix}_observable_state_correct"] = observable_state_correct
            record[f"{prefix}_observable_exact_correct"] = (
                observable_state_correct and predicted == observable_truth_set
            )
        for prefix in ("tier_a_partial", "tier_b_partial"):
            predicted = _parent_set(
                record[f"{prefix}_parent1"], record[f"{prefix}_parent2"]
            )
            record[f"{prefix}_true_edge_count"] = len(predicted & truth_set)
            record[f"{prefix}_false_edge_count"] = len(predicted - truth_set)
            record[f"{prefix}_predicted_edge_count"] = len(predicted)
            record[f"{prefix}_truth_edge_count"] = len(truth_set)
        records.append(record)

    elapsed = time.perf_counter() - started
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": spec["manifest_sha256"],
        "unit_identity": spec["unit_identity"],
        "records": records,
        "cohort_diagnostics": {
            "contigs": spec["contigs"],
            "children": len(truth_rows),
            "state_profile": spec["state_profile"],
            "state_profile_counts": spec["state_profile_counts"],
            "low_parent_depth": spec["low_parent_depth"],
            "candidate_count": N_CANDIDATES,
            "eligible_pairs_per_child": 64,
            "trio_rows": len(trios),
            "candidate_relationships": RELATIONSHIP_LABELS,
            "candidate_called_alleles": candidate_call_counts.tolist(),
            "mean_depth_by_sample": (
                sample_depth_sum / sample_depth_sites
            ).tolist(),
            "wholly_missing_bins_by_sample": wholly_missing_counts.tolist(),
            "bootstrap_workers": int(result.smart_bootstrap_worker_count),
            "numba_threads": int(numba.get_num_threads()),
        },
        "elapsed_seconds": elapsed,
        "maximum_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }


def _run_and_checkpoint(spec):
    output = _unit_payload(spec)
    _atomic_json(Path(spec["unit_path"]), output)
    return output


def _unit_name(split, scenario, effective_markers, replicate):
    marker = f"{effective_markers:g}".replace(".", "p")
    return f"{split}.{scenario}.e{marker}.replicate-{replicate:07d}.json"


def _fraction(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _sets_from_record(record, prefix):
    return _parent_set(record[f"{prefix}_parent1"], record[f"{prefix}_parent2"])


def _metric_row(rows, scenario, effective_markers, stratum):
    n = len(rows)
    raw_correct = sum(row["raw_state_correct"] for row in rows)
    raw_observable_correct = sum(
        row["raw_observable_state_correct"] for row in rows
    )
    complete_correct = sum(row["complete_exact_correct"] for row in rows)
    complete_observable_correct = sum(
        row["complete_observable_exact_correct"] for row in rows
    )
    output = {
        "scenario": scenario,
        "effective_markers": effective_markers,
        "stratum": stratum,
        "children": n,
        "raw_state_accuracy": _fraction(raw_correct, n),
        "raw_observable_state_accuracy": _fraction(raw_observable_correct, n),
        "complete_exact_accuracy": _fraction(complete_correct, n),
        "complete_observable_exact_accuracy": _fraction(
            complete_observable_correct, n
        ),
    }
    for tier in ("tier_a", "tier_b"):
        state_released = [
            row for row in rows if row[f"diagnostic_{'TierA' if tier == 'tier_a' else 'TierB'}StateCall"]
        ]
        exact_released = [
            row for row in rows if row[f"diagnostic_{'TierA' if tier == 'tier_a' else 'TierB'}ExactConfiguration"]
        ]
        output[f"{tier}_state_coverage"] = _fraction(len(state_released), n)
        output[f"{tier}_state_selective_accuracy"] = _fraction(
            sum(row[f"{tier}_state_correct"] for row in state_released),
            len(state_released),
        )
        output[f"{tier}_exact_coverage"] = _fraction(len(exact_released), n)
        output[f"{tier}_exact_selective_accuracy"] = _fraction(
            sum(row[f"{tier}_exact_correct"] for row in exact_released),
            len(exact_released),
        )
        output[f"{tier}_exact_error_count"] = sum(
            not row[f"{tier}_exact_correct"] for row in exact_released
        )
        output[f"{tier}_exact_abstention_count"] = n - len(exact_released)
        partial = f"{tier}_partial"
        true_edges = sum(row[f"{partial}_true_edge_count"] for row in rows)
        false_edges = sum(row[f"{partial}_false_edge_count"] for row in rows)
        predicted_edges = true_edges + false_edges
        truth_edges = sum(row[f"{partial}_truth_edge_count"] for row in rows)
        output[f"{tier}_partial_edge_recall"] = _fraction(true_edges, truth_edges)
        output[f"{tier}_partial_edge_precision"] = _fraction(
            true_edges, predicted_edges
        )
        output[f"{tier}_partial_false_edge_count"] = false_edges

    brier = []
    log_loss = []
    for row in rows:
        support = np.asarray([
            row["diagnostic_StateSupport0"],
            row["diagnostic_StateSupport1"],
            row["diagnostic_StateSupport2"],
        ], dtype=np.float64)
        target = np.zeros(3, dtype=np.float64)
        target[int(row["truth_state"])] = 1.0
        brier.append(float(np.sum((support - target) ** 2)))
        log_loss.append(float(-math.log(max(support[int(row["truth_state"])], 1e-300))))
    output["state_support_brier"] = (
        float(np.mean(brier)) if brier else None
    )
    output["state_support_log_loss"] = (
        float(np.mean(log_loss)) if log_loss else None
    )
    return output


def _summaries(records, scenarios, effective_markers):
    aggregate = []
    confusion = []
    strata = tuple(name for name, _, _, _ in TRUTH_STRATA)
    for scenario in scenarios:
        for effective in effective_markers:
            cell = [
                row for row in records
                if row["scenario"] == scenario
                and row["effective_markers"] == effective
            ]
            aggregate.append(_metric_row(cell, scenario, effective, "ALL"))
            for stratum in strata:
                subset = [row for row in cell if row["stratum"] == stratum]
                aggregate.append(_metric_row(subset, scenario, effective, stratum))
            for truth_state in range(3):
                truth = [row for row in cell if row["truth_state"] == truth_state]
                for selected in (0, 1, 2, None):
                    count = sum(
                        row["diagnostic_LocalObservedParentCount"] == selected
                        for row in truth
                    )
                    confusion.append({
                        "scenario": scenario,
                        "effective_markers": effective,
                        "truth_state": truth_state,
                        "raw_selected_state": (
                            "unresolved" if selected is None else selected
                        ),
                        "count": count,
                        "truth_total": len(truth),
                        "fraction": _fraction(count, len(truth)),
                    })
    return aggregate, confusion


def _source_hashes(root):
    names = (
        Path(__file__).name,
        "pedigree_inference.py",
        "pedigree_hard_painting.py",
        "pedigree_result.py",
        "thread_config.py",
        "bhd_config.py",
        "bhd_genotype_evidence.py",
    )
    return {name: _sha256(root / name) for name in names}


def _manifest(args, scenarios, effective_markers, affinity):
    root = Path(__file__).resolve().parent
    configs = {}
    for effective in effective_markers:
        config = smart.PedigreeConfig(
            bootstrap_replicates=args.bootstrap_replicates,
            minimum_informative_contigs=3,
            parent_state_effective_markers_per_information_block=effective,
        ).validated()
        configs[f"{effective:g}"] = asdict(config)
    state_profile_counts = _state_profile_counts(args.state_profile)
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_identity": ATTEMPT_IDENTITY,
        "interpretation": (
            "Known-truth validation of the current Smart raw-GL parent-state "
            "scorer and actual hierarchical prior, DAG, bootstrap, LOCO, and "
            "Tier A/B exact/partial release engine; no legacy comparison."
        ),
        "parameters": {
            "seed": args.seed,
            "split": args.split,
            "replicates": args.replicates,
            "replicate_start": args.replicate_start,
            "replicate_step": args.replicate_step,
            "state_profile": args.state_profile,
            "state_profile_counts": state_profile_counts,
            "low_parent_depth": args.low_parent_depth,
            "scenarios": list(scenarios),
            "effective_markers": list(effective_markers),
            "bootstrap_replicates": args.bootstrap_replicates,
            "contigs": args.contigs,
            "bins": args.bins,
            "snps_per_bin": args.snps_per_bin,
            "processes": args.processes,
            "threads_per_process": args.threads_per_process,
        },
        "fixed_design": {
            "founder_haplotypes": N_FOUNDERS,
            "candidate_parents": N_CANDIDATES,
            "candidate_females": len(FEMALE_CANDIDATES),
            "candidate_males": len(MALE_CANDIDATES),
            "complete_opposite_sex_pairs_per_child": 64,
            "state_profile": args.state_profile,
            "truth_stratum_counts": state_profile_counts,
            "siblings_per_family": N_SIBLINGS_PER_FAMILY,
            "truth_strata": [name for name, _, _, _ in TRUTH_STRATA],
            "baseline_read_error_probability": BASE_READ_ERROR,
            "source_minimum_depth": SOURCE_MIN_DEPTH,
            "candidate_relationships": {
                str(index): label
                for index, label in RELATIONSHIP_LABELS.items()
            },
            "missingness_fidelity": (
                "hard alleles are masked from observed depth; wholly missing "
                "paintings set both labels to -1; hom/orientation reset masks "
                "are recomputed from the masked homolog alleles"
            ),
            "scenario_observation_models": {
                "low_child": {
                    "non_child_poisson_mean_depth": 4.0,
                    "child_poisson_mean_depth": 0.25,
                    "read_error_probability": BASE_READ_ERROR,
                    "forced_dropout": False,
                },
            },
        },
        "rng_namespaces": {
            "genetics": "(seed, split, genetics, replicate, contig)",
            "observations": (
                "(seed, split, observations, replicate, scenario, contig)"
            ),
            "bootstrap": "(seed, split, bootstrap, replicate, scenario)",
            "calibration_holdout_independence": (
                "the split label is part of every stochastic namespace"
            ),
        },
        "smart_configs_by_effective_markers": configs,
        "resources": {
            "affinity_cpus": affinity,
            "outer_processes": args.processes,
            "numba_threads_per_process": args.threads_per_process,
            "aggregate_process_thread_budget": (
                args.processes * args.threads_per_process
            ),
            "inference_workers_per_outer_process": 1,
        },
        "source_sha256": _source_hashes(root),
    }


def _identity_for_spec(spec):
    return {
        key: spec[key]
        for key in (
            "seed", "split", "replicate", "scenario", "effective_markers",
            "bootstrap_replicates", "contigs", "bins", "snps_per_bin",
            "state_profile", "state_profile_counts", "low_parent_depth",
        )
    }


def _load_unit(path, manifest_hash, identity):
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("manifest_sha256") != manifest_hash
        or value.get("unit_identity") != identity
    ):
        raise ValueError(f"checkpoint identity mismatch: {path}")
    return value


def _output_identity(path, root):
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _validate_completion(root, manifest_hash):
    status_path = root / "status.computational_complete.json"
    if not status_path.exists():
        return False
    with status_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if status.get("manifest_sha256") != manifest_hash:
        raise ValueError("completion manifest identity mismatch")
    for identity in status.get("outputs", {}).values():
        path = root / identity["path"]
        if (
            not path.is_file()
            or path.stat().st_size != identity["size_bytes"]
            or _sha256(path) != identity["sha256"]
        ):
            raise ValueError(f"completed output changed or missing: {path}")
    return True


def run(args):
    scenarios, effective_markers, affinity = _validated_args(args)
    manifest = _manifest(args, scenarios, effective_markers, affinity)
    manifest_hash = _canonical_hash(manifest)
    state_profile_counts = _state_profile_counts(args.state_profile)
    output = args.output_dir.resolve()
    if output == Path.cwd().resolve():
        raise ValueError("output-dir must be an isolated directory")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError("nonempty output requires --resume")
        persisted_manifest = json.loads(json.dumps(
            manifest,
            sort_keys=True,
            allow_nan=False,
        ))
        with (output / "manifest.json").open("r", encoding="utf-8") as handle:
            if json.load(handle) != persisted_manifest:
                raise ValueError("resume manifest mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "manifest.json", manifest)
    units_root = output / "units"
    units_root.mkdir(exist_ok=True)
    if args.resume and _validate_completion(output, manifest_hash):
        print(f"[resume] validated complete: {output}")
        return output

    replicate_ids = [
        args.replicate_start + index * args.replicate_step
        for index in range(args.replicates)
    ]
    specs = []
    records = []
    resumed = 0
    for replicate in replicate_ids:
        for scenario in scenarios:
            for effective in effective_markers:
                path = units_root / _unit_name(
                    args.split, scenario, effective, replicate
                )
                spec = {
                    "seed": args.seed,
                    "split": args.split,
                    "replicate": replicate,
                    "scenario": scenario,
                    "state_profile": args.state_profile,
                    "state_profile_counts": state_profile_counts,
                    "low_parent_depth": args.low_parent_depth,
                    "effective_markers": effective,
                    "bootstrap_replicates": args.bootstrap_replicates,
                    "contigs": args.contigs,
                    "bins": args.bins,
                    "snps_per_bin": args.snps_per_bin,
                    "threads_per_process": args.threads_per_process,
                    "manifest_sha256": manifest_hash,
                    "unit_path": str(path),
                }
                spec["unit_identity"] = _identity_for_spec(spec)
                if args.resume and path.exists():
                    unit = _load_unit(path, manifest_hash, spec["unit_identity"])
                    records.extend(unit["records"])
                    resumed += 1
                else:
                    specs.append(spec)

    started = time.perf_counter()
    computed = 0
    maximum_workers = min(args.processes, max(1, len(specs)))
    if maximum_workers == 1:
        for index, spec in enumerate(specs, start=1):
            unit = _run_and_checkpoint(spec)
            records.extend(unit["records"])
            computed += 1
            print(
                f"[{index}/{len(specs)}] {spec['scenario']} "
                f"e={spec['effective_markers']:g} replicate={spec['replicate']}",
                flush=True,
            )
    elif specs:
        context = mp.get_context("forkserver")
        with ProcessPoolExecutor(
            max_workers=maximum_workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(_run_and_checkpoint, spec): spec
                for spec in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                unit = future.result()
                records.extend(unit["records"])
                computed += 1
                print(
                    f"[{computed}/{len(specs)}] {spec['scenario']} "
                    f"e={spec['effective_markers']:g} "
                    f"replicate={spec['replicate']}",
                    flush=True,
                )

    records.sort(key=lambda row: (
        row["replicate"], row["scenario"], row["effective_markers"],
        row["family"], row["within_family"],
    ))
    aggregate, confusion = _summaries(records, scenarios, effective_markers)
    per_child_path = output / "per_child_release_metrics.csv"
    aggregate_path = output / "aggregate_metrics.csv"
    confusion_path = output / "raw_state_confusion.csv"
    summary_path = output / "summary.json"
    _atomic_csv(per_child_path, records)
    _atomic_csv(aggregate_path, aggregate)
    _atomic_csv(confusion_path, confusion)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "split": args.split,
        "replicate_ids": replicate_ids,
        "scenarios": list(scenarios),
        "state_profile": args.state_profile,
        "state_profile_counts": state_profile_counts,
        "low_parent_depth": args.low_parent_depth,
        "effective_markers": list(effective_markers),
        "cohort_units": len(replicate_ids) * len(scenarios) * len(effective_markers),
        "child_records": len(records),
        "units_computed_this_invocation": computed,
        "units_resumed_this_invocation": resumed,
        "elapsed_seconds_this_invocation": elapsed,
        "production_eligible": False,
        "interpretation": (
            "Actual current Tier A/B release outputs under known simulation "
            "truth; calibration and holdout splits must be analysed separately."
        ),
    }
    _atomic_json(summary_path, summary)

    unit_identities = [
        (str(path.relative_to(output)), _sha256(path))
        for path in sorted(units_root.glob("*.json"))
    ]
    unit_digest = hashlib.sha256(
        "\n".join(f"{path} {digest}" for path, digest in unit_identities).encode()
    ).hexdigest()
    output_paths = (
        output / "manifest.json",
        per_child_path,
        aggregate_path,
        confusion_path,
        summary_path,
    )
    status = {
        "computational_complete": True,
        "attempt_identity": ATTEMPT_IDENTITY,
        "manifest_sha256": manifest_hash,
        "unit_count": len(unit_identities),
        "unit_sha256_digest": unit_digest,
        "outputs": {
            path.name: _output_identity(path, output) for path in output_paths
        },
    }
    _atomic_json(output / "status.computational_complete.json", status)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def main(argv=None):
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
