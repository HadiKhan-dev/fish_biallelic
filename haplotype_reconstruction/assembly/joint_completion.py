"""assembly / joint completion for the canonical reconstruction pipeline."""
from __future__ import annotations


from dataclasses import dataclass
import hashlib
import math
import numpy as np

from . import joint_statistics


@dataclass(frozen=True)
class HardFounderPanel:
    """Founder alleles with ``-1`` unresolved and hard calls in ``{0, 1}``."""

    positions: np.ndarray
    keys: tuple[object, ...]
    alleles: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions)
        alleles = np.asarray(self.alleles, dtype=np.int8)
        if alleles.ndim != 2:
            raise ValueError("alleles must have shape (founders, sites)")
        if positions.shape != (alleles.shape[1],):
            raise ValueError("positions must match the site dimension")
        if len(self.keys) != alleles.shape[0]:
            raise ValueError("keys must match the founder dimension")
        if np.any(~np.isin(alleles, (-1, 0, 1))):
            raise ValueError("alleles must contain only -1, 0, or 1")
        object.__setattr__(self, "positions", positions.copy())
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "alleles", alleles.copy())

    @property
    def called(self) -> np.ndarray:
        return self.alleles >= 0

    @property
    def n_founders(self) -> int:
        return int(self.alleles.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.alleles.shape[1])


@dataclass(frozen=True)
class JointBlockConfig:
    """Numerical, convergence, and conservative release controls."""

    max_unresolved_founders: int = 6
    max_iterations: int = 100
    minimum_iterations: int = 2
    elbo_absolute_tolerance: float = 1e-8
    elbo_relative_tolerance: float = 1e-8
    monotonic_tolerance: float = 1e-8
    uniform_mix: float = 0.01
    log_floor: float = -2.0
    num_starts: int = 5
    initialization_logit_scale: float = 1.0
    deterministic_seed: int = 1729
    minimum_call_probability: float = 0.90
    minimum_effective_carriers: float = 2.0

    def __post_init__(self) -> None:
        if not 0 <= self.max_unresolved_founders <= 20:
            raise ValueError("max_unresolved_founders must lie in [0, 20]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not 1 <= self.minimum_iterations <= self.max_iterations:
            raise ValueError("minimum_iterations must lie in [1, max_iterations]")
        if self.elbo_absolute_tolerance < 0 or self.elbo_relative_tolerance < 0:
            raise ValueError("ELBO tolerances must be non-negative")
        if self.monotonic_tolerance < 0:
            raise ValueError("monotonic_tolerance must be non-negative")
        if not 0 <= self.uniform_mix <= 1:
            raise ValueError("uniform_mix must lie in [0, 1]")
        if self.log_floor > 0:
            raise ValueError("log_floor must be non-positive")
        if self.num_starts < 1:
            raise ValueError("num_starts must be positive")
        if self.initialization_logit_scale < 0:
            raise ValueError("initialization_logit_scale must be non-negative")
        if not 0.5 < self.minimum_call_probability <= 1:
            raise ValueError("minimum_call_probability must lie in (0.5, 1]")
        if self.minimum_effective_carriers < 0:
            raise ValueError("minimum_effective_carriers must be non-negative")


@dataclass(frozen=True)
class StartDiagnostics:
    label: str
    converged: bool
    iterations: int
    elbo_history: np.ndarray
    minimum_elbo_increment: float


@dataclass(frozen=True)
class JointBlockFit:
    """Best variational fit plus all-start diagnostics."""

    panel: HardFounderPanel
    pairs: np.ndarray
    diplotype_probabilities: np.ndarray
    site_configurations: tuple[np.ndarray | None, ...]
    site_configuration_probabilities: tuple[np.ndarray | None, ...]
    marginal_alt_probability: np.ndarray
    enumerated_sites: np.ndarray
    unresolved_founders_per_site: np.ndarray
    effective_carriers: np.ndarray
    exchangeable_groups: tuple[tuple[int, ...], ...]
    exchangeable_founders: np.ndarray
    best_start: int
    starts: tuple[StartDiagnostics, ...]
    elbo: float


@dataclass(frozen=True)
class ReleasedPanel:
    panel: HardFounderPanel
    released: np.ndarray
    posterior_confidence: np.ndarray
    effective_carriers: np.ndarray
    exchangeable_skipped: np.ndarray
    cap_skipped: np.ndarray


@dataclass(frozen=True)
class CarrierProfiles:
    expected_copies: np.ndarray
    carrier_probability: np.ndarray
    informative_samples: np.ndarray
    informative_site_count: np.ndarray


@dataclass(frozen=True)
class CrossFitFold:
    fold: int
    training_indices: np.ndarray
    heldout_indices: np.ndarray
    fit: JointBlockFit
    released: ReleasedPanel


@dataclass(frozen=True)
class CrossFitBlock:
    profiles: CarrierProfiles
    folds: tuple[CrossFitFold, ...]


def _unordered_pairs(k: int) -> np.ndarray:
    return np.asarray(
        [(first, second) for first in range(k) for second in range(first, k)],
        dtype=np.int64,
    )


def _normalise_evidence(evidence: np.ndarray) -> np.ndarray:
    values = np.asarray(evidence, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("genotype evidence must have shape (samples, sites, 3)")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("genotype evidence must be finite and non-negative")
    totals = np.sum(values, axis=2, keepdims=True)
    result = np.full(values.shape, 1.0 / 3.0, dtype=np.float64)
    np.divide(values, totals, out=result, where=totals > 0)
    return result


def _prepare_inputs(
    panel: HardFounderPanel,
    evidence: np.ndarray,
    observed: np.ndarray | None,
    config: JointBlockConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evidence = _normalise_evidence(evidence)
    if evidence.shape[1] != panel.n_sites:
        raise ValueError("genotype evidence sites must match the panel")
    if observed is None:
        observed_mask = np.ones(evidence.shape[:2], dtype=np.bool_)
    else:
        observed_mask = np.asarray(observed, dtype=np.bool_)
        if observed_mask.shape != evidence.shape[:2]:
            raise ValueError("observed must have shape (samples, sites)")
    robust = (1.0 - config.uniform_mix) * evidence + config.uniform_mix / 3.0
    with np.errstate(divide="ignore"):
        log_emission = np.log(robust)
    np.maximum(log_emission, config.log_floor, out=log_emission)
    log_emission[~observed_mask] = 0.0
    return evidence, observed_mask, log_emission


def _softmax(log_values: np.ndarray, axis: int = -1) -> np.ndarray:
    maximum = np.max(log_values, axis=axis, keepdims=True)
    values = np.exp(log_values - maximum)
    values /= np.sum(values, axis=axis, keepdims=True)
    return values


def _entropy_term(probability: np.ndarray, log_prior: np.ndarray | float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probability = np.log(probability)
    terms = np.where(probability > 0, probability * (log_prior - log_probability), 0.0)
    return float(np.sum(terms))


def _stable_unit_interval(key: object) -> float:
    digest = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, "little")
    return (integer + 0.5) / float(2**64)


def _initial_logit(
    config: JointBlockConfig, start: int, key: object, position: object
) -> float:
    if start == 0:
        return 0.0
    family = (start + 1) // 2
    sign = 1.0 if start % 2 else -1.0
    token = (config.deterministic_seed, family, key, position)
    centered = 2.0 * _stable_unit_interval(token) - 1.0
    return sign * config.initialization_logit_scale * centered


def _configurations_for_site(
    panel: HardFounderPanel, site: int, config: JointBlockConfig
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    unknown = np.flatnonzero(panel.alleles[:, site] < 0)
    if unknown.size > config.max_unresolved_founders:
        return None, None, unknown
    n_configurations = 1 << int(unknown.size)
    configurations = np.empty((n_configurations, panel.n_founders), dtype=np.int8)
    fixed = panel.alleles[:, site]
    for index in range(n_configurations):
        configurations[index] = fixed
        for offset, founder in enumerate(unknown):
            configurations[index, founder] = (index >> offset) & 1
    return configurations, np.full(n_configurations, -unknown.size * math.log(2.0)), unknown


def _configuration_dosages(configurations: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return configurations[:, pairs[:, 0]] + configurations[:, pairs[:, 1]]


def _exchangeable_groups(panel: HardFounderPanel) -> tuple[tuple[int, ...], ...]:
    groups: list[list[int]] = []
    for founder in range(panel.n_founders):
        for group in groups:
            representative = group[0]
            same_called = np.array_equal(
                panel.called[founder], panel.called[representative]
            )
            if same_called and np.array_equal(
                panel.alleles[founder], panel.alleles[representative]
            ):
                group.append(founder)
                break
        else:
            groups.append([founder])
    return tuple(tuple(group) for group in groups if len(group) > 1)


def _site_initial_probability(
    panel: HardFounderPanel,
    site: int,
    configurations: np.ndarray,
    unknown: np.ndarray,
    config: JointBlockConfig,
    start: int,
) -> np.ndarray:
    if unknown.size == 0 or start == 0:
        return np.full(configurations.shape[0], 1.0 / configurations.shape[0])
    logits = np.asarray(
        [_initial_logit(config, start, panel.keys[k], panel.positions[site]) for k in unknown]
    )
    probability = 1.0 / (1.0 + np.exp(-logits))
    joint = np.ones(configurations.shape[0], dtype=np.float64)
    for offset, founder in enumerate(unknown):
        allele = configurations[:, founder]
        joint *= np.where(allele == 1, probability[offset], 1.0 - probability[offset])
    joint /= np.sum(joint)
    return joint


def _complete_site_state_scores(
    panel: HardFounderPanel, log_emission: np.ndarray, observed: np.ndarray, pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    complete = np.all(panel.called, axis=0)
    usable = complete[None, :] & observed
    scores = np.zeros((log_emission.shape[0], pairs.shape[0]), dtype=np.float64)
    for state, (first, second) in enumerate(pairs):
        dosage = panel.alleles[first] + panel.alleles[second]
        chosen = np.take_along_axis(log_emission, dosage[None, :, None], axis=2)[..., 0]
        scores[:, state] = np.sum(chosen * usable, axis=1)
    return scores, np.sum(usable, axis=1, dtype=np.int64)


def _site_probabilities(probabilities, offsets):
    return tuple(
        probabilities[start:stop] if stop > start else None
        for start, stop in zip(offsets[:-1], offsets[1:])
    )


def _elbo(expected_scores, log_priors, site_probabilities, state_probability):
    # Reuse the same expected scores that updated z; q has not changed.
    value = float(np.sum(state_probability * expected_scores))
    value += _entropy_term(state_probability, -math.log(state_probability.shape[1]))
    for log_prior, probability in zip(log_priors, site_probabilities):
        # A fully known site has q=[1] and log_prior=[0], hence zero entropy.
        if probability is not None and len(probability) > 1:
            value += _entropy_term(probability, log_prior)
    return value


def _fit_one_start(
    panel: HardFounderPanel,
    pairs: np.ndarray,
    configurations: tuple[np.ndarray | None, ...],
    log_priors: tuple[np.ndarray | None, ...],
    unknown_by_site: tuple[np.ndarray, ...],
    config: JointBlockConfig,
    start: int,
    statistics_workspace,
    complete_scores: np.ndarray,
    shared_result=None,
) -> tuple[np.ndarray, tuple[np.ndarray | None, ...], StartDiagnostics]:
    evidence, alleles, offsets, packed_prior = statistics_workspace
    state_probability = _softmax(complete_scores - math.log(pairs.shape[0]), axis=1)
    initial = [
        _site_initial_probability(
            panel, site, values, unknown_by_site[site], config, start)
        for site, values in enumerate(configurations) if values is not None
    ]
    probability = np.concatenate(initial) if initial else np.empty(0)
    state_scores = evidence @ joint_statistics.dosage_marginals(
        alleles, pairs, offsets, probability)
    site_probability = _site_probabilities(probability, offsets)
    elbo_history = [_elbo(state_scores, log_priors, site_probability, state_probability)]
    if shared_result is not None:
        # All starts initialize the same z and update q before z. From the
        # first update onward their trajectories are therefore identical.
        # With minimum_iterations >= 2 only the initial ELBO/check is unique;
        # minimum-one starts retain the independent path below because their
        # first increment can make them stop at different iterations.
        state_probability, site_probability, shared_diagnostics = shared_result
        increment = shared_diagnostics.elbo_history[1] - elbo_history[0]
        if increment < -config.monotonic_tolerance:
            raise RuntimeError(
                f"ELBO decreased by {increment:.6g} at start {start}, iteration 1"
            )
        history = np.r_[elbo_history, shared_diagnostics.elbo_history[1:]]
        diagnostics = StartDiagnostics(
            label=f"paired_jitter_{(start + 1) // 2}_{'plus' if start % 2 else 'minus'}",
            converged=shared_diagnostics.converged,
            iterations=shared_diagnostics.iterations,
            elbo_history=history,
            minimum_elbo_increment=float(np.min(np.diff(history))),
        )
        return state_probability, site_probability, diagnostics
    converged = False
    for iteration in range(1, config.max_iterations + 1):
        # All site-q updates condition on the same z. Their three-dosage
        # statistics can therefore be contracted together without changing
        # the coordinate-update schedule. BLAS remains worker-local/one thread.
        statistics = state_probability.T @ evidence
        probability = joint_statistics.configuration_probabilities(
            statistics, alleles, pairs, offsets, packed_prior)
        state_scores = evidence @ joint_statistics.dosage_marginals(
            alleles, pairs, offsets, probability)
        state_probability = _softmax(
            state_scores - math.log(pairs.shape[0]), axis=1
        )
        site_probability = _site_probabilities(probability, offsets)
        value = _elbo(state_scores, log_priors, site_probability, state_probability)
        increment = value - elbo_history[-1]
        if increment < -config.monotonic_tolerance:
            raise RuntimeError(
                f"ELBO decreased by {increment:.6g} at start {start}, iteration {iteration}"
            )
        elbo_history.append(value)
        threshold = config.elbo_absolute_tolerance + (
            config.elbo_relative_tolerance * abs(elbo_history[-2])
        )
        if iteration >= config.minimum_iterations and abs(increment) <= threshold:
            converged = True
            break
    history = np.asarray(elbo_history, dtype=np.float64)
    diagnostics = StartDiagnostics(
        label=("neutral" if start == 0 else f"paired_jitter_{(start + 1) // 2}_{'plus' if start % 2 else 'minus'}"),
        converged=converged,
        iterations=len(history) - 1,
        elbo_history=history,
        minimum_elbo_increment=(
            float(np.min(np.diff(history))) if history.size > 1 else math.inf
        ),
    )
    return state_probability, site_probability, diagnostics


def fit_joint_block(
    panel: HardFounderPanel,
    genotype_evidence: np.ndarray,
    observed: np.ndarray | None = None,
    config: JointBlockConfig = JointBlockConfig(),
) -> JointBlockFit:
    """Fit the bounded joint-allele variational model with deterministic starts."""

    _, observed_mask, log_emission = _prepare_inputs(
        panel, genotype_evidence, observed, config
    )
    pairs = _unordered_pairs(panel.n_founders)
    configurations: list[np.ndarray | None] = []
    log_priors: list[np.ndarray | None] = []
    unknown_by_site: list[np.ndarray] = []
    unresolved_count = np.empty(panel.n_sites, dtype=np.int64)
    for site in range(panel.n_sites):
        site_configurations, log_prior, unknown = _configurations_for_site(
            panel, site, config
        )
        configurations.append(site_configurations)
        log_priors.append(log_prior)
        unknown_by_site.append(unknown)
        unresolved_count[site] = unknown.size
    configuration_tuple = tuple(configurations)
    log_prior_tuple = tuple(log_priors)
    unknown_tuple = tuple(unknown_by_site)

    statistics_workspace = joint_statistics.prepare(
        configuration_tuple, log_prior_tuple, log_emission, panel.n_founders)
    complete_scores, _ = _complete_site_state_scores(panel, log_emission, observed_mask, pairs)
    state_results: list[np.ndarray] = []
    site_results: list[tuple[np.ndarray | None, ...]] = []
    diagnostics: list[StartDiagnostics] = []
    shared_result = None
    for start in range(config.num_starts):
        state, site, start_diagnostics = _fit_one_start(
            panel, pairs, configuration_tuple, log_prior_tuple, unknown_tuple,
            config, start, statistics_workspace, complete_scores,
            shared_result=shared_result,
        )
        state_results.append(state)
        site_results.append(site)
        diagnostics.append(start_diagnostics)
        if start == 0 and config.minimum_iterations >= 2:
            shared_result = state, site, start_diagnostics
    final_elbos = np.asarray([value.elbo_history[-1] for value in diagnostics])
    best = int(np.argmax(final_elbos))
    best_state = state_results[best]
    best_sites = site_results[best]

    marginal = np.full(panel.alleles.shape, 0.5, dtype=np.float64)
    marginal[panel.called] = panel.alleles[panel.called]
    enumerated = np.zeros(panel.n_sites, dtype=np.bool_)
    for site, (site_configurations, probability, unknown) in enumerate(
        zip(configuration_tuple, best_sites, unknown_tuple)
    ):
        if site_configurations is None or probability is None:
            continue
        enumerated[site] = True
        for founder in unknown:
            marginal[founder, site] = float(
                np.sum(probability * site_configurations[:, founder])
            )

    effective_carriers = np.zeros(panel.n_founders, dtype=np.float64)
    for state, (first, second) in enumerate(pairs):
        mass = best_state[:, state]
        effective_carriers[first] += np.sum(mass)
        if second != first:
            effective_carriers[second] += np.sum(mass)
    groups = _exchangeable_groups(panel)
    exchangeable = np.zeros(panel.n_founders, dtype=np.bool_)
    for group in groups:
        exchangeable[np.asarray(group, dtype=np.int64)] = True
    return JointBlockFit(
        panel=panel,
        pairs=pairs,
        diplotype_probabilities=best_state,
        site_configurations=configuration_tuple,
        site_configuration_probabilities=best_sites,
        marginal_alt_probability=marginal,
        enumerated_sites=enumerated,
        unresolved_founders_per_site=unresolved_count,
        effective_carriers=effective_carriers,
        exchangeable_groups=groups,
        exchangeable_founders=exchangeable,
        best_start=best,
        starts=tuple(diagnostics),
        elbo=float(final_elbos[best]),
    )


def release_joint_calls(
    fit: JointBlockFit,
    config: JointBlockConfig = JointBlockConfig(),
) -> ReleasedPanel:
    """Release conservative hard calls; posterior probabilities stay separate."""

    alleles = fit.panel.alleles.copy()
    initially_unknown = alleles < 0
    confidence = np.maximum(
        fit.marginal_alt_probability, 1.0 - fit.marginal_alt_probability
    )
    cap_skipped = initially_unknown & ~fit.enumerated_sites[None, :]
    exchangeable_skipped = initially_unknown & fit.exchangeable_founders[:, None]
    eligible = (
        initially_unknown
        & fit.enumerated_sites[None, :]
        & ~exchangeable_skipped
        & (fit.effective_carriers[:, None] >= config.minimum_effective_carriers)
        & (confidence >= config.minimum_call_probability)
    )
    alleles[eligible] = (
        fit.marginal_alt_probability[eligible] > 0.5
    ).astype(np.int8)
    released = initially_unknown & (alleles >= 0)
    return ReleasedPanel(
        panel=HardFounderPanel(fit.panel.positions, fit.panel.keys, alleles),
        released=released,
        posterior_confidence=confidence,
        effective_carriers=fit.effective_carriers.copy(),
        exchangeable_skipped=exchangeable_skipped,
        cap_skipped=cap_skipped,
    )


def infer_from_complete_sites(
    panel: HardFounderPanel,
    genotype_evidence: np.ndarray,
    observed: np.ndarray | None = None,
    config: JointBlockConfig = JointBlockConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coherent baseline posterior using the same complete sites for all states."""

    _, observed_mask, log_emission = _prepare_inputs(
        panel, genotype_evidence, observed, config
    )
    pairs = _unordered_pairs(panel.n_founders)
    scores, informative_count = _complete_site_state_scores(
        panel, log_emission, observed_mask, pairs
    )
    probabilities = _softmax(scores - math.log(pairs.shape[0]), axis=1)
    return pairs, probabilities, informative_count


def founder_profiles(
    pairs: np.ndarray,
    probabilities: np.ndarray,
    informative_site_count: np.ndarray,
    n_founders: int,
) -> CarrierProfiles:
    copies = np.zeros((probabilities.shape[0], n_founders), dtype=np.float64)
    carriers = np.zeros_like(copies)
    for state, (first, second) in enumerate(pairs):
        mass = probabilities[:, state]
        if first == second:
            copies[:, first] += 2.0 * mass
            carriers[:, first] += mass
        else:
            copies[:, first] += mass
            copies[:, second] += mass
            carriers[:, first] += mass
            carriers[:, second] += mass
    # These are exact expectations/probabilities mathematically. Summing the
    # normalized state masses can nevertheless leave a one-ULP excursion such
    # as 1.0000000000000002, which downstream validation must not mistake for
    # a material model failure.
    np.clip(copies, 0.0, 2.0, out=copies)
    np.clip(carriers, 0.0, 1.0, out=carriers)
    informative_count = np.asarray(informative_site_count, dtype=np.int64)
    return CarrierProfiles(
        expected_copies=copies,
        carrier_probability=carriers,
        informative_samples=informative_count > 0,
        informative_site_count=informative_count,
    )


def deterministic_two_folds(n_samples: int, seed: int = 1729) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("two-fold cross-fitting requires at least two samples")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    folds = np.empty(n_samples, dtype=np.int8)
    folds[order] = np.arange(n_samples, dtype=np.int64) % 2
    return folds


def crossfit_block(
    panel: HardFounderPanel,
    genotype_evidence: np.ndarray,
    observed: np.ndarray | None = None,
    *,
    fold_assignments: np.ndarray | None = None,
    config: JointBlockConfig = JointBlockConfig(),
) -> CrossFitBlock:
    """Learn calls on one fold and paint only the complementary held-out fold."""

    evidence = np.asarray(genotype_evidence)
    if evidence.ndim != 3 or evidence.shape[1] != panel.n_sites:
        raise ValueError("genotype evidence must have shape (samples, sites, 3)")
    if observed is None:
        observed_mask = np.ones(evidence.shape[:2], dtype=np.bool_)
    else:
        observed_mask = np.asarray(observed, dtype=np.bool_)
        if observed_mask.shape != evidence.shape[:2]:
            raise ValueError("observed must have shape (samples, sites)")
    if fold_assignments is None:
        folds = deterministic_two_folds(evidence.shape[0], config.deterministic_seed)
    else:
        folds = np.asarray(fold_assignments, dtype=np.int8)
        if folds.shape != (evidence.shape[0],) or set(folds.tolist()) != {0, 1}:
            raise ValueError("fold_assignments must contain both 0 and 1")

    n_states = panel.n_founders * (panel.n_founders + 1) // 2
    heldout_probability = np.empty((evidence.shape[0], n_states), dtype=np.float64)
    informative_count = np.zeros(evidence.shape[0], dtype=np.int64)
    results: list[CrossFitFold] = []
    reference_pairs: np.ndarray | None = None
    for fold in (0, 1):
        heldout = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        fit = fit_joint_block(
            panel, evidence[training], observed_mask[training], config
        )
        released = release_joint_calls(fit, config)
        pairs, probability, count = infer_from_complete_sites(
            released.panel, evidence[heldout], observed_mask[heldout], config
        )
        if reference_pairs is None:
            reference_pairs = pairs
        elif not np.array_equal(reference_pairs, pairs):
            raise RuntimeError("unordered state order changed between folds")
        heldout_probability[heldout] = probability
        informative_count[heldout] = count
        results.append(CrossFitFold(fold, training, heldout, fit, released))
    assert reference_pairs is not None
    return CrossFitBlock(
        founder_profiles(
            reference_pairs, heldout_probability, informative_count, panel.n_founders
        ),
        tuple(results),
    )
