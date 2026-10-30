"""Full-marker Mendelian exclusion and the fixed-observed-genome M0/M1 count release.

At each site jointly observed in a dyad, A is the normalised likelihood of the two incompatible dyad
genotypes ((x0 y2 + x2 y0) / 2), C the maximum over the seven compatible genotype pairs and U the
unrestricted maximum; D = (1 - delta) C + delta U bounds every compatible read model with at most
delta arbitrary genotype replacement, so A / D has null expectation at most one under calibrated
read likelihoods. Products of convex bets 1 - b + b A / D keep that bound conditional on fixed
genotypes and independent read errors (SNP inheritance need not be independent). Windows, bets and
chromosomes are averaged (never maximised). The evidence is symmetric; the release uses it only for
children whose Tier-B count is unresolved, never adds an edge, and is additionally conditional on
the already supported family directions (a supported outgoing edge rejects the reverse candidate).
Mapping and ascertainment errors remain limitations; QC-flagged samples are scored as in base.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .config import ExclusionConfig, PedigreeConfig
from .eligibility import ParentEligibility
from .records import NO_PARENT, UNRESOLVED, ExclusionEvidence
from .release import PartialEdges, TierCalls

# Exact repeated-GL-triple reuse: at most this many distinct triples per sample (temporary memory,
# ~49 MiB per dyad lookup at worst); samples with more use the direct per-site scorer.
MAX_PATTERNS = 1024
# Dyads per batch: each batch boundary lets a chromosome worker's CPU lease grow.
DYAD_BATCH = 256


def physical_windows(positions: np.ndarray, widths: tuple[int, ...]) -> np.ndarray:
    """int64 [W, 2] site ranges: the whole chromosome plus half-overlapping windows of each width."""
    spans = [(0, len(positions))]
    if len(positions):
        for width in widths:
            for start in range(0, int(positions[-1]) + 1, width // 2):
                a, b = np.searchsorted(positions, [start, start + width])
                if b > a:
                    spans.append((int(a), int(b)))
    return np.asarray(sorted(set(spans)), dtype=np.int64)


@njit(cache=True, inline="always")
def _ratio(x0, x1, x2, y0, y1, y2, delta):
    """A / D of one jointly observed site (see the module docstring)."""
    compatible = max(x0 * max(y0, y1), x1 * max(y0, y1, y2), x2 * max(y1, y2))
    unrestricted = max(x0, x1, x2) * max(y0, y1, y2)
    bound = max(compatible, (1 - delta) * compatible + delta * unrestricted)
    return (x0 * y2 + x2 * y0) / 2 / max(bound, 1e-300)


@njit(cache=True, inline="always")
def _log_bet(ratio, bet):
    return np.log(max(1 - bet + bet * ratio, 1e-300))


@njit(cache=True, parallel=True)
def _score_direct(gl, observed, rows, spans, delta, bets):
    """[D, 2]: window/bet-averaged log E and jointly observed sites, site by site."""
    output = np.empty((len(rows), 2))
    for row in prange(len(rows)):
        a, b = rows[row]
        ratios = np.ones(gl.shape[1])
        exposure = 0
        for site in range(gl.shape[1]):
            if not observed[a, site] or not observed[b, site]:
                continue
            exposure += 1
            ratios[site] = _ratio(gl[a, site, 0], gl[a, site, 1], gl[a, site, 2],
                                  gl[b, site, 0], gl[b, site, 1], gl[b, site, 2], delta)
        combined = -np.inf
        prefix = np.empty(gl.shape[1] + 1)
        prefix[0] = 0.0
        for bet in bets:
            for site in range(gl.shape[1]):
                prefix[site + 1] = prefix[site] + _log_bet(ratios[site], bet)
            for start, stop in spans:
                combined = np.logaddexp(combined, prefix[stop] - prefix[start])
        output[row, 0] = combined - np.log(len(bets) * len(spans))
        output[row, 1] = exposure
    return output


@njit(cache=True, parallel=True)
def _encode_patterns(gl, observed, cap):
    """Exact per-sample codes of distinct GL triples (-1 unobserved); count -1 = over the cap."""
    samples, sites, _ = gl.shape
    codes = np.empty((samples, sites), np.int16)
    catalog = np.empty((samples, cap, 3), gl.dtype)
    counts = np.zeros(samples, np.int64)
    for sample in prange(samples):
        # Establish the dictionary's key/value types without quantising floats.
        table = {(0., 0., 0.): -1}
        del table[(0., 0., 0.)]
        count = 0
        for site in range(sites):
            if not observed[sample, site]:
                codes[sample, site] = -1
                continue
            x0, x1, x2 = gl[sample, site]
            key = (float(x0), float(x1), float(x2))
            code = table.get(key, -1)
            if code < 0:
                if count == cap:
                    count = -1
                    break
                code = count
                table[key] = code
                catalog[sample, code] = gl[sample, site]
                count += 1
            codes[sample, site] = code
        counts[sample] = count
    return codes, catalog, counts


@njit(cache=True, parallel=True)
def _score_patterns(codes, catalog, counts, rows, spans, delta, bets):
    """Same values as _score_direct for dyads of two encoded samples, reusing each pattern pair."""
    output = np.empty((len(rows), 2))
    for row in prange(len(rows)):
        a, b = rows[row]
        width = counts[b]
        table = np.empty((counts[a] * width, len(bets)))
        seen = np.zeros(counts[a] * width, np.bool_)
        prefix = np.empty(codes.shape[1] + 1)
        prefix[0] = 0.
        exposure = 0
        combined = -np.inf
        for bet_index in range(len(bets)):
            for site in range(codes.shape[1]):
                ca, cb = codes[a, site], codes[b, site]
                value = 0.
                if ca >= 0 and cb >= 0:
                    pair = ca * width + cb
                    if bet_index == 0:
                        exposure += 1
                    if not seen[pair]:
                        ratio = _ratio(catalog[a, ca, 0], catalog[a, ca, 1], catalog[a, ca, 2],
                                       catalog[b, cb, 0], catalog[b, cb, 1], catalog[b, cb, 2],
                                       delta)
                        for index in range(len(bets)):
                            table[pair, index] = _log_bet(ratio, bets[index])
                        seen[pair] = True
                    value = table[pair, bet_index]
                prefix[site + 1] = prefix[site] + value
            for start, stop in spans:
                combined = np.logaddexp(combined, prefix[stop] - prefix[start])
        output[row, 0] = combined - np.log(len(bets) * len(spans))
        output[row, 1] = exposure
    return output


def target_dyads(tier_b: TierCalls, rows: np.ndarray, eligibility: ParentEligibility) -> np.ndarray:
    """int64 [D, 2] (child, candidate): every eligible candidate of each child whose Tier-B count
    is unresolved (no state, or fewer released parents than the called count)."""
    dyads = []
    for child in np.flatnonzero(eligibility.children):
        row = int(tier_b.row[child])
        named = 0 if row < 0 else int(np.count_nonzero(rows[row, 1:] >= 0))
        state = int(tier_b.state[child])
        if state == UNRESOLVED or named != state:
            dyads.extend((int(child), int(parent))
                         for parent in np.flatnonzero(eligibility.parents[child]))
    return np.asarray(dyads, dtype=np.int64).reshape(-1, 2)


def score_contig(contig: str, likelihoods: np.ndarray, observed: np.ndarray,
                 positions: np.ndarray, dyads: np.ndarray, config: ExclusionConfig,
                 claim_threads=lambda: None) -> ExclusionEvidence:
    """Exclusion evidence of one chromosome for the target dyads (all markers)."""
    spans = physical_windows(positions, config.window_widths_bp)
    bets = np.asarray(config.bet_sizes, dtype=np.float64)
    delta = float(config.genotype_replacement_probability)
    claim_threads()
    codes, catalog, counts = _encode_patterns(likelihoods, observed, MAX_PATTERNS)
    values = np.empty((len(dyads), 2))
    for start in range(0, len(dyads), DYAD_BATCH):
        claim_threads()
        batch = dyads[start:start + DYAD_BATCH]
        encoded = (counts[batch[:, 0]] >= 0) & (counts[batch[:, 1]] >= 0)
        batch_values = values[start:start + DYAD_BATCH]
        if encoded.any():
            batch_values[encoded] = _score_patterns(codes, catalog, counts, batch[encoded], spans,
                                                    delta, bets)
        if not encoded.all():
            batch_values[~encoded] = _score_direct(likelihoods, observed, batch[~encoded], spans,
                                                   delta, bets)
    return ExclusionEvidence(contig, dyads, values[:, 0].copy(), values[:, 1].astype(np.int64),
                             int(len(positions)), int(np.count_nonzero(counts < 0)))


def combine(evidence: list[ExclusionEvidence]) -> tuple[np.ndarray, np.ndarray]:
    """(log E [D] averaged over all configured chromosomes, jointly observed sites [D])."""
    log_e = (np.logaddexp.reduce(np.stack([value.log_e for value in evidence]), axis=0)
             - np.log(len(evidence)))
    observed = np.sum([value.observed_sites for value in evidence], axis=0).astype(np.int64)
    return log_e, observed


@dataclass(frozen=True)
class CountRelease:
    """Result of the full-marker release.

    tier_b / partial_b: the final Tier-B views (released children: row = the complete graph row,
    status "tier_b_fixed_genome_directed_exclusion", release evidence
    "mendelian_exclusion_and_supported_direction"); released bool [N]; kept: per child the
    remaining candidate (0 or 1 sample index). ``records``: one dict per target child (report).
    dyads / log_e / observed_sites: the evidence used (possibly empty).
    """

    tier_b: TierCalls
    partial_b: PartialEdges
    released: np.ndarray
    kept: tuple[tuple[int, ...], ...]
    records: tuple[dict, ...]
    dyads: np.ndarray
    log_e: np.ndarray
    observed_sites: np.ndarray
    log_e_upper: np.ndarray | None = None


def release_counts(tier_b: TierCalls, partial_b: PartialEdges, complete: TierCalls,
                   graph_conflict: np.ndarray, informative: np.ndarray, rows: np.ndarray,
                   sample_ids: tuple[str, ...], eligibility: ParentEligibility,
                   dyads: np.ndarray, log_e: np.ndarray, observed_sites: np.ndarray,
                   config: PedigreeConfig, *, log_e_upper: np.ndarray | None = None) -> CountRelease:
    """Release M0/M1 counts that the exclusion bounds and the supported directions settle.

    A target child is released when every eligible candidate was scored, it is safe (eligible,
    informative chromosomes holding enough map units: ``informative`` float64 [N]; no graph conflict) and the
    candidates that survive exclusion
    (log E below log(2 N / alpha)) minus those it is the supported parent of form at most one
    sample, equal to both its supported marginal parents and its complete graph parents.
    """
    n = len(sample_ids)
    evidence = {(int(c), int(p)): float(e) for (c, p), e in zip(dyads, log_e)}
    outgoing: list[set[int]] = [set() for _ in range(n)]
    for child in range(n):
        for slot in range(2):
            if partial_b.supported[child, slot]:
                outgoing[int(partial_b.parents[child, slot])].add(child)
    threshold = math.log(2 * n / config.exclusion.alpha)
    if log_e_upper is not None:
        upper_evidence = {(int(c), int(p)): float(e)
                          for (c, p), e in zip(dyads, log_e_upper)}
        if np.any((log_e < threshold) & (log_e_upper >= threshold) & (log_e != log_e_upper)):
            raise ValueError("unresolved exclusion interval requires exact refinement")
    state = tier_b.state.copy()
    row = tier_b.row.copy()
    status = list(tier_b.status)
    parents = partial_b.parents.copy()
    supported = partial_b.supported.copy()
    exact = partial_b.exact.copy()
    count_minimum = partial_b.count_minimum.copy()
    count_maximum = partial_b.count_maximum.copy()
    partial_state = partial_b.state.copy()
    partial_status = list(partial_b.status)
    release_evidence = list(partial_b.release_evidence)
    released = np.zeros(n, dtype=np.bool_)
    kept_by_child: list[tuple[int, ...]] = [()] * n
    records = []
    name = sample_ids.__getitem__
    for child in sorted({int(c) for c, _ in dyads}):
        candidates = [int(p) for p in np.flatnonzero(eligibility.parents[child])]
        covered = all((child, p) in evidence for p in candidates)
        compatible = {p for p in candidates if not covered or evidence[child, p] < threshold}
        reverse = compatible & outgoing[child]
        survivors = compatible - reverse
        supported_parents = {int(p) for p in partial_b.parents[child][partial_b.supported[child]]}
        complete_row = int(complete.row[child])
        selected = (set() if complete_row < 0 else
                    {int(p) for p in rows[complete_row, 1:] if p >= 0})
        complete_count = int(complete.state[child])
        safe = bool(eligibility.children[child]
                    and informative[child] >= config.release.minimum_informative_units
                    and not graph_conflict[child])
        concordant = (survivors == supported_parents == selected
                      and complete.state[child] != UNRESOLVED and complete_count == len(survivors))
        is_released = bool(covered and safe and len(survivors) <= 1 and concordant)
        excluded = [evidence[child, p] for p in candidates if covered and p not in compatible]
        names = lambda indices: tuple(sorted(name(i) for i in indices))  # noqa: E731
        records.append(dict(
            Sample=name(child), EligibleCandidates=len(candidates),
            CompatibleCandidates=names(compatible), SupportedReverseCandidates=names(reverse),
            RemainingCandidates=names(survivors),
            WeakestExclusionLogE=min(excluded, default=np.nan), ThresholdLogE=threshold,
            CompleteCandidateEvidence=covered, Released=is_released,
            ReleaseConditionalOnSupportedDirections=bool(reverse)))
        if log_e_upper is not None:
            excluded_upper = [upper_evidence[child, p]
                              for p in candidates if covered and p not in compatible]
            exact_min = all(evidence[child, p] == upper_evidence[child, p]
                            for p in candidates if covered and p not in compatible)
            records[-1].update(WeakestExclusionLogELower=min(excluded, default=np.nan),
                              WeakestExclusionLogEUpper=min(excluded_upper, default=np.nan),
                              WeakestExclusionLogEExact=exact_min)
            if not exact_min:
                records[-1]['WeakestExclusionLogE'] = np.nan
        if not is_released:
            continue
        kept = tuple(sorted(survivors, key=name))
        released[child] = True
        kept_by_child[child] = kept
        state[child] = len(kept)
        row[child] = complete_row
        status[child] = "tier_b_fixed_genome_directed_exclusion"
        partial_state[child] = len(kept)
        partial_status[child] = status[child]
        release_evidence[child] = "mendelian_exclusion_and_supported_direction"
        parents[child] = (kept[0] if kept else NO_PARENT, NO_PARENT)
        supported[child] = parents[child] >= 0
        exact[child] = True
        count_minimum[child] = count_maximum[child] = len(kept)
    final = TierCalls(state, row, tuple(status), tier_b.state_pass, tier_b.identity_pass,
                      tier_b.orientation_withheld, tier_b.parent_flags)
    final_partial = PartialEdges(parents, supported, exact, count_minimum, count_maximum,
                                 partial_state, tuple(partial_status), tuple(release_evidence))
    return CountRelease(final, final_partial, released, tuple(kept_by_child), tuple(records),
                        dyads, log_e, observed_sites, log_e_upper)
