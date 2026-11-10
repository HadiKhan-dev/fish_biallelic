"""Build the candidate bank and starting panels for each local selection block.

Owns LS17-LS20: nearest-donor completion, discovery endpoint families, nested
cavity/BIC classes and whole-family budget admission. Must not schedule workers,
read storage or choose the final path-search outcome.
Method: docs/methods/selection.md#ls17.
Donors overwrite only unknown sites before deduplication and the variant cap;
every completion enters the bank. Both fixed-K usage schedules remain. Starts
retain first/last enumeration under the explicit convention; admitted endpoint
families keep their complete membership. Start, score and round backstops affect
result identity and are reported.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ...contracts.selection import BankSummary
from ...core.ties import (
    NOISE_MOTION,
    WIDE_CLASS_QUANTA,
    census,
    census_enabled,
    fixed_policy,
    improves,
    tie_classes,
)
from ..discovery.api import (
    CavityWorkspace,
    DiscoveryParameters,
    deduplicate,
    fit_panels,
    panel_key,
    prepare_workspace,
    release_alleles,
    score_solutions,
    solution_tie_mask,
    unique_rows,
)
from .fit import canonical as canonical_start
from .fit import unique_rows as unique_integer_rows
from .parameters import RACE_KEEP, RACE_MIN_STARTS, RACE_PROBE_ITERATIONS, CandidateBankParameters

__all__ = ["complete_proposals", "nested_tie_set", "candidate_bank"]


def complete_proposals(
    original_latent, proposal_panels, variant_cap: int = CandidateBankParameters.completion_variant_cap
):
    """(sources, bank rows, missing cells, overflow): the latent panel and each proposal panel's
    completed rows as search sources, and every distinct completion as a bank row.

    A partial row takes the alleles of its nearest latent rows on its uncalled sites. When equally near
    donors give different completions, the panel makes one source per
    combination of its tied rows' completions (no donor is preferred; the cavity tie sets and the
    path objective decide). A panel with more than ``variant_cap`` combinations keeps only the rows
    with a unique completion as its source (overflow: counted, the tied completions stay bank
    rows)."""
    original = unique_rows(original_latent).astype(np.int64)
    sources: list[np.ndarray] = [original]
    bank: list[np.ndarray] = [original]
    missing = overflow = 0
    for panel in proposal_panels:
        panel = np.asarray(panel)
        fixed: list[np.ndarray] = []
        options: list[np.ndarray] = []
        alternatives: list[np.ndarray] = []
        for row in panel:
            known = row >= 0
            missing += int((~known).sum())
            if not known.any():
                continue
            if known.all():
                fixed.append(row.astype(np.int64))
                continue
            distance = np.sum(original[:, known] != row[known], axis=1)
            donors = original[distance == distance.min()].copy()
            donors[:, known] = row[known]
            donors = unique_rows(donors)
            alternatives.extend(donors)
            if len(donors) == 1:
                fixed.append(donors[0])
            else:
                options.append(donors)
        if options and math.prod(len(choices) for choices in options) > variant_cap:
            overflow += 1
            options = []
        variants: dict[bytes, np.ndarray] = {}
        for choice in itertools.product(*options):
            rows = fixed + list(choice)
            if rows:
                variant = unique_rows(np.asarray(rows))
                variants.setdefault(variant.astype(np.int8).tobytes(), variant)
        if variants:
            for key in sorted(variants):
                sources.append(variants[key])
                bank.append(variants[key])
        else:
            sources.append(np.empty((0, original.shape[1]), np.int64))
        if alternatives:
            bank.append(np.asarray(alternatives))
    return tuple(sources), unique_rows(np.vstack(bank)).astype(np.int64), missing, overflow


def _race(workspace, starts, selection_score):
    """The RACE_KEEP starts whose short fits (RACE_PROBE_ITERATIONS fixed-K iterations) reach the best BIC
    selection score, in start order; ties at the cut are kept whole.

    A neighbour round of the bank search can propose thousands of panels at low depth (drops, adds and every
    single-row replacement by a bank row) and fits each to convergence, while only the BIC-best endpoints are
    admitted. The short fits share the workspace's cached transitions with the full fits that follow, so the
    probe's work is not repeated. A start whose short fit ranks low is not fitted further: a stated approximation
    that bounds the bank's work per round instead of growing it with the bank."""

    class _Entry:
        __slots__ = ("solution",)

        def __init__(self, solution):
            self.solution = solution

    synchronized, refitted, families = fit_panels(workspace, starts, RACE_PROBE_ITERATIONS, with_families=True)
    by_key = {solution.key: solution for solution in (*synchronized, *refitted)}
    scores = []
    for family in families:
        values = [selection_score(_Entry(by_key[key])) for key in family if key in by_key]
        scores.append(max(values) if values else -np.inf)
    order = sorted(range(len(starts)), key=lambda index: -scores[index])
    if len(order) <= RACE_KEEP:
        return starts
    cut = scores[order[RACE_KEEP - 1]]
    kept = {index for index in range(len(starts)) if scores[index] >= cut}
    return [start for index, start in enumerate(starts) if index in kept]


def _neighbours(panel, bank):
    """Drops, adds of every bank row not in the panel, then replacements (closest first)."""
    k = len(panel)
    proposals = []
    present = {row.tobytes() for row in panel}
    extra = [row for row in bank if row.tobytes() not in present]
    if k > 1:
        for i in range(k):
            proposals.append(np.delete(panel, i, axis=0))
    for row in extra:
        proposals.append(np.vstack((panel, row)))
    replacements = []
    for row in extra:
        for i in range(k):
            replacements.append((int(np.sum(row != panel[i])), i, row.tobytes(), row))
    for _, i, _, row in sorted(replacements, key=lambda value: value[:3]):
        new = panel.copy()
        new[i] = row
        proposals.append(new)
    return proposals


@dataclass(frozen=True)
class _Deferred:
    """An admitted panel whose cavity score is deferred (BIC bank search)."""

    solution: object


def nested_tie_set(entries, levels, *, counter=None, kind="", reverse: bool):
    """Every scored bank entry tied with the best under nested anchored tie classes (``levels`` as
    (value, quantum, motion) triples), after the stated prior fewer founders: at each level the best class
    goes on to the next (a class wider than the width guard keeps only the entries exactly at its
    best value). A set: the result does not depend on the order of ``entries`` (returned in (K,
    canonical key) order). Counts the decision in ``counter`` at every level."""
    candidates = sorted(entries, key=lambda entry: (entry.solution.k, entry.solution.key))
    inside = False
    wide = boundary = 0
    for position, (value, quantum, motion) in enumerate(levels):
        values = [value(entry) for entry in candidates]
        classes = tie_classes(
            values, quantum, motion=motion, reverse=fixed_policy(kind, quantum, reverse=reverse).reverse
        )
        wide += classes.wide_classes
        boundary += len(classes.boundary_gaps)
        members = [int(index) for index in np.flatnonzero(classes.classes == 0)]
        if position == 0:
            inside = len(members) > 1
        if len(members) > 1 and classes.spans[0] > WIDE_CLASS_QUANTA * quantum:
            best = min(values[index] for index in members)
            members = [index for index in members if values[index] == best]
        candidates = [candidates[index] for index in members]
        if len(candidates) == 1:
            break
    fewest = min(entry.solution.k for entry in candidates)
    if counter is not None:
        counter.note(kind, inside_tie=inside, wide=wide, boundary=boundary)
    return [entry for entry in candidates if entry.solution.k == fewest]


def _admit_endpoint_families(solutions, families, scored_keys, room, quanta, *, reverse: bool):
    """Fit families are indivisible at the unchanged endpoint-count score cap.

    Overlapping families form one component, ranked by its best endpoint's exact NLL.
    Whole NLL tie classes enter together; the first nonfitting class and all lower
    ranks are omitted. No label or enumeration chooses a partial family or tie class.
    """
    by_key = {solution.key: solution for solution in solutions}
    novel = set(by_key).difference(scored_keys)
    if len(novel) <= room:
        return [by_key[key] for key in sorted(novel)], False
    leader = {key: key for key in by_key}

    def find(key):
        while leader[key] != key:
            leader[key] = leader[leader[key]]
            key = leader[key]
        return key

    for family in families:
        members = [key for key in family if key in by_key]
        if not members:
            continue
        first = find(members[0])
        for key in members[1:]:
            leader[find(key)] = first
    components: dict[bytes, set[bytes]] = {}
    for key in by_key:
        components.setdefault(find(key), set()).add(key)
    groups = [keys for keys in components.values() if keys & novel]
    values = [min(by_key[key].nll for key in keys) for keys in groups]
    classes = tie_classes(
        values, quanta.nll, reverse=fixed_policy("selection.bank.cavity", quanta.nll, reverse=reverse).reverse
    )
    kept: set[bytes] = set()
    for cls in sorted(set(classes.classes.tolist())):
        class_members: set[bytes] = set()
        for index in classes.members(cls):
            class_members.update(groups[int(index)] & novel)
        if len(kept) + len(class_members) > room:
            break
        kept.update(class_members)
    return [by_key[key] for key in sorted(kept)], len(kept) < len(novel)


def _bic_search(
    likelihoods,
    observed,
    original_latent,
    proposal_panels,
    discovery_config: DiscoveryParameters,
    bank_config,
    ties=None,
    census_key=None,
    *,
    reverse: bool,
    census_sink=None,
    threads: int = 1,
):
    """(BIC-selected tie set (solutions), released starts (calls, latent rows) of the cavity-best
    tie set over all sources' endpoints, summary)."""
    model = discovery_config.model
    likelihood = np.array(likelihoods, dtype=np.float64, order="C", copy=True)
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    active = observed.any(axis=1)
    likelihood[~observed] = 1.0 / 3.0
    sources, bank, _missing, overflow = complete_proposals(
        original_latent, proposal_panels, bank_config.completion_variant_cap
    )
    if ties is not None:
        ties.note("ls.bank.completion.overflow", inside_tie=overflow > 0)
    workspace = prepare_workspace(likelihood, observed, model, reverse=reverse, threads=threads)
    quanta = workspace.quanta
    motion = NOISE_MOTION["nll"]
    collect = census_key is not None and census_enabled(sink=census_sink)
    cavity = CavityWorkspace(likelihood if np.all(active) else likelihood[active], discovery_config.cavity)
    scored: dict = {}
    fitted_starts: set = set()
    endpoint_keys: dict = {}
    state = dict(budget_bound=False, selected=None)
    complexity = bank_config.complexity_scale * (likelihood.shape[1] / 200.0) * int(active.sum())

    def selection_score(entry) -> float:
        # The fixed-K fitter's binned switching-Viterbi NLL in the outer BIC convention
        # K * complexity - 2 log L (lower is better), negated and halved.
        solution = entry.solution
        return -0.5 * (solution.k * complexity - 2.0 * (-solution.nll))

    def select(round_label):
        """The BIC-selected tie set over everything admitted: the best class of -selection_score,
        inside it the best class of the NLL, then the prior fewer founders; retain every member."""
        entries = list(scored.values())
        if not entries:
            return ()
        if collect:
            census(
                "ls.bank.bic",
                f"{census_key}/bank/{round_label}",
                sorted(-selection_score(entry) for entry in entries)[:8],
                sink=census_sink,
            )
        return nested_tie_set(
            entries,
            (
                (lambda entry: -selection_score(entry), quanta.nll, motion),
                (lambda entry: entry.solution.nll, quanta.nll, motion),
            ),
            counter=ties,
            kind="ls.bank.bic",
            reverse=reverse,
        )

    def best_score(tied) -> float:
        return max(selection_score(entry) for entry in tied)

    def fit_and_score(panels, require_cavity=False):
        selected = state["selected"]
        k = selected[0].solution.k if selected else max((len(panel) for panel in panels), default=0)
        novel = []
        for panel in panels:
            if not len(panel):
                continue
            panel = unique_rows(panel).astype(np.int64)
            key = panel.astype(np.int8).tobytes()
            if key not in fitted_starts:
                fitted_starts.add(key)
                novel.append(panel)
        if not novel:
            keys = set()
            for panel in panels:
                if len(panel):
                    keys.update(endpoint_keys.get(panel_key(panel), ()))
            # Canonical key order (a set's iteration order depends on the hash seed).
            return tuple(scored[key] for key in sorted(keys) if key in scored)
        start_limit = bank_config.start_backstop(k)
        if len(novel) > start_limit:
            state["budget_bound"] = True
            # Budget-bound: the first starts in enumeration order (a convention; the last ones
            # under core.ties.reverse_conventions).
            if reverse:
                novel, dropped = novel[len(novel) - start_limit :], novel[: len(novel) - start_limit]
            else:
                novel, dropped = novel[:start_limit], novel[start_limit:]
            for panel in dropped:
                fitted_starts.remove(panel.astype(np.int8).tobytes())
        if not require_cavity and len(novel) > RACE_MIN_STARTS:
            novel = _race(workspace, novel, selection_score)
        synchronized, refitted, families = fit_panels(workspace, novel, model.max_iterations, with_families=True)
        unique = deduplicate((*synchronized, *refitted), reverse=reverse)
        if len(novel) == 1:
            endpoint_keys[panel_key(novel[0])] = tuple(solution.key for solution in unique)
        room = max(0, bank_config.score_backstop(k) - len(scored))
        pending, bounded = _admit_endpoint_families(unique, families, set(scored), room, quanta, reverse=reverse)
        state["budget_bound"] |= bounded
        if reverse:
            # Keep-first per key (a convention): the last admitted fit of a key replaces it.
            pending += [solution for solution in unique if solution.key in scored]
        if require_cavity:
            # score_solutions reads each endpoint's full tie membership and restricts it with
            # the same active-sample mask as the likelihoods.
            for score in score_solutions(cavity, pending, active) if pending else ():
                scored[score.solution.key] = score
        else:
            for solution in pending:
                scored[solution.key] = _Deferred(solution)
        return tuple(scored[solution.key] for solution in unique if solution.key in scored)

    # Each source separately first: fair source-only refits that protect every input basin. The
    # candidate starts are every source's cavity-best tie set (a set of panels: neither the source
    # order nor a key order decides; the path objective chooses among them, rounds.py).
    starts_by_key = {}
    for index, panel in enumerate(sources):
        values = fit_and_score([panel], require_cavity=True)
        if not values and len(panel):
            key = panel_key(panel)
            values = (scored[key],) if key in scored else ()
        if values and collect:
            census(
                "ls.bank.cavity",
                f"{census_key}/bank/source{index}",
                sorted(-score.log_score for score in values)[:8],
                sink=census_sink,
            )
        if values:
            for score in nested_tie_set(
                values,
                (
                    (lambda score: -score.log_score, quanta.nll, motion),
                    (lambda score: score.solution.nll, quanta.nll, motion),
                ),
                counter=ties,
                kind="ls.bank.cavity",
                reverse=reverse,
            ):
                starts_by_key[score.solution.key] = score
    start_set = [starts_by_key[key] for key in sorted(starts_by_key)]
    fit_and_score([bank])
    selected = state["selected"] = select("r0")
    if not selected:
        # A wholly omitted initial family supplies no BIC winner. The public bank
        # caller retains its existing original_latent incumbent and raw start.
        summary = BankSummary(
            bank_rows=len(bank),
            fitted_starts=len(fitted_starts),
            scored=0,
            selected_k=0,
            rounds=0,
            budget_bound=True,
            selected_tied=0,
            starts=0,
        )
        return (), (), summary
    rounds = 0
    while True:
        if rounds >= bank_config.round_backstop(selected[0].solution.k):
            state["budget_bound"] = True
            break
        before = best_score(selected)
        neighbours = []
        for entry in selected:
            neighbours.extend(_neighbours(entry.solution.haplotypes, bank))
        fit_and_score(neighbours)
        selected = state["selected"] = select(f"r{rounds + 1}")
        rounds += 1
        if collect:
            census(
                "ls.bank.stop", f"{census_key}/bank/stop{rounds}", (-best_score(selected), -before), sink=census_sink
            )
        if not improves(best_score(selected), before, quanta.improvement):
            break
        if len(scored) >= bank_config.score_backstop(selected[0].solution.k):
            state["budget_bound"] = True
            break
    releases = {}

    def released(score):
        """(released calls, latent rows) of a cavity-scored endpoint (raw-discovery rule)."""
        if score is None:
            return None
        solution = score.solution
        tie_mask = solution_tie_mask(solution)
        key = (solution.key, tie_mask.tobytes())
        if key not in releases:
            _, _, calls = release_alleles(
                likelihood,
                solution.haplotypes,
                solution.assignments,
                observed,
                model.unknown_copy_penalty,
                discovery_config.release,
                quanta,
                tie_mask=tie_mask,
            )
            releases[key] = calls
        return releases[key], solution.haplotypes

    summary = BankSummary(
        bank_rows=len(bank),
        fitted_starts=len(fitted_starts),
        scored=len(scored),
        selected_k=selected[0].solution.k,
        rounds=rounds,
        budget_bound=bool(state["budget_bound"]),
        selected_tied=len(selected),
        starts=len(start_set),
    )
    return (tuple(entry.solution for entry in selected), tuple(released(score) for score in start_set), summary)


def candidate_bank(
    likelihoods,
    observed,
    original_latent,
    proposal_panels,
    discovery_config: DiscoveryParameters,
    bank_config,
    incumbent_panel=None,
    *,
    ties=None,
    census_key=None,
    alternatives=(),
    reverse: bool,
    census_sink=None,
    threads: int = 1,
):
    """(start panels, bank rows, summary or None) of a block's path search.

    Without proposals (the initial round) the one start is the raw latent panel and the bank its
    rows. With proposals the starts are the released cavity-best tie set over all sources (distinct
    panels in canonical key order; the search runs from each and its objective decides), and the
    bank adds the raw latent rows, the starts and every member of the BIC tie set. An incumbent
    (the previous round's selected panel) contributes all its rows. ``ties``: a
    core.ties.TieCounter for the bank's tie decisions; ``census_key``: the block's census key.
    ``alternatives``: discovery's NLL-tied alternative latent panels: extra starts
    and bank rows in the initial round, extra sources in context rounds.
    """
    original_latent = np.asarray(original_latent, dtype=np.int64)
    proposal_panels = list(proposal_panels)
    summary = None
    alternatives = [np.asarray(panel, dtype=np.int64) for panel in alternatives if len(panel)]
    if proposal_panels:
        selected, released_starts, summary = _bic_search(
            likelihoods,
            observed,
            original_latent,
            [*proposal_panels, *alternatives],
            discovery_config,
            bank_config,
            ties,
            census_key,
            reverse=reverse,
            census_sink=census_sink,
            threads=threads,
        )
        distinct: dict[bytes, np.ndarray] = {}
        for calls, latent in released_starts:
            # A start as 2.1 passes it (duplicate rows weigh its initial frequencies); one start per
            # canonical (rows, frequencies) identity.
            panel = np.where(calls >= 0, calls, latent).astype(np.int64)
            rows, mass = canonical_start(panel)
            distinct.setdefault(rows.astype(np.int8).tobytes() + mass.tobytes(), panel)
        starts = [distinct[key] for key in sorted(distinct)] or [original_latent]
        bank = unique_integer_rows(
            np.vstack((original_latent, *starts, *(solution.haplotypes for solution in selected)))
        )
    elif alternatives:
        distinct = {}
        for panel in (original_latent, *alternatives):
            rows, mass = canonical_start(panel)
            distinct.setdefault(rows.astype(np.int8).tobytes() + mass.tobytes(), panel)
        starts = [distinct[key] for key in sorted(distinct)]
        bank = unique_integer_rows(np.vstack(starts))
    else:
        starts = [original_latent]
        bank = unique_integer_rows(original_latent)
    if incumbent_panel is not None:
        bank = unique_integer_rows(np.vstack((bank, incumbent_panel)))
    return starts, bank, summary
