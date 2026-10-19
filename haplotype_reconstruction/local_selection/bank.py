"""The candidate bank of a context round: which rows the path search may add, and where it starts.

Context proposals are candidates, never observations. Each proposal panel's partial rows are
completed from the nearest raw-discovery latent rows on their called sites (every tied
completion enters the bank; called alleles are never overwritten). Every source (the latent
panel, then each proposal panel) is refitted with discovery's hard fixed-K model and its
endpoints are cavity-scored; then the union bank and its drop / add / replace neighbourhoods are
searched under the switch-penalised BIC (cavity scores deferred: neither ranking nor stopping
reads them). The path search starts from the latest source's cavity-best endpoint released
with the raw-discovery rule (released alleles, latent rows where withheld); the bank adds the
raw latent rows, that start and the BIC-selected panel. Neither score replaces the path
objective.

Release 2.1 (noise stability): the BIC selection and each source's cavity-best endpoint are
chosen by nested anchored tie classes of the block's ``nll`` quantum (the fixed-K workspace's
BlockQuanta): the best class of the score, inside it the best class of the NLL, then fewer
founders, then the canonical key (2.0 took a strict minimum of float-first keys, so panels whose
hard-model NLLs are mathematically equal, e.g. differing only at symmetric-evidence cells, were
picked by their last bits). The neighbourhood rounds stop unless the selection improves by more
than the ``improvement`` quantum (2.0: 1e-9, at the noise scale); a budget-bound admission cuts
the NLL tie-class order (discovery's ``sort_solutions``); the endpoints' release uses the block's
quanta.

Release 2.2 (tied sets decided by the data): the BIC selection is the whole tie set of the best
class (score class, NLL class, then the stated prior fewer founders); the neighbourhoods of every
member are searched and every member's rows join the bank. The candidate path-search starts are
every source's cavity-best tie set, released with the raw-discovery rule; rounds.py fits each,
searches from the best tie class of their selection objectives (which include the context code)
and lets the selection objective decide between the outcomes. 2.1 started from the latest source's
endpoint and broke every tie by the canonical key.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from ..discovery.cavity import CavityWorkspace, score_solutions
from ..discovery.fixed_k.fit import fit_panels, prepare_workspace
from ..discovery.panel import deduplicate, panel_key, sort_solutions, unique_rows
from ..discovery.release import release_alleles
from ..execution import claim_threads
from ..maths.ties import (NOISE_MOTION, WIDE_CLASS_QUANTA, TieCounter, census, census_enabled, improves,
                          reverse_conventions, tie_classes)
from .fit import canonical as canonical_start
from .fit import unique_rows as unique_integer_rows


@dataclass(frozen=True)
class BankSummary:
    """bank_rows: distinct candidate rows; fitted_starts / scored: novel starts fitted and
    panels admitted; selected_k: the BIC-selected K; rounds: neighbourhood rounds; budget_bound:
    a reported backstop was hit (start, score or round backstop). Release 2.2: selected_tied:
    members of the final BIC tie set (all join the bank); starts: distinct path-search starts (the
    cavity-best tie set over all sources). Both default to 1 (2.1 records)."""

    bank_rows: int
    fitted_starts: int
    scored: int
    selected_k: int
    rounds: int
    budget_bound: bool
    selected_tied: int = 1
    starts: int = 1


def complete_proposals(original_latent, proposal_panels, variant_cap: int = 8):
    """(sources, bank rows, missing cells, overflow): the latent panel and each proposal panel's
    completed rows as search sources, and every distinct completion as a bank row.

    A partial row takes the alleles of its nearest latent rows on its uncalled sites. Release 2.2:
    when equally near donors give different completions, the panel makes one source per
    combination of its tied rows' completions (no donor is preferred; the cavity tie sets and the
    path objective decide). A panel with more than ``variant_cap`` combinations keeps only the rows
    with a unique completion as its source (overflow: counted, the tied completions stay bank
    rows). 2.1 completed every row from the first tied donor in content order (a convention)."""
    original = unique_rows(original_latent).astype(np.int64)
    sources = [original]
    bank = [original]
    missing = overflow = 0
    for panel in proposal_panels:
        panel = np.asarray(panel)
        fixed, options, alternatives = [], [], []
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
        variants = {}
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


def nested_winner(entries, levels, *, counter=None, kind=""):
    """The winner of scored bank entries (each with a ``solution``) under nested anchored tie
    classes: ``levels`` = ((value, quantum, motion), ...), smaller values better. At each level the
    best class (everything chained from the best value within the quantum) goes on to the next;
    the last level's class is cut by (K, canonical key). A class wider than the width guard is
    decided by value inside, exact value ties by (K, canonical key) (maths.ties). The candidates are
    taken in (K, canonical key) order and every level orders by (K, canonical key) inside its
    classes, so no outcome depends on the order the entries arrive in (admission order, or a
    Python set's hash-seed order for a repeated source). Counts the decision in ``counter`` (inside
    a tie: the first level's best class had several members; wide and boundary events of every
    level)."""
    candidates = sorted(entries, key=lambda entry: (entry.solution.k, entry.solution.key))
    inside = False
    wide = boundary = 0
    for position, (value, quantum, motion) in enumerate(levels):
        last = position == len(levels) - 1
        classes = tie_classes([value(entry) for entry in candidates], quantum, motion=motion,
                              secondary=[(entry.solution.k,) for entry in candidates],
                              content=[entry.solution.key for entry in candidates])
        wide += classes.wide_classes
        boundary += len(classes.boundary_gaps)
        members = classes.members(0)
        if position == 0:
            inside = len(members) > 1
        if last or classes.wide or len(members) == 1:
            winner = candidates[int(classes.order[0])]
            break
        candidates = [candidates[int(index)] for index in members]
    if counter is not None:
        counter.note(kind, inside_tie=inside, wide=wide, boundary=boundary)
    return winner


def nested_tie_set(entries, levels, *, counter=None, kind=""):
    """Every scored bank entry tied with the best under nested anchored tie classes (``levels`` as
    in :func:`nested_winner`), after the stated prior fewer founders: at each level the best class
    goes on to the next (a class wider than the width guard keeps only the entries exactly at its
    best value). A set: the result does not depend on the order of ``entries`` (returned in (K,
    canonical key) order). Counts the decision in ``counter`` like :func:`nested_winner`."""
    candidates = sorted(entries, key=lambda entry: (entry.solution.k, entry.solution.key))
    inside = False
    wide = boundary = 0
    for position, (value, quantum, motion) in enumerate(levels):
        values = [value(entry) for entry in candidates]
        classes = tie_classes(values, quantum, motion=motion)
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


def _bic_search(likelihoods, observed, original_latent, proposal_panels, discovery_config,
                bank_config, ties=None, census_key=None):
    """(BIC-selected tie set (solutions), released starts (calls, latent rows) of the cavity-best
    tie set over all sources' endpoints, summary)."""
    model = discovery_config.model
    likelihood = np.array(likelihoods, dtype=np.float64, order="C", copy=True)
    observed = np.ascontiguousarray(observed, dtype=np.bool_)
    active = observed.any(axis=1)
    likelihood[~observed] = 1.0 / 3.0
    sources, bank, _missing, overflow = complete_proposals(original_latent, proposal_panels,
                                                           bank_config.completion_variant_cap)
    if ties is not None:
        ties.note("ls.bank.completion.overflow", inside_tie=overflow > 0)
    workspace = prepare_workspace(likelihood, observed, model)
    quanta = workspace.quanta
    motion = NOISE_MOTION["nll"]
    collect = census_key is not None and census_enabled()
    cavity = CavityWorkspace(likelihood if np.all(active) else likelihood[active],
                             discovery_config.cavity)
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
        inside it the best class of the NLL, then the prior fewer founders (release 2.2: every
        member, not one picked by the canonical key)."""
        entries = list(scored.values())
        if collect:
            census("ls.bank.bic", f"{census_key}/bank/{round_label}",
                   sorted(-selection_score(entry) for entry in entries)[:8])
        return nested_tie_set(entries, ((lambda entry: -selection_score(entry), quanta.nll, motion),
                                        (lambda entry: entry.solution.nll, quanta.nll, motion)),
                              counter=ties, kind="ls.bank.bic")

    def best_score(tied) -> float:
        return max(selection_score(entry) for entry in tied)

    def fit_and_score(panels, require_cavity=False):
        selected = state["selected"]
        k = (selected[0].solution.k if selected
             else max((len(panel) for panel in panels), default=0))
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
            # under maths.ties.reverse_conventions).
            if reverse_conventions():
                novel, dropped = novel[len(novel) - start_limit:], novel[:len(novel) - start_limit]
            else:
                novel, dropped = novel[:start_limit], novel[start_limit:]
            for panel in dropped:
                fitted_starts.remove(panel.astype(np.int8).tobytes())
        claim_threads()
        synchronized, refitted = fit_panels(workspace, novel, model.max_iterations)
        unique = deduplicate((*synchronized, *refitted))
        if len(novel) == 1:
            endpoint_keys[panel_key(novel[0])] = tuple(solution.key for solution in unique)
        pending = [solution for solution in unique if solution.key not in scored]
        room = max(0, bank_config.score_backstop(k) - len(scored))
        if len(pending) > room:
            state["budget_bound"] = True
            pending = sort_solutions(pending, quanta)[:room]
        if reverse_conventions():
            # Keep-first per key (a convention): the last admitted fit of a key replaces it.
            pending += [solution for solution in unique if solution.key in scored]
        if require_cavity:
            for score in (score_solutions(cavity, pending, active) if pending else ()):
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
            census("ls.bank.cavity", f"{census_key}/bank/source{index}",
                   sorted(-score.log_score for score in values)[:8])
        if values:
            for score in nested_tie_set(values, ((lambda score: -score.log_score, quanta.nll, motion),
                                                 (lambda score: score.solution.nll, quanta.nll, motion)),
                                        counter=ties, kind="ls.bank.cavity"):
                starts_by_key[score.solution.key] = score
    start_set = [starts_by_key[key] for key in sorted(starts_by_key)]
    fit_and_score([bank])
    selected = state["selected"] = select("r0")
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
            census("ls.bank.stop", f"{census_key}/bank/stop{rounds}", (-best_score(selected), -before))
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
        key = (solution.key, solution.assignments.tobytes())
        if key not in releases:
            claim_threads()
            _, _, calls = release_alleles(likelihood, solution.haplotypes, solution.assignments,
                                          observed, model.unknown_copy_penalty,
                                          discovery_config.release, quanta)
            releases[key] = calls
        return releases[key], solution.haplotypes

    summary = BankSummary(bank_rows=len(bank), fitted_starts=len(fitted_starts), scored=len(scored),
                          selected_k=selected[0].solution.k, rounds=rounds,
                          budget_bound=bool(state["budget_bound"]), selected_tied=len(selected),
                          starts=len(start_set))
    return (tuple(entry.solution for entry in selected), tuple(released(score) for score in start_set),
            summary)


def candidate_bank(likelihoods, observed, original_latent, proposal_panels, discovery_config,
                   bank_config, incumbent_panel=None, *, ties=None, census_key=None, alternatives=()):
    """(start panels, bank rows, summary or None) of a block's path search.

    Without proposals (the initial round) the one start is the raw latent panel and the bank its
    rows. With proposals the starts are the released cavity-best tie set over all sources (distinct
    panels in canonical key order; the search runs from each and its objective decides), and the
    bank adds the raw latent rows, the starts and every member of the BIC tie set. An incumbent
    (the previous round's selected panel) contributes all its rows. ``ties``: a
    maths.ties.TieCounter for the bank's tie decisions; ``census_key``: the block's census key.
    ``alternatives``: discovery's NLL-tied alternative latent panels (release 2.2): extra starts
    and bank rows in the initial round, extra sources in context rounds.
    """
    original_latent = np.asarray(original_latent, dtype=np.int64)
    proposal_panels = list(proposal_panels)
    summary = None
    alternatives = [np.asarray(panel, dtype=np.int64) for panel in alternatives if len(panel)]
    if proposal_panels:
        selected, released_starts, summary = _bic_search(
            likelihoods, observed, original_latent, [*proposal_panels, *alternatives], discovery_config,
            bank_config, ties, census_key)
        distinct = {}
        for calls, latent in released_starts:
            # A start as 2.1 passes it (duplicate rows weigh its initial frequencies); one start per
            # canonical (rows, frequencies) identity.
            panel = np.where(calls >= 0, calls, latent).astype(np.int64)
            rows, mass = canonical_start(panel)
            distinct.setdefault(rows.astype(np.int8).tobytes() + mass.tobytes(), panel)
        starts = [distinct[key] for key in sorted(distinct)] or [original_latent]
        bank = unique_integer_rows(np.vstack((original_latent, *starts,
                                              *(solution.haplotypes for solution in selected))))
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
