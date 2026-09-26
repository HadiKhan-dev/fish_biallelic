"""Bounded drop/merge/add panel search with exact, block-scoped score reuse.

Defaults allow three outer rounds and eight full refits per proposal kind.
Reuse is keyed by the complete binary panel AND its frequency vector: equal
alleles with different starts must still compete. Prepared observations and
memoized fits live only for this one block search.
"""
import numpy as np
from haplotype_reconstruction.core import parallel
from . import path_scoring as scoring, path_fitting as fitting
from .path_fitting import canonical, regularizer
from .candidate_selection import CandidateSelectionConfig, select_candidate_panel


def prepare_candidate_bank(gl, observed, original_latent, proposal_panels,
                           config=None, incumbent=None):
    """Return the tested local start, binary candidate bank and diagnostics.

    Assembly calls are candidate initializations, never extra observations.
    Partial rows use the existing nearest-latent completion; known alleles are
    preserved. The latest nonempty cavity-ranked source endpoint initializes
    search; the BIC-selected endpoint augments the bank. Neither score replaces
    the normalized path objective. An incumbent contributes all its latent rows.
    """
    proposals = list(proposal_panels)
    if proposals:
        if config is None:
            config = CandidateSelectionConfig(criterion="bic")
        competing = select_candidate_panel(
            gl, observed_mask=observed, original_latent=original_latent,
            proposal_panels=proposals, config=config, defer_cavity=True)
        backbone = next(value for value in reversed(competing.cavity_source_results)
                        if value is not None)
        start = np.where(backbone["discrete_haps"] >= 0,
                         backbone["discrete_haps"], backbone["latent_haps"])
        bank = np.unique(np.vstack((original_latent, start,
                                    competing.selected_mode.haplotypes)), axis=0)
        diagnostic = competing.diagnostic
    else:
        start = np.asarray(original_latent)
        bank = np.unique(start, axis=0)
        diagnostic = dict(source="raw latent initialization only")
    if incumbent is not None:
        bank = np.unique(np.vstack((bank, incumbent["panel"])), axis=0)
    return start, bank, diagnostic


def wildcard_compatibility(pair, observed):
    """Observed-site MAP unknown-count summary, not a constant founder pair.

    With no observed site a sample has two unknown slots, but contributes no
    mass denominator. Posterior expected unknown mass is reported separately.
    """
    count_probability = np.stack((pair[:, :, :-1, :-1].sum(axis=(2, 3)),
        pair[:, :, -1, :-1].sum(axis=2) + pair[:, :, :-1, -1].sum(axis=2),
        pair[:, :, -1, -1]), axis=2)
    counts = count_probability.argmax(axis=2)
    has_depth = observed.any(axis=1)
    slots = np.where(observed, counts, 0).max(axis=1)
    slots = np.where(has_depth, slots, 2).astype(np.int8)
    mass = float(slots[has_depth].sum() / max(1, 2*has_depth.sum()))
    return dict(slots=slots, has_depth=has_depth, mass=mass)


def proposals(fit, bank):
    panel, f = fit['panel'], fit['frequencies']
    def key(p, q):
        return (p.tobytes(), np.ascontiguousarray(q).tobytes())
    seen = {key(panel, f)}
    if len(panel) > 1:
        for i in range(len(panel)):
            p, q = canonical(np.delete(panel, i, axis=0), np.delete(f, i))
            if key(p, q) not in seen:
                seen.add(key(p, q))
                yield 'drop', p, q
        # Consensus is only an initialization; the same read model refits it.
        # Both oriented deletion starts above compete with these merge starts.
        for i in range(len(panel)):
            for j in range(i + 1, len(panel)):
                p, q = np.delete(panel, j, axis=0), np.delete(f, j)
                weight = fit['occupancy'][i] + fit['occupancy'][j]
                mean = np.divide(fit['occupancy'][i]*fit['allele_probability'][i] +
                                 fit['occupancy'][j]*fit['allele_probability'][j],
                                 weight, out=np.full(panel.shape[1], .5), where=weight>0)
                p[i] = np.where(mean == .5, panel[i], mean > .5)
                q[i] += f[j]
                p, q = canonical(p, q)
                if key(p, q) not in seen:
                    seen.add(key(p, q))
                    yield 'merge', p, q
    for row in bank:
        if np.any(np.all(panel == row, axis=1)):
            continue
        p, q = canonical(np.vstack((panel, row)), np.r_[f * len(panel)/(len(panel)+1), 1/(len(panel)+1)])
        if key(p, q) not in seen:
            seen.add(key(p, q))
            yield 'add', p, q



def search_panel(start, bank, gl, observed, positions, *, generations,
                 learn_frequencies, max_updates=20, rounds=3, refits_per_kind=8,
                 initial_fit=None, bulk=False, prepared=None,
                 recombination_rate_per_bp=5e-8, interval_morgans=None,
                 wildcard_mass=.01):
    assert not bulk, 'This optimized route preserves the ordinary-neighbour search only'
    model_options = dict(generations=generations,
        recombination_rate_per_bp=recombination_rate_per_bp,
        interval_morgans=interval_morgans, wildcard_mass=wildcard_mass)
    if prepared is None:
        prepared = scoring.prepare_observations(gl, observed, positions, **model_options)
    prepared = fitting.prepare_fit_observations(prepared)
    score_cache, fit_cache = {}, {}
    work = dict(score_calls=0, score_cache_hits=0, fit_calls=0, fit_cache_hits=0)

    def fit_panel(panel, frequencies=None):
        key = (panel.tobytes(), None if frequencies is None else frequencies.tobytes())
        if key in fit_cache:
            work['fit_cache_hits'] += 1
            return fit_cache[key]
        result = fitting.fit_fixed(panel, gl, observed, positions,
            learn_frequencies=learn_frequencies,
            frequencies=frequencies, max_updates=max_updates, prepared=prepared,
            **model_options)
        work['fit_calls'] += 1
        fit_cache[key] = result
        return result

    if initial_fit is None:
        fit = fit_panel(start)
    else:
        trial = fit_panel(initial_fit['panel'], initial_fit['frequencies'])
        fit = trial if trial['objective'] >= initial_fit['objective'] else initial_fit
    trace = []
    for round_index in range(rounds):
        ranked, seen = {}, set()
        for kind, panel, frequency in proposals(fit, bank):
            if not learn_frequencies:
                frequency = np.full(len(panel), 1./len(panel))
            key = (panel.tobytes(), frequency.tobytes())
            tagged_key = (kind, *key)
            if tagged_key in seen:
                continue
            seen.add(tagged_key)
            if key in score_cache:
                work['score_cache_hits'] += 1
                ll = score_cache[key]
            else:
                parallel.apply_dynamic_threads()
                ll = scoring.score_prepared(panel, frequency, prepared)['log_likelihood']
                score_cache[key] = ll
                work['score_calls'] += 1
            score = ll + regularizer(panel, frequency, learn_frequencies)
            ranked.setdefault(kind, []).append((score, panel, frequency))
        trials = []
        for kind, options in ranked.items():
            options.sort(key=lambda value: (-value[0], value[1].tobytes()))
            for score, panel, frequency in options[:refits_per_kind]:
                trials.append((fit_panel(panel, frequency), kind))
        if not trials:
            break
        best, kind = min(trials, key=lambda value: (-value[0]['objective'],
            len(value[0]['panel']), value[0]['panel'].tobytes()))
        trace.append(dict(round=round_index, proposals={k:len(v) for k,v in ranked.items()},
            refits=len(trials), old_k=len(fit['panel']), best_k=len(best['panel']),
            gain=best['objective']-fit['objective'], kind=kind))
        if best['objective'] <= fit['objective'] + 1e-6:
            break
        fit = best
    return dict(fit, search_trace=trace, search_method='reference_neighbours',
                computational_work=work)


def release_and_wildcard(fit, gl, observed, positions, config, generations, *,
                         recombination_rate_per_bp=5e-8, interval_morgans=None,
                         wildcard_mass=.01):
    """One diagnostic E-step supplies allele release and spatial MAP-U counts."""
    from scipy.special import expit
    from . import path_model as model
    state = model.e_step(fit['panel'], gl, positions, observed,
        generations=generations, founder_frequencies=fit['frequencies'],
        return_pair_posteriors=True, return_sample_stats=True,
        recombination_rate_per_bp=recombination_rate_per_bp,
        interval_morgans=interval_morgans, wildcard_mass=wildcard_mass)
    probability = expit(-state['conditional_flip_gains'])
    favorable = state['sample_conditional_flip_gains'] < -config.score_tolerance
    support = np.sum(state['sample_carrier_probability'] * favorable, axis=0)
    confident = ((probability >= config.min_hard_call_pseudo_probability)
                 & (support >= config.min_directional_supporters))
    release = dict(calls=np.where(confident, fit['panel'], -1).astype(np.int8),
        current_allele_probability=probability, directional_support=support,
        positive_sample_count=np.sum(favorable, axis=0),
        unknown_copy_mass=float(state['occupancy'][-1].sum()),
        total_copy_mass=float(state['occupancy'].sum()),
        unknown_copy_mass_by_site=state['occupancy'][-1],
        known_partner_support=state['current_support_mass'],
        expected_state_changes=float(state['expected_state_changes'].sum()),
        interpretation='Exact one-bit probability conditional on other panel alleles and fitted parameters; expected posterior-carrier directional support replaces hard-assigned support. Not a full allele posterior.')
    wildcard = wildcard_compatibility(state['pair_posteriors'], observed)
    return release, wildcard
