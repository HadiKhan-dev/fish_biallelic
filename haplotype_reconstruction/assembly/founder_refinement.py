"""Refine final founder chromosomes against the original prepared local panels.

Hierarchy supplies the initial founder count and fixed phase-component boundaries.
Final refinement reopens local row choices and compares bounded count reductions
without inventing alleles, merging components or using pedigree information.
The sample-level fitting HMM is internal to assembly; it does not replace painting
or publish sample ancestry. Path edits improve the same full-site
cohort objective; count changes compete under the existing complexity penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .founder import count_criterion as founder_count_criterion
from . import paths
from ..core.haplotypes import BlockResults


# Count decisions of the converged search (founder/converged_sweep.py) converge the
# COUNT_RANKED_REFITS top-ranked candidates of a round (deletions ranked by the
# criterion's margin on their exact unrefitted scores, duplicate seeds by their copy's
# exact best-response gain) concurrently and decide on the best-F of them (ties: lower
# candidate index); a round with at most this many candidates converges every one.
COUNT_RANKED_REFITS = 9


@dataclass(frozen=True)
class FounderRefinementConfig:
    enabled: bool = True
    beam_width: int = 64
    maximum_beam_width: int = 1024
    focal_quota: int = 1
    max_iterations: int = 20
    branch_cap: int = 16
    proposal_max_bins: int = 2000
    # Proposal resolution only; canonical acceptance still uses every site.
    proposal_min_sites_per_bin: int = 200
    dual_search_sweeps: int = 20
    window_blocks: int = 100
    count_repair_sweeps: int = 3
    # Final-component count-up budget; zero disables additions only.
    count_max_additions: int = 2
    # Count-up acceptance is leave-one-sample-out: an addition must still pay
    # its complexity cost after removing any single sample's score gain.
    count_addition_leave_one_sample_out: bool = True
    # Heuristic deep-refit screen in units of the existing per-founder cost.
    # None retains unscreened deep count search; this is not an upper bound.
    count_refit_deficit_multiple: float | None = 8.0
    # Retained in saved configuration records; tract search now ranks all pairs.
    interval_partners: int = 3
    # Founder-count decisions use the effective-carrier criterion
    # (founder/count_criterion.py): the per-sample cost is charged only to a
    # founder's effective carriers, with a leave-one-sample-out margin. Calls whose sample-bootstrap interval of the margin contains zero are
    # reported as ambiguous; they keep the point decision unless "keep_larger".
    ambiguous_count_action: str = "point_estimate"
    count_bootstrap_resamples: int = 2000
    count_bootstrap_level: float = 0.95
    # Refinement search (founder/converged_sweep.py): Gauss-Seidel sweeps of exact
    # window proposals and suffix exchanges until a sweep accepts nothing; every count
    # decision is made between converged fits: deletions at every level, additions at
    # the final count step (the last level). The count decisions choose K; the panel
    # kept at that K is the best-F converged fit computed at K during the count pass;
    # a count cycle (an accepted addition whose deletion round then accepts a
    # deletion) keeps the smaller K with its better-F panel and is reported as
    # ambiguous. Its caps are safety backstops only.
    # Window proposals solve the binned surrogate ("bin"); "site" (exact full-site
    # windows) is a validation option of the converged search.
    window_emission_resolution: str = "bin"
    # Guard on coarse (L1-span copy) moves of the converged search (environment
    # override HAPREC_COARSE_MOVE_GUARD). "genotype_fit" is production's: the move must
    # not lower E = F + penalty x switches. "shared_junction" vetoes only what that guard
    # exists for: a move that lowers E is still accepted when its F gain exceeds TOL
    # without credit for any saved switch at a one-way junction shared by two or more
    # samples (the same founder transition with overlapping evidence intervals: an
    # inherited crossover an artificial recombinant founder would absorb; see
    # founder/converged_sweep.py). "none" disables it.
    coarse_move_guard: str = "shared_junction"
    # Optional paired-interval exchange family of the converged search (not built).
    interval_moves: bool = False
    # Screens before every exact window programme decide only the work spent, never a
    # proposal: the box bound (the dual at zero multipliers) always runs; window_screen
    # adds the Lagrangian-dual iterations (screen_sweeps bounds that work).
    window_screen: bool = False
    screen_sweeps: int = 200
    # Safety backstops: reaching sweep_backstop marks a fit non-converged in its
    # diagnostics, frontier_backstop raises, count_round_backstop bounds K at the
    # incoming count K_in plus this multiple of K_in (K_in + K_in by default) and is
    # reported when reached; additions otherwise stop at the criterion (bound or
    # rejected best seed).
    sweep_backstop: int = 100
    frontier_backstop: int = 1_000_000
    count_round_backstop: int = 1

    def __post_init__(self):
        for name in ("beam_width", "maximum_beam_width", "focal_quota",
                     "max_iterations", "branch_cap", "proposal_max_bins", "proposal_min_sites_per_bin", "dual_search_sweeps",
                     "window_blocks", "count_repair_sweeps", "interval_partners"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.count_max_additions, bool)
                or int(self.count_max_additions) != self.count_max_additions
                or self.count_max_additions < 0):
            raise ValueError("count_max_additions must be a nonnegative integer")
        multiple = self.count_refit_deficit_multiple
        if multiple is not None and (not np.isfinite(multiple) or multiple <= 0):
            raise ValueError("count_refit_deficit_multiple must be positive or None")
        if self.maximum_beam_width < self.beam_width:
            raise ValueError("maximum_beam_width must be at least beam_width")
        founder_count_criterion.check_config(self)
        choices = dict(window_emission_resolution=("bin", "site"),
                       coarse_move_guard=("shared_junction", "genotype_fit", "none"))
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(allowed)}")
        for name in ("screen_sweeps", "sweep_backstop", "frontier_backstop", "count_round_backstop"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.interval_moves:
            raise NotImplementedError("the paired-interval family is not built for converged_sweep")


class _LeafKeyMap:
    def __init__(self, blocks):
        self.keys = [tuple(sorted(block.haplotypes)) for block in blocks]

    def get_key_from_dense(self, block, row):
        return self.keys[block][row]


def _local_selection(component, blocks):
    """Decode composed source provenance back to rows of the prepared inputs."""
    atomic, _, atomic_lengths, _ = paths.missing_aware_atomic_source_provenance(component)
    selected = np.empty((len(component.haplotypes), len(blocks)), np.int64)
    offset = 0
    for column, block in enumerate(blocks):
        local, _, lengths, _ = paths.missing_aware_atomic_source_provenance(block)
        end = offset + len(lengths)
        if not np.array_equal(atomic_lengths[offset:end], lengths):
            raise ValueError("founder refinement source spans do not match prepared blocks")
        lookup = {}
        for row, source in enumerate(local):
            lookup.setdefault(tuple(source), row)
        for founder, source in enumerate(atomic[:, offset:end]):
            selected[founder, column] = lookup[tuple(source)]
        offset = end
    if offset != atomic.shape[1]:
        raise ValueError("founder refinement did not cover the component's source rows")
    original = np.concatenate([
        block.discrete_haps[selected[:, column]]
        for column, block in enumerate(blocks)
    ], axis=1)
    if not np.array_equal(original, component.discrete_haps):
        raise ValueError("founder refinement provenance does not reproduce input alleles")
    return selected


def _macro_context(batch, l1_blocks):
    """Combine alternative assembled rows inside each unchanged L1 span."""
    if l1_blocks is None:
        return None
    starts = {int(block.positions[0]): index for index, block in enumerate(batch)}
    ends = {int(block.positions[-1]): index + 1 for index, block in enumerate(batch)}
    by_span = {}
    for block in l1_blocks:
        if int(block.positions[0]) not in starts or int(block.positions[-1]) not in ends:
            continue
        start, end = starts[int(block.positions[0])], ends[int(block.positions[-1])]
        rows = by_span.setdefault((start, end), {})
        # Stable deduplication preserves the preferred refined-path ordering.
        # Only complete prepared-row paths are offered, never inferred alleles.
        for row in _local_selection(block, batch[start:end]):
            rows.setdefault(tuple(row), row)
    groups = sorted(by_span)
    if not groups or all(end - start == 1 for start, end in groups):
        return None
    if (groups[0][0] != 0 or groups[-1][1] != len(batch)
            or any(left[1] != right[0] for left, right in zip(groups, groups[1:]))):
        raise ValueError("L1 founder context does not partition the final component")
    context = [np.asarray(list(by_span[group].values()), np.int64) for group in groups]
    return groups, context


def refine_components(prepared_blocks, components, neutral_probs, global_sites, *,
                      config=FounderRefinementConfig(), num_threads=1,
                      checkpoints=None, l1_blocks=None, cc_scale=0.5,
                      count_additions=True):
    """Run the converged refinement independently inside each phase component.

    count_additions: the count pass adds founders only at the final count step
    (the last level); earlier levels make deletion decisions only.
    checkpoints: the level's records (assembly.checkpoints.AssemblyCheckpoints
    scoped to ``<assembly>/l<k>/refine``): every component's refinement is
    recorded as it completes and reused on resume (founder/checkpoints.py), and
    long components record their converged-sweep states.
    """
    if not config.enabled:
        return components, {"enabled": False, "components": []}
    from .founder.checkpoints import RefinementCheckpoints
    records = None if checkpoints is None else RefinementCheckpoints(checkpoints, prepared_blocks)
    if len(components) > 1:
        from .founder.components import refine_independent_components
        return refine_independent_components(
            prepared_blocks, components, neutral_probs, global_sites,
            config=config, num_threads=num_threads, records=records,
            l1_blocks=l1_blocks, cc_scale=cc_scale, count_additions=count_additions)
    # A single component is refined in this process with the whole budget.
    from .founder.converged_sweep import level_diagnostics
    cached = None if records is None else records.load(0)
    if cached is not None:
        block, diagnostic = cached
        return BlockResults([block]), level_diagnostics([diagnostic], count_additions)
    output, diagnostics = _refine_serial_components(
        prepared_blocks, components, neutral_probs, global_sites,
        config=config, num_threads=num_threads,
        checkpoints=None if records is None else records.sweeps(0),
        l1_blocks=l1_blocks, cc_scale=cc_scale, count_additions=count_additions)
    if records is not None:
        records.save(0, output[0], diagnostics["components"][0])
    return output, diagnostics


def _refine_serial_components(prepared_blocks, components, neutral_probs, global_sites, *,
                      config=FounderRefinementConfig(), num_threads=1,
                      checkpoints=None, l1_blocks=None, cc_scale=0.5,
                      count_additions=True):
    """Converged founder refinement and count decisions (founder/converged_sweep.py).

    Every accepted endpoint retains the canonical full-site objective.
    Independent components and inner candidates share dynamic thread budgets.
    No truth, generation labels or pedigree enters this assembly fit.
    checkpoints: the converged-sweep records of a single component.
    """
    if not config.enabled:
        return components, {"enabled": False, "components": []}
    from .founder import converged_sweep
    return converged_sweep.refine_components(
        prepared_blocks, components, neutral_probs, global_sites, config=config,
        num_threads=num_threads, checkpoints=checkpoints, l1_blocks=l1_blocks,
        cc_scale=cc_scale, additions=count_additions)
