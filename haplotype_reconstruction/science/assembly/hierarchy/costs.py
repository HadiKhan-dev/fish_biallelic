"""Estimate batch working sets without selecting scientific outcomes.

The measured working-set envelope is reported to the engine. Branches use their maximum,
not their sum because they execute sequentially. No cap truncates a family here.
"""

import math
from dataclasses import replace

from haplotype_reconstruction.contracts.assembly import JoinTask
from haplotype_reconstruction.core.kernels import samples_per_thread

from ..join.emissions import sites_per_bin

# Cost only: estimated Python beam/archive residence in bytes per path-pair.
POOL_BYTES_PER_PATH_PAIR = 200


def batch_threads(task: JoinTask) -> int:
    """Useful sample width from the binned pair-recursion work of the batch."""
    sites = [sum(panel.sites for panel in task.prepared[c.start : c.stop]) for c in task.components]
    bin_size = sites_per_bin(sites, task.settings.join.max_bins)
    bins = sum(math.ceil(count / bin_size) for count in sites)
    branches = task.branches or (task.components,)
    founders = max(component.founders for branch in branches for component in branch)
    # Forward/backward pair recurrences make high-K samples substantial work.
    # The common work-aware grain keeps small pair grids coarse and lets large
    # ones use the live share even when the cohort contains few samples.
    grain = samples_per_thread(bins * founders * founders)
    return max(1, len(task.samples) // grain)


def batch_work(task: JoinTask) -> int:
    """Dimension-only ready priority; composition still visits genomic batch order.

    Sample/site work scales with the pair-state recursion and its O(K) refits.
    Incoming alternatives are separate executions on the same evidence domain.
    """
    branches = task.branches or (task.components,)
    sites = sum(panel.sites for panel in task.prepared)
    return len(task.samples) * sites * sum(max(c.founders for c in branch) ** 3 for branch in branches)


def batch_memory_bytes(task: JoinTask) -> int:
    """Upper estimate of one batch join's peak bytes over its informative samples S.

    The batch keeps a float32 evidence slice of its samples over its sites; as much again covers
    transient copies (the search's flat emission layout while it is built). Linker scans keep seven
    float32 log emissions and seven float64 weights per sample and proxy site, plus uint8 pair codes per
    proxy site; its concurrent gap fits keep forward and backward pair messages (2 x 8 S K_b^2 per
    block and gap). The search keeps float64 binned emissions (S, K_b * (K_b + 1) // 2, m_b) with their static
    totals (counted 1.25 times: the batch's peak is close to it when it dominates), an int32 painting
    (S, sum m_b), painting back pointers bounded by tensor_budget_mb, its exclusion forwards
    (S, sum m_b) per excluded row (at most K + 1), one chunk of per-sample bounds, and move arrays and
    beam lists of O(B q K^2) entries up to the saturation ladder's normal top (quota 16 x 2^4); a
    search whose ladder climbs on to the backstop, and its concurrent seed refits, reserve their own
    bytes (join.search). The linker's concurrent steps reserve nothing beyond this estimate.
    Measured peak/estimate ratios at K = 30, S = 320 are 0.37-0.80 across hierarchy levels,
    including 2000 bins and a quota-1024 pool; this headroom supports the working-set allowance.
    Worker private residence is held separately for the pool lifetime.
    """
    if task.branches:
        return max(batch_memory_bytes(replace(task, components=branch, branches=())) for branch in task.branches)
    from ..join import emissions as join_emissions
    from ..join import moves as join_moves
    from ..parameters import POOL_QUOTA_DOUBLINGS

    samples = len(task.samples)
    config = task.settings.join
    max_sites = task.settings.linker.max_sites
    site_counts = []
    founders = []
    for component in task.components:
        panels = task.prepared[component.start : component.stop]
        site_counts.append(sum(panel.sites for panel in panels))
        founders.append(component.founders)
    spb = join_emissions.sites_per_bin(site_counts, config.max_bins)
    span = sum(site_counts)
    total = config.tensor_budget_mb * 1024**2 + 2 * 12 * samples * span
    total_bins = 0
    for sites, haps in zip(site_counts, founders, strict=False):
        proxy_sites = min(sites, max_sites)
        ordered = haps * haps
        folded = haps * (haps + 1) // 2
        bins = math.ceil(sites / spb)
        total_bins += bins
        total += (
            samples * proxy_sites * 7 * 12
            + proxy_sites * (ordered + folded)
            + 10 * samples * folded * (bins + 1)
            + 4 * samples * bins
        )
    rows = max(founders)
    quota = config.paths_per_endpoint * 2**POOL_QUOTA_DOUBLINGS
    fits = (len(task.components) - 1) * 16 * samples * sum(k * k for k in founders)
    total += (
        8 * samples * total_bins * (rows + 2)
        + 8 * samples * join_moves.MOVES_PER_CHUNK
        + POOL_BYTES_PER_PATH_PAIR * len(task.components) * quota * rows * rows
        + fits
    )
    return int(total)
