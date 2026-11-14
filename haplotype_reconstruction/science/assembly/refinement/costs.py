"""Declare dimension-only memory estimates for refinement task admission.

Owns component, fit, proposal, beam and guard working-set models; never changes a
scientific cap, candidate, order or fallback. Method: docs/methods/founder_refinement.md#memory
(RC01/RC20, RS06). Parent tasks reserve all simultaneously live child estimates,
including the fixed exact frontier and the seeded search's dimension-derived arrays.
"""

from __future__ import annotations

import math

import numpy as np

from . import checkpoints
from .parameters import (
    COUNT_FANOUT_BYTES,
    FAMILY_BYTES,
    HYBRID_FRONTIER,
    INTERVAL_SEED_STARTS,
    POLISH_SHARE_MAX_BYTES,
    POLISH_SHARE_MIN_BYTES,
    POLISH_SHARE_TEMPLATE_BYTES,
    SCORE_LEDGER_BYTES,
    SEED_STARTS,
    WINDOW_STORE_MAX_BYTES,
    WINDOW_STORE_MIN_BYTES,
    WINDOW_STORE_TEMPLATE_BYTES,
)
from .polish_cache import search_store_bytes
from .window import PARTIAL_SITE_BYTES, RESIDENT_BYTES, resident_bytes, window_sites

__all__ = [
    "component_bytes",
    "component_work",
    "component_store_bytes",
    "fit_bytes",
    "traceback_bytes",
    "component_task_bytes",
]


def component_store_bytes(templates: int, founders: int) -> tuple[int, int]:
    """Budgets (bytes) of a component's window-result store and of its shared polish answers.

    Dimension-only: proportional to the number of window templates and K^2 (a count pass
    fits O(K) candidates in each of up to K rounds, each sweep solving every template),
    clamped to [WINDOW_STORE_MIN_BYTES, WINDOW_STORE_MAX_BYTES] and [POLISH_SHARE_MIN_BYTES,
    POLISH_SHARE_MAX_BYTES]. The stage reserves both with the component; they bound
    retention only (eviction recomputes the identical problem)."""
    scale = max(1, int(templates)) * max(1, int(founders)) ** 2
    window = min(WINDOW_STORE_MAX_BYTES, max(WINDOW_STORE_MIN_BYTES, WINDOW_STORE_TEMPLATE_BYTES * scale))
    polish = min(POLISH_SHARE_MAX_BYTES, max(POLISH_SHARE_MIN_BYTES, POLISH_SHARE_TEMPLATE_BYTES * scale))
    return int(window), int(polish)


# Bookkeeping and transients of one component task beyond its arrays (bytes); measured on std_n80 L1-L2 components
# (N = 80): whole-task peaks of 62-310 MiB above the worker's starting RSS, fits and window solves included.
COMPONENT_BASE_BYTES = 64 << 20


def seeded_search_bytes(samples, units, bins, rows, founders):
    """Dimension-only peak storage of one seeded focal solve, in bytes.

    Two gathered emission sets and contexts coexist while relevant samples are
    selected. Restriction scores, validity/taken masks, wishes, forward states
    and suffix covectors remain dimension-derived. O(N (B R K+B Q+B^2)+N W^2
    +W(B+K)) bytes; no live-memory or scientific decision enters this estimate.
    """
    states = founders * (founders + 1) // 2
    emissions = 16 * samples * (bins * rows * founders + bins * states + bins**2)
    flanks = 16 * samples * states * (units + 3)
    restrictions = 64 * (samples + 1) * units * (units + 1)
    covectors = 32 * (units + 1) * (bins + founders)
    endpoints = 24 * (1 + SEED_STARTS + INTERVAL_SEED_STARTS) * units
    return emissions + flanks + restrictions + covectors + endpoints


def component_bytes(panels, samples, *, min_sites_per_bin, max_bins) -> int:
    """Peak bytes of one component's workspace for the stage memory reservation: its log
    evidence and switch costs (window.py: the resident tables of a small component, 24 S L float64 codes 0-2, the
    partial sites' codes 3-6 and up to COST_ARRAYS float32 cost arrays; for a streaming component one window of the
    count pass's own kernels: each search's window is in its fit_bytes), a component evidence copy
    (12 S L float32, reserved even when a contiguous whole-span view is possible), the binned emissions
    (8 S R_b(R_b + 1)/2 bins_b per block) and the coded leaves (R_b x L_b int16, forward and reversed
    orientation), plus COMPONENT_BASE_BYTES. The fits' message arrays, the window solves and the score tables
    reserve their own bytes (additive estimates); the estimates depend only on component dimensions."""
    sites = sum(panel.sites for panel in panels)
    bin_size = max(int(min_sites_per_bin), math.ceil(sites / max_bins))
    emissions = sum(
        8 * samples * (panel.rows * (panel.rows + 1) // 2) * ((panel.sites + bin_size - 1) // bin_size)
        for panel in panels
    )
    leaves = 4 * sum(panel.rows * panel.sites for panel in panels)
    # One more cost array: an evicted array can stay referenced by an active search while its
    # replacement is built.
    # The polish beam's resident per-bin table (8 S J (3 + Rmax(Rmax + 1)/2), beam.rank_table)
    # and its bin resets (8 S J).
    bins = sum((panel.sites + bin_size - 1) // bin_size for panel in panels)
    rmax = max(panel.rows for panel in panels)
    beam_table = 8 * samples * bins * (4 + rmax * (rmax + 1) // 2)
    partials = sum(int(np.count_nonzero(np.any(panel.inference < 0, axis=0))) for panel in panels)
    tables = resident_bytes(samples, sites, partials)
    if not (resident_bytes(samples, sites, partials) <= RESIDENT_BYTES):
        width = min(sites, window_sites(samples))
        tables = samples * width * 28 + samples * min(partials, width) * PARTIAL_SITE_BYTES
    return tables + 12 * samples * sites + emissions + leaves + beam_table + COMPONENT_BASE_BYTES


def component_work(samples: int, sites: int, founders: int) -> int:
    """Dimension-based work of one component's count search (samples x sites x K^3): a deletion round
    fits O(K) panels whose message passes cost O(K^2) per sample and site. A ranking and threshold
    quantity, not a time: it orders component tasks and decides which record their fits."""
    return int(samples) * int(sites) * int(founders) ** 3


def fit_bytes(workspace, founders, guarded, cached_boundaries, polish=False):
    """Memory reservation of one fixed-K search: the message arrays its pass holds
    (stored backward values, + counts and the forward history of the guard), its proposal cache's
    boundaries, and the pass transients per junction (f/g starts and ends of up to 3K candidates, the
    lookahead and backward chain), its streaming window of a large component (window.py; 0 for a
    resident one), plus 32 MiB. Guard traceback scratch is
    reserved separately during acceptance; it is not resident during window proposals. A polish fit's chains
    and lookahead can span a long repair's 64 blocks."""
    states = founders * (founders + 1) // 2
    arrays = 4 if guarded else 2
    step = workspace.message_step or checkpoints.stride(founders, workspace.blocks, workspace.samples, guarded)
    ncheck = len(checkpoints.boundaries(workspace.blocks, step))
    # A pass has one two-segment score/count cache; proposals materialise their
    # flanks temporarily beside the previous sweep's proposal-cache copies.
    if step == 1:
        boundaries = (workspace.blocks + 1) * arrays + cached_boundaries
    else:
        boundaries = (
            ncheck * arrays
            + 2 * cached_boundaries
            + (2 if guarded else 1) * checkpoints.CACHE_SEGMENTS * (min(workspace.blocks, step) + 1)
        )
    if guarded:
        boundaries += 2 * ((workspace.blocks - 1) // 16 + 1)
    samples = workspace.samples
    span = max(48, 64) if polish else 48
    transients = 8 * samples * states * (3 * 3 * founders + 2 * span)
    return 8 * boundaries * samples * states + transients + workspace.windows.window_bytes() + (32 << 20)


def traceback_bytes(samples: int, sites: int, founders: int, threads: int) -> int:
    """Conservative peak of one guard's full-width tracing beyond its message arrays.

    Both old/new int64 event tensors coexist (48 N L bytes). Up to N L removed
    events add 24 N L. shared_savings expands each event into at most four
    transitions: five int64 entry arrays, two interval coordinates and an order
    index add 256 N L, and two masks add 2 N L. Canonicalisation, sorting and
    one-sample FIFO scratch are covered by another 200 L bytes. Each active thread
    holds int8[L,Q] traceback routes, int16[L,K] founder-maximum partners and int64 + float64[L]
    source and cost vectors, while boundary, reconstructed-prefix and per-thread messages/costs add
    64 N Q. Event costs add 24 N L (old and new traceback cost arrays and the flattened removed-event
    costs), and the per-event transition index (ev_first, ev_count: 16 N L) and
    per-transition shared flags (3 N L) another 19 N L. Width may grow to the entire component;
    using the whole width in admission is deliberate.
    """
    states = founders * (founders + 1) // 2
    active = min(samples, max(1, threads))
    return int((373 * samples + 200) * sites + active * sites * (states + 2 * founders + 16) + 64 * samples * states)


def component_task_bytes(panels, samples, founders, config, *, min_sites_per_bin, context, cores, incoming=()) -> int:
    """Parent estimate including simultaneously live child work, from dimensions.

    The count pass only deletes founders, so no panel exceeds the largest root K:
    the incoming K or an alternative root panel's own K. Each declared core may own
    one independent fit, proposal or shadow worker; sequential branch histories share
    these buffers. No free-memory reading or admission result enters a scientific decision.
    """
    blocks = len(panels)
    if blocks < 2 or founders < 2:
        return COMPONENT_BASE_BYTES
    sites = sum(panel.sites for panel in panels)
    roots = [founders]
    for family in incoming:
        roots.extend(len(panel) for panel in family.get("panels", ()) if len(panel))
    maximum_k = max(roots)
    q = maximum_k * (maximum_k + 1) // 2
    bin_size = max(int(min_sites_per_bin), math.ceil(sites / config.proposal_max_bins))
    bins = [(panel.sites + bin_size - 1) // bin_size for panel in panels]
    guarded = context is not None
    step = checkpoints.stride(maximum_k, blocks, samples, guarded)
    checks = len(checkpoints.boundaries(blocks, step))
    # Every distinct proposal flank is at a block boundary (two orientations).
    cached_boundaries = 2 * (blocks + 1)
    arrays = 4 if guarded else 2
    stored = (
        (blocks + 1) * arrays + cached_boundaries
        if step == 1
        else checks * arrays
        + 2 * cached_boundaries
        + (2 if guarded else 1) * checkpoints.CACHE_SEGMENTS * (min(blocks, step) + 1)
    )
    if guarded:
        stored += 2 * ((blocks - 1) // 16 + 1)
    search = 8 * stored * samples * q + 8 * samples * q * (9 * maximum_k + 128) + (32 << 20)
    fine_bins = [sum(bins[a : min(blocks, a + 8)]) for a in range(blocks)]
    maximum_bin = max(bins)
    maximum_units = min(8, blocks)
    maximum_total_bins = max(fine_bins)
    coarse = 0
    maximum_rows = max(panel.rows for panel in panels)
    if context is not None:
        groups, paths = context
        coarse_bins = [sum(bins[a:c]) for a, c in groups]
        maximum_bin = max(maximum_bin, max(coarse_bins))
        maximum_total_bins = max(maximum_total_bins, max(sum(coarse_bins[a : a + 4]) for a in range(len(groups))))
        widths = [maximum_k + len(rows) for rows in paths]
        maximum_rows = max(maximum_rows, max(widths))
        coarse = sum(
            sorted(
                (
                    8 * samples * (r * (r + 1) // 2) * b + 16 * samples * q * b
                    for r, b in zip(widths, coarse_bins, strict=True)
                ),
                reverse=True,
            )[:4]
        )
    proposal = (
        8 * samples * q * (4 * maximum_total_bins + 2 * maximum_units + 1)
        + 20 * samples * (q + maximum_k + maximum_bin + 2) * HYBRID_FRONTIER
        + seeded_search_bytes(samples, maximum_units, maximum_total_bins, maximum_rows, maximum_k)
        + coarse
    )
    width = min(sites, window_sites(samples))
    window = samples * width * (28 + PARTIAL_SITE_BYTES)
    guard = traceback_bytes(samples, sites, maximum_k, 1) if guarded else 0
    # Beam long repairs may span 64 blocks; coarse spans can be larger.
    beam_bins = max(maximum_total_bins, max(sum(bins[a : a + 64]) for a in range(blocks)))
    beam = 8 * samples * q * (6 * 64 + 12 + beam_bins) + 8 * samples * beam_bins * beam_bins
    score = 2 * maximum_k * sites + min(sites * q, 1 << 31) + 8 * samples
    # Import at call time: proposals itself imports seeded_search_bytes above.
    # Admission owns only dimensions; it never creates a proposer or a store.
    from .proposals import window_templates

    templates = window_templates(blocks, None if context is None else context[0])
    polish_stores = search_store_bytes(maximum_k, blocks, templates) if config.polish else 0
    child = search + window + max(proposal, guard, beam, score) + polish_stores
    # The count driver evaluates shadow families in waves whose TOTAL scratch is
    # bounded by COUNT_FANOUT_BYTES. Charging a full FAMILY_BYTES to every CPU
    # made small components reserve tens of GiB for scratch that cannot coexist.
    # One oversized score is already covered by the per-child score estimate.
    families = min(COUNT_FANOUT_BYTES, max(1, cores) * FAMILY_BYTES)
    # One window-result store and one shared polish-answer store serve every fit of the
    # component (roots, branches, polish); concurrent count histories share them.
    stores = sum(component_store_bytes(len(templates), maximum_k))
    # Count histories that run concurrently (count.count_pass: roots, then one continuation generation, each at most
    # alternative_branches) hold, between their fits, one round's candidate panels and score vectors each; their fits
    # are CPU-share children inside the per-core child reservation and their decisions take turns (one fan-out).
    # Histories at different counts can keep two more switch-cost arrays of a resident component referenced each.
    resident = resident_bytes(samples, sites, 0) <= RESIDENT_BYTES
    histories = int(config.alternative_branches) * (
        8 * maximum_k * blocks * (maximum_k + 2)
        + 16 * samples * (maximum_k + 2)
        + (8 * samples * sites if resident else 0)
    )
    return (
        component_bytes(panels, samples, min_sites_per_bin=min_sites_per_bin, max_bins=config.proposal_max_bins)
        + max(1, cores) * child
        + families
        # Optional score residence includes Python keys/metadata, not just N
        # floats. One shared ledger serves every concurrent fit and branch.
        + SCORE_LEDGER_BYTES
        + stores
        + histories
    )
