"""Release selected founder alleles using direct support and context witness evidence.

Owns LS21-LS22: conditional one-bit probability, favourable-carrier support,
context witness rescue and the assembly-facing LocalPanel. Must not fit a panel,
read records, schedule work or inspect process state.
Method: docs/methods/local_selection.md#release (WP_SE2 section 4.10).
Probability must pass min_call_probability; support is the maximum of carrier
count and mass. A matching context founder's actually released witness can
rescue insufficient direct support. Completion-filled calls are never witnesses.
Unknown-copy summaries preserve the no-depth slot and unknown_count_band policy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from ...contracts.panels import LocalPanel, unknown_copy_mass
from ...contracts.selection import PathFit, PathRelease
from ...core.ties import TieCounter, census, census_enabled
from .diagnostics import CensusSink
from .objective import ContextFounders
from .parameters import FAVOURABLE_CENSUS_WINDOW, UNKNOWN_CENSUS_WINDOW, PathReleaseConfig
from .path_model import PreparedBlock, release_statistics

__all__ = ["context_witnessed", "release", "local_panel"]


def context_witnessed(panel: np.ndarray, context: ContextFounders, witness: np.ndarray) -> np.ndarray:
    """bool[K, L]: cells whose allele equals the call of the context founder claiming the row in the panel's
    shortest description, where the context refit released that call (``witness``: one row per assembled founder,
    aligned with the founders ``context`` was built from)."""
    panel = np.asarray(panel, dtype=np.int64)
    out = np.zeros(panel.shape, dtype=np.bool_)
    for founder, row in context.matched(panel):
        out[row] |= (panel[row] == context.founders[founder]) & witness[context.source[founder]]
    return out


def release(
    fit: PathFit,
    prepared: PreparedBlock,
    config: PathReleaseConfig,
    ties: TieCounter | None = None,
    census_key: str | None = None,
    context: ContextFounders | None = None,
    witness: np.ndarray | None = None,
    *,
    census_sink: CensusSink | None = None,
) -> PathRelease:
    """The PathRelease of a selected fit. ``ties`` (maths.ties.TieCounter) counts the favourable band
    and the unknown-count band (one decision per release each; inside: some observed carrier cell's
    gain in the favourable band, some observed site's count lowered by the unknown-count band);
    ``census_key`` records the carrier cells' gains within FAVOURABLE_CENSUS_WINDOW of zero and the
    samples' smallest count gaps within UNKNOWN_CENSUS_WINDOW when the census is on. ``context``
    (objective.ContextFounders) and ``witness`` (the context refit's released mask per assembled founder) enable
    the context-witness rescue of the module docstring."""
    collect = census_key is not None and census_enabled(sink=census_sink)
    statistics = release_statistics(
        fit.panel, fit.frequencies, prepared, config.favourable_margin, config.unknown_count_band, keep_gains=collect
    )
    if ties is not None:
        # One decision per release; inside the tie when some observed carrier cell's gain lies in the
        # band (their number is PathRelease.favourable_band_cells).
        ties.note("ls.favourable.band", inside_tie=statistics["band_cells"] > 0)
        ties.note("ls.unknown.band", inside_tie=statistics["unknown_band_sites"] > 0)
    if collect:
        gains = statistics["sample_gains"]
        for sample, founder, site in np.argwhere(np.abs(gains) <= FAVOURABLE_CENSUS_WINDOW):
            census(
                "ls.favourable.band",
                f"{census_key}/n{sample}/a{founder}/s{site}",
                (0.0, float(gains[sample, founder, site])),
                sink=census_sink,
            )
        for sample in np.flatnonzero(statistics["unknown_gaps"] <= UNKNOWN_CENSUS_WINDOW):
            census(
                "ls.unknown.band",
                f"{census_key}/n{sample}",
                (0.0, float(statistics["unknown_gaps"][sample])),
                sink=census_sink,
            )
    probability = expit(-statistics["conditional_flip_gains"])
    supporters, mass = statistics["supporters"], statistics["carrier_mass"]
    confident = (probability >= config.min_call_probability) & (np.maximum(supporters, mass) >= config.min_supporters)
    context_cells = 0
    need = (probability >= config.min_call_probability) & ~confident
    if need.any() and context is not None and witness is not None and len(context):
        rescue = context_witnessed(fit.panel, context, witness) & need
        context_cells = int(rescue.sum())
        confident |= rescue
    occupancy = statistics["occupancy"]
    has_depth = statistics["has_depth"]
    slots = statistics["unknown_slots"]
    return PathRelease(
        calls=np.where(confident, fit.panel, -1).astype(np.int8),
        probability=probability,
        supporters=supporters,
        unknown_copy_fraction=float(occupancy[-1].sum()) / float(occupancy.sum()),
        has_depth=has_depth,
        unknown_slots=slots,
        unknown_mass=unknown_copy_mass(slots, has_depth),
        favourable_band_cells=int(statistics["band_cells"]),
        context_cells=context_cells,
    )


def local_panel(
    block: int, site_start: int, positions: np.ndarray, kept: np.ndarray, fit: PathFit, path_release: PathRelease
) -> LocalPanel:
    """The LocalPanel of a selected fit: released calls, the fit's allele probabilities and the
    favourable carrier counts at the discovery sites; -1 / 0.5 / 0 elsewhere."""
    k = len(fit.panel)
    length = len(positions)
    calls = np.full((k, length), -1, dtype=np.int8)
    calls[:, kept] = path_release.calls
    alt_probability = np.full((k, length), 0.5)
    alt_probability[:, kept] = fit.allele_probability
    support = np.zeros((k, length))
    support[:, kept] = path_release.supporters
    return LocalPanel(
        block=int(block),
        site_start=int(site_start),
        positions=np.ascontiguousarray(positions, dtype=np.int64),
        kept=np.ascontiguousarray(kept, dtype=np.bool_),
        calls=calls,
        alt_probability=alt_probability,
        support=support,
        has_depth=path_release.has_depth,
        unknown_slots=path_release.unknown_slots,
        unknown_mass=path_release.unknown_mass,
    )
