"""Run the diplotype exchange on the refined components of a final assembly and keep its record.

Owns: execution of science.assembly.exchange on each component of the final role's last level (the refinement
workspace and engine that score a panel exactly), the ordinary refinement fit that follows a pass with
commitments, the component's equal-F family after it, and the product's exchange record (prepared rows appended
after completion). Science decides every move; nothing here estimates an allele.
Method: docs/methods/founder_refinement.md#diplotype-exchange.

A component's step: exchange passes (science.exchange_pass: one commitment per round until none passes); after a
pass with commitments the component gets the ordinary fixed-K fit of its level (SweepEngine.converge, with the
polish when the run's refinement has one) on the panels that now hold the exchange rows; if that fit changes the
panel the exchange pass runs again; the step ends when a pass commits nothing. Every commitment raises the exact
objective by more than the acceptance quantum and the fit never lowers it, so the step ends.

Components are worked one after the other inside one task: every final assembly seen so far has exactly one
component per contig (119 of 119 contigs of the 4.21 development roots), and each component's kernels use the
task's whole team.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from haplotype_reconstruction.contracts.assembly import ReleaseFamily
from haplotype_reconstruction.contracts.panels import FounderComponent, PreparedPanel
from haplotype_reconstruction.core.errors import InvariantError
from haplotype_reconstruction.science.assembly import orders
from haplotype_reconstruction.science.assembly.exchange import exchange_pass, extended_identity, with_exchange_row
from haplotype_reconstruction.science.assembly.refinement.component_workspace import (
    ComponentWorkspace,
    component_evidence,
)
from haplotype_reconstruction.science.assembly.refinement.costs import component_store_bytes
from haplotype_reconstruction.science.assembly.refinement.count import _terminal_family
from haplotype_reconstruction.science.assembly.refinement.fit import SweepEngine
from haplotype_reconstruction.science.assembly.refinement.parameters import ACCEPTANCE_GAIN_FLOOR
from haplotype_reconstruction.science.assembly.refinement.reductions import panel_objective
from haplotype_reconstruction.science.assembly.refinement.run import MacroContext, RefinementSettings
from haplotype_reconstruction.science.assembly.refinement.window import RESIDENT_BYTES, resident_bytes, window_sites

from .storage import Windows

SCHEMA = 3
# The reason of an equal-F family of one fixed-K fit (count._terminal_family); such a family describes the panel
# it was computed on and is replaced when the exchange changes that panel.
_EQUAL_F = "same_k_equal_F"


def _windows(evidence, kept, partial, hazards):
    samples, sites = evidence.shape[:2]
    resident = resident_bytes(samples, sites, int(partial.sum())) <= RESIDENT_BYTES
    return Windows(evidence, kept, partial, hazards, resident=resident, capacity=window_sites(samples))


def _workspace(panels, latent, evidence, settings: RefinementSettings) -> ComponentWorkspace:
    config = settings.config
    return ComponentWorkspace(
        tuple(panels),
        evidence,
        min_sites_per_bin=settings.min_sites_per_bin,
        max_bins=config.proposal_max_bins,
        genetic_map=settings.genetic_map,
        genotype_error=config.genotype_error,
        reset_rate=config.reset_rate,
        map_probability_floor=config.map_probability_floor,
        windows_factory=_windows,
        latent=None if all(item is None for item in latent) else tuple(latent),
    )


def has_candidates(component: FounderComponent) -> bool:
    """Whether a component can hold an exchange candidate: a third founder row to anchor on."""
    return int(component.founders) >= 3 and int(component.stop) > int(component.start)


def _empty_report(number: int, component: FounderComponent) -> dict[str, Any]:
    return dict(
        component=int(number),
        founders=int(component.founders),
        blocks=int(component.stop - component.start),
        passes=[],
        commitments=0,
    )


def empty_exchange(components: Sequence[FounderComponent] | int = 0) -> dict[str, Any]:
    """The exchange record of a product without exchange rows."""
    record = encode_exchange([])
    reports = [] if isinstance(components, int) else [_empty_report(n, c) for n, c in enumerate(components)]
    record["report"] = dict(components=reports, commitments=0)
    return record


def exchange_component(
    number: int,
    component: FounderComponent,
    panels: Sequence[PreparedPanel],
    evidence: np.ndarray,
    context: MacroContext | None,
    settings: RefinementSettings,
    families: Sequence[ReleaseFamily],
    team: Any,
) -> tuple[FounderComponent, list[tuple], tuple[ReleaseFamily, ...], dict[str, Any]]:
    """One component's exchange step: (component, exchange rows, the component's release families, report).

    panels: the completion's prepared panels of the component; evidence: float32 (S, L_c, 3) its centred scoring
    view (component_evidence); context, settings: those of its level's refinement; families: the release families
    lying in the component; team: the task's thread team. Exchange rows: [(component-local block, source row,
    start, stop, alleles)] in commitment order. Without a commitment everything is returned unchanged.
    """
    rows = np.ascontiguousarray(component.rows, np.int64)
    founders, blocks = rows.shape
    report = _empty_report(number, component)
    families = tuple(families)
    if not has_candidates(component):
        return component, [], families, report
    panels = list(panels)
    latent: list[np.ndarray | None] = [None] * blocks
    entries: list[tuple] = []
    scoring = replace(settings.config, polish=False)  # exact panel scores only: no proposer is built
    floor = float(ACCEPTANCE_GAIN_FLOOR)
    with team.scope():
        base = _workspace(panels, latent, evidence, settings)

        def build(trial_panels, trial_latent):
            return SweepEngine(_workspace(trial_panels, trial_latent, evidence, settings), None, scoring)

        live: tuple = ()
        while True:
            start = rows.copy()
            rows, panels, latent, added, outcome = exchange_pass(
                rows, panels, latent, evidence, base.kept, base.hazards, build, team
            )
            entries.extend(added)
            passed = dict(outcome, rows_start=start.tolist(), rows_committed=rows.tolist(), polish=None)
            report["passes"].append(passed)
            if not outcome["accepted"]:
                break
            report["commitments"] += len(outcome["accepted"])
            engine = SweepEngine(
                _workspace(panels, latent, evidence, settings),
                context,
                settings.config,
                None,
                f"component{number}.exchange",
                counter=orders.SharedCounter(),
                census_prefix=f"L{settings.level}/{int(component.start)}-{int(component.stop)}/K{int(founders)}/exchange",
                census=settings.census,
                source_start=component.start,
                reverse=settings.reverse,
            )
            before = float(panel_objective(np.asarray(engine.per_sample(rows, team), np.float64)))
            if before < float(outcome["F_committed"]) - floor:
                raise InvariantError("an exchange commitment lowered the exact objective")
            label = f"exchange.polish{len(report['passes'])}"
            with engine.component_store(*component_store_bytes(len(engine.windows.templates), founders)):
                fit = engine.converge(rows, label, team, polish=bool(getattr(settings.config, "polish", False)))
                chosen, scores, live = _terminal_family(
                    engine, [fit], fit, engine.per_sample(fit.rows, team), team
                )
            polished = np.ascontiguousarray(chosen.rows, np.int64)
            after = float(panel_objective(np.asarray(scores, np.float64)))
            if after < before - floor:
                raise InvariantError("the fit after an exchange lowered the exact objective")
            changed = orders.panel_key(polished) != orders.panel_key(rows)
            passed["polish"] = dict(
                label=label, F_before=before, F_after=after, changed=bool(changed), converged=bool(fit.summary.converged)
            )
            rows = polished
            if not changed:
                break
    if not entries:
        return component, [], families, report
    rows, _ = orders.canonical_rows(rows)
    report["rows_final"] = rows.tolist()
    kept = tuple(family for family in families if family.get("reason") != _EQUAL_F)
    fresh = tuple(
        ReleaseFamily.from_mapping(
            dict(family, start=int(component.start), stop=int(component.stop), source_level=int(settings.level))
        )
        for family in live
    )
    report["families"] = dict(replaced=len(families) - len(kept), equal_F=len(fresh))
    changed_component = replace(component, rows=np.ascontiguousarray(rows, np.int32))
    return changed_component, entries, (*kept, *fresh), report


def exchange_components(
    components: Sequence[FounderComponent],
    prepared: Sequence[PreparedPanel],
    scoring: np.ndarray,
    contexts: Sequence[MacroContext | None],
    settings: RefinementSettings,
    families: Sequence[ReleaseFamily],
    team: Any,
) -> tuple[tuple[FounderComponent, ...], dict[str, Any], tuple[ReleaseFamily, ...]]:
    """The final components after the exchange step, the product's exchange record and its release families.

    components: the refined components of the last level; prepared: the completion's prepared panels (never
    changed; the record rebuilds the product's panels from them, apply_exchange); scoring: float32 (S, L, 3) the
    contig's centred scoring view; contexts: each component's L1 context; settings: the level's refinement
    settings; families: the level's release families; team: the task's thread team.
    """
    result, reports, rows = [], [], []
    remaining = list(families)
    for number, component in enumerate(components):
        panels = tuple(prepared[component.start : component.stop])
        if not has_candidates(component):
            result.append(component)
            reports.append(_empty_report(number, component))
            continue
        own = [f for f in remaining if int(f["start"]) >= component.start and int(f["stop"]) <= component.stop]
        team.sample_work = int(sum(panel.sites for panel in panels)) * int(component.founders) ** 2
        changed, added, own_after, report = exchange_component(
            number, component, panels, component_evidence(scoring, panels), contexts[number], settings, own, team
        )
        if added:
            # the component's families, replaced in place (families are compared by identity: they hold arrays)
            held = [index for index, f in enumerate(remaining) if any(f is g for g in own)]
            position = held[0] if held else len(remaining)
            remaining = [f for f in remaining if not any(f is g for g in own)]
            remaining[position:position] = list(own_after)
        result.append(changed)
        reports.append(report)
        rows.extend((component.start + block, source, start, stop, alleles) for block, source, start, stop, alleles in added)
    record = encode_exchange(rows)
    record["report"] = dict(components=reports, commitments=sum(r["commitments"] for r in reports))
    return tuple(result), record, tuple(remaining)


def encode_exchange(rows: Sequence[tuple[int, int, int, int, np.ndarray]]) -> dict[str, Any]:
    """The exchange rows in commitment order, as a few arrays: per row its prepared panel, the source row it
    copies, the block-local site range of the run and the alleles on that range. Everything a row publishes follows
    from these and the completion's panel (science.assembly.exchange.with_exchange_row)."""
    return dict(
        schema=SCHEMA,
        block=np.asarray([r[0] for r in rows], np.int64),
        source=np.asarray([r[1] for r in rows], np.int64),
        start=np.asarray([r[2] for r in rows], np.int64),
        stop=np.asarray([r[3] for r in rows], np.int64),
        alleles=np.concatenate([np.asarray(r[4], np.int8) for r in rows]) if rows else np.empty(0, np.int8),
    )


def exchange_rows(record: Mapping[str, Any] | None) -> list[tuple[int, int, int, int, np.ndarray]]:
    """[(prepared panel, source row, start, stop, alleles)] of a record, in commitment order."""
    if not record or not len(record["block"]):
        return []
    if int(record.get("schema", 0)) != SCHEMA:
        raise ValueError("unsupported exchange record")
    rows, cell = [], 0
    for block, source, start, stop in zip(
        np.asarray(record["block"]).tolist(),
        np.asarray(record["source"]).tolist(),
        np.asarray(record["start"]).tolist(),
        np.asarray(record["stop"]).tolist(),
        strict=True,
    ):
        rows.append((block, source, start, stop, np.asarray(record["alleles"][cell : cell + stop - start], np.int8)))
        cell += stop - start
    return rows


def apply_exchange(prepared: Sequence[PreparedPanel], record: Mapping[str, Any] | None) -> tuple[PreparedPanel, ...]:
    """The completion's prepared panels with the record's exchange rows appended, in commitment order (a later row
    may copy an earlier exchange row of its block)."""
    panels = list(prepared)
    for block, source, start, stop, alleles in exchange_rows(record):
        panels[block] = with_exchange_row(panels[block], source, start, stop, alleles)
    return tuple(panels)


def latent_identities(
    prepared: Sequence[PreparedPanel], record: Mapping[str, Any] | None
) -> tuple[np.ndarray | None, ...]:
    """Per prepared panel of the completion: the latent identities its exchange rows give the extended panel
    (None for a panel without exchange rows); what a refinement workspace on the product's panels needs
    (ComponentWorkspace(latent=...))."""
    counts = [panel.rows for panel in prepared]
    latent: list[np.ndarray | None] = [None] * len(counts)
    for block, source, start, stop, _ in exchange_rows(record):
        latent[block] = extended_identity(latent[block], counts[block], prepared[block].sites, source, start, stop)
        counts[block] += 1
    return tuple(latent)


__all__ = [
    "SCHEMA",
    "apply_exchange",
    "empty_exchange",
    "encode_exchange",
    "exchange_component",
    "exchange_components",
    "exchange_rows",
    "has_candidates",
    "latent_identities",
]
