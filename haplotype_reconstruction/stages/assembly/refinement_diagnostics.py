"""Summarise refinement diagnostics after the pure solver has returned.

Per-component and per-fit detail remains in the existing refinement records. The
level census carries only totals and counts of affected components; the terminal
assembly logs one summary per chromosome and fit stage across all levels.
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from haplotype_reconstruction.contracts.assembly import ComponentRefinement
from haplotype_reconstruction.engine.records import record_diagnostic

_DESCRIPTIONS = {
    "nonconverged_fits": "sweep limit",
    "exchange_cap": "exchange round cap",
    "round_cap": "content round cap",
    "escape_beam_binding": "beam-escape width",
    "long_beam_binding": "long-repair width",
    "beam_fallbacks": "window beam fallback",
    "beam_truncations": "window beam truncation",
    "frontier_fallbacks": "frontier cap (incumbent retained)",
}


def refinement_diagnostics(results: Iterable[ComponentRefinement]) -> dict:
    """Reduce exactly the events reported by component RuntimeWarnings.

    Totals count every fit event, grouping polish alternatives together and all
    count-search fits together. Each component is counted once per diagnostic/stage.
    All individual fit operands stay in component records.
    """
    totals: dict[str, Counter] = {}
    affected: dict[str, dict[str, set[int]]] = {}
    for result in results:
        for fit in result.fits:
            counts = dict(fit.work_counters)
            counts["nonconverged_fits"] = int(not fit.converged)
            counts["frontier_fallbacks"] = fit.frontier_fallbacks
            stage = "polish" if fit.label.split(".", 1)[0] == "polish" else "count"
            _accumulate(totals, affected, stage, result.component, counts)
    return {
        label: {name: dict(total=count, components=len(affected[label][name])) for name, count in counts.items()}
        for label, counts in totals.items()
    }


def _accumulate(
    totals: dict[str, Counter],
    affected: dict[str, dict[str, set[int]]],
    label: str,
    component: int,
    counts: Mapping[str, int],
) -> None:
    for name in _DESCRIPTIONS:
        count = counts.get(name, 0)
        if count:
            totals.setdefault(label, Counter())[name] += int(count)
            affected.setdefault(label, {}).setdefault(name, set()).add(component)


def report_refinement(contig: str, levels: Iterable[Mapping[str, Any]]) -> None:
    """Log one chromosome/stage line with event totals and affected component counts.

    Components at different hierarchy levels are separate refinement units.
    """
    totals: dict[str, dict[str, Counter]] = {}
    for level in levels:
        for label, diagnostics in level.get("cap_diagnostics", {}).items():
            for name, counts in diagnostics.items():
                totals.setdefault(label, {}).setdefault(name, Counter()).update(counts)
    for label, diagnostics in sorted(totals.items()):
        detail = ", ".join(
            f"{_DESCRIPTIONS[name]} {counts['total']} ({counts['components']} components)"
            for name, counts in diagnostics.items()
        )
        record_diagnostic(
            f"[refinement {contig} {label}] {detail}; exact acceptance retained; "
            "detail in assembly/l*/refined component records"
        )
