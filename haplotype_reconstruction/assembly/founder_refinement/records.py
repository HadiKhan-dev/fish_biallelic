"""Checkpoint records of one founder-refinement level (plain dicts of arrays and builtins).

Names under the level's record scope (``<P>/l<k>/refine``, P = the final assembly's section, passed
by the assembly loop):
  ``<n:05d>``              component n: {"component": the refined component
                           (assembly.records.encode_components of one component, count call
                           included), "refinement": encode_refinement(ComponentRefinement)};
                           written by the coordinating process as the component completes
  ``<n:05d>/fit/<label>``  resumable fit states (search.SweepEngine) of components with more than
                           RECORD_FITS_ABOVE_BLOCKS prepared blocks: ``<fit>.done`` (rows, summary,
                           weak junctions), ``<fit>.<fwd|rev>.done`` (rows, search record with its
                           sweep history, stored once), ``<fit>.<fwd|rev>.sweep<s>`` (rows, F, that
                           sweep's history entry, weak junctions)
  ``level``                the level summary; presence = the level is complete
The assembly section identity (bound by the assembly stage over its config, inputs and code) covers
every record. Readers for other stages and the evaluator: founder_panels, refinements,
refined_levels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from .. import records as assembly_records
from ...model.components import materialise
from .count import BestFReplacement, CountDecisionRecord
from .search import FitSummary, WeakJunction

# Components with more blocks than this also record their fit states (small components finish
# quickly; thousands of per-fit records would dominate the early levels' I/O).
RECORD_FITS_ABOVE_BLOCKS = 100
LEVEL = "level"


@dataclass(frozen=True)
class ComponentRefinement:
    """One component's refinement at one level (the content of its component record).

    component: index in the level; start, stop: prepared block range. changed: rows or K changed (release 2.1:
    the refined rows are in canonical founder order, as are the incoming join rows, so a pure reordering is no
    change).
    reason: None when refined; "single_block_or_founder" when the component has one block or one
    founder (passed through unchanged; no decision). founders_before / founders: K in and out.
    F0: exact objective of the incoming panel; F: of the returned panel (log units; None when not
    refined). changed_local_rows: cells that differ when K is unchanged (None otherwise).
    bin_size: surrogate bin (sites); coarse_units: L1 spans offering coarse moves (None without
    context); cost: the per-founder cost c. decisions: the count rounds in pass order. fits: every
    converged fit of the pass. count_cycle / cycle_undecided: the pass closed a count cycle (and
    whether it is undecided); count_limit: the reported limit K (2 x incoming K) and whether
    additions reached it. ambiguous: the call is ambiguous (any decision, a cycle or a restored
    panel failing its own deletion test). final_fit: label of the returned panel's fit;
    best_f_replacement: see BestFReplacement. weak_label_junctions: of the returned fit.
    alternatives (release 2.2): sets of alternative panels (panel keys) equally supported by the data -- the
    final fit's tied forward/reverse results and the count decisions' alternatives -- released flagged; the kept
    panel is one of them (decided by a stated prior, else the canonical key). Empty when the data decide.
    elapsed_seconds: wall time of the component's refinement. tie_events: maths.ties.TieCounter record of
    the component's tie-class decisions ({kind: {decisions, inside_tie, wide, boundary}}, release 2.1).
    """

    component: int
    start: int
    stop: int
    changed: bool
    reason: str | None
    founders_before: int
    founders: int
    F0: float | None = None
    F: float | None = None
    changed_local_rows: int | None = None
    bin_size: int | None = None
    coarse_units: int | None = None
    cost: float | None = None
    decisions: tuple = ()
    fits: tuple = ()
    count_cycle: bool = False
    cycle_undecided: bool = False
    count_limit: int | None = None
    count_limit_reached: bool = False
    ambiguous: bool = False
    final_fit: str | None = None
    best_f_replacement: BestFReplacement | None = None
    weak_label_junctions: tuple = ()
    elapsed_seconds: float = 0.0
    tie_events: dict = field(default_factory=dict)
    alternatives: tuple = ()


@dataclass(frozen=True)
class LevelSummary:
    """Census of one refinement level (its level record; logged as one line, LevelSummary.line).

    fits / nonconverged_fits: converged fits of every count pass and those that reached the sweep
    limit; max_sweeps, mean_sweeps: of their kept searches; max_frontier: the largest window
    programme frontier; frontier_fallbacks: programmes that kept the incumbent at the frontier cap
    (FounderRefinementConfig.frontier_bytes);
    count_limit_hits: passes whose additions reached 2 x incoming K; count_cycles (undecided);
    best_f_replacements; weak_label_junctions; deletions_without_sequence (deletion rounds whose
    incumbent had founders without a private cell); decided_by_sequence_part (calls the total-gain
    test passed and the sequence test rejected); hidden_sequence_failures (kept founders whose
    sequence part fails at a kept deletion round); ambiguous_count_calls: (component, kind, round,
    margin, band) of every ambiguous call. elapsed_seconds: the level's wall time.
    """

    level: int
    additions: bool
    min_sites_per_bin: int
    elapsed_seconds: float
    components: int
    fits: int
    nonconverged_fits: int
    max_sweeps: int
    mean_sweeps: float
    max_frontier: int
    frontier_fallbacks: int
    count_limit_hits: int
    count_cycles: int
    count_cycles_undecided: int
    best_f_replacements: int
    weak_label_junctions: int
    deletions_without_sequence: int
    decided_by_sequence_part: int
    hidden_sequence_failures: int
    ambiguous_count_calls: tuple

    def line(self) -> str:
        """The level's census log line (backstops, cycles and ambiguity are never silent)."""
        return (f"  [founder refinement L{self.level}] converged search: {self.fits} fits, "
                f"{self.nonconverged_fits} at the sweep limit, max sweeps {self.max_sweeps}, "
                f"max window frontier {self.max_frontier} ({self.frontier_fallbacks} frontier-cap "
                f"fallbacks), count limit hits {self.count_limit_hits}, count cycles "
                f"{self.count_cycles} ({self.count_cycles_undecided} undecided), ambiguous count "
                f"calls {len(self.ambiguous_count_calls)}, best-F panel replacements "
                f"{self.best_f_replacements}, weak label junctions {self.weak_label_junctions}, "
                f"deletions without sequence {self.deletions_without_sequence}, decided by the "
                f"sequence part {self.decided_by_sequence_part}, kept founders failing the "
                f"sequence test {self.hidden_sequence_failures}")


def encode_refinement(refinement) -> dict:
    """A ComponentRefinement as nested plain dicts, lists and builtins."""
    return asdict(refinement)


def decode_refinement(value):
    """Inverse of encode_refinement."""
    value = dict(value)
    value["decisions"] = tuple(CountDecisionRecord(**item) for item in value["decisions"])
    value["fits"] = tuple(FitSummary(**item) for item in value["fits"])
    value["weak_label_junctions"] = tuple(WeakJunction(**item)
                                          for item in value["weak_label_junctions"])
    if value["best_f_replacement"] is not None:
        value["best_f_replacement"] = BestFReplacement(**value["best_f_replacement"])
    return ComponentRefinement(**value)


class RefinementRecords:
    """The records of one refinement level on a record scope (store.RecordScope)."""

    def __init__(self, scope):
        self.scope = scope

    def load_component(self, number):
        """(refined FounderComponent, ComponentRefinement) of component `number`, or None."""
        payload = self.scope.load(f"{number:05d}")
        if payload is None:
            return None
        component, = assembly_records.decode_components(payload["component"])
        return component, decode_refinement(payload["refinement"])

    def save_component(self, number, component, refinement):
        self.scope.save(f"{number:05d}", {
            "component": assembly_records.encode_components([component]),
            "refinement": encode_refinement(refinement)})

    def fits(self, number, blocks):
        """Component `number`'s fit-state scope (one writer thread), or None for a component of at
        most RECORD_FITS_ABOVE_BLOCKS blocks."""
        if blocks <= RECORD_FITS_ABOVE_BLOCKS:
            return None
        return self.scope.scoped(f"{number:05d}/fit", nthreads=1)

    def load_level(self):
        """The LevelSummary when the level is complete, else None."""
        value = self.scope.load(LEVEL)
        if value is None:
            return None
        return LevelSummary(**dict(value, ambiguous_count_calls=tuple(value["ambiguous_count_calls"])))

    def save_level(self, summary):
        self.scope.save(LEVEL, asdict(summary))


# ------------------------------------------------------------------ readers
def refined_levels(file, prefix=assembly_records.FINAL_PREFIX):
    """The levels of the assembly section `prefix` whose refinement is complete, ascending."""
    pattern = re.compile(re.escape(f"{prefix}/l") + r"(\d+)/refine/" + LEVEL)
    return sorted(int(match.group(1)) for key in file.keys(f"{prefix}/l")
                  if (match := pattern.fullmatch(key)))


def _level_components(file, level, prefix):
    scope = f"{prefix}/l{level}/refine/"
    numbers = sorted(int(key[len(scope):]) for key in file.keys(scope)
                     if key[len(scope):].isdigit())
    return [file.get(f"{scope}{number:05d}") for number in numbers]


def refinements(file, level=None, prefix=assembly_records.FINAL_PREFIX):
    """The ComponentRefinements of refined level `level` (None: the last refined level), or None."""
    levels = refined_levels(file, prefix)
    if not levels or (level is not None and level not in levels):
        return None
    level = levels[-1] if level is None else level
    return tuple(decode_refinement(payload["refinement"])
                 for payload in _level_components(file, level, prefix))


def founder_panels(file, level=None, prefix=assembly_records.FINAL_PREFIX):
    """One contig's refined founder panels as ComponentPanels (positions, released and inference
    calls, founder labels 0..K-1 = row indices), materialised over the assembly's prepared panels,
    or None while absent. level k: the refined level k (base stage assembly_l<k>); None: the last
    refined level, the final count step (base stage founder_refinement)."""
    levels = refined_levels(file, prefix)
    if not levels or (level is not None and level not in levels):
        return None
    level = levels[-1] if level is None else level
    completion = f"{prefix}/{assembly_records.COMPLETION}"
    if not file.has(completion):
        return None
    prepared = assembly_records.decode_prepared(file.get(completion, items=("prepared",))["prepared"])
    components = tuple(assembly_records.decode_components(payload["component"])[0]
                       for payload in _level_components(file, level, prefix))
    return materialise(components, prepared)
