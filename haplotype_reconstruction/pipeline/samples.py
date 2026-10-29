"""The sample axis of a run, its optional metadata table, and parent eligibility built from them.

The axis is every sample of the source, in source order (TroMau keeps its G0 rows, as base). A metadata workbook
([samples]) supplies generation and sex labels per sample: labels are read as text, generation labels starting
with a listed prefix read as that prefix, and allowed labels are checked when listed. Eligibility
([pedigree.eligibility]) is built here, before any computation, so a wrong workbook, sheet or sample list fails at
once (base bug: the crosses read their workbooks only after painting).

Cohort policies describe what the design asserts, never individual parentage: a target's candidates are the
samples of its parent generation; generation steps are design chronology; opposite-sex pairs constrain
two-parent configurations only.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class SampleTable:
    """The run's samples: ids in axis order, and per sample its metadata labels (None without metadata)."""

    sample_ids: tuple[str, ...]
    generations: tuple[str, ...] | None = None
    sexes: tuple[str, ...] | None = None
    reference_generation: str = ""

    @property
    def reference_rows(self) -> tuple[int, ...]:
        """Rows of the reference generation (e.g. G0) for the reference-consistency report."""
        if not self.reference_generation or self.generations is None:
            return ()
        return tuple(row for row, label in enumerate(self.generations) if label == self.reference_generation)

    def composition(self) -> dict:
        """Samples per generation label."""
        counts = {}
        for label in self.generations or ():
            counts[label] = counts.get(label, 0) + 1
        return counts


def _text(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value)


def read_sample_table(section, sample_ids) -> SampleTable:
    """The sample table of a run ([samples] section); without a workbook, ids only.

    Every sample of the axis must occur exactly once in the workbook (by the id column); rows of other samples
    are ignored.
    """
    sample_ids = tuple(str(name) for name in sample_ids)
    if section.metadata is None:
        return SampleTable(sample_ids)
    import pandas as pd
    path = Path(section.metadata)
    if path.suffix.lower() in (".tsv", ".txt", ".csv"):
        # Delimited text (2.17; e.g. a simulation's explicit pedigree table): no sheet, every value as text.
        frame = pd.read_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", dtype=str,
                            keep_default_na=False)
    else:
        try:
            frame = pd.read_excel(path, sheet_name=section.sheet)
        except ValueError as error:
            raise ValueError(f"{path}: cannot read sheet {section.sheet!r}: {error}") from None
    columns = (section.id_column, section.generation_column, section.sex_column)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} [{section.sheet}]: missing columns {', '.join(missing)}")
    rows = frame.loc[frame[section.id_column].notna() & (frame[section.id_column] != ""), list(columns)]
    ids = [_text(value) for value in rows[section.id_column]]
    by_id, axis = {}, set(sample_ids)
    for name, generation, sex in zip(ids, rows[section.generation_column], rows[section.sex_column]):
        if name in by_id and name in axis:
            raise ValueError(f"{path} [{section.sheet}]: duplicate rows for sample {name!r}")
        by_id[name] = (_text(generation), _text(sex).upper())
    absent = [name for name in sample_ids if name not in by_id]
    if absent:
        shown = ", ".join(absent[:5]) + (" ..." if len(absent) > 5 else "")
        raise ValueError(f"{path} [{section.sheet}]: {len(absent)} input sample(s) missing: {shown}")
    generations, sexes = [], []
    for name in sample_ids:
        generation, sex = by_id[name]
        for prefix in section.generation_prefixes:
            if generation.startswith(prefix):
                generation = prefix
                break
        generations.append(generation)
        sexes.append(sex)
    if section.generations:
        invalid = sorted(set(generations) - set(section.generations))
        if invalid:
            raise ValueError(f"{path}: unsupported generation labels: {', '.join(map(repr, invalid))}")
    if section.sexes:
        invalid = sorted(set(sexes) - set(section.sexes))
        if invalid:
            raise ValueError(f"{path}: unsupported sex labels: {', '.join(map(repr, invalid))}")
    return SampleTable(sample_ids, tuple(generations), tuple(sexes), section.reference_generation)


def cohort_policy(parent_generations: dict, opposite_sex_pairs: bool) -> tuple[str, tuple[str, ...]]:
    """Policy name and stated assumptions of a cohort design (they enter the eligibility identity and reports)."""
    steps = sorted(parent_generations.items())
    name = ("cohorts_" + "_".join(f"{child.lower()}_from_{parent.lower()}" for child, parent in steps)
            + ("_opposite_sex" if opposite_sex_pairs else "_any_pair") + "_v1")
    targets = " and ".join(child for child, _ in steps)
    assumptions = [
        f"Only metadata-labelled {targets} samples are inference targets.",
        *(f"{child} samples admit {parent} samples as observed-parent candidates." for child, parent in steps),
        "Generation labels constrain eligibility but do not establish individual parentage.",
        "The explicit generation steps support candidate direction, not individual parentage truth.",
        "Samples of other generations are neither targets nor candidates.",
    ]
    if opposite_sex_pairs:
        assumptions.append("Two-observed-parent configurations require one recorded female and one recorded male "
                           "in the eligible parent cohort; parent order remains arbitrary.")
    return name, tuple(assumptions)


def build_eligibility(config, table: SampleTable):
    """ParentEligibility of the run ([pedigree.eligibility]); None = unrestricted (every sample a candidate)."""
    from ..pedigree.eligibility import ParentEligibility
    section = config.pedigree.eligibility
    if section.policy == "none":
        return None
    if section.policy == "constraints":
        constraints = json.loads(Path(section.constraints).read_text())
        return ParentEligibility.from_constraints(table.sample_ids, constraints)
    if table.generations is None:
        raise ValueError("cohort eligibility needs a metadata workbook ([samples] metadata)")
    steps = dict(section.parent_generations)
    policy, assumptions = cohort_policy(steps, section.opposite_sex_pairs)
    samples = config.samples
    return ParentEligibility.from_cohorts(
        table.sample_ids, table.generations, steps, sexes=table.sexes,
        require_opposite_sex_pairs=section.opposite_sex_pairs, policy=policy, assumptions=assumptions,
        source_fields=(samples.id_column, samples.generation_column, samples.sex_column))
