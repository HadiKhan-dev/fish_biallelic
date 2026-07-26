"""Explicit Tropheops cross design and candidate eligibility.

The compatibility design reproduces the candidate construction used by the
selected V7-margin analysis: every sequenced male G0 may be paired with every
sequenced female G0 for F1 reconstruction, and every sequenced male F1 may be
paired with every sequenced female F1 for F2 inference. Generation and sex are
eligibility fields, not parentage ground truth.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CONTIGS = tuple(
    [f"chr{chromosome}" for chromosome in range(1, 21)] + ["chr22", "chr23"]
)


@dataclass(frozen=True)
class TropheopsV7Design:
    """Auditable candidate rules for exact V7-margin compatibility."""

    g0_generation: str = "G0"
    f1_generation: str = "F1"
    f2_generation: str = "F2"
    father_sex: str = "M"
    mother_sex: str = "F"
    expected_g0_count: int = 4
    expected_f1_count: int = 16
    expected_f2_count: int = 96
    expected_g0_fathers: int = 1
    expected_g0_mothers: int = 3
    expected_f1_fathers: int = 8
    expected_f1_mothers: int = 8

    def as_dict(self):
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


COMPATIBILITY_DESIGN = TropheopsV7Design()


def candidate_sets(metadata, design=COMPATIBILITY_DESIGN):
    """Build all sex-compatible candidate pairs under an explicit design.

    This function deliberately does not use family names, alias content,
    inferred relationships, or previous pedigree outputs for eligibility. It
    applies the stated generation and sex rules and checks compatibility-mode
    cohort sizes. Aliases are nevertheless required to be non-empty and unique
    because they are emitted as auditable output identifiers.
    """
    required = {"sample_index", "Sample", "Alias", "Generation", "Sex"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata lacks required columns: {sorted(missing)}")
    if metadata["sample_index"].duplicated().any():
        raise ValueError("metadata sample_index values must be unique")
    aliases = metadata["Alias"].fillna("").astype(str).str.strip()
    if aliases.eq("").any() or aliases.duplicated().any():
        raise ValueError("metadata aliases must be non-empty and unique")

    def select(generation, sex=None):
        selected = metadata["Generation"].eq(generation)
        if sex is not None:
            selected &= metadata["Sex"].eq(sex)
        return metadata.loc[selected, "sample_index"].to_numpy(np.int64)

    g0_children = select(design.g0_generation)
    f1_children = select(design.f1_generation)
    f2_children = select(design.f2_generation)
    g0_fathers = select(design.g0_generation, design.father_sex)
    g0_mothers = select(design.g0_generation, design.mother_sex)
    f1_fathers = select(design.f1_generation, design.father_sex)
    f1_mothers = select(design.f1_generation, design.mother_sex)
    observed = {
        "g0": len(g0_children),
        "f1": len(f1_children),
        "f2": len(f2_children),
        "g0_fathers": len(g0_fathers),
        "g0_mothers": len(g0_mothers),
        "f1_fathers": len(f1_fathers),
        "f1_mothers": len(f1_mothers),
    }
    expected = {
        "g0": design.expected_g0_count,
        "f1": design.expected_f1_count,
        "f2": design.expected_f2_count,
        "g0_fathers": design.expected_g0_fathers,
        "g0_mothers": design.expected_g0_mothers,
        "f1_fathers": design.expected_f1_fathers,
        "f1_mothers": design.expected_f1_mothers,
    }
    if observed != expected:
        raise RuntimeError(
            "metadata does not match the exact Tropheops V7 compatibility "
            f"design: observed={observed}, expected={expected}"
        )

    def pair_arrays(fathers, mothers):
        parents = np.concatenate((fathers, mothers))
        local = {int(value): index for index, value in enumerate(parents)}
        global_pairs = np.asarray(
            [(int(father), int(mother))
             for father in fathers for mother in mothers],
            dtype=np.int64,
        )
        local_pairs = np.asarray(
            [(local[int(father)], local[int(mother)])
             for father, mother in global_pairs],
            dtype=np.int64,
        )
        return parents, global_pairs, local_pairs

    g0_parents, g0_pairs, g0_pairs_local = pair_arrays(
        g0_fathers, g0_mothers
    )
    f1_parents, f1_pairs, f1_pairs_local = pair_arrays(
        f1_fathers, f1_mothers
    )
    if not np.array_equal(f1_children, f1_parents):
        raise RuntimeError(
            "exact V7 compatibility requires the ordered F1 cohort to equal "
            "the male-then-female candidate-parent order; interleaved F1 "
            "metadata/BCF order requires an explicitly validated reorder"
        )
    return {
        "g0_parents": g0_parents,
        "g0_pairs": g0_pairs,
        "g0_pairs_local": g0_pairs_local,
        "f1_parents": f1_parents,
        "f1_pairs": f1_pairs,
        "f1_pairs_local": f1_pairs_local,
        "f1_children": f1_children,
        "f2_children": f2_children,
    }



SEED_STATUSES = frozenset({
    "exact_stable_inferred",
    "documented_breeding_record",
    "computational_seed_only",
})


def _strict_boolean(value, field):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field} must contain strict true/false values")


def load_g0_seed_assignments(path, candidates, metadata):
    """Load explicit F1 G0-pair seeds and return local pair indices.

    The preferred compact schema has sample IDs plus structured ``seed_status``
    and ``report_parent_edges`` fields. Historical V5 assignments are accepted
    only when their exact-stability flag parses strictly as true for every row.
    The seed file is computational input; its status never becomes breeding
    truth merely because it was supplied to this function.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("G0 seed assignment file is empty")
    by_index = metadata.set_index("sample_index")
    index_by_sample = metadata.set_index("Sample")["sample_index"].to_dict()
    compact = {
        "child", "father", "mother", "seed_basis", "seed_status",
        "report_parent_edges",
    }
    historical = {
        "child_index", "joint_MAP_father_index", "joint_MAP_mother_index",
        "exact_pair_stable",
    }
    if compact <= set(frame.columns):
        basis = frame["seed_basis"].fillna("").astype(str).map(
            lambda value: " ".join(value.split())
        )
        if basis.eq("").any():
            raise ValueError("every compact seed row requires a non-empty seed_basis")
        statuses = frame["seed_status"].fillna("").astype(str).str.strip()
        invalid_status = sorted(set(statuses) - SEED_STATUSES)
        if invalid_status:
            raise ValueError(f"unrecognized compact seed_status: {invalid_status}")
        report_edges = np.asarray([
            _strict_boolean(value, "report_parent_edges")
            for value in frame["report_parent_edges"]
        ], dtype=np.bool_)
        invalid_reporting = (
            statuses.eq("computational_seed_only").to_numpy()
            & report_edges
        )
        if np.any(invalid_reporting):
            raise ValueError(
                "computational_seed_only rows cannot report pedigree parent edges"
            )
        mapped = frame[["child", "father", "mother"]].apply(
            lambda column: column.map(index_by_sample)
        )
        if mapped.isna().any().any():
            raise ValueError("compact G0 seed sample IDs must occur in metadata")
        children = mapped["child"].to_numpy(np.int64)
        fathers = mapped["father"].to_numpy(np.int64)
        mothers = mapped["mother"].to_numpy(np.int64)
        basis_values = basis.to_numpy(object)
        status_values = statuses.to_numpy(object)
    elif historical <= set(frame.columns):
        children = frame["child_index"].to_numpy(np.int64)
        fathers = frame["joint_MAP_father_index"].to_numpy(np.int64)
        mothers = frame["joint_MAP_mother_index"].to_numpy(np.int64)
        stable = np.asarray([
            _strict_boolean(value, "exact_pair_stable")
            for value in frame["exact_pair_stable"]
        ], dtype=np.bool_)
        if not np.all(stable):
            raise RuntimeError(
                "historical compatibility seeds must all be exact-pair stable"
            )
        basis_values = np.repeat(
            "historical exact-stable inferred seed; not asserted truth",
            len(frame),
        )
        status_values = np.repeat("exact_stable_inferred", len(frame))
        report_edges = np.ones(len(frame), dtype=np.bool_)
    else:
        raise ValueError(
            "G0 seeds require compact child/father/mother/seed_basis/"
            "seed_status/report_parent_edges columns or the historical V5 "
            "index and exact_pair_stable columns"
        )

    expected_children = np.asarray(candidates["f1_children"], dtype=np.int64)
    if (
        len(children) != len(expected_children)
        or len(np.unique(children)) != len(children)
    ):
        raise RuntimeError("G0 seeds must contain exactly one row per eligible F1")
    row_for_child = {int(child): row for row, child in enumerate(children)}
    if set(row_for_child) != set(expected_children.tolist()):
        raise RuntimeError("G0 seed children do not equal the eligible F1 cohort")
    pair_lookup = {
        (int(father), int(mother)): index
        for index, (father, mother) in enumerate(candidates["g0_pairs"])
    }
    local_pairs = []
    audit_rows = []
    for child in expected_children:
        row = row_for_child[int(child)]
        pair = (int(fathers[row]), int(mothers[row]))
        if pair not in pair_lookup:
            raise RuntimeError(f"ineligible G0 seed pair for F1 {int(child)}: {pair}")
        pair_index = pair_lookup[pair]
        local_pairs.append(candidates["g0_pairs_local"][pair_index])
        audit_rows.append({
            "child_index": int(child),
            "child": by_index.at[int(child), "Sample"],
            "child_alias": by_index.at[int(child), "Alias"],
            "father_index": pair[0],
            "father": by_index.at[pair[0], "Sample"],
            "father_alias": by_index.at[pair[0], "Alias"],
            "mother_index": pair[1],
            "mother": by_index.at[pair[1], "Sample"],
            "mother_alias": by_index.at[pair[1], "Alias"],
            "eligible_pair_index": int(pair_index),
            "seed_basis": str(basis_values[row]),
            "seed_status": str(status_values[row]),
            "report_parent_edges": bool(report_edges[row]),
            "source_seed_file": str(path.resolve()),
        })
    return (
        np.asarray(local_pairs, dtype=np.int64),
        pd.DataFrame(audit_rows),
    )
