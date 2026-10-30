"""The stage vocabulary: names in pipeline order (one name = record prefix = output directory = stop point).

Records of a stage have keys equal to its name or starting with ``<name>/``; its bound identities are
``identity/<name>[/...]``; genome-wide completion is ``complete/<name>`` in run.ckpt; its published files are
under ``OUTPUT/<name>/``. ``scope`` says where the stage's products live (per-contig files or run.ckpt);
``packages`` are the packages whose code the stage executes (they enter its code identity together with
model/ and maths/). Simulated inputs use the prefix ``simulation/`` (an input step, not a pipeline stage).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One pipeline stage."""

    name: str
    scope: str          # "contig": products in contigs/<contig>.ckpt; "genome": in run.ckpt
    packages: tuple[str, ...]


STAGES: tuple[Stage, ...] = (
    Stage("evidence", "contig", ("evidence",)),
    # Release 3.3: genome-wide, after every contig's evidence and before any assembly (the relatedness barrier).
    Stage("relatedness", "genome", ("relatedness",)),
    Stage("discovery", "contig", ("discovery",)),
    Stage("local_selection", "contig", ("local_selection", "discovery", "assembly")),
    Stage("assembly", "contig", ("assembly",)),
    Stage("painting", "contig", ("painting",)),
    Stage("pedigree", "genome", ("pedigree", "painting")),
    Stage("founder_refit", "contig", ("founder_refit", "painting")),
    Stage("family_phase", "contig", ("family_phase", "painting")),
    Stage("recombination", "contig", ("recombination", "family_phase")),
    Stage("export", "contig", ("export",)),
)

STAGE_NAMES: tuple[str, ...] = tuple(entry.name for entry in STAGES)

# Where a run may stop early (`[run] stop_after`): after relatedness (every contig's evidence and the genome-wide
# relatedness product recorded; what shards need first), after discovery, or after painting (per-contig work done,
# later genome-wide stages not started).
STOP_POINTS: tuple[str, ...] = ("relatedness", "discovery", "painting")

SIMULATION_PREFIX = "simulation"

# Input steps: they write records (and OUTPUT/<name>/ files) like a stage but are not part of the pipeline order.
INPUT_STEPS: tuple[Stage, ...] = (
    Stage(SIMULATION_PREFIX, "contig", ("simulation",)),
)

_BY_NAME = {entry.name: entry for entry in (*INPUT_STEPS, *STAGES)}


def stage(name: str) -> Stage:
    """The registered stage (or input step) called ``name`` (ValueError naming the valid stages otherwise)."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown stage {name!r}; stages: {', '.join(STAGE_NAMES)}") from None


def position(name: str) -> int:
    """Index of a stage in pipeline order."""
    if name not in STAGE_NAMES:
        raise ValueError(f"{name!r} is not a pipeline stage; stages: {', '.join(STAGE_NAMES)}")
    return STAGE_NAMES.index(name)
