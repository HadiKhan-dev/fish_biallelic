"""refinement / conditioning for the canonical reconstruction pipeline."""
from __future__ import annotations
from haplotype_reconstruction import PACKAGE_ROOT

import numpy as np
from dataclasses import dataclass

import hashlib
import json
from pathlib import Path
import pandas as pd
from numba import njit, prange
import haplotype_reconstruction.painting.components as painting_components

def paint_final_phase(t09, positions, phase_map):
    """Split at actual phase changes, including changes inside an old bin.

Coordinates are half-open, as in the source painting. Original gaps and
component boundaries survive unchanged; no founder namespace is merged and
no inferred allele fill is represented as a new founder label.
"""
    positions = np.asarray(positions, dtype=np.int64)
    changes = [positions[1:][row[1:] != row[:-1]] for row in phase_map]
    output = []
    for component in t09.painting_bundle.components:
        samples = []
        for index, sample in enumerate(component.painting.samples):
            chunks = []
            cuts = changes[index]
            for source in sample.chunks:
                lo = np.searchsorted(cuts, source.start, side='right')
                hi = np.searchsorted(cuts, source.end, side='left')
                boundaries = np.r_[source.start, cuts[lo:hi], source.end]
                for left, right in zip(boundaries[:-1], boundaries[1:]):
                    marker = int(np.searchsorted(positions, left))
                    flip = marker < len(positions) and positions[marker] < right and phase_map[index, marker] == 1
                    first, second = (source.hap2, source.hap1) if flip else (source.hap1, source.hap2)
                    if chunks and chunks[-1].end == left and (chunks[-1].hap1, chunks[-1].hap2) == (first, second):
                        old = chunks[-1]
                        chunks[-1] = painting_components.PaintedChunk(old.start, int(right), first, second)
                    else:
                        chunks.append(painting_components.PaintedChunk(int(left), int(right), first, second))
            samples.append(painting_components.SamplePainting(index, chunks))
        output.append(
            painting_components.BlockPainting((component.painting.start_pos, component.painting.end_pos), samples)
        )
    return tuple(output)


SCHEMA = "stage11-stable-final-phase-v2"


@dataclass(frozen=True)
class PhaseScaffold:
    reference_alleles: np.ndarray
    phase_bins: np.ndarray
    component_ids: np.ndarray
    component_bin_edges: tuple[np.ndarray, ...]


@njit(cache=True, parallel=True)
def _map_reference(positions, source_columns, frozen, offsets, chunks):
    samples = len(offsets) - 1
    result = np.full((samples, len(positions), 2), -1, dtype=np.int8)
    for sample in prange(samples):
        chunk = offsets[sample]
        for index in range(len(positions)):
            pos = positions[index]
            while chunk < offsets[sample + 1] and chunks[chunk, 1] <= pos:
                chunk += 1
            if chunk >= offsets[sample + 1] or chunks[chunk, 0] > pos:
                continue
            column = source_columns[index]
            if column < 0:
                continue
            for track in range(2):
                founder = chunks[chunk, track + 2]
                if 0 <= founder < frozen.shape[0]:
                    result[sample, index, track] = frozen[founder, column]
    return result


def prepare_phase_scaffold(t09, positions):
    """Decode only frozen hard alleles; unknowns are never argmaxed to REF."""
    t09 = painting_checkpoints.validate_t09_component_checkpoint(t09)
    pos = np.asarray(positions, dtype=np.int64)
    blocks = core_runtime.validate_phase_component_manifest(t09.component_manifest)
    reference = np.full((len(t09.sample_ids), len(pos), 2), -1, dtype=np.int8)
    phase_bins = np.full(len(pos), -1, dtype=np.int64)
    component_ids = np.full(len(pos), -1, dtype=np.int32)
    all_edges = []
    offset = 0
    for index, (block, component) in enumerate(zip(blocks, t09.painting_bundle.components)):
        source_pos = np.asarray(block.positions, dtype=np.int64)
        frozen = np.asarray(block.missing_aware_inference_discrete_haps)
        if frozen.shape != (len(component.founder_keys), len(source_pos)) or np.any(~np.isin(frozen, (-1, 0, 1))):
            raise ValueError("frozen component allele matrix has incompatible rows/sites")
        lo, hi = np.searchsorted(pos, [source_pos[0], source_pos[-1]], side="left")
        hi = int(np.searchsorted(pos, source_pos[-1], side="right"))
        lo = int(lo)
        local = pos[lo:hi]
        cols = np.searchsorted(source_pos, local)
        valid = cols < len(source_pos)
        valid[valid] &= source_pos[cols[valid]] == local[valid]
        cols = np.where(valid, cols, -1).astype(np.int64)
        chunks = np.asarray([tuple(chunk) for sample in component.painting.samples for chunk in sample.chunks], dtype=np.int64).reshape((-1, 4))
        starts = np.concatenate(([0], np.cumsum([len(sample.chunks) for sample in component.painting.samples]))).astype(np.int64)
        reference[:, lo:hi] = _map_reference(local, cols, np.asarray(frozen, dtype=np.int8), starts, chunks)
        diagnostic = component.ragged_diagnostics
        if diagnostic is not None:
            edges = np.asarray(diagnostic.bin_edges, dtype=np.int64)
        else:
            # Complete/empty reference components still use real SNP coordinates.
            edges = np.unique(np.r_[source_pos[::100], source_pos[-1] + 1])
        all_edges.append(edges)
        bins = np.searchsorted(edges, local, side="right") - 1
        good = (bins >= 0) & (bins < len(edges) - 1)
        phase_bins[lo:hi] = np.where(good, offset + bins, -1)
        component_ids[lo:hi] = index
        offset += len(edges) - 1
    return PhaseScaffold(reference, phase_bins, component_ids, tuple(all_edges))


def refinement_code_identity():
    files = (
        "refinement/pipeline.py", "refinement/conditioning.py", "refinement/model.py",
        "refinement/messages.py", "refinement/factors.py", "refinement/selector_chains.py",
        "refinement/phase_chains.py", "refinement/polish.py", "refinement/evidence.py",
        "core/genetic_map.py", "core/raw_evidence.py",
    )
    return {name: hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
            for name in files}

def relationship_identity(relationships):
    frame = relationships.loc[:, ["Sample", "ParentState", "Parent1", "Parent2"]].astype(object)
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_summary_table(output, stage, summaries):
    temporary=output/f".{stage}.csv.tmp"
    pd.DataFrame(summaries).to_csv(temporary, index=False)
    temporary.replace(output/f"{stage}.csv")

import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.painting.checkpoints as painting_checkpoints
