"""Metadata, BCF, checkpoint-facing, and atomic output helpers."""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from cyvcf2 import VCF
from founder_alleles import founder_allele_matrix


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path, frame):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(value)
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def pl_to_likelihoods(pl_values):
    """Convert biallelic VCF PL values to relative genotype likelihoods."""
    pl = np.asarray(pl_values)
    if pl.shape[-1] != 3:
        raise ValueError("PL input must have final dimension 3")
    invalid = np.any(pl < 0, axis=-1)
    safe = np.where(invalid[..., None], 0.0, pl.astype(np.float64))
    safe -= np.min(safe, axis=-1, keepdims=True)
    likelihoods = np.power(10.0, -0.1 * safe)
    likelihoods[invalid] = 1.0
    return likelihoods

def load_metadata(bcf_path, metadata_path):
    vcf = VCF(str(bcf_path))
    sample_names = list(vcf.samples)
    contig_lengths = dict(zip(vcf.seqnames, vcf.seqlens))
    vcf.close()
    metadata = pd.read_excel(metadata_path, sheet_name="main_data")
    metadata = metadata.loc[metadata["primary_ID"].notna()].copy()
    metadata["Sample"] = metadata["primary_ID"].astype(str)
    metadata = metadata.loc[metadata["Sample"].isin(sample_names)].copy()
    if len(metadata) != len(sample_names) or metadata["Sample"].duplicated().any():
        missing = sorted(set(sample_names) - set(metadata["Sample"]))
        raise RuntimeError(
            f"Metadata matched {len(metadata)}/{len(sample_names)} samples; "
            f"missing={missing[:10]}"
        )
    metadata = metadata.set_index("Sample").loc[sample_names].reset_index()
    metadata["sample_index"] = np.arange(len(metadata), dtype=np.int64)
    metadata["Alias"] = metadata["alias_ID"].fillna("").astype(str)
    metadata["SantosID"] = metadata["santos_ID"].fillna("").astype(str)
    metadata["Generation"] = metadata["generation"].astype(str)
    metadata["Sex"] = metadata["sex"].fillna("").astype(str)
    if "collection_date" not in metadata:
        metadata["collection_date"] = pd.NaT
    return sample_names, contig_lengths, metadata[
        [
            "sample_index",
            "Sample",
            "Alias",
            "SantosID",
            "Generation",
            "Sex",
            "collection_date",
        ]
    ]

def founder_hard_alleles(founder_block):
    keys = sorted(founder_block.haplotypes)
    if keys != list(range(max(keys) + 1)):
        raise RuntimeError(f"Non-contiguous founder labels {keys}")
    return founder_allele_matrix(
        founder_block.haplotypes,
        len(founder_block.positions),
        dtype=np.int8,
    )


def select_founder_indices(founder_block, markers_per_contig):
    positions = np.asarray(founder_block.positions, dtype=np.int64)
    hard = founder_hard_alleles(founder_block)
    informative = np.max(hard, axis=0) != np.min(hard, axis=0)
    if founder_block.keep_flags is not None:
        informative &= np.asarray(founder_block.keep_flags, dtype=np.bool_)
    candidate_indices = np.flatnonzero(informative)
    if len(candidate_indices) < markers_per_contig:
        raise RuntimeError(
            f"Only {len(candidate_indices)} informative founder sites available"
        )
    candidate_positions = positions[candidate_indices]
    centers = np.linspace(
        float(candidate_positions[0]),
        float(candidate_positions[-1]),
        markers_per_contig,
    )
    insertion = np.searchsorted(candidate_positions, centers)
    selected = []
    for center, right in zip(centers, insertion):
        choices = []
        if right < len(candidate_indices):
            choices.append(right)
        if right > 0:
            choices.append(right - 1)
        best = min(
            choices,
            key=lambda local: abs(float(candidate_positions[local]) - center),
        )
        selected.append(int(candidate_indices[best]))
    selected = np.unique(np.asarray(selected, dtype=np.int64))
    return selected, hard

def scalar_af(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        value = value[0] if len(value) else math.nan
    return math.nan if value is None else float(value)


def select_scoring_indices(founder_block, markers_per_contig):
    """Select retained founder-informative sites; zero means use them all."""
    if markers_per_contig < 0:
        raise ValueError("markers_per_contig must be non-negative")
    if markers_per_contig:
        return select_founder_indices(founder_block, markers_per_contig)

    hard = founder_hard_alleles(founder_block)
    informative = np.max(hard, axis=0) != np.min(hard, axis=0)
    if founder_block.keep_flags is not None:
        informative &= np.asarray(founder_block.keep_flags, dtype=np.bool_)
    selected = np.flatnonzero(informative)
    if not len(selected):
        raise RuntimeError("No retained founder-informative sites available")
    return selected.astype(np.int64, copy=False), hard


def load_selected_likelihoods(
    bcf_path, contig, founder_block, markers_per_contig, bcf_threads
):
    selected_indices, hard = select_scoring_indices(
        founder_block, markers_per_contig
    )
    founder_positions = np.asarray(founder_block.positions, dtype=np.int64)
    position_to_index = {
        int(founder_positions[index]): int(index) for index in selected_indices
    }
    values_by_position = {}
    vcf = VCF(str(bcf_path), threads=bcf_threads)
    for variant in vcf(contig):
        position = int(variant.POS)
        if position not in position_to_index:
            continue
        pl = variant.format("PL")
        if pl is None or pl.ndim != 2 or pl.shape[1] < 3:
            continue
        values_by_position[position] = (
            np.asarray(pl[:, :3], dtype=np.int32),
            scalar_af(variant.INFO.get("AF")),
        )
    vcf.close()
    if not values_by_position:
        raise RuntimeError(f"No selected PL markers found on {contig}")
    positions = np.asarray(sorted(values_by_position), dtype=np.int64)
    block_indices = np.asarray(
        [position_to_index[int(position)] for position in positions],
        dtype=np.int64,
    )
    pl = np.stack(
        [values_by_position[int(position)][0] for position in positions], axis=1
    )
    frequencies = np.asarray(
        [values_by_position[int(position)][1] for position in positions],
        dtype=np.float64,
    )
    missing = ~np.isfinite(frequencies)
    if np.any(missing):
        frequencies[missing] = np.mean(
            hard[:, block_indices[missing]], axis=0
        )
    frequencies = np.clip(frequencies, 1e-4, 1.0 - 1e-4)
    likelihoods = pl_to_likelihoods(pl)
    return likelihoods, frequencies, positions, block_indices, hard

def local_founder_equivalence(
    founder_positions,
    hard_alleles,
    selected_positions,
    window_bp,
    max_diff_fraction,
    min_diff_sites,
):
    n_founders = hard_alleles.shape[0]
    n_selected = len(selected_positions)
    equivalence = np.zeros(
        (n_selected, n_founders, n_founders), dtype=np.bool_
    )
    for founder in range(n_founders):
        equivalence[:, founder, founder] = True
    bin_start = (selected_positions // window_bp) * window_bp
    bin_end = bin_start + window_bp
    start_indices = np.searchsorted(founder_positions, bin_start, side="left")
    end_indices = np.searchsorted(founder_positions, bin_end, side="left")
    sites_in_bin = end_indices - start_indices
    for first in range(n_founders):
        for second in range(first + 1, n_founders):
            difference = (hard_alleles[first] != hard_alleles[second]).astype(
                np.int64
            )
            cumulative = np.empty(len(difference) + 1, dtype=np.int64)
            cumulative[0] = 0
            np.cumsum(difference, out=cumulative[1:])
            differences = cumulative[end_indices] - cumulative[start_indices]
            allowed = np.maximum(
                (max_diff_fraction * sites_in_bin).astype(np.int64),
                min_diff_sites,
            )
            equivalent = differences <= allowed
            equivalence[:, first, second] = equivalent
            equivalence[:, second, first] = equivalent
    return equivalence


def smoothed_founder_probabilities(
    founder_block,
    block_indices,
    equivalence,
):
    keys = sorted(founder_block.haplotypes)
    probabilities = np.asarray([
        np.asarray(founder_block.haplotypes[key])[block_indices]
        for key in keys
    ], dtype=np.float64)
    smoothed = np.empty_like(probabilities)
    for site in range(len(block_indices)):
        for founder in range(len(keys)):
            equivalents = equivalence[site, founder]
            smoothed[founder, site] = np.mean(
                probabilities[equivalents, site], axis=0
            )
    smoothed /= np.sum(smoothed, axis=2, keepdims=True)
    return probabilities, smoothed


def painted_parent_tracks(
    painting,
    selected_positions,
    founder_probabilities,
    sample_indices,
):
    n_samples = len(sample_indices)
    n_sites = len(selected_positions)
    tracks = np.full((n_samples, n_sites, 2, 2), 0.5, dtype=np.float64)
    coverage = np.zeros((n_samples, n_sites), dtype=np.bool_)
    for local_sample, sample_index in enumerate(sample_indices):
        chunks = painting.samples[int(sample_index)].chunks
        if not chunks:
            continue
        starts = np.asarray([chunk.start for chunk in chunks], dtype=np.int64)
        ends = np.asarray([chunk.end for chunk in chunks], dtype=np.int64)
        hap1 = np.asarray([chunk.hap1 for chunk in chunks], dtype=np.int64)
        hap2 = np.asarray([chunk.hap2 for chunk in chunks], dtype=np.int64)
        chunk_index = np.searchsorted(ends, selected_positions, side="right")
        valid = chunk_index < len(chunks)
        clipped = np.clip(chunk_index, 0, len(chunks) - 1)
        valid &= selected_positions >= starts[clipped]
        valid &= selected_positions < ends[clipped]
        valid &= hap1[clipped] >= 0
        valid &= hap2[clipped] >= 0
        site_indices = np.flatnonzero(valid)
        if len(site_indices):
            labels1 = hap1[clipped[site_indices]]
            labels2 = hap2[clipped[site_indices]]
            tracks[local_sample, site_indices, 0] = founder_probabilities[
                labels1, site_indices
            ]
            tracks[local_sample, site_indices, 1] = founder_probabilities[
                labels2, site_indices
            ]
            coverage[local_sample, site_indices] = True
    return tracks, coverage
