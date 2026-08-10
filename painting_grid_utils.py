"""Shared painting-to-grid and founder-ID-to-allele transformations."""

import numpy as np

from founder_alleles import founder_allele_matrix, hard_alleles


def discretize_painting_to_bins(painting, bin_edges):
    """Discretize half-open painting chunks at bin centres."""
    num_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    id_grid = np.full((num_bins, 2), -1, dtype=np.int32)

    chunks = painting.chunks if hasattr(painting, "chunks") else []
    if not chunks:
        return id_grid, np.ones(num_bins, dtype=np.bool_)

    chunk_ends = np.array([chunk.end for chunk in chunks], dtype=np.int64)
    chunk_starts = np.array([chunk.start for chunk in chunks], dtype=np.int64)
    chunk_hap1 = np.array([chunk.hap1 for chunk in chunks], dtype=np.int32)
    chunk_hap2 = np.array([chunk.hap2 for chunk in chunks], dtype=np.int32)
    indices = np.searchsorted(chunk_ends, bin_centers, side="right")
    indices = np.clip(indices, 0, len(chunks) - 1)
    valid = ((bin_centers >= chunk_starts[indices])
             & (bin_centers < chunk_ends[indices]))
    id_grid[:, 0] = np.where(valid, chunk_hap1[indices], -1)
    id_grid[:, 1] = np.where(valid, chunk_hap2[indices], -1)
    hom_mask = ((id_grid[:, 0] == id_grid[:, 1])
                | (id_grid[:, 0] == -1)
                | (id_grid[:, 1] == -1))
    return id_grid, hom_mask


def id_grid_to_allele_grid(id_grid, bin_centers, positions, haplotypes,
                           bin_width_bp=None, check_founder_bounds=True):
    """Convert a diploid founder-ID grid to one allele per bin and track."""
    num_bins = id_grid.shape[0]
    bin_indices = np.searchsorted(positions, bin_centers)
    bin_indices = np.clip(bin_indices, 0, len(positions) - 1)
    if bin_width_bp is None:
        bin_width_bp = (bin_centers[1] - bin_centers[0]
                        if len(bin_centers) > 1 else 10000)

    found_positions = positions[bin_indices]
    valid_snp = np.abs(found_positions - bin_centers) <= (bin_width_bp / 2.0)
    allele_lookup = np.full((max(haplotypes) + 1 if haplotypes else 1,
                             num_bins), -1, dtype=np.int8)
    for founder_id, haplotype in haplotypes.items():
        extracted = hard_alleles(haplotype)[bin_indices].copy()
        extracted[~valid_snp] = -1
        allele_lookup[founder_id, :] = extracted

    allele_grid = np.full_like(id_grid, -1, dtype=np.int8)
    columns = np.arange(num_bins)
    for track in (0, 1):
        founder_ids = id_grid[:, track]
        if check_founder_bounds:
            valid = ((founder_ids >= 0)
                     & (founder_ids < allele_lookup.shape[0]))
        else:
            # Compatibility for pedigree_inference's historical wrapper.
            valid = founder_ids != -1
        safe_ids = founder_ids.copy()
        safe_ids[~valid] = 0
        alleles = allele_lookup[safe_ids, columns]
        alleles[~valid] = -1
        allele_grid[:, track] = alleles
    return allele_grid


def id_grid_to_allele_grid_multisnp(
    id_grid,
    bin_centers,
    positions,
    haplotypes,
    bin_width_bp=None,
    max_snps_per_bin=10,
    check_founder_bounds=True,
):
    """Convert a diploid founder-ID grid to sampled alleles within each bin."""
    num_bins = id_grid.shape[0]
    n_snps = len(positions)
    allele_grid = np.full((num_bins, 2, max_snps_per_bin), -1,
                          dtype=np.int8)
    if n_snps == 0:
        return allele_grid
    if bin_width_bp is None:
        bin_width_bp = (bin_centers[1] - bin_centers[0]
                        if len(bin_centers) > 1 else 10000)

    founder_alleles = founder_allele_matrix(
        haplotypes, n_snps, dtype=np.int8
    )
    half_width = bin_width_bp / 2.0
    start_indices = np.searchsorted(
        positions, bin_centers - half_width, side="left"
    )
    end_indices = np.searchsorted(
        positions, bin_centers + half_width, side="right"
    )
    n_founders = founder_alleles.shape[0]
    for bin_index in range(num_bins):
        start = start_indices[bin_index]
        end = end_indices[bin_index]
        count = end - start
        if count == 0:
            continue
        if count <= max_snps_per_bin:
            sampled = range(start, end)
        else:
            step = count / max_snps_per_bin
            sampled = [start + int(index * step)
                       for index in range(max_snps_per_bin)]
        for output_index, snp_index in enumerate(sampled):
            if output_index >= max_snps_per_bin:
                break
            for track in (0, 1):
                founder_id = id_grid[bin_index, track]
                valid = founder_id >= 0
                if check_founder_bounds:
                    valid = valid and founder_id < n_founders
                if valid:
                    allele_grid[bin_index, track, output_index] = \
                        founder_alleles[founder_id, snp_index]
    return allele_grid


def ibs_homozygosity_mask(allele_grid):
    """Return bins whose two tracks are indistinguishable at valid alleles."""
    if allele_grid.ndim == 3:
        hom_mask = np.ones(allele_grid.shape[0], dtype=np.bool_)
        for bin_index in range(allele_grid.shape[0]):
            first = allele_grid[bin_index, 0, :]
            second = allele_grid[bin_index, 1, :]
            valid = (first != -1) & (second != -1)
            if np.any(valid):
                hom_mask[bin_index] = np.all(first[valid] == second[valid])
        return hom_mask
    return ((allele_grid[:, 0] == allele_grid[:, 1])
            | (allele_grid[:, 0] == -1)
            | (allele_grid[:, 1] == -1))
