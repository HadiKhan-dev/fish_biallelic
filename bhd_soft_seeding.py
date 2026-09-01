"""Soft-clustered data seeds for reversible founder discovery."""

import numpy as np

from bhd_config import DEFAULT_SOFT_SEED_MIN_CLUSTER_SIZE


def soft_cluster_seed_haplotypes(
    genotype_likelihoods,
    n_seeds,
    min_cluster_size=DEFAULT_SOFT_SEED_MIN_CLUSTER_SIZE,
):
    """Return up to ``n_seeds`` pooled binary haplotype seeds.

    Samples are clustered by expected-genotype agreement. Each retained
    cluster contributes one pooled-alt consensus, providing deterministic,
    denoised starting basins without using sample metadata or truth.
    """

    import bhd_kernels
    import hdbscan

    likelihoods = np.asarray(genotype_likelihoods, dtype=np.float64)
    if likelihoods.ndim != 3 or likelihoods.shape[2] != 3:
        raise ValueError(
            "genotype_likelihoods must have shape (samples, sites, 3)"
        )
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if likelihoods.shape[0] < int(min_cluster_size):
        return []

    similarity = bhd_kernels.soft_agreement_similarity(likelihoods)
    distance = similarity.max() - similarity
    np.fill_diagonal(distance, 0.0)
    distance = np.ascontiguousarray(distance, dtype=np.float64)

    labels = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(min_cluster_size),
    ).fit(distance).labels_
    clusters = [
        np.flatnonzero(labels == label)
        for label in np.unique(labels)
        if label != -1
    ]
    clusters.sort(key=lambda members: -len(members))

    alt_fraction = bhd_kernels.alt_fractions(likelihoods)
    return [
        bhd_kernels.pooled_alt_to_hap(
            alt_fraction[members].mean(axis=0)
        ).astype(np.int64)
        for members in clusters[:n_seeds]
    ]


__all__ = ["soft_cluster_seed_haplotypes"]
