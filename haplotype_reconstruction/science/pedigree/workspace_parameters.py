"""Workspace limits for pedigree scoring, screening and family inference."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PedigreeWorkspace:
    """Storage dimensions only; estimates never reserve memory or limit concurrency."""

    relationship_sequence_chunk: int = field(
        default=32_768,
        metadata={
            "units": "count",
            "provenance": "PD28; bound binary sequence workspaces per contig task; execution grain only.",
        },
    )

    common_emission_cache_bytes: int = field(
        default=256 << 20,
        metadata={
            "provenance": "PS09; 268435456 bytes is the cutoff for retaining common child-emission caches.",
        },
    )

    class_transfer_cache_bytes: int = field(
        default=1 << 30,
        metadata={
            "provenance": (
                "LR phase 2 Tier 2; 1073741824 bytes bound one child batch's class-transfer cache of the read state "
                "(read_transfer.class_transfers: bins x 2 views x S^2 x C^2 float64 per child); a child whose cache "
                "exceeds it is scored per site (the same values)."
            ),
        },
    )

    bound_workspace_bytes: int = field(
        default=512 << 20,
        metadata={
            "provenance": (
                "PS15; workspace target 536870912 bytes sizes pair-bound batches without limiting concurrency."
            ),
        },
    )
    bound_batch_rows: int = field(
        default=4_000_000,
        metadata={
            "provenance": "PS15; at most 4000000 pair-bound workspace rows per batch.",
        },
    )
    exclusion_pattern_limit: int = field(
        default=1024,
        metadata={
            "provenance": (
                "PS20; cache at most 1024 distinct genotype-likelihood patterns per sample before direct "
                "fallback."
            ),
        },
    )

    emission_sample_block: int = field(
        default=16,
        metadata={
            "provenance": "PS01; batches of 16 samples bound live recomputed-emission and genotype-likelihood storage.",
        },
    )

    family_lanes_per_thread: int = field(
        default=8,
        metadata={
            "provenance": (
                "PD09; 8 interleaved independent-item lanes per supplied thread count distribute unequal family "
                "row counts."
            ),
        },
    )

    family_claim_work: int = field(
        default=20_000_000,
        metadata={
            "provenance": "PD09; target 20000000 row-scan work terms per cavity iteration chunk.",
        },
    )

    family_transpose_block: int = field(
        default=64,
        metadata={
            "provenance": "PD09; transpose cavity messages in blocks of 64 samples for contiguous child reads.",
        },
    )

    selective_bound_group_bins: int = field(
        default=8,
        metadata={
            "provenance": "PS22; group 8 consecutive bins when constructing selective-bound geometry.",
        },
    )

    exclusion_dyad_batch: int = field(
        default=256,
        metadata={
            "provenance": "PS20; batches of 256 dyads bound exclusion scanner workspaces.",
        },
    )
PEDIGREE_WORKSPACE = PedigreeWorkspace()
COMMON_EMISSION_CACHE_BYTES = PEDIGREE_WORKSPACE.common_emission_cache_bytes
CLASS_TRANSFER_CACHE_BYTES = PEDIGREE_WORKSPACE.class_transfer_cache_bytes
BOUND_WORKSPACE_BYTES = PEDIGREE_WORKSPACE.bound_workspace_bytes
BOUND_BATCH_ROWS = PEDIGREE_WORKSPACE.bound_batch_rows
EXCLUSION_PATTERN_LIMIT = PEDIGREE_WORKSPACE.exclusion_pattern_limit
EMISSION_SAMPLE_BLOCK = PEDIGREE_WORKSPACE.emission_sample_block
FAMILY_LANES_PER_THREAD = PEDIGREE_WORKSPACE.family_lanes_per_thread
FAMILY_CLAIM_WORK = PEDIGREE_WORKSPACE.family_claim_work
FAMILY_TRANSPOSE_BLOCK = PEDIGREE_WORKSPACE.family_transpose_block
SELECTIVE_BOUND_GROUP_BINS = PEDIGREE_WORKSPACE.selective_bound_group_bins
EXCLUSION_DYAD_BATCH = PEDIGREE_WORKSPACE.exclusion_dyad_batch
