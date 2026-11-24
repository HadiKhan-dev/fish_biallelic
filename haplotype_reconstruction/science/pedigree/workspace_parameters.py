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
                "(read_transfer.class_transfers: bins x S^2 x C^2 float64 per child); a child whose cache "
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

    emission_sample_block: int = field(
        default=16,
        metadata={
            "provenance": "PS01; batches of 16 samples bound live recomputed-emission and genotype-likelihood storage.",
        },
    )

    selective_bound_group_bins: int = field(
        default=8,
        metadata={
            "provenance": "PS22; group 8 consecutive bins when constructing selective-bound geometry.",
        },
    )

    exclusion_dyad_batch: int = field(
        default=4096,
        metadata={
            "provenance": (
                "PS20; batches of 4096 dyads bound exclusion scanner outputs (each dyad's window workspace is its "
                "own). 4.22 A11: the observation-model scan has no likelihood-pattern cache and certificates stop "
                "dyads at very different sites, so a batch must give every thread many dyads; kernel core use "
                "0.52 -> 0.76 (f3 roots) and 0.78 -> 0.86-0.91 (famhs) against 256 with the round-robin dealing of "
                "exclusion_bound.score_contig (lab422/r4a); execution grain only, results do not depend on it."
            ),
        },
    )
PEDIGREE_WORKSPACE = PedigreeWorkspace()
COMMON_EMISSION_CACHE_BYTES = PEDIGREE_WORKSPACE.common_emission_cache_bytes
CLASS_TRANSFER_CACHE_BYTES = PEDIGREE_WORKSPACE.class_transfer_cache_bytes
BOUND_WORKSPACE_BYTES = PEDIGREE_WORKSPACE.bound_workspace_bytes
BOUND_BATCH_ROWS = PEDIGREE_WORKSPACE.bound_batch_rows
EMISSION_SAMPLE_BLOCK = PEDIGREE_WORKSPACE.emission_sample_block
SELECTIVE_BOUND_GROUP_BINS = PEDIGREE_WORKSPACE.selective_bound_group_bins
EXCLUSION_DYAD_BATCH = PEDIGREE_WORKSPACE.exclusion_dyad_batch
