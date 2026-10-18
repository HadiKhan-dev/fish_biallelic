"""Settings of the export stage (the pipeline's final products) and of `haplotypes export`."""
from __future__ import annotations

from dataclasses import dataclass

# Products: founders = per-contig founder VCF; samples = per-contig phased sample VCF; tracks = per-component
# founder-track TSV.gz (export command only).
PRODUCTS = ("founders", "samples", "tracks")
FORMATS = ("vcf", "bcf")


@dataclass(frozen=True)
class ExportConfig:
    """What the export writes (part of the export stage's identity; results of other stages never depend on it).

    full_length: every input biallelic SNP that reached the site-quality filter, with FILTER = PASS or the rules
        that removed it (False: only the kept sites, FILTER '.').
    file_format: "vcf" (bgzipped VCF + tabix index; CSI beyond 2^29 bp) or "bcf" (BCF + CSI index).
    products: subset of PRODUCTS.
    """

    full_length: bool = True
    file_format: str = "vcf"
    products: tuple[str, ...] = ("founders", "samples")
