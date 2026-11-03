"""Represent export options and the selected rows being formatted (P0 OU01/OU08).

Owns formatting projections only; source reads and released calls belong to stages.
Method: docs/methods/export.md. Constants retain export/sites.py and config.py provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...contracts.release import ReleasedSites

# Rule names and their bit (bit k = rule rk; evidence.site_quality.RULES), with the FILTER descriptions.
RULE_DESCRIPTIONS = {
    "r1": "Site quality r1: total depth outside the contig's clean depth core (either tail)",
    "r2": "Site quality r2: per-sample depth heterogeneity above the contig's clean core",
    "r3": "Site quality r3: wrong-allele reads in homozygous samples (a negative-binomial artefact class is more "
    "likely than the clean Poisson core)",
    "r4": "Site quality r4: heterozygote allele balance (a beta-binomial artefact class is more likely than the clean "
    "binomial core)",
    "r5": "Uninformative r5: fewer than two expected carriers of a non-major genotype",
}
R5_ONLY = np.uint8(1 << 5)
COMMON_ALLELE_RULE = (
    "at sites removed by r5 alone, every founder column whose component span contains the site "
    "carries the allele with more reads in total over the analysed samples not flagged by the "
    "contamination QC (the samples r5 counts; FORMAT/AD of the input at export time; GT allele "
    "counts for a record without AD); equal totals give '.'; sites removed by r1-r4 and sites "
    "outside every component are '.'"
)


@dataclass(frozen=True)
class SiteTable:
    """The rows of one chromosome's VCF products (see the module docstring).

    positions int64 [M] (increasing); kept bool [M]; removed_bits uint8 [M] (bit k: applied rule rk removed the
    site; 0 at kept sites); ref, alt <U1 [M] (input alleles, or A/C for a simulation's abstract 0/1); common int8 [M]
    (0/1 at sites removed by r5 alone with a majority allele, else -1); contig_length (bp, from the input header;
    None when unknown); site_filter: the filter's state on this contig; dropped: loader records never written;
    common_counts: r5-only sites given REF or ALT, left '.' (tie), and decided from reads (AD, or simulated
    reads) or from GT.
    """

    contig: str
    positions: np.ndarray
    kept: np.ndarray
    removed_bits: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    common: np.ndarray
    full_length: bool
    contig_length: int | None = None
    site_filter: str = "unknown"
    applied_rules: tuple[str, ...] = ()
    dropped: Mapping[str, int] = field(default_factory=dict)
    common_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def r5_only(self) -> np.ndarray:
        return self.removed_bits == R5_ONLY

    def filters(self) -> list[str]:
        """The FILTER text of every row: '.' (kept sites, kept-site table), PASS, or the rules that removed it."""
        if not self.full_length:
            return ["."] * len(self.positions)
        names: dict[int, str] = {}
        result = []
        for bits in self.removed_bits.tolist():
            text = names.get(bits)
            if text is None:
                text = names[bits] = (
                    "PASS" if bits == 0 else ";".join(rule for rule in RULE_DESCRIPTIONS if bits >> int(rule[1:]) & 1)
                )
            result.append(text)
        return result

    def summary(self) -> dict:
        """The manifest entry of the site set."""
        removed = ~self.kept
        return dict(
            written=int(len(self.positions)),
            kept=int(self.kept.sum()),
            removed=int(removed.sum()),
            removed_by={rule: int(((self.removed_bits >> int(rule[1:])) & 1).sum()) for rule in RULE_DESCRIPTIONS},
            removed_only_by_r5=int(self.r5_only.sum()),
            common_allele=dict(self.common_counts),
            site_filter=self.site_filter,
            applied_rules=list(self.applied_rules),
            loader_dropped=dict(self.dropped),
        )


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


PRODUCTS = ("founders", "samples", "tracks")
FORMATS = ("vcf", "bcf")
SCHEMA = 4


def project_sites(record: ReleasedSites, *, full_length: bool) -> SiteTable:
    """Select the 3.5 full/kept formatting table without making a site decision.

    Kept-only files clear removed bits and common calls and have empty common counts.
    Filter state, applied rules, source alleles, drops and header length are retained.
    """
    selected = slice(None) if full_length else record.kept
    size = len(record.positions[selected])
    return SiteTable(
        contig=record.contig,
        positions=record.positions[selected],
        kept=record.kept[selected] if full_length else np.ones(size, dtype=bool),
        removed_bits=record.removed_bits[selected] if full_length else np.zeros(size, dtype=np.uint8),
        ref=record.ref[selected],
        alt=record.alt[selected],
        common=record.common[selected] if full_length else np.full(size, -1, dtype=np.int8),
        full_length=full_length,
        contig_length=record.contig_length,
        site_filter=record.site_filter,
        applied_rules=record.applied_rules,
        dropped=record.dropped,
        common_counts=record.common_counts if full_length else {},
    )


@dataclass(frozen=True)
class ExportRecord:
    """Output-only completion: source/option identity and ordered manifest entry."""

    identity: str
    entry: Mapping[str, Any]
    schema: int = SCHEMA


def output_identity(upstream: Mapping[str, str], settings: ExportConfig, code_identity: str) -> str:
    """Bind rendered record identities, formatting options and the output code closure."""
    from ...core.digest import json_digest

    return json_digest(
        dict(
            schema=SCHEMA,
            upstream=dict(upstream),
            code=code_identity,
            full_length=settings.full_length,
            file_format=settings.file_format,
            products=settings.products,
        )
    )


def complete(record: ExportRecord | None, identity: str, destination) -> bool:
    """A matching output identity is complete only while every recorded file exists."""
    from pathlib import Path

    return (
        record is not None
        and record.identity == identity
        and all((Path(destination) / str(name)).is_file() for name in record.entry["files"])
    )
