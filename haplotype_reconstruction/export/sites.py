"""The rows of one chromosome's VCF products: which input sites, their REF/ALT, FILTER and r5 common alleles.

Two site sets:

* kept sites (``full_length=False``): the evidence stage's sites, i.e. the input biallelic SNPs the site-quality
  filter kept; FILTER is ``.``;
* full length (``full_length=True``): every input biallelic SNP that reached the site-quality filter (the evidence
  report's input positions); FILTER is PASS at kept sites, else the applied rules that fired, e.g. ``r1;r5``
  (RULE_DESCRIPTIONS; r3 and r4 are only evaluated at sites passing r1 and r2, r4 only at sites passing r3).

Records the VCF loader drops never reach the filter and are never written: records that are not biallelic SNPs
(r0: multiallelic, indels, MNPs, symbolic alleles) and biallelic SNPs sharing a position (duplicate_position). The
evidence report counts them; the VCF header repeats the counts.

Common allele (founder VCF, full length): at a site removed by r5 alone (uninformative: fewer than two expected
carriers of a non-major genotype) it is the allele with more reads in total over the samples r5 counts, i.e. the
run's analysed samples that the contamination QC did not flag (the evidence record's ``founder_excluded``), from
FORMAT/AD of the input file read at export time. A record without AD values (no AD, or HTSlib's all-missing AD)
counts those samples' GT alleles instead. Equal totals (including no reads at all) give no common allele (``.``).
A simulation's reads come from its ``simulation/reads`` record. Nothing here changes a stage's records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

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
COMMON_ALLELE_RULE = ("at sites removed by r5 alone, every founder column whose component span contains the site "
                      "carries the allele with more reads in total over the analysed samples not flagged by the "
                      "contamination QC (the samples r5 counts; FORMAT/AD of the input at export time; GT allele "
                      "counts for a record without AD); equal totals give '.'; sites removed by r1-r4 and sites "
                      "outside every component are '.'")
_BASES = frozenset("ACGTacgt")


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
        names = {}
        result = []
        for bits in self.removed_bits.tolist():
            text = names.get(bits)
            if text is None:
                text = names[bits] = ("PASS" if bits == 0 else
                                      ";".join(rule for rule in RULE_DESCRIPTIONS if bits >> int(rule[1:]) & 1))
            result.append(text)
        return result

    def summary(self) -> dict:
        """The manifest entry of the site set."""
        removed = ~self.kept
        return dict(written=int(len(self.positions)), kept=int(self.kept.sum()), removed=int(removed.sum()),
                    removed_by={rule: int(((self.removed_bits >> int(rule[1:])) & 1).sum())
                                for rule in RULE_DESCRIPTIONS},
                    removed_only_by_r5=int(self.r5_only.sum()), common_allele=dict(self.common_counts),
                    site_filter=self.site_filter, applied_rules=list(self.applied_rules),
                    loader_dropped=dict(self.dropped))


def _biallelic_snp(record) -> bool:
    """One REF and one ALT base, both A/C/G/T (the evidence loader's site rule, evidence.vcf)."""
    return len(record.ALT) == 1 and record.REF in _BASES and record.ALT[0] in _BASES


def read_input_sites(reference, contig: str, positions: np.ndarray, count: np.ndarray, sample_ids):
    """One pass over the input file's records of ``contig``.

    Returns (ref, alt, ref_reads, alt_reads, source, contig_length): the alleles of each position's biallelic SNP
    record (<U1 [M]); where ``count``, the REF and ALT totals over ``sample_ids`` (int64 [M]) and how they were
    counted (int8 [M]: 1 AD reads, 2 GT alleles of a record without AD, 0 not counted).
    """
    from cyvcf2 import VCF
    header = VCF(str(reference))
    try:
        if set(sample_ids) - set(header.samples) or contig not in header.seqnames:
            raise ValueError("export samples/contig are absent from the original VCF")
        try:
            contig_length = int(header.seqlens[header.seqnames.index(contig)])
        except AttributeError:
            contig_length = None    # A source header without lengths stays unspecified.
    finally:
        header.close()
    positions = np.asarray(positions, dtype=np.int64)
    count = np.asarray(count, dtype=bool)
    size = len(positions)
    ref, alt = np.empty(size, dtype="<U1"), np.empty(size, dtype="<U1")
    ref_reads, alt_reads = np.zeros(size, np.int64), np.zeros(size, np.int64)
    source = np.zeros(size, np.int8)
    reader = VCF(str(reference), samples=list(sample_ids)) if count.any() else VCF(str(reference))
    try:
        variants = iter(reader(contig))
        record = None
        for site, position in enumerate(positions.tolist()):
            # The analysed site is the position's biallelic SNP record (the loader drops the others).
            while record is None or record.POS < position or (record.POS == position
                                                                and not _biallelic_snp(record)):
                record = next(variants, None)
                if record is None:
                    raise ValueError(f"input VCF lacks a biallelic SNP at {contig}:{position}")
            if record.POS != position:
                raise ValueError(f"input VCF lacks a biallelic SNP at {contig}:{position}")
            ref[site], alt[site] = record.REF, record.ALT[0]
            if count[site]:
                depths = record.format("AD")
                if depths is None or (depths.shape[1] == 1 and np.all(depths < 0)):
                    types = record.gt_types          # 0 HOM_REF, 1 HET, 2 UNKNOWN, 3 HOM_ALT
                    ref_reads[site] = 2 * int(np.sum(types == 0)) + int(np.sum(types == 1))
                    alt_reads[site] = 2 * int(np.sum(types == 3)) + int(np.sum(types == 1))
                    source[site] = 2
                else:
                    depths = np.maximum(depths[:, :2], 0)       # missing values are no reads
                    ref_reads[site] = int(depths[:, 0].sum())
                    alt_reads[site] = int(depths[:, 1].sum())
                    source[site] = 1
    finally:
        reader.close()
    return ref, alt, ref_reads, alt_reads, source, contig_length


def _simulated_read_totals(file, positions, sample_ids):
    """REF and ALT read totals over ``sample_ids`` at ``positions`` (from the simulation/reads record)."""
    from ..simulation import truth
    reads = truth.load_reads(file)
    simulated = np.asarray(reads["positions"], dtype=np.int64)
    index = np.searchsorted(simulated, positions)
    if np.any(index >= len(simulated)) or not np.array_equal(simulated[np.minimum(index, len(simulated) - 1)],
                                                             positions):
        raise ValueError(f"{file.label}: simulated reads lack exported sites")
    chosen = set(sample_ids)
    rows = np.array([name in chosen for name in reads["sample_ids"]], dtype=bool)
    totals = np.asarray(reads["counts"])[rows][:, index, :].sum(axis=0, dtype=np.int64)
    return totals[:, 0], totals[:, 1]


def _common_alleles(r5_only, ref_reads, alt_reads, source):
    common = np.full(len(r5_only), -1, dtype=np.int8)
    common[r5_only & (ref_reads > alt_reads)] = 0
    common[r5_only & (alt_reads > ref_reads)] = 1
    decided = r5_only & (common >= 0)
    counts = dict(ref=int(np.sum(common == 0)), alt=int(np.sum(common == 1)), tie=int(np.sum(r5_only & (common < 0))),
                  from_reads=int(np.sum(decided & (source == 1))), from_gt=int(np.sum(decided & (source == 2))))
    return common, counts


def kept_site_table(contig: str, positions, *, reference, synthetic: bool, sample_ids=()) -> SiteTable:
    """A kept-site table for given positions (callers holding products without the evidence report): REF/ALT
    from ``reference`` (checked to hold ``sample_ids`` and the contig) or A/C for a simulation."""
    positions = np.asarray(positions, dtype=np.int64)
    size = len(positions)
    if synthetic:
        ref, alt, length = np.full(size, "A"), np.full(size, "C"), None
    else:
        ref, alt, _, _, _, length = read_input_sites(reference, contig, positions, np.zeros(size, bool), sample_ids)
    return SiteTable(contig=contig, positions=positions, kept=np.ones(size, bool),
                     removed_bits=np.zeros(size, np.uint8), ref=ref, alt=alt, common=np.full(size, -1, np.int8),
                     full_length=False, contig_length=length)


def evidence_sites(file) -> dict:
    """What the evidence record says about a contig's sites: sample_ids, counted_samples (those the contamination
    QC did not flag: r5's carrier count and the common allele use them), kept positions, input positions, kept
    mask, removed_bits (applied rules that fired), site-filter state, applied rules and loader drops."""
    from ..evidence import records as evidence_records
    payload = file.get(evidence_records.EVIDENCE, items=("report", "positions", "sample_ids", "founder_excluded"))
    report = payload["report"]
    kept_positions = np.asarray(payload["positions"], dtype=np.int64)
    input_positions = np.asarray(report["input_positions"], dtype=np.int64)
    quality = report["site_quality"]
    kept = np.isin(input_positions, kept_positions, assume_unique=True)
    if int(kept.sum()) != len(kept_positions):
        raise ValueError(f"{file.label}: evidence sites are not among the input sites")
    applied = tuple(quality["applied_rules"])
    mask = np.uint8(sum(1 << int(rule[1:]) for rule in applied))
    bits = quality["rule_bits"]
    removed_bits = (np.zeros(len(input_positions), np.uint8) if bits is None
                    else np.asarray(bits, dtype=np.uint8) & mask)
    if not np.array_equal(removed_bits == 0, kept):
        raise ValueError(f"{file.label}: site-filter rule bits disagree with the kept sites")
    if not quality["enabled"]:
        state = "off"
    elif not quality["applied"]:
        state = f"not applied ({quality['reason']})"
    else:
        state = "applied"
    sample_ids = tuple(payload["sample_ids"])
    flagged = np.asarray(payload["founder_excluded"], dtype=bool)
    return dict(sample_ids=sample_ids,
                counted_samples=tuple(name for name, out in zip(sample_ids, flagged.tolist()) if not out),
                kept_positions=kept_positions,
                input_positions=input_positions, kept=kept, removed_bits=removed_bits, site_filter=state,
                applied_rules=applied, dropped={str(key): int(value) for key, value in report["dropped"].items()})


def build_site_table(file, contig: str, *, full_length: bool, reference, synthetic: bool,
                     evidence: dict | None = None) -> SiteTable:
    """The rows of one contig's VCF products from its evidence record and the input file (or simulated reads)."""
    evidence = evidence_sites(file) if evidence is None else evidence
    if full_length:
        positions, kept, bits = evidence["input_positions"], evidence["kept"], evidence["removed_bits"]
    else:
        positions = evidence["kept_positions"]
        kept, bits = np.ones(len(positions), bool), np.zeros(len(positions), np.uint8)
    r5_only = bits == R5_ONLY
    size = len(positions)
    if synthetic:
        ref, alt, length = np.full(size, "A"), np.full(size, "C"), None
        ref_reads, alt_reads = np.zeros(size, np.int64), np.zeros(size, np.int64)
        source = np.where(r5_only, 1, 0).astype(np.int8)
        if r5_only.any():
            ref_reads[r5_only], alt_reads[r5_only] = _simulated_read_totals(file, positions[r5_only],
                                                                            evidence["counted_samples"])
    else:
        ref, alt, ref_reads, alt_reads, source, length = read_input_sites(reference, contig, positions, r5_only,
                                                                          evidence["counted_samples"])
    common, counts = _common_alleles(r5_only, ref_reads, alt_reads, source)
    return SiteTable(contig=contig, positions=positions, kept=kept, removed_bits=bits, ref=ref, alt=alt,
                     common=common, full_length=full_length, contig_length=length,
                     site_filter=evidence["site_filter"], applied_rules=evidence["applied_rules"],
                     dropped=evidence["dropped"], common_counts=counts if full_length else {})
