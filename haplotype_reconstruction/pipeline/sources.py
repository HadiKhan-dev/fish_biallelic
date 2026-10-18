"""Observation sources: where a run's allele depths come from (an indexed VCF/BCF, or a simulation).

A source has ``kind``, ``contigs()``, ``sample_ids()``, ``identity() -> dict`` (bound into the evidence identity),
``materialise(store, contigs, budget)`` (writes simulated inputs; nothing for a VCF) and
``reads(contig, budget) -> ContigReads``. The simulation source is the simulation package's; the VCF source wraps
the evidence stage's reader and checks the input before any computation.
"""
from __future__ import annotations

from pathlib import Path


class VcfSource:
    """An indexed VCF/BCF with FORMAT/AD; every header sample is analysed."""

    kind = "vcf"

    def __init__(self, path, contigs):
        from ..evidence import vcf
        self.path = Path(path)
        self._contigs = tuple(contigs)
        self.header = vcf.inspect(self.path)

    def check(self) -> None:
        """Fail before any computation: AD present, >= 3 unique samples, every contig declared and populated."""
        header = self.header
        if not header.has_allele_depths:
            raise ValueError(f"{self.path}: FORMAT/AD allele depths are required (GT- or PL-only input is not "
                             "supported)")
        if len(header.sample_ids) < 3 or len(set(header.sample_ids)) != len(header.sample_ids):
            raise ValueError(f"{self.path}: at least three uniquely named samples are required")
        missing = [contig for contig in self._contigs if contig not in header.contigs]
        if missing:
            raise ValueError(f"{self.path}: contigs not in the header: {', '.join(missing)}")
        from cyvcf2 import VCF
        reader = VCF(str(self.path))
        try:
            empty = [contig for contig in self._contigs if next(iter(reader(contig)), None) is None]
        finally:
            reader.close()
        if empty:
            raise ValueError(f"{self.path}: no indexed records for {', '.join(empty)}; select populated contigs")

    def contigs(self) -> tuple[str, ...]:
        return self._contigs

    def sample_ids(self) -> tuple[str, ...]:
        return self.header.sample_ids

    def identity(self) -> dict:
        """The input file (path, size, modification time) and its sample axis."""
        status = self.path.stat()
        return dict(kind=self.kind, vcf=str(self.path.resolve()), size=status.st_size,
                    mtime_ns=status.st_mtime_ns, sample_ids=list(self.header.sample_ids))

    def materialise(self, store, contigs, budget) -> None:
        return None

    def reads(self, contig, budget):
        from ..evidence import vcf
        return vcf.read_contig(self.path, contig, budget)


def observation_source(config):
    """The run's source: VcfSource, or the simulation package's SimulationSource."""
    if config.input.kind == "vcf":
        return VcfSource(config.input.vcf, config.input.contigs)
    from ..config import stages
    from ..simulation.source import SimulationSource
    return SimulationSource(stages.simulation(config))
