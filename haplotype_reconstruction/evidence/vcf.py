"""Indexed VCF/BCF input: the allele depths (FORMAT/AD) of one contig's biallelic SNPs.

One loader policy for every input (real crosses and generic VCFs):

* records that are not biallelic SNPs are dropped and counted by reason: ``multiallelic`` (more
  than one ALT allele) and ``not_snp`` (REF or ALT is not one of A, C, G, T: indels, MNPs,
  symbolic alleles, ``*``, no ALT). Multi-allelic depths are never truncated into biallelic ones;
* when two or more biallelic SNP records share a position, all of them are dropped
  (``duplicate_position``): which one is the site is not decidable from the file;
* a missing AD value is zero reads, as is a record without AD;
* everything else that is not as expected is an error: unreadable files, unknown contigs or
  samples, records out of position order, AD with the wrong number of values. No exception is
  swallowed.

The contig is read in parallel over disjoint physical windows (a record belongs to the window
that contains its POS), so the result does not depend on the number of workers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from cyvcf2 import VCF

from ..execution import BudgetedPool
from ..execution.budget import as_budget
from ..model.evidence import (ContigReads, DROPPED_DUPLICATE_POSITION, DROPPED_MULTIALLELIC,
                              DROPPED_NOT_SNP)

_BASES = frozenset("ACGTacgt")
# Physical windows per worker: enough tasks to balance uneven record density.
_WINDOWS_PER_WORKER = 4
_MINIMUM_WINDOW_BP = 100_000


@dataclass(frozen=True)
class VcfHeader:
    """What the header of an input file declares."""

    sample_ids: tuple[str, ...]
    contigs: tuple[str, ...]
    lengths: dict          # contig -> length in bp (empty when the header gives none)
    has_allele_depths: bool


def inspect(path) -> VcfHeader:
    """Read the header of an indexed VCF/BCF."""
    reader = VCF(str(path))
    try:
        try:
            lengths = dict(zip(reader.seqnames, map(int, reader.seqlens)))
        except AttributeError:          # cyvcf2: "no sequence lengths found in header"
            lengths = {}
        return VcfHeader(sample_ids=tuple(reader.samples), contigs=tuple(reader.seqnames),
                         lengths=lengths, has_allele_depths="##FORMAT=<ID=AD," in reader.raw_header)
    finally:
        reader.close()


def _is_snp(reference, alternates):
    return (len(alternates) == 1 and len(reference) == 1 and len(alternates[0]) == 1
            and reference in _BASES and alternates[0] in _BASES)


def _read_window(path, contig, start, stop, samples):
    """Biallelic SNP depths of the records with start <= POS < stop (stop None: to the end).

    Returns (positions int64[k], depths int32[k, N, 2], not_snp, multiallelic).
    """
    reader = VCF(str(path)) if samples is None else VCF(str(path), samples=list(samples))
    try:
        n_samples = len(reader.samples)
        if stop is None:
            region = contig if start <= 1 else f"{contig}:{start}-"
        else:
            region = f"{contig}:{start}-{stop - 1}"
        positions, depths = [], []
        not_snp = multiallelic = 0
        previous = 0
        for record in reader(region):
            position = record.POS
            if position < start or (stop is not None and position >= stop):
                continue            # starts outside this window (overlap query)
            if position < previous:
                raise ValueError(f"{path}: {contig}:{position}: records are not sorted by position")
            previous = position
            alternates = record.ALT
            if len(alternates) > 1:
                multiallelic += 1
                continue
            if not _is_snp(record.REF, alternates):
                not_snp += 1
                continue
            ad = record.format("AD")
            if ad is None or (ad.shape[1] == 1 and np.all(ad < 0)):
                # No AD, or HTSlib's single all-missing sentinel column: zero observed reads.
                ad = np.zeros((n_samples, 2), np.int32)
            elif ad.shape != (n_samples, 2):
                raise ValueError(f"{path}: {contig}:{position}: AD must hold REF and ALT depths "
                                 f"for every sample (found shape {ad.shape})")
            positions.append(position)
            depths.append(ad)
    finally:
        reader.close()
    if not positions:
        return np.empty(0, np.int64), np.empty((0, n_samples, 2), np.int32), not_snp, multiallelic
    depths = np.stack(depths).astype(np.int32, copy=False)
    depths[depths < 0] = 0          # missing values
    return np.asarray(positions, np.int64), depths, not_snp, multiallelic


def _read_window_task(task, shared, budget):
    """Pool task wrapper of _read_window (module level, picklable)."""
    return _read_window(*task)


def _windows(length, workers):
    """Disjoint [start, stop) windows covering 1..length (one open window when length is unknown)."""
    if length is None or workers <= 1:
        return [(1, None)]
    count = max(1, min(workers * _WINDOWS_PER_WORKER, length // _MINIMUM_WINDOW_BP))
    bounds = np.linspace(1, length + 1, count + 1).astype(np.int64)
    windows = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    # Records beyond the declared length (a malformed header) still belong to the last window.
    windows[-1] = (windows[-1][0], None)
    return windows


def read_contig(path, contig: str, budget, *, samples: Sequence[str] | None = None) -> ContigReads:
    """Biallelic SNP allele depths of ``contig``; ``samples`` selects a subset (header order kept).

    ``budget`` (a CpuBudget or a thread count) bounds the worker processes reading disjoint physical
    windows in parallel; the result is identical for any budget.
    """
    workers = as_budget(budget).threads()
    header = inspect(path)
    if contig not in header.contigs:
        raise ValueError(f"{path}: contig {contig!r} is not in the header")
    if not header.has_allele_depths:
        raise ValueError(f"{path}: FORMAT/AD allele depths are required (GT- or PL-only input is "
                         "not supported)")
    sample_ids = header.sample_ids
    if samples is not None:
        unknown = sorted(set(samples) - set(header.sample_ids))
        if unknown:
            raise ValueError(f"{path}: samples not in the header: {unknown}")
        chosen = set(samples)
        sample_ids = tuple(name for name in header.sample_ids if name in chosen)
    tasks = [(str(path), contig, start, stop, None if samples is None else sample_ids)
             for start, stop in _windows(header.lengths.get(contig), workers)]
    if len(tasks) == 1:
        results = [_read_window(*tasks[0])]
    else:
        with BudgetedPool(budget, processes=min(workers, len(tasks)), label="vcf windows") as pool:
            results = pool.map(_read_window_task, tasks)
    positions = np.concatenate([result[0] for result in results])
    depths = np.concatenate([result[1] for result in results])        # (L, N, 2)
    dropped = {DROPPED_NOT_SNP: sum(result[2] for result in results),
               DROPPED_MULTIALLELIC: sum(result[3] for result in results)}
    unique, counts = np.unique(positions, return_counts=True)
    duplicated = unique[counts > 1]
    keep = ~np.isin(positions, duplicated)
    dropped[DROPPED_DUPLICATE_POSITION] = int((~keep).sum())
    return ContigReads(contig=contig, sample_ids=sample_ids, positions=positions[keep],
                       counts=np.ascontiguousarray(depths[keep].transpose(1, 0, 2)),
                       dropped=dropped)
