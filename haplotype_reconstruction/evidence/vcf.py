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

from array import array
from dataclasses import dataclass
from typing import Sequence
import weakref

import numpy as np
from cyvcf2 import VCF

from ..execution import (BudgetedPool, attach_shared_array, close_shared_memory,
                         empty_shared_array, reserve_memory)
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


def _scan_window(path, contig, start, stop, samples):
    """Find the output site axis without decoding sample FORMAT fields."""
    reader = VCF(str(path), lazy=True)
    try:
        if stop is None:
            region = contig if start <= 1 else f"{contig}:{start}-"
        else:
            region = f"{contig}:{start}-{stop - 1}"
        positions = array("q")
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
            positions.append(position)
    finally:
        reader.close()
    positions = np.asarray(positions, np.int64)
    # Equal positions always belong to the same physical window. Drop every member of each
    # duplicate run, just as the former whole-contig unique/isin pass did.
    keep = np.ones(len(positions), bool)
    repeated = positions[1:] == positions[:-1]
    keep[1:] &= ~repeated
    keep[:-1] &= ~repeated
    duplicated = int(len(positions) - np.count_nonzero(keep))
    return positions[keep], not_snp, multiallelic, duplicated


def _read_window_into(task, positions, counts):
    """Decode directly into this window's columns of the final sample-major count array."""
    path, contig, start, stop, samples = task
    reader = (VCF(str(path), lazy=True) if samples is None
              else VCF(str(path), samples=list(samples), lazy=True))
    try:
        n_samples = len(reader.samples)
        if stop is None:
            region = contig if start <= 1 else f"{contig}:{start}-"
        else:
            region = f"{contig}:{start}-{stop - 1}"
        site = 0
        for record in reader(region):
            position = record.POS
            if position < start or (stop is not None and position >= stop):
                continue
            if not _is_snp(record.REF, record.ALT):
                continue
            ad = record.format("AD")
            if ad is None or (ad.shape[1] == 1 and np.all(ad < 0)):
                # No AD, or HTSlib's single all-missing sentinel column: zero observed reads.
                ad = None
            elif ad.shape != (n_samples, 2):
                raise ValueError(f"{path}: {contig}:{position}: AD must hold REF and ALT depths "
                                 f"for every sample (found shape {ad.shape})")
            # Validate AD even at duplicate positions: these errors were visible before filtering.
            if site < len(positions) and position == positions[site]:
                if ad is None:
                    counts[:, site, :] = 0
                else:
                    np.maximum(ad, 0, out=counts[:, site, :])
                site += 1
        if site != len(positions):
            raise ValueError(f"{path}: {contig}: input changed while decoding allele depths")
    finally:
        reader.close()


def _scan_window_task(task, shared, budget):
    return _scan_window(*task)


def _read_window_task(task, shared, budget):
    """Workers attach the final output; only small task results cross the pool's pipes."""
    window, positions, start, reference = task
    handle, counts = attach_shared_array(reference)
    try:
        _read_window_into(window, positions, counts[:, start:start + len(positions), :])
    finally:
        del counts
        close_shared_memory([handle])


def _decode_windows(tasks, results, sample_ids, contig, pool):
    positions = np.concatenate([result[0] for result in results])
    dropped = {DROPPED_NOT_SNP: sum(result[1] for result in results),
               DROPPED_MULTIALLELIC: sum(result[2] for result in results),
               DROPPED_DUPLICATE_POSITION: sum(result[3] for result in results)}
    n_samples, n_sites = len(sample_ids), len(positions)
    # The output is the only sample-by-site allocation. Workers need one decoded AD record,
    # plus HTSlib buffers; neither window depths nor counts travel through pickle.
    scratch = (1 << 20) + 32 * n_samples
    output_bytes = 8 * n_samples * n_sites + positions.nbytes
    with reserve_memory(output_bytes, label="vcf decoded counts"):
        if pool is None:
            counts = np.empty((n_samples, n_sites, 2), np.int32)
            with reserve_memory(scratch, label="vcf record"):
                _read_window_into(tasks[0], positions, counts)
        else:
            handle, reference, counts = empty_shared_array((n_samples, n_sites, 2), np.int32)
            try:
                offset = 0
                decoding = []
                for task, result in zip(tasks, results):
                    decoding.append((task, result[0], offset, reference))
                    offset += len(result[0])
                pool.map(_read_window_task, decoding, task_bytes=scratch)
            except BaseException:
                del counts
                close_shared_memory([handle], unlink=True)
                raise
            # ContigReads freezes a view whose base retains this array. Its finalizer owns the
            # shared-memory handle, so descendants can keep views without a whole-contig copy.
            weakref.finalize(counts, close_shared_memory, [handle], unlink=True)
        return ContigReads(contig=contig, sample_ids=sample_ids, positions=positions,
                           counts=counts, dropped=dropped)


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
        return _decode_windows(tasks, [_scan_window(*tasks[0])], sample_ids, contig, None)
    with BudgetedPool(budget, processes=min(workers, len(tasks)), label="vcf windows") as pool:
        results = pool.map(_scan_window_task, tasks)
        return _decode_windows(tasks, results, sample_ids, contig, pool)
