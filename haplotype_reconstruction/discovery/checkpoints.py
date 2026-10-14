"""A chromosome's block discovery records in its contig checkpoint file.

``discovery/prepared``  the prepared chromosome, recorded as soon as its
    preparation finishes: its evidence (``global_probs``, the chromosome's
    only stored genotype-likelihood copy, ``global_sites``,
    ``global_observed_mask``), read calibration, site quality, evidence and
    observation-mask modes, the discovery backend and configuration, and the
    workflow's further fields (e.g. ``founder_evidence_excluded``).
``discovery/plan``  {blocks: input block count, spans: (blocks, 2) first and
    last input positions, global_probs_nbytes}.
``discovery/evidence_sha256``  SHA-256 identities of the three evidence
    arrays, recorded by the assembly release when it hashed exactly these
    arrays, so their consumers need not hash them again.
``discovery/blocks/<n:06d>``  up to CHUNK_BLOCKS completed block tasks in
    completion order, {indices, blocks} (None for a block without informative
    data). ``n`` counts the tasks recorded once this chunk is, so the largest
    ``n`` is the discovery's progress; it is complete at the plan's count.

Block results are recorded without ``probs_array``: loading rebuilds it from
the prepared evidence exactly as discovery made it (block_probabilities). An
interrupted discovery resumes with its missing block tasks only.
"""
from __future__ import annotations

import numpy as np

from ..core.haplotypes import BlockResults


PREPARED = "discovery/prepared"


PLAN = "discovery/plan"


CHUNKS = "discovery/blocks/"


EVIDENCE_SHA256 = "discovery/evidence_sha256"


EVIDENCE_ARRAYS = ("global_probs", "global_sites", "global_observed_mask")


CHUNK_BLOCKS = 64


def recorded_tasks(file):
    """Block tasks recorded so far (no payload reads)."""
    counts = [int(key[len(CHUNKS):]) for key in file.keys(CHUNKS)]
    return max(counts, default=0)


def complete(file):
    """Whether every planned block task of the contig is recorded."""
    return file.has(PLAN) and recorded_tasks(file) == file.get(PLAN)["blocks"]


def require_complete(store, contigs):
    """Require every listed contig's discovery records (e.g. before completion)."""
    missing = [str(contig) for contig in contigs if not complete(store.contig(contig))]
    if missing:
        raise OSError("Failed to checkpoint block discovery: " + ", ".join(missing))


def block_keys(file):
    """The records load_blocks decodes, or None until discovery is complete."""
    if not complete(file):
        return None
    return (*file.keys(CHUNKS), PREPARED)


def decoded_bytes(file):
    """Decoded bytes of the contig's discovery product with its blocks loaded.

    A planning allowance from record metadata (and the small plan): the
    discovery records plus the probs_array copies rebuilt on load, which
    together hold the chromosome's genotype likelihoods once more.
    """
    return (sum(info.decoded_bytes for info in file.infos("discovery/"))
            + file.get(PLAN)["global_probs_nbytes"])


def block_probabilities(evidence, positions):
    """A block's probs_array, as discovery made it from the prepared evidence.

    ``evidence``: the prepared record's global_probs, global_sites,
    global_observed_mask and optional founder_evidence_excluded. Discovery
    copies the block's global_probs columns as read-only float64 and makes
    every cell without discovery reads uniform (1/3): unobserved cells and
    the samples excluded from founder discovery.
    """
    sites = evidence["global_sites"]
    indices = np.searchsorted(sites, positions)
    if np.any(indices >= len(sites)) or not np.array_equal(sites[indices], positions):
        raise ValueError("discovery block positions do not match the chromosome's sites")
    if len(indices) and indices[-1] - indices[0] == len(indices) - 1:
        columns = slice(indices[0], indices[-1] + 1)
    else:
        columns = indices
    result = np.array(evidence["global_probs"][:, columns], dtype=np.float64, order="C",
                      copy=True)
    without_reads = ~np.asarray(evidence["global_observed_mask"][:, columns], dtype=np.bool_)
    excluded = evidence.get("founder_evidence_excluded")
    if excluded is not None:
        without_reads[np.asarray(excluded, dtype=np.bool_)] = True
    result[without_reads] = 1.0 / 3.0
    result.setflags(write=False)
    return result


def record_prepared(file, prepared):
    """Record a prepared contig (prepared record, then plan) unless already recorded.

    A resumed contig is prepared again to discover its missing blocks; its
    records are kept (their identities bind later stages) after checking that
    the preparation reproduced its evidence arrays exactly, so blocks
    discovered now and the probs_array rebuilt from the record agree.
    """
    fields = prepared.fields
    if file.has(PREPARED):
        stored = file.get(PREPARED, items=EVIDENCE_ARRAYS)
        if any(stored[name].dtype != fields[name].dtype
               or not np.array_equal(stored[name], fields[name]) for name in EVIDENCE_ARRAYS):
            raise RuntimeError(
                f"{file.label}: prepared discovery input differs from its checkpoint; "
                "use a new output root")
    else:
        file.put(PREPARED, fields)
    spans = np.array([(positions[0], positions[-1]) if len(positions) else (-1, -1)
                      for positions, _reads, _keep in prepared.genomic_data],
                     dtype=np.int64).reshape(-1, 2)
    if file.has(PLAN):
        if not np.array_equal(file.get(PLAN)["spans"], spans):
            raise RuntimeError(
                f"{file.label}: discovery blocks differ from the recorded plan; "
                "use a new output root")
    else:
        file.put(PLAN, dict(blocks=len(spans), spans=spans,
                            global_probs_nbytes=int(fields["global_probs"].nbytes)))


def recorded_indices(file):
    """Indices of the block tasks already recorded (decodes only the index arrays)."""
    indices = [file.get(key, items=("indices",))["indices"] for key in file.keys(CHUNKS)]
    return set(np.concatenate(indices).tolist()) if indices else set()


class Chunks:
    """Completed block tasks recorded CHUNK_BLOCKS at a time in completion order.

    Each block's probs_array is checked to equal its rebuild from the
    prepared evidence (block_probabilities), then dropped. ``flush`` records
    the remainder (at the end, and on an interruption that leaves Python
    running).
    """

    def __init__(self, file, evidence):
        self.file = file
        self.evidence = evidence
        self.recorded = recorded_tasks(file)
        self.pending = []

    def add(self, index, block):
        if block is not None:
            rebuilt = block_probabilities(self.evidence, block.positions)
            probs = block.probs_array
            if probs.dtype != rebuilt.dtype or not np.array_equal(probs, rebuilt):
                raise RuntimeError(
                    f"{self.file.label}: block {index} genotype likelihoods differ from "
                    "the chromosome's prepared evidence")
            block.probs_array = None
        self.pending.append((index, block))
        if len(self.pending) == CHUNK_BLOCKS:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        self.recorded += len(self.pending)
        self.file.put(f"{CHUNKS}{self.recorded:06d}", dict(
            indices=np.array([index for index, _ in self.pending], dtype=np.int64),
            blocks=[block for _, block in self.pending]), nthreads=1)
        self.pending = []


# The prepared entries block_probabilities reads.
PROBABILITY_SOURCES = (*EVIDENCE_ARRAYS, "founder_evidence_excluded")


def attach_probabilities(blocks, evidence):
    """Rebuild every block's probs_array from the prepared evidence (block_probabilities)."""
    for block in blocks:
        block.probs_array = block_probabilities(evidence, block.positions)
    return blocks


def load_blocks(file, *, probs=True, nthreads=None):
    """The contig's discovered blocks in genomic order, as discovery produced them.

    ``probs=False`` leaves probs_array unset (None); attach_probabilities adds
    it later from already loaded evidence.
    """
    planned = file.get(PLAN)["blocks"]
    by_index = {}
    for key in file.keys(CHUNKS):
        chunk = file.get(key, nthreads=nthreads)
        by_index.update(zip(chunk["indices"].tolist(), chunk["blocks"]))
    if len(by_index) != planned or set(by_index) != set(range(planned)):
        raise RuntimeError(
            f"{file.label}: block discovery is incomplete ({len(by_index)} of "
            f"{planned} blocks recorded)")
    blocks = BlockResults([block for index in range(planned)
                           if (block := by_index[index]) is not None and len(block.positions)])
    if probs:
        attach_probabilities(blocks, file.get(PREPARED, items=PROBABILITY_SOURCES,
                                              nthreads=nthreads))
    return blocks
