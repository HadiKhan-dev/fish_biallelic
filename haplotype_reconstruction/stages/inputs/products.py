"""Open a task's checked reader and its separate writable store capability."""

from __future__ import annotations

from ...engine.cpu import TASK_THREADS
from ...results.protocol import ProductInputs
from ...store.store import CheckpointStore


def task_inputs(root):
    """Use the task's live decoder share for reading and publishing."""
    return ProductInputs(root, nthreads=TASK_THREADS), CheckpointStore(root, nthreads=TASK_THREADS)


def required_product(reader, adapter, product, contig):
    """Read a required product through its independently checked request."""
    request = reader.request(adapter, product, contig)
    return reader.require(request, contig)


def required_dense_evidence(reader, contig):
    """Project a checked dense view only for a whole-contig consumer."""
    from ..evidence import records

    request = reader.request(records, "evidence", contig)
    reader.require_receipt(request, contig)
    file = CheckpointStore(reader.checkpoints, nthreads=TASK_THREADS).contig(contig)
    return records.load_dense(file, expected_identity=request.identity)
