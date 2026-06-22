"""checkpoint_io.py - compressed, multithreaded per-contig checkpoint I/O.

Shared serialization layer for the pipeline stages.  Each checkpoint is a
pickle.dumps(obj) compressed with blosc2 (ZSTD + the default SHUFFLE filter)
and written atomically (tmp file + os.replace).  blosc2 releases the GIL while
(de)compressing and parallelises internally across `nthreads`, so writes scale
both ACROSS contigs (one thread per contig, via save_contigs_parallel) and
WITHIN one contig (nthreads per compress call).

On-disk layout (the checkpoint root is supplied by the caller, so the same
helpers serve pipelines that use different checkpoint directories):

    {ckpt_dir}/{stage}/{r_name}.pkl.b2   - one file per contig
    {ckpt_dir}/{stage}/_global.pkl.b2    - one per-stage global

The ".pkl.b2" suffix keeps the format from colliding with any stale plain
pickles; the format is intentionally not backward compatible with the old
uncompressed .pkl checkpoints (delete an old checkpoint dir before first use).

Importing this module has no side effects beyond importing blosc2.
"""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import blosc2

SUFFIX = ".pkl.b2"
CLEVEL = 5
CODEC = blosc2.Codec.ZSTD


def contig_path(ckpt_dir, stage, r_name):
    """Path of one contig's checkpoint: {ckpt_dir}/{stage}/{r_name}.pkl.b2."""
    return os.path.join(ckpt_dir, stage, r_name + SUFFIX)


def global_path(ckpt_dir, stage):
    """Path of a stage's global checkpoint: {ckpt_dir}/{stage}/_global.pkl.b2."""
    return os.path.join(ckpt_dir, stage, "_global" + SUFFIX)


def write(path, obj, nthreads=1):
    """Pickle (HIGHEST_PROTOCOL) + blosc2-compress `obj` with `nthreads`,
    written atomically (tmp + os.replace).  Returns the number of bytes
    written.  On failure the temp file is removed and the error re-raised."""
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    blob = blosc2.compress2(
        raw,
        cparams=blosc2.CParams(nthreads=max(1, int(nthreads)),
                               clevel=CLEVEL, codec=CODEC),
    )
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return len(blob)


def read(path, nthreads=1):
    """Inverse of write(): read + blosc2-decompress (`nthreads`) + unpickle."""
    with open(path, "rb") as f:
        blob = f.read()
    raw = blosc2.decompress2(
        blob, dparams=blosc2.DParams(nthreads=max(1, int(nthreads))))
    return pickle.loads(raw)


def save_contigs_parallel(ckpt_dir, stage, items, total_cores):
    """Write many contigs' checkpoints concurrently under {ckpt_dir}/{stage}/.

    `items` is an iterable of (r_name, data_object).  Two-level parallelism --
    a ThreadPoolExecutor across contigs x blosc2 `nthreads` within each call --
    which is genuine (not GIL-bound) because blosc2 releases the GIL while
    compressing.  Use this for stages whose contigs are all resident at once;
    the per-contig serial-compute stages write via write() one at a time.

    Returns a list of (r_name, n_bytes, error_or_None), and prints a one-line
    summary plus a warning per failed contig.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []
    os.makedirs(os.path.join(ckpt_dir, stage), exist_ok=True)
    total = max(1, int(total_cores))
    workers = min(n, total)
    threads = max(1, total // workers)

    def _one(item):
        r_name, data = item
        try:
            nbytes = write(contig_path(ckpt_dir, stage, r_name), data, nthreads=threads)
            return (r_name, nbytes, None)
        except OSError as e:
            return (r_name, 0, e)

    if workers == 1:
        results = [_one(it) for it in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_one, items))

    ok = [(r, b) for r, b, e in results if e is None]
    mb = sum(b for _, b in ok) / (1024 * 1024)
    print(f"  [Checkpoint] {stage}: {len(ok)}/{n} contigs written "
          f"({workers} parallel x {threads} blosc2 threads, {mb:.1f} MB total)")
    for r, _b, e in results:
        if e is not None:
            print(f"  [Checkpoint] WARNING: {stage}/{r}: {e}")
    return results