"""checkpoint_io.py - compressed, multithreaded per-contig checkpoint I/O.

Shared serialization layer for the pipeline stages.  Each checkpoint is a
pickle.dumps(obj) compressed with blosc2 (ZSTD + the default SHUFFLE filter)
and written atomically (tmp file + os.replace).  The pickle is split into
<2 GiB segments behind a small frame (see the _CHUNK_* constants) so payloads
of any size clear blosc2's single-buffer limit.  blosc2 releases the GIL while
(de)compressing and parallelises internally across `nthreads`, so writes scale
both ACROSS contigs (one thread per contig, via save_contigs_parallel) and
WITHIN one contig (nthreads per compress call).

On-disk layout (the checkpoint root is supplied by the caller, so the same
helpers serve pipelines that use different checkpoint directories):

    {ckpt_dir}/{stage}/{r_name}.pkl.b2   - one file per contig
    {ckpt_dir}/{stage}/_global.pkl.b2    - one per-stage global

The ".pkl.b2" suffix keeps the format from colliding with any stale plain
pickles; the frame is not backward compatible with earlier checkpoints (plain
.pkl, or the pre-chunking single-frame .pkl.b2), so delete any existing
checkpoint dir before first use.

Importing this module has no side effects beyond importing blosc2.
"""

import os
import pickle
import struct
from concurrent.futures import ThreadPoolExecutor

import blosc2

SUFFIX = ".pkl.b2"
CLEVEL = 5
CODEC = blosc2.Codec.ZSTD

# blosc2.compress2 takes the source size as a signed 32-bit int, so a single
# call caps at INT32_MAX (~2 GiB) and raises "negative count" beyond it.  Large
# checkpoints (e.g. a whole simulated contig) exceed that, so write() splits the
# pickle into <2 GiB segments, compresses each, and stores them behind a small
# self-describing frame that read() parses:
#     _CHUNK_MAGIC (8B) | uint64 n_chunks | (uint64 seg_len | seg) * n_chunks
_CHUNK_MAGIC = b"BHDB2CK1"          # 8-byte format marker
_CHUNK_BYTES = 1 << 30              # 1 GiB of raw input per compressed segment (safe < 2 GiB)


def _compress(raw, nthreads):
    return blosc2.compress2(
        raw,
        cparams=blosc2.CParams(nthreads=max(1, int(nthreads)),
                               clevel=CLEVEL, codec=CODEC),
    )


def _decompress(blob, nthreads):
    return blosc2.decompress2(
        blob, dparams=blosc2.DParams(nthreads=max(1, int(nthreads))))


def contig_path(ckpt_dir, stage, r_name):
    """Path of one contig's checkpoint: {ckpt_dir}/{stage}/{r_name}.pkl.b2."""
    return os.path.join(ckpt_dir, stage, r_name + SUFFIX)


def global_path(ckpt_dir, stage):
    """Path of a stage's global checkpoint: {ckpt_dir}/{stage}/_global.pkl.b2."""
    return os.path.join(ckpt_dir, stage, "_global" + SUFFIX)


def write(path, obj, nthreads=1):
    """Pickle (HIGHEST_PROTOCOL) + blosc2-compress `obj` with `nthreads`,
    written atomically (tmp + os.replace).  Returns the number of bytes
    written.  On failure the temp file is removed and the error re-raised.

    The pickle is split into <2 GiB segments (blosc2.compress2 caps a single
    buffer at INT32_MAX), each compressed independently, so checkpoints of any
    size are handled.  Segments stream straight to the temp file rather than
    accumulating one giant in-memory blob.
    """
    raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    n_chunks = max(1, (len(raw) + _CHUNK_BYTES - 1) // _CHUNK_BYTES)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(_CHUNK_MAGIC)
            f.write(struct.pack("<Q", n_chunks))
            for i in range(n_chunks):
                seg = _compress(raw[i * _CHUNK_BYTES:(i + 1) * _CHUNK_BYTES], nthreads)
                f.write(struct.pack("<Q", len(seg)))
                f.write(seg)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return os.path.getsize(path)


def read(path, nthreads=1):
    """Inverse of write(): parse the chunk frame, blosc2-decompress each segment
    (`nthreads`), join, and unpickle."""
    with open(path, "rb") as f:
        blob = f.read()
    m = len(_CHUNK_MAGIC)
    if blob[:m] != _CHUNK_MAGIC:
        raise ValueError(f"{path}: not a {SUFFIX} checkpoint (bad magic)")
    (n_chunks,) = struct.unpack_from("<Q", blob, m)
    off = m + 8
    parts = []
    for _ in range(n_chunks):
        (seg_len,) = struct.unpack_from("<Q", blob, off)
        off += 8
        parts.append(_decompress(blob[off:off + seg_len], nthreads))
        off += seg_len
    raw = parts[0] if len(parts) == 1 else b"".join(parts)
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