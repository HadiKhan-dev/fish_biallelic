#!/usr/bin/env python3
"""Read-only, batched allocated-space measurement (Python 3.6+, POSIX)."""

import argparse
import ctypes
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
import json
import os
from queue import Empty, Queue
import stat
import sys
import threading
import time


class _NativeEntry(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint64), ('blocks', ctypes.c_uint64),
                ('device', ctypes.c_uint64), ('inode', ctypes.c_uint64),
                ('mode', ctypes.c_uint32), ('error', ctypes.c_int32)]


class _NativeBatch:
    """Optional local C helper; CDLL releases the GIL once for the whole batch."""

    def __init__(self):
        library = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_recursive_du_native.so')
        self.library = ctypes.CDLL(library)
        self.scan = self.library.du_batch
        self.scan.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t,
                              ctypes.POINTER(_NativeEntry), ctypes.POINTER(ctypes.c_uint64)]
        self.scan.restype = ctypes.c_size_t

    def __call__(self, fd, names):
        encoded = (ctypes.c_char_p * len(names))(*(os.fsencode(name) for name in names))
        special = (_NativeEntry * len(names))()
        totals = (ctypes.c_uint64 * 2)()
        used = self.scan(fd, encoded, len(names), special, totals)
        return totals[0], totals[1], special, used


def _load_native(backend):
    if backend not in ('auto', 'python', 'native'):
        raise ValueError('backend must be auto, python, or native')
    if backend == 'python':
        return None
    try:
        return _NativeBatch()
    except OSError:
        if backend == 'native':
            raise
        return None


class _Directory:
    """Keep a directory FD alive until enumeration and all its batches finish."""

    def __init__(self, path):
        self.path = path
        self.pending = 0
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            self.entries = os.scandir(path)
        except BaseException:
            os.close(self.fd)
            raise

    def finish_listing(self):
        if self.entries is not None:
            self.entries.close()
            self.entries = None
        self.release()

    def release(self):
        if self.entries is None and self.pending == 0 and self.fd is not None:
            os.close(self.fd)
            self.fd = None


def measure(path, workers=128, batch_size=64, progress_seconds=30, progress=None, backend="auto"):
    """Count allocated bytes once per inode; never follow entry symlinks.

    Progress and results include only completed batches. Files changing during
    traversal make this a live estimate, even when no lookup reports an error.
    """
    if workers < 1 or batch_size < 1 or progress_seconds < 0:
        raise ValueError('workers/batch_size must be positive; progress_seconds nonnegative')
    native = _load_native(backend)
    path = os.path.abspath(path)
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    seen = set()
    lock = threading.Lock()
    stop = threading.Event()
    error_count = 0
    errors = []
    total = count = 0
    directories = deque()

    def record_errors(failures):
        nonlocal error_count
        error_count += len(failures)
        errors.extend(failures[:max(0, 20 - len(errors))])

    def claim(st):
        if stat.S_ISDIR(st.st_mode) or st.st_nlink > 1:
            key = (st.st_dev, st.st_ino)
            with lock:
                if key in seen:
                    return False
                seen.add(key)
        return True

    def scan_native_batch(directory, names):
        if stop.is_set():
            return 0, 0, [], []
        subtotal, entries, special, used = native(directory.fd, names)
        children = []
        failures = []
        # Ordinary single-link files were summed in C. Only directories,
        # multiply linked entries, and errors cross back into Python here.
        with lock:
            for i in range(used):
                item = special[i]
                name = names[item.index]
                if item.error:
                    path = os.path.join(directory.path, name)
                    failures.append('{}: {}'.format(path, OSError(item.error, os.strerror(item.error))))
                    continue
                key = (item.device, item.inode)
                if key in seen:
                    continue
                seen.add(key)
                subtotal += item.blocks * 512
                entries += 1
                if stat.S_ISDIR(item.mode):
                    children.append(os.path.join(directory.path, name))
        return subtotal, entries, children, failures

    def scan_python_batch(directory, names):
        subtotal = entries = 0
        children = []
        failures = []
        for name in names:
            if stop.is_set():
                break
            try:
                # Unlike DirEntry.stat() on Python 3.6, os.stat releases the
                # GIL during the lookup. dir_fd avoids repeated prefix walks.
                st = os.stat(name, dir_fd=directory.fd, follow_symlinks=False)
                if not claim(st):
                    continue
                subtotal += st.st_blocks * 512
                entries += 1
                if stat.S_ISDIR(st.st_mode):
                    children.append(os.path.join(directory.path, name))
            except OSError as exc:
                failures.append('{}: {}'.format(os.path.join(directory.path, name), exc))
        return subtotal, entries, children, failures

    try:
        root = os.lstat(path)
        claim(root)
        total = root.st_blocks * 512
        count = 1
        if stat.S_ISDIR(root.st_mode):
            directories.append(path)
    except OSError as exc:
        record_errors([str(exc)])

    scan_batch = scan_native_batch if native is not None else scan_python_batch
    completed = Queue()
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {}
    current = None
    last_progress = start
    try:
        while directories or current is not None or pending:
            # One enumerator and at most 2*workers queued/running batches.
            # A wide directory can use the entire pool, unlike one task/dir.
            while len(pending) < workers * 2 and (current is not None or directories):
                if current is None:
                    try:
                        current = _Directory(directories.popleft())
                    except OSError as exc:
                        record_errors([str(exc)])
                        continue
                try:
                    names = [entry.name for entry in itertools.islice(current.entries, batch_size)]
                except OSError as exc:
                    record_errors(['{}: {}'.format(current.path, exc)])
                    names = []
                if not names:
                    current.finish_listing()
                    current = None
                    continue
                future = pool.submit(scan_batch, current, names)
                current.pending += 1
                pending[future] = current
                future.add_done_callback(completed.put)
            if pending:
                # A completion queue avoids rebuilding a waiter across every
                # pending future whenever even one batch finishes.
                try:
                    done = [completed.get(timeout=min(5, progress_seconds or 5))]
                except Empty:
                    done = []
                while True:
                    try:
                        done.append(completed.get_nowait())
                    except Empty:
                        break
                for future in done:
                    directory = pending.pop(future)
                    try:
                        size, entries, children, failures = future.result()
                    finally:
                        directory.pending -= 1
                        directory.release()
                    total += size
                    count += entries
                    directories.extend(children)
                    record_errors(failures)
            now = time.monotonic()
            if progress is not None and progress_seconds and now - last_progress >= progress_seconds:
                print('PROGRESS entries={} counted_TB={:.6f} queued_dirs={} '
                      'pending_batches={} elapsed_s={:.0f}'.format(
                          count, total / 1e12, len(directories), len(pending), now - start),
                      file=progress, flush=True)
                last_progress = now
    finally:
        # On interruption, stop issuing metadata requests and wait only for
        # requests already in flight before closing the FDs they reference.
        stop.set()
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True)
        if current is not None:
            current.finish_listing()
        for directory in set(pending.values()):
            directory.pending = 0
            directory.finish_listing()
    return dict(path=path, allocated_bytes=total, TB=total / 1e12, TiB=total / 2**40,
                entries=count, elapsed_seconds=time.monotonic() - start,
                started_at_utc=started_at, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                workers=workers, batch_size=batch_size, backend="native" if native is not None else "python",
                complete=error_count == 0,
                error_count=error_count, errors=errors)


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', help='file or directory to measure')
    parser.add_argument('--workers', type=_positive, default=128, help='metadata I/O threads (default: 128)')
    parser.add_argument('--batch-size', type=_positive, default=64, help='entries per task (default: 64)')
    parser.add_argument('--progress-seconds', type=_nonnegative, default=30, help='stderr interval; 0 disables')
    parser.add_argument('--backend', choices=('auto', 'python', 'native'), default='auto',
                        help='auto uses the optional compiled helper when available')
    parser.add_argument('--json', action='store_true', help='timestamped final result as JSON on stdout')
    args = parser.parse_args(argv)
    try:
        result = measure(args.path, args.workers, args.batch_size, args.progress_seconds, sys.stderr, args.backend)
    except OSError as exc:
        parser.error('cannot load native helper: {}; see tools/README.md to build it'.format(exc))
    except KeyboardInterrupt:
        print('Interrupted; no completed total.', file=sys.stderr)
        return 130
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print('{}\t{}'.format(result['allocated_bytes'], result['path']))
    print('{}: {:.3f} TB / {:.3f} TiB; {} entries; {:.1f}s; {} lookup errors'.format(
        'COMPLETE' if result['complete'] else 'PARTIAL', result['TB'], result['TiB'],
        result['entries'], result['elapsed_seconds'], result['error_count']), file=sys.stderr)
    for error in result['errors']:
        print(error, file=sys.stderr)
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
