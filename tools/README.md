# Storage tools

## Recursive allocated-space measurement

`recursive_du.py` is a standalone utility for Python 3.6+ on POSIX systems.
Its Python backend uses only the standard library; an optional small C helper
accelerates metadata batches without installing Python packages. It measures
allocated disk space in a file or directory tree, including hidden files,
directories, and symlink inodes. It does not
follow symlink targets, including when the argument itself names a symlink.
Hardlinked inodes are counted once across the whole scan; sparse files use
allocated blocks rather than apparent length. Like GNU `du -s -B1`, the total
is the sum of `st_blocks * 512`.

### Current implementation and latest full scan

The implementation is **Python plus C99**, with no Rust component:

- [`recursive_du.py`](recursive_du.py) owns traversal, thread scheduling,
  hardlink deduplication, progress, errors, and the command-line interface.
- [`recursive_du_native.c`](recursive_du_native.c) performs batches of metadata
  lookups and totals ordinary single-link entries. Python loads its compiled
  shared library with standard-library `ctypes`.
- [`test_recursive_du.py`](test_recursive_du.py) checks both backends against
  GNU `du`, including sparse files, hardlinks, symlinks, and partial failures.

The latest full scan finished **27 September 2026 at 12:47:01 UTC** using the
native backend, 128 I/O threads, and 64-entry batches:

| Measurement | Result |
| --- | --- |
| Elapsed time | 128.12 seconds (2 minutes 8 seconds) |
| Entries counted | 4,656,930 |
| Allocated space | 15,446,858,137,600 bytes (15.447 TB / 14.049 TiB) |
| Lookup errors | 0 |
| Average CPU use / allocation | 2.57 / 4 cores |
| Peak memory | 48.2 MiB |

The preceding Python scan took 201.06 seconds: the latest observed run used
36% less elapsed time. This comparison includes uncontrolled cache/filesystem
conditions and changes to the active tree; it is not a guaranteed speedup.
The seven focused tests passed on both Python 3.6.8 and 3.12.0.

The timestamped JSON, resource report, progress log, and exit status are saved
locally under `.work/du-native-20260927/full-scan-124453/` (ignored by Git).
Reuse that result when sufficient; the size is a live measurement, not a
persistent quota or an atomic filesystem snapshot.

### Usage

From the repository root, on an appropriate allocated compute node:

```bash
/usr/bin/python3 tools/recursive_du.py /path/to/directory
/usr/bin/python3 tools/recursive_du.py /path/to/directory --json > /tmp/disk-usage.json
/usr/bin/python3 tools/recursive_du.py /path/to/directory --workers 32 --batch-size 64
```

For the optional native backend, build once on the machine where it will run:

```bash
cc -O3 -std=c99 -Wall -Wextra -fPIC -shared tools/recursive_du_native.c \
  -o tools/_recursive_du_native.so
```

The helper is already built in the current CSD3 checkout. The default
`--backend auto` uses it when available and falls back to Python otherwise.
Use `--backend python` for the portable reference path or `--backend native`
to require the helper. JSON records the actual backend. Compilation is never
performed implicitly; the local shared library is ignored by Git. Rebuild it
when the C source changes or when moving to a different platform. The C source
uses the local system's `struct stat`, with a fixed-width interface to Python.

The default is **128 metadata I/O threads**, with **64 entries per batch**.
These are threads within one process, not 128 allocated CPU cores. They were
benchmarked within an existing four-core allocation. The program does not
submit or manage Slurm jobs. Coordinate scans, lower concurrency when needed,
and reuse timestamped results instead of repeatedly scanning large trees.

Progress goes to stderr every 30 seconds; `--progress-seconds 0` disables it.
Counts advance when batches finish, so reported partial bytes can pause and
then jump. The size of the unvisited tree is unknown: there is no percentage
or reliable ETA. Normal stdout is `allocated_bytes<TAB>absolute_path`.
`--json` instead emits bytes, decimal TB, binary TiB, entry count, UTC start/end
times, elapsed time, settings, and error information. Save stdout outside the
scanned tree when practical.

Exit status is **0** when all lookups succeed, **1** for a partial result
(including vanished temporary files or unreadable paths), **2** for invalid
arguments, and **130** after Ctrl-C. JSON `complete` means no reported lookup
errors; it does not mean that an actively changing directory was frozen.
Error count is exact; at most 20 example errors are retained. On interruption,
queued work is cancelled and in-flight filesystem calls must return before
shutdown can finish. Native calls finish their current batch (64 entries by
default) before observing cancellation.

## Why this implementation

One whole directory per worker leaves a wide directory or the last large
directory using only one worker. This utility shares batches from that same
directory across the pool. Directory-relative `os.stat` calls avoid resolving
the long path prefix again for every file. Do not replace them with
`DirEntry.stat()` without checking interpreter behavior: Python 3.6 held the
GIL during that lookup, defeating I/O concurrency
([CPython implementation](https://github.com/python/cpython/blob/v3.6.8/Modules/posixmodule.c#L10535)).

The native backend releases Python's GIL once per batch instead of once per
file. C performs the same `fstatat(..., AT_SYMLINK_NOFOLLOW)` lookup and sums
ordinary single-link entries. Directories, multiply linked entries, and errors
return to Python for traversal, global inode deduplication, and reporting.
Both backends use a completion queue so the coordinator does not rebuild a
waiter across every outstanding future after each completion.

There are at most twice as many outstanding batches as workers. Directory FDs
stay open until their final batch finishes, then close. The discovered-directory
queue and inode-deduplication set can still grow with the tree. This is a
read-only metadata traversal; it does not read checkpoint contents or change
scientific calculations.

## Validation and measured performance

Run the small local-scratch tests (GNU `du` required):

```bash
/usr/bin/python3 tools/test_recursive_du.py -v
```

They check sparse and hidden files, nested paths, cross-directory hardlinks,
hardlinked symlinks, symlink loops and external targets, root files/symlinks,
concurrency within one wide directory, disappearing temporary files, FD
cleanup, JSON, and missing-path exit status. Native/Python equivalence also
covers non-UTF8 filenames, FIFOs, native lookup failures, and automatic fallback.
All seven tests passed on Python 3.6.8 and 3.12.0 on CSD3 with the helper built;
the native-specific test is skipped when it is unavailable. Fixtures use local
`/tmp`: newly written files in the runner's Lustre `TMPDIR` showed changing
allocated-block counts between scans.
That is also a reason to treat measurements of active output trees as estimates.

Measurements on 27 September 2026:

- The original corrected helper measured 4,612,472 entries in 2,521 seconds
  (42 minutes), reporting 15,427,753,622,528 bytes and two vanished temporary
  files. Earlier unsuccessful approaches added delay; two hours is not a
  measured requirement for this utility.
- On a real 14,223-entry directory, the prototype, original helper, and GNU
  `du` all returned **8,376,967,168 bytes**. The prototype's first pass took
  4.38 seconds. Repeated cached runs at 16 workers/256-entry batches took
  0.153 seconds versus 0.204–0.206 seconds for the original helper.
- A controlled local fixture with 512 files and an injected 2 ms delay per
  lookup took **1.066 seconds** with one directory per worker, **0.165 seconds**
  with 64-entry batches, and **0.116 seconds** with 32-entry batches, all at
  16 workers and with identical totals. This isolates the batching benefit;
  it is not an end-to-end Lustre speedup claim.
- Disjoint randomized 8,192-file Lustre samples took 3.85, 1.50, 1.47, and
  0.66 seconds at 16, 32, 64, and 128 workers, respectively. A separate
  4,016-file comparison took 0.33 seconds at 128 and 0.26 seconds at 256.
  Every sample was checked against independent allocated-block sums.
  These are different samples with uncontrolled cache/server conditions;
  they support the configurable default, not an exact scaling guarantee.

- A subsequent full scan with the batched Python utility took **201.06 seconds**
  (3 minutes 21 seconds): 4,650,789 entries, 15,445,563,604,992 bytes, no lookup
  errors. It used 349.69 CPU-seconds, 46.6 MiB peak memory, and 14,001,436
  voluntary context switches within the existing four-core allocation.
- On the same unchanged 44,817-entry real directory, the native backend took
  **0.396 seconds** after its initial pass, versus **0.680–0.683 seconds** for
  the previous Python implementation. The completion-queue Python path took
  0.683–0.685 seconds. All versions and GNU `du` returned **1,622,945,792 bytes**.
  Native context switches fell to about 3,000 versus 77,000–81,000 for the
  previous implementation. Tests at 32 workers and 256-entry batches did not
  improve elapsed time on this sample; the 128-worker/64-entry defaults remain.

- The native full-project scan took **128.12 seconds** (2 minutes 8 seconds):
  4,656,930 entries, 15,446,858,137,600 bytes, no lookup errors. That is 36% less
  elapsed time than the preceding 201-second Python scan. It used 328.95
  CPU-seconds (about 6% less), 48.2 MiB peak memory, and 9,505,633 voluntary
  context switches (about 32% fewer). Average CPU use was 2.57 of the four
  allocated cores. Metadata I/O remained the principal constraint; no extra
  allocation or increase in the 128 I/O threads was needed. The active tree
  grew between measurements, and filesystem/cache conditions were uncontrolled.

A 4,096-file first pass took 43 seconds while later cached passes took around
0.04 seconds. Cache effects can overwhelm implementation effects. Full scans
of this active project at different times are observed runtimes, not controlled
speedup measurements. Benchmark scripts and detailed results are retained
locally under the ignored `.work/du-optimization-20260927/` and
`.work/du-native-20260927/` directories.
