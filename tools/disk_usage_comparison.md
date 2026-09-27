# Disk-usage tool comparison on CSD3

Measured on 27 September 2026. **dua beat dwalk in this four-core setup, but
neither established a consistent speed advantage over the project utility.**
The fastest full runs of dua and our native-backed utility were effectively
tied at 99 seconds. Our utility was faster on the unchanged directory sample.

## Setup and full-project results

All runs used the existing four-core allocation on cpu-q-68, with affinity
`[1,3,32,34]`. Runs were sequential, with no new allocations or cache dropping.
Our Python/C utility and dua used 128 metadata I/O threads each; dwalk used
four MPI processes. This compares useful configurations within that allocation,
not equal numbers of simultaneous metadata requests or multi-node scaling.

| Run order | Tool | Wall time | CPU-seconds | Peak RSS | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | Project utility, native backend | 221.74 s | 289.38 | 60.1 MiB | Completed, 0 lookup errors |
| 2 | dua 2.45.0, 128 threads | 98.83 s | 306.02 | 14.5 MiB | Completed, 0 reported errors |
| 3 | dwalk 0.12, 4 MPI ranks | >600 s | — | — | Stopped at ten-minute limit |
| 4 | dua 2.45.0, 128 threads | 122.72 s | 310.30 | 15.4 MiB | Traversal ended with 3 I/O errors; partial total |
| 5 | Project utility, native backend | 98.70 s | 243.27 | 59.0 MiB | Completed, 0 lookup errors |

Wall times include process startup. Resource measurements are from
`/usr/bin/time -v`; the stopped MPI run did not produce its final resource report.
Its last progress message recorded 366,766 entries at 570.7 seconds, under 8%
of the approximately 4.7-million-entry tree. Do not extrapolate this to dwalk
with a much larger MPI allocation.

The initial native-to-dua comparison suggested a large dua advantage; reversing
the pair removed that conclusion. Cache state, filesystem load, and active
pipeline writes were uncontrolled. These are observed run times, not precise
or universal speedup factors.

The two error-free native scans reported 15,473,342,559,744 and
15,512,099,905,024 allocated bytes. The first dua scan reported
15,475,175,393,280 bytes. The tree was changing between and during scans, so
these are live measurements rather than snapshots. The three errors in dua's
second run were counted but their paths were not reported by the selected
output mode; their causes were not established.

## Correctness and unchanged-directory comparison

Both allocated-space tools matched GNU `du -s -B1` exactly on a local fixture:
**876,544 bytes**. It covered regular/hidden/sparse files, directories,
cross-directory hardlinks, symlinks including a long target and a hardlinked
symlink, a symlink loop, a FIFO, and a non-UTF8 filename.

On the unchanged 44,817-entry directory
`work/runs/seed_8001/checkpoints/feedback_path_initial`, the order was:

| Tool | Wall time |
| --- | ---: |
| dua, 128 threads | 1.418 s |
| Project utility, native backend | 0.585 s |
| Project utility, native backend | 0.539 s |
| dua, 128 threads | 0.813 s |

Every run and GNU `du` returned **1,622,945,792 bytes**. This supports the
project utility's counting and good performance on a large flat directory;
it does not establish an advantage for all directory shapes or cold caches.

Entry counts are not directly interchangeable: our utility counts deduplicated
inodes, while dua's traversal statistics count visited paths. The zero-byte
extra input described below also adds one visited entry to dua's statistics.

## Semantics and invocation details

**dwalk is not a replacement for the allocated-space total.** Stock dwalk
reports apparent regular-file bytes per pathname, counts hardlinks multiple
times, and excludes directory/symlink bytes from that total. Its timing above
is a metadata-traversal comparison. `--lite` was not used, since it skips stat.
Both tools' source confirms that they can distribute metadata lookups within
a wide directory; the result is not evidence of one worker owning all its stats.

For dua, a single directory argument is expanded into its children. To preserve
the root and top-level symlinks for this comparison, we passed the project path
and a separate zero-byte file. That extra file contributes zero allocated bytes.
An ordinary single-directory invocation can therefore differ from `du -s`.

The commands, with paths abbreviated, were:

```bash
/usr/bin/python3 tools/recursive_du.py "$project" --backend native --json
"$dua" --threads 128 --format bytes aggregate --stats "$project" "$zero_byte_file"
env -u I_MPI_PMI_LIBRARY I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm \
  I_MPI_PIN=0 mpirun -n 4 "$dwalk" --progress 30 -v "$project"
```

Before the full runs, two four-thread dua attempts on the flat 44,817-entry
directory exceeded a 90-second limit, with and without root expansion. These
were preliminary configuration checks, excluded from the full-run table.
We did not isolate their slowdown from cache and filesystem conditions.

## Versions and retained evidence

- [dua 2.45.0](https://github.com/Byron/dua-cli/releases/tag/v2.45.0): official
  Linux x86_64 musl binary, checked against the release asset's SHA256.
- [mpiFileUtils 0.12](https://github.com/hpc/mpifileutils/releases/tag/v0.12):
  isolated Release build with Lustre support and existing Intel MPI 2021.6.0.
  All four ranks were verified to inherit the allocation's CPU affinity.
- The project utility used its existing native helper and 64-entry batches.
  Its source was not changed for this comparison.

No Conda, system installation, shell configuration, or scientific code was
changed. Downloaded/built executables remain in the node-local
`/tmp/du-comparison-20260927/` and `/tmp/du-comparison-dwalk-20260927/`
directories; these are not permanent installations or PATH additions.

Reproducible commands, release metadata/checksums, build notes, fixture setup,
comparison scripts, stdout/stderr, resource reports, and timestamped JSON are
retained locally in the ignored `.work/du-comparison-20260927/` directory.
The three drivers are `compare.py`, `reverse.py`, and `stable.py`; they contain
the exact executable paths and invocation options used here.
