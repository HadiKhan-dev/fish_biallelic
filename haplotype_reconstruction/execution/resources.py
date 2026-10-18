"""CPUs and memory available to this process (the only probes of either).

Both respect the scheduler: CPU affinity (a Slurm job's cpuset) rather than the node size, and memory as the
tightest of the node's available memory, every enclosing cgroup limit (v1 and v2, the job's and its steps') and
SLURM_MEM_PER_NODE. Stages size worker counts from these; they never read /proc or the cgroup themselves.
"""
from __future__ import annotations

import os
from pathlib import Path


def affinity_cpus() -> int:
    """CPUs this process may run on (its affinity mask; the node's count only where no mask exists)."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


def _inactive_file_bytes(directory: Path, version1: bool) -> int:
    """Reclaimable inactive file cache charged to a cgroup (hierarchical total in v1)."""
    key = "total_inactive_file" if version1 else "inactive_file"
    try:
        for line in (directory / "memory.stat").read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except (OSError, ValueError):
        pass
    return 0


def available_memory_bytes() -> int | None:
    """Bytes this process may still allocate: the minimum over all known limits, or None if none is known.

    Candidates: MemAvailable of the node; for every cgroup level from this process's own up to the root, its
    limit minus its working set (usage minus reclaimable inactive file cache, so a plan does not depend on how
    much recently read checkpoint data the page cache still holds); SLURM_MEM_PER_NODE (MiB).
    """
    candidates = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        entries = []
    for entry in entries:
        _, controllers, relative = entry.split(":", 2)
        if "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory")
            maximum_name, current_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        elif not controllers:
            root = Path("/sys/fs/cgroup")
            maximum_name, current_name = "memory.max", "memory.current"
        else:
            continue
        directory = root / relative.lstrip("/")
        while True:
            try:
                maximum = int((directory / maximum_name).read_text().strip())
                current = int((directory / current_name).read_text().strip())
                if 0 < maximum < (1 << 60):
                    current -= _inactive_file_bytes(directory, bool(controllers))
                    candidates.append(max(0, maximum - current))
            except (OSError, ValueError):
                pass
            if directory == root:
                break
            directory = directory.parent
    slurm_mib = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mib and slurm_mib.isdigit() and int(slurm_mib) > 0:
        candidates.append(int(slurm_mib) << 20)
    return min(candidates) if candidates else None


def resident_bytes() -> int:
    """This process's current resident set size in bytes (0 where /proc is unavailable)."""
    try:
        with open("/proc/self/statm", "rt", encoding="ascii") as handle:
            return int(handle.readline().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0
