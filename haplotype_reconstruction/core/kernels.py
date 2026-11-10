"""Declare Numba overloads without forcing signatures or changing lazy dispatch.

A declaration records exact Numba argument types, including layout, read-only arrays,
literals and optionals. The signature codec stores exact Numba types.
Runtime code explicitly requests warm-up; importing a declaration compiles
nothing and creates no files. The only registry is this small kernel inventory.
"""

from __future__ import annotations

import base64
import os
import pickle
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numba
from numba.core.serialize import dumps

# Numba names a parallel loop's native function after hash(parfor), the loop's IR object (numba's IR objects hash by
# id(), i.e. by address), followed by the mangled argument types: "__numba_parfor_gufunc_0x<address>...". Worker
# processes started alike allocate alike, so two different kernels with the same argument types compiled in two
# workers can receive the same symbol. A process that later loads both from the kernel cache links both callers to
# one of the two loops and silently runs the wrong code. Each process therefore
# hashes parfors with its own random salt and numba's per-process parfor counter: names differ between processes,
# and within a process a reused address no longer repeats a name. Parfor equality and identity semantics are
# unchanged (distinct parfors keep distinct hashes); no computed value depends on the name.
_PARFOR_SALT = [int.from_bytes(os.urandom(8), "little")]


def _new_parfor_salt() -> None:
    _PARFOR_SALT[0] = int.from_bytes(os.urandom(8), "little")


def _salted_parfor_hash(parfor: Any) -> int:
    return hash((_PARFOR_SALT[0], id(parfor), getattr(parfor, "id", -1)))


def _unique_parfor_symbols() -> None:
    """Install the salted parfor hash (above) once; forked children (fork and forkserver workers) draw a new salt."""
    from numba.parfors.parfor import Parfor

    if Parfor.__hash__ is not _salted_parfor_hash:
        Parfor.__hash__ = _salted_parfor_hash
        os.register_at_fork(after_in_child=_new_parfor_salt)


_unique_parfor_symbols()

__all__ = [
    "KernelDeclaration",
    "SAMPLES_PER_THREAD",
    "bounded_team",
    "sample_team",
    "kernel",
    "enable_caches",
    "declarations",
    "encode_signature",
    "decode_signature",
    "manifest",
    "requests_from_manifest",
]


@dataclass(frozen=True)
class KernelDeclaration:
    """One importable dispatcher and its intended argument-type tuples."""

    module: str
    function: str
    overloads: tuple[tuple[Any, ...], ...]
    dispatcher: Any
    cache: bool = False


_DECLARATIONS: dict[tuple[str, str], KernelDeclaration] = {}
_CACHED_DISPATCHERS: set[Any] = set()


def kernel(*, overloads: Iterable[tuple[Any, ...]], **options: Any) -> Callable:
    """Decorate a function with lazy njit and record its intended overloads.

    ``overloads`` contains argument-type tuples, never explicit return signatures.
    Compilation options go unchanged to njit except cache=True, which is activated
    explicitly by enable_caches at runtime. Fastmath is refused because the run's
    numerical policy excludes it. No compilation happens here, and actual dispatch
    stays open to additional signatures so the audit can find missing declarations.
    """
    signatures = tuple(tuple(signature) for signature in overloads)
    cache = bool(options.pop("cache", False))
    if options.get("fastmath", False):
        raise ValueError("kernel numerical policy requires fastmath=False")

    def decorate(function: Callable) -> Any:
        # Numba's cache=True decorator creates its directory immediately, even
        # without a compilation. Defer that filesystem effect to the runtime.
        dispatcher = numba.njit(cache=False, **options)(function)
        key = function.__module__, function.__qualname__
        declaration = KernelDeclaration(*key, signatures, dispatcher, cache)
        _DECLARATIONS[key] = declaration
        dispatcher.kernel_declaration = declaration
        return dispatcher

    return decorate


@contextmanager
def bounded_team(items: int) -> Iterator[None]:
    """Run the enclosed parallel kernels on at most ``items`` threads of the calling thread's current team, then
    restore the team. A parallel region wakes and joins every thread of its team, tens of microseconds of CPU each,
    whether or not the thread has a work item; a kernel with fewer items than threads should wake only its items."""
    previous = numba.get_num_threads()
    bounded = max(1, min(previous, int(items)))
    if bounded == previous:
        yield  # already within the bound (always on one thread): no team change to make or undo
        return
    numba.set_num_threads(bounded)
    try:
        yield
    finally:
        numba.set_num_threads(previous)


# The minimum grain of a kernel parallel over samples, when the work per sample is unknown: the team is sized so that
# each thread would take at least this many samples (the static OpenMP schedule gives each thread one contiguous
# slice). A slice of fewer samples of short work costs less than waking and joining its thread, tens of microseconds
# of CPU per parallel region (over a hundred on nodes held at base clock). Sample order and values never depend on
# the team.
SAMPLES_PER_THREAD = 8

# The grain above stands in for a work floor. When the caller supplies a scheduling estimate of one sample's work
# (site-diplotype cells: sites x K^2 for a component's messages), the grain is the samples that carry
# THREAD_WORK_FLOOR cells, g = min(SAMPLES_PER_THREAD, max(1, ceil(THREAD_WORK_FLOOR / work))), and a kernel's team
# is max(1, min(share, samples // g)); a limited share can still give a thread more samples than g: a 40-sample component of 300,000 sites and
# 6 founders can wake 40 threads instead of 5, and one of 2,000 sites (72,000 cells per sample) 10 instead of 5. On
# 8 CPUs that 2,000-site component runs 1.4x faster at 4 samples per thread than at 8 with the same CPU time, and
# spends 4% more CPU at 1. Threads change scheduling only; sample order and values never depend on them.
THREAD_WORK_FLOOR = 250_000


def samples_per_thread(work_per_sample: int = 0) -> int:
    """Samples one thread needs: SAMPLES_PER_THREAD, or fewer when each carries ``work_per_sample`` cells (> 0)."""
    if work_per_sample <= 0:
        return SAMPLES_PER_THREAD
    return min(SAMPLES_PER_THREAD, max(1, -(-THREAD_WORK_FLOOR // int(work_per_sample))))


def sample_team(samples: int, share: int, work_per_sample: int = 0) -> int:
    """Threads for parallel regions over ``samples`` samples within a share of ``share`` CPUs: at most one thread per
    samples_per_thread(work_per_sample) samples, never fewer than one."""
    return max(1, min(int(share), int(samples) // samples_per_thread(work_per_sample)))


def _writable_directory(directory: str) -> bool:
    """Whether ``directory`` exists or can be created, and can be written (as Numba's own cache locator requires)."""
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return False
    return os.access(directory, os.W_OK | os.X_OK)


def enable_caches() -> tuple[str, ...]:
    """Activate requested disk caches explicitly, after runtime chooses its namespace.

    The engine calls this before warm-up or task execution, after importing kernels.
    A newly imported kernel is picked up on the next call. No compilation is forced.
    Unavailable disk caches leave dispatch lazy and return their names for the
    runtime diagnostic; other kernels can still use their caches. A configured
    cache directory (NUMBA_CACHE_DIR) that cannot be created or written leaves every
    kernel lazy: Numba would otherwise fall back to a cache beside the source, inside
    the installed package.
    """
    directory = getattr(numba.config, "CACHE_DIR", "")
    if directory and not _writable_directory(directory):
        return tuple(
            f"{entry.module}.{entry.function}"
            for entry in declarations()
            if entry.cache and entry.dispatcher not in _CACHED_DISPATCHERS
        )
    failures = []
    for declaration in declarations():
        if declaration.cache and declaration.dispatcher not in _CACHED_DISPATCHERS:
            try:
                declaration.dispatcher.enable_caching()
            except (OSError, RuntimeError):
                failures.append(f"{declaration.module}.{declaration.function}")
            else:
                _CACHED_DISPATCHERS.add(declaration.dispatcher)
    return tuple(failures)


def declarations() -> tuple[KernelDeclaration, ...]:
    """Return the currently imported declarations in module/function order."""
    return tuple(_DECLARATIONS[key] for key in sorted(_DECLARATIONS))


def encode_signature(signature: tuple[Any, ...]) -> dict[str, str]:
    """Encode exact trusted Numba types using the original base85 pickle format."""
    return {"text": str(signature), "pickle": base64.b85encode(dumps(signature)).decode("ascii")}


def decode_signature(encoded: dict[str, str]) -> tuple[Any, ...]:
    """Decode a trusted signature produced by the kernel manifest codec."""
    return pickle.loads(base64.b85decode(encoded["pickle"]))


def manifest() -> dict:
    """Describe imported kernels in the original format-1 release manifest shape."""
    entries = [
        dict(
            module=entry.module,
            function=entry.function,
            signatures=[encode_signature(s) for s in sorted(entry.overloads, key=str)],
            seconds=0.0,
        )
        for entry in declarations()
    ]
    return dict(format=1, numba_version=numba.__version__, overloads=entries)


def requests_from_manifest(data: dict) -> tuple[dict, dict]:
    """Decode a trusted format-1 manifest; incompatible declarations raise instead of disappearing.

    The engine decides how to report an incompatible toolchain; this codec raises
    decoding errors without choosing a reporting policy.
    """
    if data["format"] != 1:
        raise ValueError("unknown release manifest format")
    overloads: dict[tuple[str, str], set[tuple[Any, ...]]] = {}
    seconds: dict[tuple[str, str], float] = {}
    for entry in data["overloads"]:
        key = entry["module"], entry["function"]
        overloads.setdefault(key, set()).update(decode_signature(s) for s in entry["signatures"])
        seconds[key] = entry.get("seconds", 0.0)
    return overloads, seconds
