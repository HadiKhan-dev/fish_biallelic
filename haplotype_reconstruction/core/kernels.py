"""Declare Numba overloads without forcing signatures or changing lazy dispatch.

A declaration records exact Numba argument types, including layout, read-only arrays,
literals and optionals. The signature codec is the 3.5 kernel_manifest codec (P0
ENG-28). Runtime code explicitly requests warm-up; importing a declaration compiles
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

__all__ = [
    "KernelDeclaration",
    "SAMPLES_PER_THREAD",
    "bounded_team",
    "sample_team",
    "kernel",
    "enable_caches",
    "declarations",
    "warmup_requests",
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
    numba.set_num_threads(max(1, min(previous, int(items))))
    try:
        yield
    finally:
        numba.set_num_threads(previous)


# A kernel parallel over samples gives every thread at least this many samples (the static OpenMP schedule gives each
# thread one contiguous slice): a slice of fewer samples costs less than waking and joining its thread, tens of
# microseconds of CPU per parallel region (over a hundred on nodes held at base clock), so a wider team only adds
# wake-ups. Sample order and values never depend on the team.
SAMPLES_PER_THREAD = 8


def sample_team(samples: int, share: int) -> int:
    """Threads for parallel regions over ``samples`` samples within a share of ``share`` CPUs: at most one thread per
    SAMPLES_PER_THREAD samples, never fewer than one."""
    return max(1, min(int(share), int(samples) // SAMPLES_PER_THREAD))


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


def warmup_requests() -> dict[tuple[str, str], set[tuple[Any, ...]]]:
    """Return the 3.5 warm-up manifest's function-to-signatures shape, without compiling."""
    return {(entry.module, entry.function): set(entry.overloads) for entry in declarations()}


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

    The engine decides how to report an incompatible toolchain. 3.5's silent skip
    and diagnostic printing are release-tool behavior, not a numerical kernel rule.
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
