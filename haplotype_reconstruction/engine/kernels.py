"""Keep compiled kernels disposable, keyed by their full code closure and target.

The caller supplies roots and applies numerical settings before importing Numba.
This module never discovers scientific stages or imports them. The caller preloads
its roots before warm_up; every executing process activates declared caches only
at that explicit boundary. Persistent cache failures cost compilation time, never
scientific results. ENG-19/27 describe the learned manifest and signature guard.
"""

from __future__ import annotations

import atexit
import fcntl
import hashlib
import io
import logging
import os
import pickle
import re
import shutil
import socket
import sys
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any, cast

from ..core.errors import InvariantError
from ..store.identity import PACKAGE_ROOT, CodeIdentity, code_dependency_identity

# ENG-19/27: operational kernel-cache storage limits, in bytes.
KEEP_KEYS = 6
KERNEL_SCRATCH = Path("/dev/shm")
KERNEL_SCRATCH_FREE_BYTES = 4 << 30
_SNAPSHOT = "cache.tar"
_MANIFEST = "manifest.pkl"
_PACKAGE = "haplotype_reconstruction"
_INDEX_SUFFIX = f".py{sys.version_info[0]}{sys.version_info[1]}{getattr(sys, 'abiflags', '')}.nbi"
_LOG = logging.getLogger(__name__)


def _host_cpu_model() -> str:
    """CPU vendor, model and instruction-set flags (Linux), else the platform name."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            first = handle.read(1 << 16).split("\n\n", 1)[0]
    except OSError:
        import platform

        return f"{platform.machine()} {platform.processor()}"
    fields = {
        name.strip(): value.strip()
        for name, value in (line.split(":", 1) for line in first.splitlines() if ":" in line)
    }
    return " | ".join(fields.get(name, "") for name in ("vendor_id", "model name", "flags"))


def _remove_orphaned_caches(parent: Path, host: str) -> None:
    """Remove private run caches in ``parent`` whose creating process on this host has ended (a run killed before
    its exit handler ran), so memory-backed caches cannot accumulate."""
    if os.name != "posix":  # process-existence checks below are POSIX semantics
        return
    pattern = re.compile(r"-" + re.escape(re.sub(r"[^A-Za-z0-9_.-]", "_", host)) + r"-pid(\d+)$")
    try:
        entries = list(parent.iterdir())
    except OSError:
        return
    for entry in entries:
        match = pattern.search(entry.name)
        if match is None:
            continue
        pid = int(match.group(1))
        try:
            os.kill(pid, 0)  # the creator still runs (or exists under another user)
            continue
        except ProcessLookupError:
            pass
        except OSError:
            continue
        shutil.rmtree(entry, ignore_errors=True)


def _kernel_scratch(parent: str, host: str) -> tuple[Path, bool]:
    """(directory for the automatic run-local kernel cache, whether it is private node-local memory); ``parent``
    names this user's directory of private run caches."""
    job = os.environ.get("SLURM_TMPDIR") if os.environ.get("SLURM_JOB_ID") else None
    if job:
        return Path(job), False
    try:
        if os.access(KERNEL_SCRATCH, os.W_OK):
            # Ended runs' caches are removed before the free-space test: they may be what fills the space.
            _remove_orphaned_caches(KERNEL_SCRATCH / parent, host)
            if shutil.disk_usage(KERNEL_SCRATCH).free >= KERNEL_SCRATCH_FREE_BYTES:
                return KERNEL_SCRATCH, True
    except OSError:
        pass
    return Path(tempfile.gettempdir()), False


def _batch_cache_scope() -> str | None:
    """A scheduler job (and array task) identifier for the automatic cache directory, if any."""
    schedulers: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("slurm", "SLURM_JOB_ID", ("SLURM_ARRAY_TASK_ID",)),
        ("pbs", "PBS_JOBID", ("PBS_ARRAY_INDEX", "PBS_ARRAYID")),
        ("lsf", "LSB_JOBID", ("LSB_JOBINDEX",)),
    )
    if os.environ.get("SGE_ROOT"):
        schedulers += (("sge", "JOB_ID", ("SGE_TASK_ID",)),)
    for name, variable, task_variables in schedulers:
        job = os.environ.get(variable)
        if job:
            task = next((os.environ[key] for key in task_variables if os.environ.get(key)), None)
            return f"{name}-{job}" + (f"-task-{task}" if task else "")
    return None


def cache_key(roots: Iterable[str], *, package_root: Path = PACKAGE_ROOT) -> str:
    """Hash source closure, its location and every code-generation input; no writes.

    Call once in the parent after applying numerical policy. Workers inherit its
    key and directory, rather than choosing scratch or recalculating the key.
    Missing roots raise ValueError. Thread counts/layer/cache path are operational
    and do not affect this key; every other NUMBA_* setting does.
    """
    code = code_dependency_identity(roots, package_root=package_root)
    digest = hashlib.sha256()

    def add(*values):
        for value in values:
            digest.update(str(value).encode("utf-8", "surrogateescape") + b"\0")

    add(str(Path(package_root).resolve()), sys.version, sys.implementation.cache_tag, getattr(sys, "abiflags", ""))
    for name in ("numpy", "numba", "llvmlite"):
        add(name, metadata.version(name))
    for name in sorted(os.environ):
        if name.startswith("NUMBA_") and name not in {"NUMBA_CACHE_DIR", "NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER"}:
            add(name, os.environ[name])
    add(_host_cpu_model(), code.digest)
    return digest.hexdigest()[:24]


def cache_directory(key: str, base: str | Path | None = None) -> Path:
    """Create a private mode-0700 namespace at the explicit runtime boundary.

    Explicit base, Slurm scratch, memory scratch and temporary storage are tried
    in that order. Other scheduler variables only identify the scope. Workers
    reuse the returned path supplied by their parent. Filesystem errors propagate
    so the boundary can fall back to lazy, uncached compilation.
    """
    host = socket.gethostname()
    private = False
    run = f"run-{uuid.uuid4().hex}"
    if base is None:
        parent = f"haplotype-reconstruction-numba-{os.getuid()}"
        scratch, private = _kernel_scratch(parent, host)
        scope = run if private else (_batch_cache_scope() or run)
        scope = re.sub(r"[^A-Za-z0-9_.-]", "_", scope + f"-{host}" + (f"-pid{os.getpid()}" if private else ""))
        root = scratch / parent / scope
    else:
        root = Path(base) / re.sub(r"[^A-Za-z0-9_.-]", "_", _batch_cache_scope() or run)
    directory = root / key
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    if private:
        atexit.register(shutil.rmtree, str(root), True)
    return directory


def check_closure(roots: Iterable[str], key_closure: CodeIdentity | Iterable[str]) -> None:
    """Refuse cacheable declarations that the selected key does not cover.

    No science module is imported here. The caller already imported its roots.
    Non-cacheable declarations are exempt, including test audit controls.
    """
    from ..core.kernels import declarations

    modules = set(key_closure.modules if isinstance(key_closure, CodeIdentity) else key_closure)
    if not set(code_dependency_identity(roots).modules) <= modules:
        raise InvariantError("kernel roots exceed the cache key's dependency closure")
    missing = sorted({entry.module for entry in declarations() if entry.cache and entry.module not in modules})
    if missing:
        raise InvariantError("cacheable kernels outside the keyed closure: " + ", ".join(missing))


def store_directory() -> Path | None:
    """The persistent store, or None without a home or XDG cache directory."""
    base: str | Path | None = os.environ.get("XDG_CACHE_HOME")
    if not base:
        try:
            base = Path.home() / ".cache"
        except RuntimeError:
            return None
    return Path(base) / "haplotype-reconstruction" / "numba"


def _listing(directory) -> dict:
    """{relative path: (bytes, mtime ns)} of the Numba cache files in ``directory``."""
    files = {}
    for parent, _, names in os.walk(directory):
        for name in names:
            if name.endswith((".nbi", ".nbc")):
                path = os.path.join(parent, name)
                try:
                    status = os.stat(path)
                except FileNotFoundError:
                    continue
                files[os.path.relpath(path, directory)] = (status.st_size, status.st_mtime_ns)
    return files


def _write_atomically(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed(cache_directory: Path, snapshot: Path) -> int:
    """Install the snapshot's functions that the run-local cache lacks; return the files written.

    A function's index (``.nbi``) and data files (``.nbc``, named after the index) are installed together and
    only when no local index exists, data files first, each by an atomic rename: a reader never meets a partial
    entry, and a local index is never mixed with another one's data files.
    """
    payload = snapshot.read_bytes()
    try:
        os.utime(snapshot)  # most recently used, for pruning
    except OSError:
        pass
    functions: dict[tuple[str, str], list[tarfile.TarInfo]] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) != 2 or ".." in parts:
                continue
            name = parts[1]
            if name.endswith(".nbi"):
                base = name[:-4]
            elif name.endswith(".nbc"):
                base = name[:-4].rpartition(".")[0]
            else:
                continue
            functions.setdefault((parts[0], base), []).append(member)
        written = 0
        for (directory, base), members in functions.items():
            target = cache_directory / directory
            if (target / f"{base}.nbi").exists():
                continue
            target.mkdir(parents=True, exist_ok=True)
            for member in sorted(members, key=lambda item: item.name.endswith(".nbi")):
                handle = archive.extractfile(member)
                assert handle is not None
                _write_atomically(target / Path(member.name).name, handle.read())
                written += 1
    return written


def _module_packages() -> dict:
    """Numba cache subdirectory -> dotted package, for this package's directories.

    Numba files a kernel under ``<directory name>_<sha1 of its absolute directory>``
    (numba.core.caching._CacheLocator.get_suitable_cache_subpath).
    """
    root = str(PACKAGE_ROOT)
    packages = {}
    for directory, subdirectories, _ in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if name != "__pycache__"]
        relative = os.path.relpath(directory, root)
        dotted = _PACKAGE if relative == "." else f"{_PACKAGE}.{relative.replace(os.sep, '.')}"
        name = os.path.basename(directory) + "_" + hashlib.sha1(directory.encode()).hexdigest()
        packages[name] = dotted
    return packages


def _cached_overloads(cache_directory: Path, target=None) -> dict:
    """{(module, function): signatures} of this package's kernels in a cache directory.

    Index files are named ``<module>.<qualified name>-<line>.pyXY.nbi`` and hold the Numba version, then
    (source stamp, {(signature, target, code hashes): data file}). ``target`` keeps only overloads compiled for
    that code-generation target (Numba's magic tuple). Functions without an importable name are skipped.
    """
    import numba

    packages = _module_packages()
    overloads: dict[tuple[str, str], set] = {}
    for subdirectory in os.listdir(cache_directory):
        package = packages.get(subdirectory)
        if package is None:
            continue
        for name in os.listdir(cache_directory / subdirectory):
            if not name.endswith(_INDEX_SUFFIX):
                continue
            stem, _, line = name[: -len(_INDEX_SUFFIX)].rpartition("-")
            module, _, function = stem.partition(".")
            if not function or not line.isdigit() or "locals" in function.split("."):
                continue
            module = package if module == "__init__" else f"{package}.{module}"
            try:
                with open(cache_directory / subdirectory / name, "rb") as handle:
                    if pickle.load(handle) != numba.__version__:
                        continue
                    _, index = pickle.loads(handle.read())
            except Exception:
                continue  # Numba would not use an index it cannot read either
            signatures = {key[0] for key in index if target is None or key[1] == target}
            if signatures:
                overloads.setdefault((module, function), set()).update(signatures)
    return overloads


def _read_manifest(store: Path):
    try:
        with open(store / _MANIFEST, "rb") as handle:
            manifest = pickle.load(handle)
        return dict(manifest["overloads"]), dict(manifest["seconds"])
    except FileNotFoundError:
        return {}, {}
    except Exception as error:
        _LOG.warning("kernel learned manifest unreadable (%s); ignored", type(error).__name__)
        return {}, {}


def install_signature_guard() -> None:
    """Use a cached Numba kernel only when it was compiled for the requested argument types.

    Numba trusts its cache index. Processes that save different overloads of one function at the same time (the
    kernel warm-up's parallel tasks also compile their callees; pool workers compile on first use) can leave an
    index entry naming another overload's data file, and loading it returns that other kernel, so the calling
    kernel fails to type. A cold-cache K30 simulation failed this way on 21 Oct 2026. Such an entry is now a cache
    miss: the kernel is compiled again.
    """
    from numba.core import caching, sigutils

    load = caching.Cache._load_overload
    if getattr(load, "checks_signature", False):
        return

    def load_checked(self, sig, target_context):
        data = load(self, sig, target_context)
        signature = getattr(data, "signature", None)
        if signature is not None and tuple(signature.args) != tuple(sigutils.normalize_signature(sig)[0]):
            return None
        return data

    cast(Any, load_checked).checks_signature = True
    cast(Any, caching.Cache)._load_overload = load_checked


def install_cache_path_check_once() -> None:
    """Check each kernel cache directory once per process.

    Numba checks a function's cache directory (it creates it and writes a test file there) whenever it creates that
    function's cache, and enable_caches creates one per declared kernel: about 600 round trips to the filesystem, twice
    before a run's first task (in the coordinator and in the forkserver), 12 s each where the cache is on a network
    filesystem. A directory that passed stays writable for the process, so later calls (every save) only check that it
    still exists and run the full check again, recreating it, if it was removed; one that failed is checked again, so
    Numba still falls back to its next cache location."""
    from numba.core import caching

    ensure = caching._CacheLocator.ensure_cache_path
    if getattr(ensure, "once", False):
        return
    checked: set[str] = set()

    def ensure_once(self):
        path = self.get_cache_path()
        if path not in checked or not os.path.isdir(path):
            ensure(self)
            checked.add(path)

    cast(Any, ensure_once).once = True
    cast(Any, caching._CacheLocator).ensure_cache_path = ensure_once


# Compile locks this process holds: lock path -> [descriptor, depth]. Touched only under Numba's global compiler lock,
# which serializes this process's compilations (the index save lock is taken inside it too).
_COMPILE_LOCKS: dict[str, list[int]] = {}
# Functions (cache index paths) with an overload this process is compiling under its overload lock -> depth.
_COMPILING: dict[str, int] = {}


def install_compile_lock() -> None:
    """Compile each cached kernel overload once per cache, in whichever process needs it first; save atomically.

    Numba checks a function's disk cache, then compiles and saves on a miss. Processes that missed one overload at
    the same time each compiled it, and processes saving overloads of one function at the same time rewrote its cache
    index from stale copies and could give two overloads one data file (install_signature_guard's case). Here:

    * a process holds an overload's lock file (beside its function's cache index) from its cache check to its save:
      another process needing that overload meanwhile waits, then loads what the first saved. Overloads of one
      function compile in parallel;
    * every save of a function's index and data files holds the function's index lock: one writer at a time, each
      re-reading the index under the lock, so no entry is lost and no data file is shared.

    The file locks are taken inside Numba's global compiler lock (the order compile takes them), so this process's
    threads never wait on each other's file locks; a compilation that compiles a callee takes the callee's overload
    lock in turn, and kernels never call one another in a cycle. A recursive kernel compiling another overload of
    itself proceeds without that overload's lock (two overload locks of one function never nest). An overload
    already in this process takes no lock; where a lock cannot be taken, compilation proceeds unlocked.
    """
    from numba.core import caching, sigutils
    from numba.core.compiler_lock import global_compiler_lock
    from numba.core.dispatcher import Dispatcher

    compile = Dispatcher.compile
    if getattr(compile, "locks_cache", False):
        return

    def compile_once(self, sig):
        index = getattr(getattr(getattr(self, "_cache", None), "_cache_file", None), "_index_path", None)
        if index is None:  # no disk cache: nothing is shared with other processes
            return compile(self, sig)
        with global_compiler_lock:
            args, _ = sigutils.normalize_signature(sig)
            if tuple(args) in self.overloads or index in _COMPILING:
                return compile(self, sig)
            digest = hashlib.sha256(str(tuple(args)).encode("utf-8", "surrogateescape")).hexdigest()[:16]
            with _compile_lock(f"{index}.{digest}.lock"):
                _COMPILING[index] = _COMPILING.get(index, 0) + 1
                try:
                    return compile(self, sig)
                finally:
                    _COMPILING[index] -= 1
                    if not _COMPILING[index]:
                        del _COMPILING[index]

    save = caching.IndexDataCacheFile.save

    def save_locked(self, key, data):
        with global_compiler_lock, _compile_lock(self._index_path + ".lock"):
            return save(self, key, data)

    cast(Any, compile_once).locks_cache = True
    cast(Any, Dispatcher).compile = compile_once
    cast(Any, caching.IndexDataCacheFile).save = save_locked


@contextmanager
def _compile_lock(path: str):
    """Hold ``path``'s exclusive file lock, re-entrantly within this process (a recursive kernel compiles itself)."""
    held = _COMPILE_LOCKS.get(path)
    if held is not None:
        held[1] += 1
        try:
            yield
        finally:
            held[1] -= 1
        return
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        descriptor = None
    if descriptor is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError:  # a filesystem without locks
            os.close(descriptor)
            descriptor = None
    if descriptor is None:
        yield
        return
    _COMPILE_LOCKS[path] = [descriptor, 1]
    try:
        yield
    finally:
        del _COMPILE_LOCKS[path]
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass
class CacheState:
    """One run's disposable cache publication and warm-up accounting."""

    key: str
    cache_directory: Path
    store: Path | None
    baseline: dict = field(default_factory=dict)
    published: bool = False
    stale: bool = False
    seconds: dict = field(default_factory=dict)
    requested: int = 0
    compiled: int = 0
    skipped: int = 0
    verified: int = 0
    seeded: int = 0
    # The plan (plan_warm_up): per function, its union of declared and learned overloads, its declared ones, and the
    # overloads the cache does not hold yet.
    union: dict = field(default_factory=dict)
    required: dict = field(default_factory=dict)
    missing: dict = field(default_factory=dict)


def _compile_function(name: tuple[str, str], encoded: bytes, encoded_declared: bytes):
    """Compile one imported dispatcher's union; declared failures are invariants."""
    from ..core.kernels import declarations, enable_caches

    dispatcher = next(
        (entry.dispatcher for entry in declarations() if (entry.module, entry.function) == name), None
    )
    signatures, declared = pickle.loads(encoded), pickle.loads(encoded_declared)
    install_signature_guard()
    install_compile_lock()
    install_cache_path_check_once()
    enable_caches()
    started = time.perf_counter()
    compiled = 0
    if dispatcher is None:
        if declared:
            raise InvariantError(f"declared kernel is missing: {name}")
        return name, 0, time.perf_counter() - started
    for signature in signatures:
        try:
            dispatcher.compile(signature)
            compiled += 1
        except Exception as error:
            if signature in declared:
                raise InvariantError(f"declared kernel {name} overload {signature} failed: {error}") from error
    return name, compiled, time.perf_counter() - started


def _compile_requests(state: CacheState, requests: Mapping, cores: int) -> tuple[int, dict]:
    """Run the same overload tasks as deferred warm-up in a fresh worker group."""
    from .graph import run_graph

    tasks = sorted(
        compile_tasks(replace(state, missing=requests)),
        key=lambda task: (-state.seconds.get(task.args[0], 0.0), -len(requests[task.args[0]]), task.args[0]),
    )
    count, timings = 0, {}
    if tasks:
        values = run_graph(
            tasks, cores=cores, max_workers=max(1, min(cores, len(tasks))), in_process=False,
            priorities={task.key: float(len(tasks) - index) for index, task in enumerate(tasks)},
        )
        for name, compiled, elapsed in values.values():
            count += compiled
            timings[name] = timings.get(name, 0.0) + elapsed
    return count, timings


def plan_warm_up(
    roots: Iterable[str],
    requests: Mapping,
    *,
    cache: Path | None = None,
    store: Path | None = None,
    key: str | None = None,
    key_closure: CodeIdentity | None = None,
) -> CacheState:
    """Seed the cache and find what warm-up must compile: per function, the union of declared and learned overloads
    and those the cache lacks (``state.missing``). Compiles nothing.

    The caller preloads roots and sets NUMBA_CACHE_DIR before importing Numba. Store I/O failures only log; a cache
    that cannot be prepared leaves an empty plan (dispatch stays lazy)."""
    from numba.core.registry import cpu_target

    from ..core.kernels import declarations, enable_caches

    roots = tuple(roots)
    closure = code_dependency_identity(roots) if key_closure is None else key_closure
    check_closure(roots, closure)
    install_signature_guard()
    install_compile_lock()
    install_cache_path_check_once()
    failures = enable_caches()
    if failures:
        _LOG.warning("some kernel caches unavailable; dispatch remains lazy: %s", failures)
    directory = Path(os.environ["NUMBA_CACHE_DIR"]) if cache is None else Path(cache)
    storage = store_directory() if store is None else Path(store)
    state = CacheState(directory.name if key is None else key, directory, storage)
    snapshot = None if storage is None else storage / state.key / _SNAPSHOT
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if snapshot is not None and snapshot.is_file():
            state.seeded = _seed(directory, snapshot)
            state.published = True
        state.baseline = _listing(directory)
    except (OSError, tarfile.TarError) as error:
        _LOG.warning("kernel snapshot unavailable; using lazy compilation: %s", error)
        return state
    # Snapshot this invocation's declarations before unpickling learned signatures: unpickling imports the
    # modules that define their types, which this invocation's workers never import. Learned overloads are
    # kept per cache key, so one command's learning never reaches another command's warm-up.
    declared = {(d.module, d.function): set(d.overloads) for d in declarations()}
    learned, state.seconds = ({}, {}) if storage is None else _read_manifest(storage / state.key)
    # Never omit an imported declaration just because a caller supplied a partial request set.
    state.required = {name: set(sigs) for name, sigs in declared.items()}
    state.union = {name: set(sigs) for name, sigs in learned.items()}
    for source in (requests, state.required):
        for name, signatures in source.items():
            state.union.setdefault(name, set()).update(signatures)
    target = cpu_target.target_context.codegen().magic_tuple()
    present = _cached_overloads(directory, target)
    missing = {name: set(sigs) - present.get(name, set()) for name, sigs in state.union.items()}
    state.missing = {name: sigs for name, sigs in missing.items() if sigs}
    state.requested = sum(map(len, state.missing.values()))
    return state


def warm_up(
    roots: Iterable[str],
    requests: Mapping,
    cores: int,
    *,
    cache: Path | None = None,
    store: Path | None = None,
    key: str | None = None,
    key_closure: CodeIdentity | None = None,
) -> CacheState:
    """Seed and warm declarations plus learned signatures before scientific pools.

    The caller preloads roots and sets NUMBA_CACHE_DIR before importing Numba.
    One graph task per overload compiles the missing set; a second graph with
    fresh workers verifies the complete union after a cold compile. Learned failures are counted/skipped;
    declared overload failures raise InvariantError. Store I/O failures only log.
    Call publish(state) in a finally block, including after scientific failure.
    """
    state = plan_warm_up(roots, requests, cache=cache, store=store, key=key, key_closure=key_closure)
    state.compiled, timings = _compile_requests(state, state.missing, cores)
    state.seconds.update(timings)
    total = sum(map(len, state.union.values()))
    state.verified = total
    if state.requested:
        # The first graph has closed its workers. Verify the full union in fresh
        # processes so a loaded dispatcher cannot hide a missing cache overload.
        state.verified, _ = _compile_requests(state, state.union, cores)
    state.stale = state.compiled < state.requested or state.verified < total
    state.skipped = total - state.verified
    return state


def kernel_task_key(name: tuple[str, str], overload: int = 0):
    """The graph key of the compile task of function ``name``'s ``overload``-th missing overload (compile_tasks)."""
    from .graph import TaskKey

    return TaskKey("kernels", "compile", "run", f"{name[0]}:{name[1]}#{overload}")


def compile_kernel(name: tuple[str, str], signatures: bytes, declared: bytes) -> tuple[tuple[str, str], int, float]:
    """Graph task: compile one function's missing overloads into the run's kernel cache (eager and deferred warm-up share this action)."""
    return _compile_function(name, signatures, declared)


def compile_tasks(state: CacheState) -> tuple:
    """The plan's compilations as graph tasks, one per missing overload, for a caller that runs them beside its
    stages instead of before them (interface.run). The caller attaches each consumer's missing compile keys as
    dependencies before admitting that stage; unrelated work can proceed. Overload locks still protect shared
    callees and lazy callers, and index locks every save (install_compile_lock). There is no verify pass on this
    route. One task per missing overload allows a function's overloads to compile in parallel."""
    from numba.core.serialize import dumps

    from .graph import ResourceRequest, Task

    return tuple(
        Task(
            kernel_task_key(name, index),
            compile_kernel,
            args=(name, dumps((signature,)), dumps(set(state.required.get(name, ())) & {signature})),
            resources=ResourceRequest(1, 1),
        )
        for name, signatures in sorted(state.missing.items())
        for index, signature in enumerate(sorted(signatures, key=str))
    )


def record_compiled(state: CacheState, values: Mapping) -> None:
    """Fold compile_tasks' results into the warm-up accounting before publication (seconds: per function, summed
    over its overloads' tasks)."""
    for name, signatures in state.missing.items():
        seconds = 0.0
        for index in range(len(signatures)):
            value = values.get(kernel_task_key(name, index))
            if value is not None:
                _, compiled, elapsed = value
                state.compiled += compiled
                seconds += elapsed
        if seconds:
            state.seconds[name] = seconds
    state.verified = sum(map(len, state.union.values())) - (state.requested - state.compiled)
    state.stale = state.compiled < state.requested
    state.skipped = state.requested - state.compiled


def publish(state: CacheState | None) -> None:
    """Publish cache and learned overloads atomically; cache-store failures only log."""
    if state is None or state.store is None:
        return
    try:
        _publish(state)
    except Exception as error:
        _LOG.warning("could not publish kernel cache to %s (%s: %s)", state.store, type(error).__name__, error)


def _publish(state: CacheState) -> None:
    from numba.core.serialize import dumps

    cache_directory, store, key = state.cache_directory, state.store, state.key
    assert store is not None
    files = _listing(cache_directory)
    changed = bool(files) and not (state.published and files == state.baseline)
    if not (changed or state.stale):
        return
    overloads = _cached_overloads(cache_directory)
    store.mkdir(parents=True, exist_ok=True)
    if changed:
        directory = store / key
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory / _SNAPSHOT
        temporary = directory / f".{_SNAPSHOT}.{os.getpid()}.tmp"
        try:
            with tarfile.open(temporary, "w") as archive:
                for name in sorted(files):
                    archive.add(cache_directory / name, arcname=name)
            os.replace(temporary, snapshot)
        finally:
            temporary.unlink(missing_ok=True)
    (store / key).mkdir(parents=True, exist_ok=True)
    _write_atomically(
        store / key / _MANIFEST,
        dumps(
            dict(
                overloads={name: tuple(sigs) for name, sigs in overloads.items()},
                seconds={name: value for name, value in state.seconds.items() if name in overloads},
            )
        ),
    )
    recent = []
    for path in store.glob(f"*/{_SNAPSHOT}"):
        try:
            recent.append((path.stat().st_mtime, path.parent))
        except FileNotFoundError:
            pass
    for _, stale in sorted(recent, reverse=True)[KEEP_KEYS:]:
        shutil.rmtree(stale, ignore_errors=True)
