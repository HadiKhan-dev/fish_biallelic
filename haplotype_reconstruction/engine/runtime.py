"""Apply numerical policy only at an explicit application or worker boundary.

The isolated entry starts a clean Python interpreter before unpickling user tasks.
Forkserver preload names and hooks are caller-supplied; no stage names occur here.
Memory observations preserve ENG-19/25 (3.5 execution/resources.py).
"""

from __future__ import annotations

import base64
import importlib
import logging
import os
import pickle
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from ..core.errors import ResourceError, SettingsError

__all__ = [
    "affinity_cpus",
    "configure_worker",
    "thread_team",
    "start",
    "start_run",
    "RunSettings",
    "WorkerSettings",
    "worker_settings",
    "apply_worker_settings",
    "isolated",
    "run_isolated",
    "warm_kernels",
    "kernel_run",
    "start_forkserver",
    "signal_boundary",
    "available_memory_bytes",
    "resident_bytes",
    "memory_pressure",
    "profiling",
    "task_boundary",
]
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
BASE_PRELOAD = ("numpy", "numba", "scipy")
THREADING_LAYER = "omp"


def affinity_cpus() -> int:
    """Return the process's available CPUs using the original 3.5 affinity rule."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


def configure_worker(max_threads: int) -> None:
    """Set the 3.5 numerical policy before a fresh worker imports task modules.

    If Numba was already initialized, refuse incompatible settings rather than
    silently changing numerical execution. The caller owns its cache directory.
    """
    from threadpoolctl import threadpool_info

    for library in threadpool_info():
        if library["user_api"] == "blas" and library["num_threads"] != 1:
            raise SettingsError(f"{library['internal_api']} has {library['num_threads']} BLAS threads; expected 1")
    numba = sys.modules.get("numba")
    if numba is not None:
        if not numba.config.DISABLE_INTEL_SVML:
            raise SettingsError("Numba SVML is enabled; the numerical policy requires scalar libm")
        try:
            layer = numba.threading_layer()
        except ValueError:
            layer = numba.config.THREADING_LAYER
        if layer not in ("omp", "default"):
            raise SettingsError(f"Numba threading layer {layer!r}; expected 'omp'")
        if max_threads > numba.config.NUMBA_NUM_THREADS:
            raise ResourceError(f"Numba pool has {numba.config.NUMBA_NUM_THREADS} threads; task needs {max_threads}")
    for variable in _THREAD_VARIABLES:
        os.environ[variable] = "1"
    os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
    os.environ["NUMBA_THREADING_LAYER"] = "omp"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    if numba is None:
        os.environ["NUMBA_NUM_THREADS"] = str(max_threads)
    else:
        numba.config.THREADING_LAYER = "omp"


@contextmanager
def thread_team(cores: int):
    """Lease a bounded Numba team for one task and restore the previous team afterward."""
    import numba

    from ..core.kernels import enable_caches

    enable_caches()
    previous = numba.get_num_threads()
    numba.set_num_threads(cores)
    try:
        yield
    finally:
        numba.set_num_threads(previous)


@contextmanager
def task_team(default: int):
    """A parallel section's Numba team: under an engine CPU budget (a graph task's, or a thread map worker's one CPU),
    that budget's live share when the section starts (a boundary: a section started after peers finished uses their
    CPUs); elsewhere ``default``. On exit the team never returns above the budget's share then (it may have shrunk).
    Sections sized by the share their task started with ran on one CPU while the node emptied."""
    import numba

    from ..core.kernels import enable_caches
    from .scheduler import _THREAD_BUDGET

    enable_caches()
    budget = getattr(_THREAD_BUDGET, "value", None)
    previous = numba.get_num_threads()
    numba.set_num_threads(max(1, int(budget.threads() if budget is not None else default)))
    try:
        yield
    finally:
        numba.set_num_threads(previous if budget is None else max(1, min(previous, int(budget.threads()))))


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
            maximum_name, current_name = ("memory.limit_in_bytes", "memory.usage_in_bytes")
        elif not controllers:
            root = Path("/sys/fs/cgroup")
            maximum_name, current_name = ("memory.max", "memory.current")
        else:
            continue
        directory = root / relative.lstrip("/")
        while True:
            try:
                maximum = int((directory / maximum_name).read_text().strip())
                current = int((directory / current_name).read_text().strip())
                if 0 < maximum < 1 << 60:
                    current -= _inactive_file_bytes(directory, bool(controllers))
                    candidates.append(max(0, maximum - current))
            except (OSError, ValueError):
                pass
            if directory == root:
                break
            directory = directory.parent
    slurm_mib = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mib and slurm_mib.isdigit() and (int(slurm_mib) > 0):
        candidates.append(int(slurm_mib) << 20)
    return min(candidates) if candidates else None


def resident_bytes() -> int:
    """This process's current resident set size in bytes (0 where /proc is unavailable)."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.readline().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def memory_pressure() -> tuple[int, int, int] | None:
    """(working set, capacity, available bytes) for this job, with MemAvailable fallback.

    Prefer the enclosing Slurm job cgroup (including all steps), then the tightest bounded cgroup. Inactive file
    cache is reclaimable and excluded, in both v1 and v2. Physical node availability remains an independent
    bound when another job consumes memory. A configured lower run limit is applied by the governor.
    """
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                info[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    groups = []
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        entries = []
    for entry in entries:
        _, controllers, relative = entry.split(":", 2)
        v1 = "memory" in controllers.split(",")
        if not v1 and controllers:
            continue
        root = Path("/sys/fs/cgroup/memory" if v1 else "/sys/fs/cgroup")
        directory = root / relative.lstrip("/")
        while directory == root or root in directory.parents:
            try:
                cap = int((directory / ("memory.limit_in_bytes" if v1 else "memory.max")).read_text())
                used = int((directory / ("memory.usage_in_bytes" if v1 else "memory.current")).read_text())
                if 0 < cap < 1 << 60:
                    used = max(0, used - _inactive_file_bytes(directory, v1))
                    groups.append((directory.name.startswith("job_"), cap, used))
            except (OSError, ValueError):
                pass
            if directory == root:
                break
            directory = directory.parent
    if groups:
        jobs = [g for g in groups if g[0]]
        _, cap, used = min(jobs or groups, key=lambda g: g[1])
        available = min((c - u for _, c, u in groups))
        available = min(available, info.get("MemAvailable", cap))
        return (max(used, cap - available), cap, max(0, available))
    if "MemTotal" in info and "MemAvailable" in info:
        cap, available = (info["MemTotal"], info["MemAvailable"])
        slurm = os.environ.get("SLURM_MEM_PER_NODE", "")
        if slurm.isdigit() and int(slurm) > 0:
            cap = min(cap, int(slurm) << 20)
        return (max(0, cap - available), cap, min(cap, available))
    return None


@dataclass(frozen=True)
class RunSettings:
    """Picklable operational settings, fixed before a runtime starts its workers."""

    cores: int
    profile: bool = False
    preload: tuple[str, ...] = ()
    worker_hooks: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    cache_directory: str | None = None
    code_roots: tuple[str, ...] = ()
    closure_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkerSettings:
    run: RunSettings
    governor_handle: str


_RUN: RunSettings | None = None
_BOOTSTRAP_DIRECTORY: tempfile.TemporaryDirectory | None = None
_CHECKED_DECLARATIONS: tuple[Any, ...] | None = None


def _hooks(settings: RunSettings) -> None:
    for name, arguments in settings.worker_hooks:
        module, separator, function = name.partition(":")
        if not separator or not module or (not function) or ("." in function):
            raise SettingsError(f"worker hook must be module:function: {name!r}")
        getattr(importlib.import_module(module), function)(*arguments)


def _preload(settings: RunSettings, *, hooks: bool) -> None:
    from ..store.identity import fix_installed_sources

    fix_installed_sources()  # forkserver and workers: the code imported under this runtime, for their lifetime
    if settings.cache_directory is not None:
        os.environ["NUMBA_CACHE_DIR"] = settings.cache_directory
    configure_worker(settings.cores)
    for module in settings.preload:
        importlib.import_module(module)
    if settings.cache_directory is not None:
        import numba

        cast(Any, numba.config).CACHE_DIR = settings.cache_directory
    from ..core.kernels import enable_caches
    from .kernels import install_cache_path_check_once, install_compile_lock, install_signature_guard

    _check_declarations(settings)
    install_signature_guard()
    install_compile_lock()
    install_cache_path_check_once()
    enable_caches()
    if hooks:
        _hooks(settings)


def _forkserver_boundary(encoded: str) -> None:
    """Called by the explicit generated preload entry, never by package import."""
    global _RUN
    _RUN = pickle.loads(base64.b64decode(encoded))
    _preload(_RUN, hooks=True)


def start(
    cores: int,
    *,
    preload: Sequence[str] = (),
    profile: bool = False,
    memory_limit: int | None = None,
    memory_diagnostics: str | Path | None = None,
    worker_hooks: Sequence[tuple[str, tuple[Any, ...]]] = (),
    cache_directory: str | Path | None = None,
) -> RunSettings:
    """Configure one invocation before it imports task roots or starts pools.

    The small generated preload entry is an application bootstrap, created here,
    outside the package. It runs the closure check and test hooks in the actual
    forkserver after its preloads. Worker initializers repeat these boundaries.
    """
    global _RUN, _BOOTSTRAP_DIRECTORY
    if not 1 <= cores <= affinity_cpus():
        raise ResourceError(f"cores {cores} exceed affinity {affinity_cpus()}")
    configure_worker(cores)
    roots = tuple(dict.fromkeys(preload))
    package_roots = tuple(
        dict.fromkeys(
            (
                "haplotype_reconstruction.engine.workers",
                *(name for name in roots if name.startswith("haplotype_reconstruction.")),
            )
        )
    )
    from ..store.identity import code_dependency_identity, fix_installed_sources
    from .kernels import cache_directory as make_cache_directory
    from .kernels import cache_key

    # From here this process runs one code snapshot (its forkserver keeps the modules imported now): its identities,
    # this closure first, are computed once.
    fix_installed_sources()
    closure = code_dependency_identity(package_roots) if package_roots else None
    if cache_directory is None and _RUN is not None and (_RUN.code_roots == package_roots):
        cache_directory = _RUN.cache_directory
    if cache_directory is None and package_roots:
        key = cache_key(package_roots)
        base = os.environ.get("NUMBA_CACHE_DIR")
        try:
            cache_directory = make_cache_directory(key, base=base)
        except OSError as error:
            # Keep an explicit failed namespace: cache failures may cost compile
            # time, never abort scientific work or select a source-tree cache.
            cache_directory = error.filename or str(Path(base or tempfile.gettempdir()) / "unavailable-cache" / key)
            logging.getLogger(__name__).warning("kernel cache unavailable; compiling lazily: %s", error)
    if cache_directory is not None:
        cache_directory = str(Path(cache_directory).resolve())
        try:
            Path(cache_directory).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logging.getLogger(__name__).warning("kernel cache unavailable; compiling lazily: %s", error)
        os.environ["NUMBA_CACHE_DIR"] = cache_directory
    settings = RunSettings(
        cores,
        profile,
        tuple(dict.fromkeys((*BASE_PRELOAD, "haplotype_reconstruction.engine.workers", *roots))),
        tuple(worker_hooks),
        cache_directory,
        package_roots,
        () if closure is None else closure.modules,
    )
    _preload(settings, hooks=False)
    from ..core.kernels import declarations

    # Caller imports may execute lazy branches absent from a fresh preload import.
    # Carry those already checked declarations into the worker's registry; do not
    # discover new science modules or enlarge the keyed dependency closure.
    declared_modules = tuple(dict.fromkeys(entry.module for entry in declarations()))
    settings = replace(settings, preload=tuple(dict.fromkeys((*settings.preload, *declared_modules))))
    import multiprocessing.forkserver

    server = multiprocessing.forkserver._forkserver
    if cast(Any, server)._forkserver_pid is not None and _RUN != settings:
        raise SettingsError("forkserver already started with different settings; use isolated execution")
    if _BOOTSTRAP_DIRECTORY is None:
        _BOOTSTRAP_DIRECTORY = tempfile.TemporaryDirectory(prefix="haplotypes-runtime-")
    directory = Path(_BOOTSTRAP_DIRECTORY.name)
    encoded = base64.b64encode(pickle.dumps(settings)).decode("ascii")
    import hashlib

    name = "haplotypes_runtime_" + hashlib.sha256(encoded.encode()).hexdigest()[:16]
    (directory / (name + ".py")).write_text(
        "from haplotype_reconstruction.engine.runtime import _forkserver_boundary\n"
        + f"_forkserver_boundary({encoded!r})\n"
    )
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    multiprocessing.set_forkserver_preload([*BASE_PRELOAD, name])
    _RUN = settings
    from .scheduler import start_governor

    start_governor(memory_limit, diagnostic_directory=memory_diagnostics)
    return settings


def start_forkserver() -> None:
    """Launch this runtime's forkserver now: it imports the run's preload modules and prepares its kernel caches (tens
    of seconds on a cold filesystem) in its own process while the caller plans its graph, instead of when the graph
    spawns its first worker. The runtime's settings must be final: a later start with other settings is refused
    while the forkserver lives."""
    if _RUN is None:
        raise SettingsError("start the runtime before its forkserver")
    import multiprocessing.forkserver

    from .workers import main_module_guard

    with main_module_guard():  # the socket directory and the parent's __main__, as for every engine process start
        multiprocessing.forkserver.ensure_running()


def start_run(cores: int, **options: Any) -> RunSettings:
    """The explicit run boundary used by the graph and kernel warm-up caller."""
    return start(cores, **options)


def worker_settings() -> WorkerSettings:
    from .scheduler import governor_handle

    settings = _RUN
    if settings is None:
        settings = RunSettings(int(os.environ.get("NUMBA_NUM_THREADS", "1")))
    return WorkerSettings(settings, governor_handle())


def apply_worker_settings(settings: WorkerSettings) -> None:
    global _RUN
    _RUN = settings.run
    _preload(_RUN, hooks=True)
    from .scheduler import attach_governor

    attach_governor(settings.governor_handle)
    import numba

    numba.set_num_threads(1)


def _check_declarations(settings: RunSettings) -> None:
    """Recheck newly imported/replaced declarations against the immutable run closure.

    Task unpickling may import another kernel. Unchanged declarations were checked
    at preload; parsing every source file again per task changes no evidence.
    The application must not edit its running code snapshot.
    """
    global _CHECKED_DECLARATIONS
    from ..core.kernels import declarations
    from .kernels import check_closure

    current = (settings.code_roots, settings.closure_modules, declarations())
    if current != _CHECKED_DECLARATIONS:
        if settings.code_roots:
            check_closure(settings.code_roots, settings.closure_modules)
        _CHECKED_DECLARATIONS = current


def task_boundary() -> None:
    """Check declarations discovered while unpickling, then enable their caches."""
    from ..core.kernels import enable_caches

    if _RUN is not None:
        _check_declarations(_RUN)
    enable_caches()


def warm_kernels(requests: Any = None, *, store: str | Path | None = None, defer: bool = False) -> Any:
    """Warm the caller-declared closure before task execution; return C2 CacheState.

    start must have configured and preloaded the complete invocation. The caller
    publishes this disposable state in finally, or uses kernel_run below. With ``defer``
    nothing is compiled here: the caller runs the plan's compilations as graph tasks
    (kernels.compile_tasks) and folds their results in (kernels.record_compiled).
    """
    if _RUN is None:
        raise SettingsError("start the runtime before warming kernels")
    from .kernels import cache_key, plan_warm_up, warm_up

    assert _RUN.cache_directory is not None
    cache, key = Path(_RUN.cache_directory), cache_key(_RUN.code_roots)
    seeds = None if store is None else Path(store)
    requests = {} if requests is None else requests
    if defer:
        return plan_warm_up(_RUN.code_roots, requests, cache=cache, store=seeds, key=key)
    return warm_up(_RUN.code_roots, requests, _RUN.cores, cache=cache, store=seeds, key=key)


@contextmanager
def kernel_run(requests: Any = None, *, store: str | Path | None = None, defer: bool = False) -> Iterator[Any]:
    """Warm first (or, with ``defer``, plan: see warm_kernels) and publish learned signatures even if scientific tasks
    fail."""
    from .kernels import publish

    state = None
    with signal_boundary(lambda: None):
        try:
            state = warm_kernels(requests, store=store, defer=defer)
            yield state
        finally:
            publish(state)


def profiling() -> bool:
    return _RUN is not None and _RUN.profile


@contextmanager
def signal_boundary(cancel: Callable[[], None]) -> Iterator[None]:
    """Stop scheduling and unwind verified-worker cleanup on SIGINT or SIGTERM."""
    import threading

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum: int, frame: Any) -> None:
        cancel()
        raise SystemExit(130)

    saved = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for signum, handler in saved.items():
            signal.signal(signum, handler)


def task_failure(error: BaseException) -> Exception:
    """A task cannot stop its coordinator, including by exiting with status zero."""
    if isinstance(error, Exception):
        return error
    return RuntimeError(f"task aborted with {type(error).__name__}: {error}")


def isolated(
    function: Callable[..., Any], *args: Any, cores: int = 1, preload: Sequence[str] = (), **kwargs: Any
) -> Any:
    """Call an importable entry in a clean numerical interpreter, returning its value.

    Trusted pickle carries arguments only after the child has set its environment.
    Stdout/stderr remain attached to the invocation (including its run-log tee).
    Parent interruption terminates and reaps this exact process group.
    """
    if function.__module__ == "__main__" or "<locals>" in function.__qualname__:
        raise SettingsError("isolated entry must be a module-level importable function")
    if not 1 <= cores <= affinity_cpus():
        raise ResourceError(f"cores {cores} exceed affinity {affinity_cpus()}")
    environment = dict(os.environ)
    for name in _THREAD_VARIABLES:
        environment[name] = "1"
    environment.update(
        NUMBA_DISABLE_INTEL_SVML="1",
        NUMBA_THREADING_LAYER=THREADING_LAYER,
        OMP_WAIT_POLICY="PASSIVE",
        NUMBA_NUM_THREADS=str(cores),
    )
    with tempfile.TemporaryDirectory(prefix="haplotypes-isolated-") as directory:
        request, result = (Path(directory) / "request.pkl", Path(directory) / "result.pkl")
        request.write_bytes(pickle.dumps((pickle.dumps((function, args, kwargs)), cores, tuple(preload))))
        program = "from haplotype_reconstruction.engine.runtime import _isolated_entry; _isolated_entry()"
        process = subprocess.Popen(
            [sys.executable, "-c", program, str(request), str(result)], env=environment, start_new_session=True
        )
        try:
            with signal_boundary(lambda: None):
                code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise
        if result.exists():
            kind, value = pickle.loads(result.read_bytes())
            if kind == "error":
                raise task_failure(value)
            if kind != "result":
                raise RuntimeError(f"unknown isolated runtime reply: {kind}")
            if code == 0:
                return value
        if code:
            if code in (130, -signal.SIGTERM, -signal.SIGINT):
                raise SystemExit(130)
            raise RuntimeError(f"isolated runtime {process.pid} exited with status {code}")
        raise RuntimeError(f"isolated runtime {process.pid} exited without a successful result")


def _isolated_entry() -> None:
    request, result = map(Path, sys.argv[1:3])
    try:
        payload, cores, preload = pickle.loads(request.read_bytes())
        start(cores, preload=preload)
        function, args, kwargs = pickle.loads(payload)
        task_boundary()
        value = function(*args, **kwargs)
    except BaseException as error:
        failure = task_failure(error)
        result.write_bytes(pickle.dumps(("error", failure)))
        raise failure from error
    result.write_bytes(pickle.dumps(("result", value)))


def run_isolated(
    function: Callable[..., Any],
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    cores: int = 1,
    preload: Sequence[str] = (),
) -> Any:
    """I1's structural entry to a fresh numerical interpreter."""
    return isolated(function, *args, cores=cores, preload=preload, **({} if kwargs is None else kwargs))
