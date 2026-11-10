"""Apply numerical policy only at an explicit application or worker boundary.

The isolated entry starts a clean Python interpreter before unpickling user tasks.
Forkserver preload names and hooks are caller-supplied; no stage names occur here.
Memory observations support process bootstrap and pressure admission (ENG-19/25).
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
    "start",
    "start_run",
    "RunSettings",
    "WorkerSettings",
    "worker_settings",
    "current_settings",
    "apply_worker_settings",
    "isolated",
    "warm_kernels",
    "kernel_run",
    "start_forkserver",
    "signal_boundary",
    "RunInterrupted",
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
    """Return the process's available CPUs from affinity, then the OS CPU count."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


def configure_worker(max_threads: int) -> None:
    """Set the numerical policy before a fresh worker imports task modules.

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


def current_settings() -> RunSettings | None:
    """Observe this process's configured runtime without starting one."""
    return _RUN


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
    from .cpu import install_runtime_hooks
    from .memory import start_governor

    install_runtime_hooks()
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
    from .memory import governor_handle

    settings = _RUN
    if settings is None:
        settings = RunSettings(int(os.environ.get("NUMBA_NUM_THREADS", "1")))
    return WorkerSettings(settings, governor_handle())


def apply_worker_settings(settings: WorkerSettings) -> None:
    global _RUN
    _RUN = settings.run
    _preload(_RUN, hooks=True)
    from .cpu import install_runtime_hooks
    from .memory import attach_governor

    install_runtime_hooks()
    attach_governor(settings.governor_handle)
    import numba

    _unnamed_numba_init_lock()
    numba.set_num_threads(1)


def _unnamed_numba_init_lock() -> None:
    """Give Numba's threading-backend initialisation lock an unnamed semaphore before the layer starts.

    Numba creates that lock lazily (numba.np.ufunc.parallel._set_init_process_lock) from the default start method's
    context. Under forkserver that semaphore is named and tracked, so a worker killed inside a
    native call can leave it behind. A lock from the fork context is unlinked as it is made and serves the same purpose inside this process. Numba versions
    without this module attribute are left alone."""
    import multiprocessing

    from numba.np.ufunc import parallel

    if getattr(parallel, "_backend_init_process_lock", 0) is None:
        parallel._backend_init_process_lock = multiprocessing.get_context("fork").RLock()


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
    """Warm the caller-declared closure before task execution; return the kernel CacheState.

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


class RunInterrupted(SystemExit):
    """A coordinator received a termination request; committed products remain resumable."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(130)

    def __str__(self) -> str:
        return signal.Signals(self.signum).name


@contextmanager
def signal_boundary(cancel: Callable[[], None]) -> Iterator[None]:
    """Stop scheduling and unwind verified-worker cleanup on SIGINT or SIGTERM."""
    import threading

    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum: int, frame: Any) -> None:
        cancel()
        raise RunInterrupted(signum)

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


# After forwarding a termination request, the parent waits this long for the isolated coordinator before
# killing its process group. It must exceed the coordinator's own bounded cleanup (terminated workers get
# workers.TERMINATE_GRACE_SECONDS, then a forced kill and join, then pool and transport close): a group kill
# during that cleanup leaves the coordinator's named semaphores to the resource tracker. It stays inside a
# batch system's usual SIGTERM-to-SIGKILL window (30 s by default in Slurm).
COORDINATOR_GRACE_SECONDS = 20.0


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
        # The child imports from the parent's module search path, as multiprocessing's forkserver does: with a bare
        # -c its working directory came first, so a run started in a source checkout's root imported that checkout's
        # package instead of the parent's, giving the child a different source identity.
        search = Path(directory) / "path.pkl"
        search.write_bytes(pickle.dumps(list(sys.path)))
        program = (
            "import pickle, sys; sys.path[:] = pickle.loads(open(sys.argv[3], 'rb').read()); "
            "from haplotype_reconstruction.engine.runtime import _isolated_entry; _isolated_entry()"
        )
        process = subprocess.Popen(
            [sys.executable, "-P", "-c", program, str(request), str(result), str(search)],
            env=environment,
            start_new_session=True,
        )
        try:
            with signal_boundary(lambda: None):
                code = process.wait()
        except BaseException:
            # Let the coordinator unwind pools and unlink their semaphores first. Signalling its whole
            # session also kills the forkserver/resource tracker before owners can release their resources.
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=COORDINATOR_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
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
                raise RunInterrupted(signal.SIGTERM if code == -signal.SIGTERM else signal.SIGINT)
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
    except (KeyboardInterrupt, RunInterrupted):
        raise SystemExit(130) from None
    except BaseException as error:
        failure = task_failure(error)
        result.write_bytes(pickle.dumps(("error", failure)))
        raise failure from error
    result.write_bytes(pickle.dumps(("result", value)))
