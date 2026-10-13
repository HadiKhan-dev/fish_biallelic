"""Persistent store of compiled Numba kernels and parallel warm-up for workflow runs.

core.environment gives every process tree a run-local Numba cache directory per
code key (package source, toolchain versions, code-generation settings, CPU
model). Starting empty, every run compiled its ~300-450 kernel overloads again,
each pool worker on first use and on its own, which put several minutes of
compilation on the critical path of every reconstruction lane. This module keeps
the compiled kernels between runs and compiles a new key's kernels up front:

* Store: ``$XDG_CACHE_HOME/haplotype-reconstruction/numba`` (``~/.cache/...``
  by default; on a cluster a home directory shared by every node). One
  directory per code key holds ``cache.tar``, a complete snapshot of a
  run-local cache (about 60-100 MB), published by an atomic rename: concurrent
  runs on several hosts read only complete snapshots, and the last publication
  of a key wins. Only the KEEP_KEYS most recently used keys are kept.
* Manifest: ``manifest.pkl`` beside them lists the (module, function,
  signature) overloads of the latest publication and their compile seconds.
  Before a workflow starts its pools, the overloads it lists that the
  run-local cache lacks (all of them for a new key) are compiled in a pool of
  the run's core budget: one task per function, so each Numba index file has a
  single writer, longest first.

A run publishes its cache when the workflow ends (also after a failure) if it
compiled anything. Stored kernels can only meet the code they were compiled
from, because a code key names them; Numba still checks source timestamps and
the CPU on load. Store failures (no writable home, full quota) cost only
compilation time and are reported in the run log.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import hashlib
import importlib
import io
import os
from pathlib import Path
import pickle
import shutil
import sys
import tarfile
import time

from . import environment
from .run_record import timed_stage


KEEP_KEYS = 6
_SNAPSHOT = "cache.tar"
_MANIFEST = "manifest.pkl"
_PACKAGE = __name__.split(".")[0]
_INDEX_SUFFIX = (f".py{sys.version_info[0]}{sys.version_info[1]}"
                 f"{getattr(sys, 'abiflags', '')}.nbi")


def store_directory():
    """The persistent store, or None without a home or XDG cache directory."""
    base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        try:
            base = Path.home() / ".cache"
        except RuntimeError:
            return None
    return Path(base) / "haplotype-reconstruction" / "numba"


def _listing(directory):
    """{relative path: (bytes, mtime ns)} of the Numba cache files in directory."""
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


def _write_atomically(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed(cache_directory, snapshot):
    """Install the snapshot's functions that the run-local cache lacks.

    A function's index (``.nbi``) and data files (``.nbc``, named after the
    index) are installed together and only when no local index exists, data
    files first, each by an atomic rename: a reader never meets a partial
    entry, and a local index is never mixed with another one's data files.
    Returns the number of files written.
    """
    payload = snapshot.read_bytes()
    try:
        os.utime(snapshot)  # most recently used, for pruning
    except OSError:
        pass
    functions = {}
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
                _write_atomically(target / Path(member.name).name,
                                  archive.extractfile(member).read())
                written += 1
    return written


def _module_packages():
    """Numba cache subdirectory -> dotted package, for this package's directories.

    Numba files a kernel under ``<directory name>_<sha1 of its absolute
    directory>`` (numba.core.caching._CacheLocator.get_suitable_cache_subpath).
    """
    root = os.path.dirname(os.path.abspath(sys.modules[_PACKAGE].__file__))
    packages = {}
    for directory, subdirectories, _ in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if name != "__pycache__"]
        relative = os.path.relpath(directory, root)
        dotted = _PACKAGE if relative == "." else f"{_PACKAGE}.{relative.replace(os.sep, '.')}"
        name = os.path.basename(directory) + "_" + hashlib.sha1(directory.encode()).hexdigest()
        packages[name] = dotted
    return packages


def _cached_overloads(cache_directory, target=None):
    """{(module, function): signatures} of this package's kernels in a cache directory.

    Index files are named ``<module>.<qualified name>-<line>.pyXY.nbi`` and hold
    the Numba version, then (source stamp, {(signature, target, code hashes):
    data file}). ``target`` keeps only overloads compiled for that code-generation
    target (Numba's magic tuple: triple, CPU name, features). Functions without
    an importable name (closures) are skipped.
    """
    import numba
    packages = _module_packages()
    overloads = {}
    for subdirectory in os.listdir(cache_directory):
        package = packages.get(subdirectory)
        if package is None:
            continue
        for name in os.listdir(cache_directory / subdirectory):
            if not name.endswith(_INDEX_SUFFIX):
                continue
            stem, _, line = name[:-len(_INDEX_SUFFIX)].rpartition("-")
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
                # Numba would not use an index it cannot read either.
                continue
            signatures = {key[0] for key in index if target is None or key[1] == target}
            if signatures:
                overloads.setdefault((module, function), set()).update(signatures)
    return overloads


def _read_manifest(store):
    try:
        with open(store / _MANIFEST, "rb") as handle:
            manifest = pickle.load(handle)
        return dict(manifest["overloads"]), dict(manifest["seconds"])
    except FileNotFoundError:
        return {}, {}
    except Exception as error:
        # Written by another Numba version or interrupted: compile lazily.
        print(f"  [numba cache] manifest unreadable ({type(error).__name__}); ignored",
              flush=True)
        return {}, {}


def _compile_function(task):
    """Pool task: compile one function's signatures into the run-local cache.

    A manifest recorded by older code may name functions that no longer exist
    or signatures that no longer type; those are skipped (counted by the caller).
    """
    module, function, signatures = task
    started = time.perf_counter()
    try:
        dispatcher = functools.reduce(getattr, function.split("."),
                                      importlib.import_module(module))
        compile_signature = dispatcher.compile
    except Exception:
        return module, function, 0, None
    compiled = 0
    for signature in signatures:
        try:
            compile_signature(signature)
            compiled += 1
        except Exception:
            pass
    return module, function, compiled, time.perf_counter() - started


def _warm_up(missing, seconds, cores):
    """Compile the missing overloads, one function per task, longest first.

    Returns (compiled overloads, process count, {function: seconds}).
    """
    from . import parallel
    tasks = sorted(((module, function, tuple(signatures))
                    for (module, function), signatures in missing.items()),
                   key=lambda task: (-seconds.get(task[:2], 0.0), -len(task[2]), task[:2]))
    processes = max(1, min(cores, len(tasks)))
    compiled, timings = 0, {}
    with parallel.safe_forkserver_pool(processes) as pool:
        for module, function, count, elapsed in pool.imap_unordered(
                _compile_function, tasks, chunksize=1):
            compiled += count
            if elapsed is not None:
                timings[(module, function)] = elapsed
    return compiled, processes, timings


@timed_stage("numba_warmup")
def _prepare(cores):
    """Seed the run-local cache and compile what the manifest lists beyond it."""
    started = time.perf_counter()
    key = environment.numba_code_key()
    cache_directory = Path(os.environ.get("NUMBA_CACHE_DIR", ""))
    store = store_directory()
    if store is None or cache_directory.name != key:
        print("  [numba cache] no persistent store for this run", flush=True)
        return None
    snapshot = store / key / _SNAPSHOT
    seeded = 0
    try:
        if snapshot.is_file():
            seeded = _seed(cache_directory, snapshot)
    except (OSError, tarfile.TarError) as error:
        print(f"  [numba cache] could not read {snapshot}: {error}", flush=True)
    baseline = _listing(cache_directory)
    listed, seconds = _read_manifest(store)
    from numba.core.registry import cpu_target
    present = _cached_overloads(cache_directory, cpu_target.target_context.codegen().magic_tuple())
    missing = {}
    for function, signatures in listed.items():
        lacking = set(signatures) - present.get(function, set())
        if lacking:
            missing[function] = lacking
    requested = sum(map(len, missing.values()))
    compiled, processes, timings = _warm_up(missing, seconds, cores) if missing else (0, 0, {})
    seconds.update(timings)
    print(f"  [numba cache] key {key}: {seeded} files from {store}; warm-up compiled "
          f"{compiled} of {requested} manifest overloads missing here ({requested - compiled} "
          f"skipped: no longer in this code) with {processes} processes in "
          f"{time.perf_counter() - started:.1f} s", flush=True)
    return dict(key=key, cache_directory=cache_directory, store=store, baseline=baseline,
                published=snapshot.is_file(), stale=compiled < requested, seconds=seconds)


@timed_stage("numba_publish")
def _publish(state):
    """Publish the run-local cache as its key's snapshot if it gained kernels.

    The manifest is rewritten from the published cache, and also when it listed
    overloads this code no longer has, so later runs stop retrying them.
    """
    cache_directory, store, key = state["cache_directory"], state["store"], state["key"]
    files = _listing(cache_directory)
    changed = bool(files) and not (state["published"] and files == state["baseline"])
    if not (changed or state["stale"]):
        return
    overloads = _cached_overloads(cache_directory)
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
        print(f"  [numba cache] published {len(files)} files "
              f"({snapshot.stat().st_size / 2**20:.0f} MB) as {snapshot}", flush=True)
    _write_atomically(store / _MANIFEST, pickle.dumps(dict(
        overloads={function: tuple(signatures) for function, signatures in overloads.items()},
        seconds={function: value for function, value in state["seconds"].items()
                 if function in overloads}), protocol=pickle.HIGHEST_PROTOCOL))
    recent = []
    for path in store.glob(f"*/{_SNAPSHOT}"):
        try:
            recent.append((path.stat().st_mtime, path.parent))
        except FileNotFoundError:
            pass
    for _, stale in sorted(recent, reverse=True)[KEEP_KEYS:]:
        shutil.rmtree(stale, ignore_errors=True)


@contextmanager
def persistent_numba_cache(cores):
    """Run a workflow with stored and warmed-up kernels; publish new ones at its end.

    ``cores`` bounds the warm-up pool (the run's core budget). Call in the main
    process before the workflow creates any pool. The store only saves
    compilation: no failure of it stops the workflow (kernels then compile on
    first use, as without a store).
    """
    try:
        state = _prepare(cores)
    except Exception as error:
        print(f"  [numba cache] store not used ({type(error).__name__}: {error})", flush=True)
        state = None
    try:
        yield
    finally:
        if state is not None:
            try:
                _publish(state)
            except Exception as error:
                print(f"  [numba cache] could not publish to {state['store']} "
                      f"({type(error).__name__}: {error})", flush=True)
