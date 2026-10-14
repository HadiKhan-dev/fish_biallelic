"""Small per-invocation provenance and non-overlapping wall-time accounting.

Only the coordinator writes the record. Worker CPU time is not confused with
elapsed time; nested stage durations report exclusive time as well as totals.
Records describe completed work and resume overhead, not hypothetical cold runs.
"""
from __future__ import annotations

import bisect
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
from functools import wraps
from importlib import metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import threading

_ACTIVE = None
SETTINGS = (
    "BHD_NUM_PROCESSES", "NUMBA_NUM_THREADS", "NUMBA_CACHE_DIR", "HAPLOTYPES_BATCH_QUEUE",
    "BHD_RECOMBINATION_RATE_CM_PER_MB",
    "BHD_RECOMBINATION_MAP", "BHD_RECOMBINATION_SHARED_FAMILY", "BHD_RECOMBINATION_ORDER_BREAKPOINTS",
    "HAPLOTYPES_OUTPUT_DIR", "HAPLOTYPES_CHECKPOINT_DIR", "HAPLOTYPES_CONTIGS",
    "HAPLOTYPES_VCF", "HAPLOTYPES_TEMPLATES", "HAPLOTYPES_METADATA",
    "HAPLOTYPES_METADATA_SHEET", "HAPLOTYPES_ELIGIBILITY",
    "HAPLOTYPES_FOUNDER_REFINEMENT", "HAPLOTYPES_STOP_AFTER_STAGE",
    "HAPLOTYPES_READ_CALIBRATION", "HAPLOTYPES_SITE_QUALITY_FILTER",
    "HAPLOTYPES_SAMPLE_QUALITY_FILTER", "HAPLOTYPES_PEDIGREE_CALIBRATION", "HAPLOTYPES_CONTIG_WORKERS",
    "HAPLOTYPES_SERIAL_CONTIGS", "HAPLOTYPES_MAX_CONTIGS", "BHD_SIMULATION_SEED", "BHD_SIM_READ_DEPTH", "HAPLOTYPES_GENERATIONS",
    "BHD_SIMULATION_RECOMBINATION_MAP", "BHD_SIMULATION_RECOMBINATION_RATE_CM_PER_MB",
    "HAPLOTYPES_SIMULATION_DESIGN", "HAPLOTYPES_READ_MODEL",
    "BHD_SIM_CONTIGS", "BHD_SIM_STOP_AFTER_STAGE",
)
DEPENDENCIES = ("numpy", "numba", "scipy", "pandas", "hdbscan", "cyvcf2",
                "blosc2", "matplotlib", "networkx", "tqdm", "openpyxl", "tbb")


def source_record():
    package = Path(__file__).resolve().parents[1]
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(package.rglob("*.py"))}
    root = package.parent
    def git(*arguments):
        try:
            result = subprocess.run(["git", "-C", str(root), *arguments],
                                    text=True, capture_output=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"),
                python_files_sha256=hashes)


def environment_record():
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                executable=os.sys.executable, packages=versions)


class RunRecord:
    def __init__(self, output, label, log):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.path = Path(output)/"run_records"/f"{stamp}_{os.getpid()}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pid, self.start, self.stack = os.getpid(), time.monotonic(), []
        self.thread = threading.get_ident()
        self.data = dict(schema="run-record-v1", workflow=label, state="running",
                         started_utc=datetime.now(timezone.utc).isoformat(), pid=self.pid,
                         host=platform.node(), log=str(log),
                         command=os.sys.argv, cpu_affinity=sorted(os.sched_getaffinity(0)),
                         resolved_settings={key: os.environ[key] for key in SETTINGS if key in os.environ},
                         source=source_record(), environment=environment_record(), stages=[])
        self.publish()

    def publish(self):
        self.data['updated_utc'] = datetime.now(timezone.utc).isoformat()
        self.data["wall_seconds"] = time.monotonic()-self.start
        self.data["active_stages"] = [dict(stage=row["stage"], contig=row["contig"],
                                          elapsed_seconds=time.monotonic()-row["start"]) for row in self.stack]
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2)+"\n")
        temporary.replace(self.path)

    @contextmanager
    def stage(self, name, contig):
        frame = dict(stage=name, contig=contig, start=time.monotonic(), child_seconds=0.)
        self.stack.append(frame)
        self.publish()
        outcome = "complete"
        try:
            yield
        except BaseException:
            outcome = "failed"
            raise
        finally:
            duration = time.monotonic()-frame["start"]
            self.stack.pop()
            if self.stack:
                self.stack[-1]["child_seconds"] += duration
            self.data["stages"].append(dict(
                stage=name, contig=contig, state=outcome,
                inclusive_seconds=duration, exclusive_seconds=max(0., duration-frame["child_seconds"])))
            self.publish()

    def finish(self, error=None):
        self.data["state"] = "failed" if error else "complete"
        self.data["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if error is not None:
            self.data["error"] = f"{type(error).__name__}: {error}"
        root = self.data["resolved_settings"].get("HAPLOTYPES_CHECKPOINT_DIR")
        if root:
            from .checkpoints import CheckpointFile
            from .runtime import RUN_FILE
            run = CheckpointFile(os.path.join(root, RUN_FILE), kind="run")
            self.data["checkpoint_identities"] = {
                key[len("identity/"):]: run.get(key) for key in run.keys("identity/")}
        exclusive = sum(row["exclusive_seconds"] for row in self.data["stages"])
        self.data["unclassified_seconds"] = max(0., time.monotonic()-self.start-exclusive)
        self.publish()


@contextmanager
def record_run(output, label, log):
    global _ACTIVE
    previous = _ACTIVE
    record = RunRecord(output, label, log)
    _ACTIVE = record
    try:
        yield record
    except BaseException as error:
        # Supported stop-after-stage commands exit zero at a checkpoint boundary.
        clean_stop = isinstance(error, SystemExit) and error.code in (None, 0)
        record.finish(None if clean_stop else error)
        raise
    else:
        record.finish()
    finally:
        _ACTIVE = previous


class _WorkerStages(RunRecord):
    """Stage timer for a worker process; it writes nothing to the run directory."""

    def __init__(self):
        self.pid, self.start, self.stack = os.getpid(), time.monotonic(), []
        self.thread = threading.get_ident()
        self.data = dict(stages=[])

    def publish(self):
        pass


@contextmanager
def worker_stages():
    """Collect a worker's timed stages; its coordinator records the returned rows."""
    global _ACTIVE
    previous = _ACTIVE
    record = _WorkerStages()
    _ACTIVE = record
    try:
        yield record.data["stages"]
    finally:
        _ACTIVE = previous


def record_product(name, contig, summary):
    """Attach a small JSON-compatible summary of one contig's product to the active record."""
    record = _ACTIVE
    if record is None or record.pid != os.getpid() or record.thread != threading.get_ident():
        return
    record.data.setdefault("products", {}).setdefault(name, {})[str(contig)] = copy.deepcopy(summary)
    record.publish()


def timed_stage(name, contig_arg=None):
    """Time a coordinator function; nested stage time is not counted twice."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            record = _ACTIVE
            if record is None or record.pid != os.getpid() or record.thread != threading.get_ident():
                return function(*args, **kwargs)
            contig = None
            if contig_arg:
                contig = signature.bind_partial(*args, **kwargs).arguments.get(contig_arg)
            if contig is None and record.stack:
                contig = record.stack[-1]["contig"]
            with record.stage(name, None if contig is None else str(contig)):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def run_status(output, *, checkpoints=None, contigs=None):
    """Summarize a run without decoding scientific checkpoint payloads.

    Run records (OUTPUT/run_records) give each invocation's last persisted
    state; the checkpoint store (``checkpoints``, default OUTPUT/checkpoints)
    gives per-contig progress, live work states and genome-wide stage
    completion (checkpoint_status). ``contigs`` selects and orders the contig
    rows. Problems found (another release's checkpoint layout, an unreadable
    container) are listed under "errors" instead of raised.
    """
    directory = Path(output).expanduser().resolve()
    records = []
    for path in sorted((directory/"run_records").glob("*.json")):
        data = json.loads(path.read_text())
        records.append(dict(record=str(path), workflow=data["workflow"], state=data["state"],
                            host=data["host"], pid=data["pid"], wall_seconds=data["wall_seconds"],
                            updated_utc=data['updated_utc'],
                            exclusive_stage_seconds={name: sum(row['exclusive_seconds'] for row in data['stages'] if row['stage']==name)
                                                     for name in sorted({row['stage'] for row in data['stages']})},
                            active_stages=data["active_stages"], error=data.get("error")))
    root = (directory/"checkpoints" if checkpoints is None
            else Path(checkpoints).expanduser().resolve())
    progress = checkpoint_status(root, contigs)
    errors = progress.pop("errors")
    if not directory.is_dir() and not progress["exists"]:
        errors.insert(0, f"{directory}: no such run directory")
    return dict(output=str(directory), records=records,
                note="running is the last persisted state; a killed process cannot publish completion. "
                     "No process liveness or Slurm state is inferred.",
                checkpoints=progress, errors=errors)


# Record keys read by status (the store layout: core.runtime, discovery and
# assembly checkpoints, downstream stages). Progress comes from key names and
# record metadata; the only payloads read are the entries named below, of
# small plans and manifests, never products.
_RUN_STAGES = ("simulation/templates", "simulation/reads", "discovery", "painting", "pedigree",
               "allele_refit", "family_phase", "recombination")  # run file complete/<stage>
_DISCOVERY_MANIFEST = "manifest/discovery"   # run file; entry "contigs" orders the rows
_SIMULATION = ("simulation/templates", "simulation/reads", "simulation/truth")
_DISCOVERY_PREPARED = "discovery/prepared"
_DISCOVERY_PLAN = "discovery/plan"           # entry "blocks": planned block tasks
_DISCOVERY_BLOCKS = "discovery/blocks/"      # <n:06d>, n = block tasks recorded so far
_FEEDBACK_ROUNDS = ("initial", "l1", "l2")   # feedback/<round>/: blocks/<k:06d> (64 selected
                                             # blocks each), proposals, selected (round done),
                                             # assembly/ (the l1, l2 trial assemblies)
_ASSEMBLY = "assembly/"                      # preprocess, l<k>/..., final (assembly done)
_LEVEL_PLAN = "plan"                         # l<k>/plan entry "batches": [[first, end, digest|None]]
_LEVEL_JOIN = "join"                         # level joined; entry "components": refinement total
_LEVEL_REFINE = "refine"                     # l<k>/refine/<n:05d> component refined (deeper keys:
                                             # sweep states); l<k>/refine level refined
_JOINS = "joins/"                            # joins/<batch input digest>, shared by all assemblies
_PAINTING = "painting"
_DOWNSTREAM = (("pedigree/evidence/", "pedigree evidence"), ("pedigree/scores/", "scores"),
               ("pedigree/exclusion/", "exclusion"), ("allele_refit", "allele_refit"),
               ("family_phase", "family_phase"), ("recombination", "recombination"))  # "/": versioned


_VERSIONED = re.compile(r"(.+)/[0-9a-f]{64}")


_LEVEL = re.compile(r"l(\d+)/")


_COMPONENT = re.compile(_LEVEL_REFINE + r"/(\d+)(/.+)?")


class _Listing:
    """The live records of one container, from a single index refresh."""

    def __init__(self, infos):
        self.infos = {info.key: info for info in infos}
        self.keys = sorted(self.infos)

    def __contains__(self, key):
        return key in self.infos

    def under(self, prefix):
        """The keys starting with ``prefix``, sorted."""
        start = end = bisect.bisect_left(self.keys, prefix)
        while end < len(self.keys) and self.keys[end].startswith(prefix):
            end += 1
        return self.keys[start:end]

    def suffixes(self, prefix):
        return [key[len(prefix):] for key in self.under(prefix)]

    def updated_utc(self):
        latest = max(self.infos.values(), key=lambda info: info.seq, default=None)
        return None if latest is None else latest.written_utc


def checkpoint_status(root, contigs=None):
    """Checkpoint progress from record keys and metadata (see run_status).

    One listing per container: run.ckpt's completed and pending genome-wide
    stages, identities and manifests; per contig, simulation inputs,
    discovery, the feedback rounds, the final assembly, painting and the
    downstream products, plus its live work states (work/<contig>/). Rows
    default to the configured contigs (the discovery manifest), then any
    others with checkpoints.

    Returns {root, exists, run, contigs: {name: row}, errors}. Each container
    row has file (None if absent), size_bytes, records, updated_utc (last
    record written), or file and error if unreadable. Contig rows add
    identities, simulation, discovery {prepared, blocks_done, blocks_total,
    complete}, feedback {round: {started, chunks, proposals, done, trial}},
    assembly {preprocessed, done, levels: [{level, state (started, joining,
    refining or done), batches, joins_done, joins_total, refine_done,
    refine_total, refine_partial}]} (None: not started), painting,
    downstream {name: present}, versions {versioned name: count} and work
    [{name, size_bytes, modified_utc}].
    """
    from .checkpoints import PAYLOAD_SUFFIX
    from .runtime import WORK_DIRECTORY, CheckpointStore
    root = os.fspath(root)
    status = dict(root=root, exists=os.path.isdir(root), run=None, contigs={}, errors=[])
    if not status["exists"]:
        return status  # never create a checkpoint directory
    try:
        store = CheckpointStore(root)
    except RuntimeError as error:  # another release's layout
        status["errors"].append(str(error))
        return status
    run = status["run"] = _inspect(store.run, _run_progress, status["errors"])
    work = _work_states(os.path.join(root, WORK_DIRECTORY), PAYLOAD_SUFFIX)
    if contigs:
        names = dict.fromkeys(map(str, contigs))
    else:
        found = sorted({*store.contig_names(), *work}, key=_natural_key)
        names = dict.fromkeys([*(run.get("contigs") or ()), *found])
    for name in names:
        try:
            file = store.contig(name)
        except ValueError as error:  # not a contig name
            status["errors"].append(f"{name!r}: {error}")
            continue
        status["contigs"][name] = row = _inspect(file, _contig_progress, status["errors"])
        row["work"] = work.get(name, [])
    return status


def _inspect(file, describe, errors):
    """Describe one container from a single listing of its live records."""
    size = file.size()
    try:
        listing = _Listing(file.infos())
        row = dict(file=file.path if size else None, size_bytes=size,
                   records=len(listing.infos), updated_utc=listing.updated_utc())
        row.update(describe(file, listing))
    except (OSError, ValueError) as error:  # unreadable or foreign container
        message = str(error)
        errors.append(message if file.path in message else f"{file.path}: {message}")
        return dict(file=file.path, size_bytes=size, error=message)
    return row


def _unversioned(name):
    match = _VERSIONED.fullmatch(name)
    return name if match is None else match.group(1)


def _run_progress(file, listing):
    complete = listing.suffixes("complete/")
    done = {_unversioned(stage) for stage in complete}
    simulation = any(listing.under(prefix) for prefix in (
        "simulation/", "identity/simulation/", "complete/simulation/"))
    contigs = None
    if _DISCOVERY_MANIFEST in listing:
        contigs = file.get(_DISCOVERY_MANIFEST, items=("contigs",)).get("contigs")
    return dict(complete=complete,
                pending=[stage for stage in _RUN_STAGES if stage not in done
                         and (simulation or not stage.startswith("simulation/"))],
                identities=listing.suffixes("identity/"), manifests=listing.suffixes("manifest/"),
                contigs=None if contigs is None else [str(name) for name in contigs])


def _contig_progress(file, listing):
    joins = set(listing.suffixes(_JOINS))
    simulation = {key.rpartition("/")[2]: key in listing for key in _SIMULATION}
    downstream, versions = {}, {}
    for key, _label in _DOWNSTREAM:
        name = key.rstrip("/")
        if key.endswith("/"):
            versions[name] = len({suffix.split("/", 1)[0] for suffix in listing.suffixes(key)})
            downstream[name] = versions[name] > 0
        else:
            downstream[name] = key in listing
    return dict(
        identities=listing.suffixes("identity/"),
        simulation=simulation if any(simulation.values()) else None,
        discovery=_discovery(file, listing),
        feedback={name: _feedback(file, listing, name, joins) for name in _FEEDBACK_ROUNDS},
        assembly=_assembly(file, listing, _ASSEMBLY, "identity/assembly", joins, refinement=True),
        painting=_PAINTING in listing, downstream=downstream, versions=versions)


def _discovery(file, listing):
    """Block tasks recorded out of the plan's count, once preparation is recorded."""
    done = max((int(suffix) for suffix in listing.suffixes(_DISCOVERY_BLOCKS)
                if suffix.isdigit()), default=0)
    prepared = _DISCOVERY_PREPARED in listing
    if not (prepared or done or _DISCOVERY_PLAN in listing):
        return None
    total = None
    if _DISCOVERY_PLAN in listing:
        total = int(file.get(_DISCOVERY_PLAN, items=("blocks",))["blocks"])
    return dict(prepared=prepared, blocks_done=done, blocks_total=total, complete=done == total)


def _feedback(file, listing, name, joins):
    """One feedback round: chunks of selected blocks until its selected record.

    Round identities can be bound before the round starts, so a round has
    started once it has records (its trial assembly's identity included).
    """
    prefix = f"feedback/{name}/"
    chunks = sum(suffix.isdigit() for suffix in listing.suffixes(prefix + "blocks/"))
    row = dict(started=bool(listing.under(prefix)), chunks=chunks,
               proposals=prefix + "proposals" in listing, done=prefix + "selected" in listing,
               trial=None)
    if name != "initial" and not (row["done"] or row["proposals"] or chunks):
        row["trial"] = _assembly(file, listing, prefix + "assembly/",
                                 f"identity/{prefix}assembly", joins, refinement=False)
        row["started"] = row["started"] or row["trial"] is not None
    return row


def _assembly(file, listing, prefix, identity, joins, *, refinement):
    """An assembly: per hierarchy level its joins, then (final) its refinement."""
    under = listing.under(prefix)
    if not under and identity not in listing:
        return None
    done = prefix + "final" in listing
    levels = sorted({int(match.group(1)) for key in under
                     if (match := _LEVEL.match(key, len(prefix)))})
    # A joined level followed by another without any refinement record: the
    # run's founder refinement is off, so joined levels are complete.
    refinement = refinement and not any(
        f"{prefix}l{level}/{_LEVEL_JOIN}" in listing
        and not listing.under(f"{prefix}l{level}/{_LEVEL_REFINE}") for level in levels[:-1])
    return dict(preprocessed=prefix + "preprocess" in listing, done=done, levels=[
        _level(file, listing, f"{prefix}l{level}/", level, joins, refinement=refinement,
               finished=done or level < levels[-1])
        for level in levels])


def _level(file, listing, base, level, joins, *, refinement, finished):
    """One level: plan, a joins/<digest> record per joined batch, the join
    manifest, then per component a refine/<n> record and the refine record.
    ``finished``: a later level or the assembly's final record exists."""
    refined, partial = set(), set()
    for key in listing.under(f"{base}{_LEVEL_REFINE}/"):
        match = _COMPONENT.fullmatch(key, len(base))
        if match:
            (refined if match.group(2) is None else partial).add(match.group(1))
    row = dict(level=level, state="started", batches=None, joins_done=None, joins_total=None,
               refine_done=len(refined), refine_total=None, refine_partial=len(partial - refined))
    joined = base + _LEVEL_JOIN in listing
    if base + _LEVEL_REFINE in listing or (joined and (finished or not refinement)):
        row["state"] = "done"
    elif joined:
        row["state"] = "refining"
        total = file.get(base + _LEVEL_JOIN, items=("components",)).get("components")
        row["refine_total"] = None if total is None else int(total)
    elif base + _LEVEL_PLAN in listing:
        row["state"] = "joining"
        batches = file.get(base + _LEVEL_PLAN, items=("batches",))["batches"]
        digests = [batch[2] for batch in batches if batch[2] is not None]
        row.update(batches=len(batches), joins_total=len(digests),
                   joins_done=sum(digest in joins for digest in digests))
    return row


def _work_states(directory, suffix):
    """Live transient resume states by contig: work/<contig>/<name><suffix>."""
    states = {}
    try:
        contigs = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    except FileNotFoundError:
        return states
    for contig in contigs:
        rows = []
        try:
            entries = sorted(os.scandir(os.path.join(directory, contig)),
                             key=lambda entry: entry.name)
        except FileNotFoundError:  # cleared meanwhile
            continue
        for entry in entries:
            if not entry.name.endswith(suffix):
                continue  # e.g. a replacement being written
            try:
                stat = entry.stat()
            except FileNotFoundError:
                continue
            rows.append(dict(name=entry.name[:-len(suffix)], size_bytes=stat.st_size,
                             modified_utc=_utc(stat.st_mtime)))
        if rows:
            states[contig] = rows
    return states


def _utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _natural_key(name):
    """chr2 before chr10."""
    return [int(part) if index % 2 else part
            for index, part in enumerate(re.split(r"(\d+)", name))]


def format_status(report):
    """The run_status report as text: run records, genome-wide stages, one line per contig."""
    records = report["records"]
    lines = [f"Run records: {len(records)} in {report['output']}/run_records"]
    for row in records:
        lines.append(f"  {Path(row['record']).stem}  {row['workflow']} {row['state']}  "
                     f"{row['host']} pid {row['pid']}  wall {_duration(row['wall_seconds'])}  "
                     f"updated {_age(row['updated_utc'])}")
        if row["error"]:
            lines.append(f"    error: {row['error']}")
        if row["active_stages"]:
            lines.append("    active: " + " > ".join(
                f"{stage['stage']}{'' if stage['contig'] is None else ' ' + stage['contig']} "
                f"{_duration(stage['elapsed_seconds'])}" for stage in row["active_stages"]))
        times = sorted(row["exclusive_stage_seconds"].items(), key=lambda item: -item[1])
        if times:
            lines.extend(_wrap("    stage times: ",
                               [f"{name} {_duration(seconds)}" for name, seconds in times]))
    if records:
        lines.append(f"  ({report['note']})")
    lines.extend(_checkpoint_lines(report["checkpoints"]))
    lines.extend(f"ERROR: {message}" for message in report["errors"])
    return "\n".join(lines)


def _checkpoint_lines(status):
    lines = [f"Checkpoints: {status['root']}"]
    if not status["exists"]:
        return [*lines, "  no checkpoint directory"]
    run = status["run"]
    if run is None:
        return lines  # unreadable layout: see the errors
    if "error" in run:
        lines.append("  run.ckpt  unreadable (error below)")
    elif run["file"] is None:
        lines.append("  run.ckpt  - (no genome-wide records yet)")
    else:
        lines += [f"  run.ckpt  {_size(run['size_bytes'])}, {run['records']} records, "
                  f"last write {_age(run['updated_utc'])}",
                  "  complete: " + _stage_names(run["complete"]),
                  "  pending: " + (" ".join(run["pending"]) or "-"),
                  "  identities: " + _stage_names(run["identities"]),
                  "  manifests: " + _stage_names(run["manifests"])]
    rows = status["contigs"]
    if not rows:
        return [*lines, "  no contig checkpoints"]
    lines.append(
        f"  {len(rows)} contigs: {sum(bool(row.get('file')) for row in rows.values())} with "
        f"checkpoints, {sum(bool((row.get('discovery') or {}).get('complete')) for row in rows.values())} "
        f"discovered, {sum(bool((row.get('assembly') or {}).get('done')) for row in rows.values())} "
        f"assembled, {sum(bool(row.get('painting')) for row in rows.values())} painted")
    cells = [(name, _size(row["size_bytes"]) if row.get("file") else "-",
              _age(row.get("updated_utc")), _contig_text(row)) for name, row in rows.items()]
    widths = [max(len(cell[column]) for cell in cells) for column in range(3)]
    lines += [f"  {name:<{widths[0]}}  {size:>{widths[1]}}  {age:>{widths[2]}}  {text}"
              for name, size, age, text in cells]
    return lines


def _contig_text(row):
    if "error" in row:
        return "unreadable checkpoint file (error below)"
    if row["file"] is None:
        parts = ["not started"]
    else:
        parts = [_simulation_text(row["simulation"])] if row["simulation"] else []
        parts += [_discovery_text(row["discovery"]),
                  "feedback " + _feedback_text(row["feedback"]),
                  "final " + _assembly_text(row["assembly"]),
                  "painting " + ("done" if row["painting"] else "-"),
                  _downstream_text(row["downstream"], row["versions"])]
    if row["work"]:
        parts.append("work: " + ", ".join(
            f"{state['name']} {_size(state['size_bytes'])} {_age(state['modified_utc'])}"
            for state in row["work"]))
    return " | ".join(parts)


def _simulation_text(simulation):
    if all(simulation.values()):
        return "simulation done"
    return "simulation " + " ".join(
        f"{name} {'done' if present else '-'}" for name, present in simulation.items())


def _discovery_text(discovery):
    if discovery is None:
        return "discovery -"
    if discovery["blocks_total"] is None:
        return "discovery prepared"
    return f"discovery {discovery['blocks_done']}/{discovery['blocks_total']} blocks"


def _feedback_text(rounds):
    states = [_round_text(row) for row in rounds.values()]
    if len(set(states)) == 1 and states[0] in ("done", "-"):
        return states[0]
    return " ".join(f"{name} {state}" for name, state in zip(rounds, states))


def _round_text(row):
    if row["done"]:
        return "done"
    if row["chunks"] or row["proposals"]:
        return f"{row['chunks']} chunk{'' if row['chunks'] == 1 else 's'}"
    if row["trial"] is not None:
        return "trial " + _assembly_text(row["trial"])
    return "started" if row["started"] else "-"


def _assembly_text(assembly):
    if assembly is None:
        return "-"
    if assembly["done"]:
        return "done"
    levels = assembly["levels"]
    if not levels:
        return "preprocessed" if assembly["preprocessed"] else "started"
    done = 0  # completed levels first, as one range: L1-L3 done
    while done < len(levels) and levels[done]["state"] == "done":
        done += 1
    parts = [] if not done else [
        f"L{levels[0]['level']}{'' if done == 1 else '-L' + str(levels[done - 1]['level'])} done"]
    return " ".join(parts + [_level_text(level) for level in levels[done:]])


def _level_text(level):
    if level["state"] == "joining":
        text = f"joins {level['joins_done']}/{level['joins_total']}"
    elif level["state"] == "refining":
        total = "?" if level["refine_total"] is None else level["refine_total"]
        text = f"refine {level['refine_done']}/{total}"
        if level["refine_partial"]:
            text += f" (+{level['refine_partial']} partial)"
    else:
        text = level["state"]
    return f"L{level['level']} {text}"


def _downstream_text(present, versions):
    if not any(present.values()):
        return "downstream -"
    if all(present.values()) and all(count <= 1 for count in versions.values()):
        return "downstream done"
    states = []
    for key, label in _DOWNSTREAM:
        name = key.rstrip("/")
        state = "done" if present[name] else "-"
        if versions.get(name, 0) > 1:
            state += f" ({versions[name]} versions)"
        states.append(f"{label} {state}")
    return "downstream: " + " ".join(states)


def _stage_names(names):
    """Names in pipeline order, versions of one stage grouped (e.g. 'pedigree/scores (2 versions)')."""
    counts = {}
    for name in names:
        stage = _unversioned(name)
        counts[stage] = counts.get(stage, 0) + 1
    order = {stage: index for index, stage in enumerate(_RUN_STAGES)}
    stages = sorted(counts, key=lambda stage: (order.get(stage, len(order)), stage))
    return " ".join(stage if counts[stage] == 1 else f"{stage} ({counts[stage]} versions)"
                    for stage in stages) or "-"


def _size(nbytes):
    """Bytes in binary units, e.g. 1.2 GiB."""
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            break
        value /= 1024
    else:
        unit = "TiB"
    if unit == "B":
        return f"{int(value)} B"
    return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"


def _duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _age(stamp):
    """Time since an ISO UTC time stamp, e.g. '3m ago'."""
    if not stamp:
        return "-"
    then = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    seconds = max(0., (datetime.now(timezone.utc) - then).total_seconds())
    for limit, scale, unit in ((90, 1, "s"), (90 * 60, 60, "m"), (48 * 3600, 3600, "h")):
        if seconds < limit:
            return f"{seconds / scale:.0f}{unit} ago"
    return f"{seconds / 86400:.0f}d ago"


def _wrap(prefix, items, width=110):
    """``prefix`` then comma-separated items, wrapped between items."""
    lines, current = [], []
    indent = " " * len(prefix)
    for item in items:
        lead = indent if lines else prefix
        if current and len(lead + ", ".join([*current, item])) > width:
            lines.append(lead + ", ".join(current) + ",")
            current = []
        current.append(item)
    lines.append((indent if lines else prefix) + ", ".join(current))
    return lines
