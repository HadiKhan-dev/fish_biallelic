"""Small per-invocation provenance and non-overlapping wall-time accounting.

Only the coordinator writes the record. Worker CPU time is not confused with
elapsed time; nested stage durations report exclusive time as well as totals.
Records describe completed work and resume overhead, not hypothetical cold runs.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from importlib import metadata
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import threading

_ACTIVE = None
SETTINGS = (
    "BHD_NUM_PROCESSES", "NUMBA_NUM_THREADS", "NUMBA_CACHE_DIR", "HAPLOTYPES_BATCH_QUEUE",
    "BHD_RECOMBINATION_RATE_CM_PER_MB",
    "BHD_RECOMBINATION_MAP", "BHD_RECOMBINATION_SHARED_FAMILY",
    "HAPLOTYPES_OUTPUT_DIR", "HAPLOTYPES_CHECKPOINT_DIR", "HAPLOTYPES_CONTIGS",
    "HAPLOTYPES_VCF", "HAPLOTYPES_TEMPLATES", "HAPLOTYPES_METADATA",
    "HAPLOTYPES_METADATA_SHEET", "HAPLOTYPES_ELIGIBILITY",
    "HAPLOTYPES_ASSEMBLY_MODEL", "HAPLOTYPES_ASSEMBLY_SEARCH",
    "HAPLOTYPES_FOUNDER_REFINEMENT", "HAPLOTYPES_DISCOVERY_SEARCH",
    "HAPLOTYPES_FEEDBACK_SELECTION", "HAPLOTYPES_STOP_AFTER_STAGE",
    "HAPLOTYPES_READ_CALIBRATION",
    "HAPLOTYPES_PEDIGREE_CALIBRATION",
    "BHD_SIMULATION_SEED", "BHD_SIM_READ_DEPTH", "HAPLOTYPES_GENERATIONS",
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
            self.data["checkpoint_identities"] = {
                path.parent.name: json.loads(path.read_text())
                for path in sorted(Path(root).glob("*/_identity.json"))}
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


def run_status(output):
    """Inspect progress without loading large scientific checkpoints."""
    directory = Path(output)
    records = []
    for path in sorted((directory/"run_records").glob("*.json")):
        data = json.loads(path.read_text())
        records.append(dict(record=str(path), workflow=data["workflow"], state=data["state"],
                            host=data["host"], pid=data["pid"], wall_seconds=data["wall_seconds"],
                            updated_utc=data['updated_utc'],
                            exclusive_stage_seconds={name: sum(row['exclusive_seconds'] for row in data['stages'] if row['stage']==name)
                                                     for name in sorted({row['stage'] for row in data['stages']})},
                            active_stages=data["active_stages"]))
    return dict(records=records,
                note="running is the last persisted state; a killed process cannot publish completion. "
                     "No process liveness or Slurm state is inferred.")
