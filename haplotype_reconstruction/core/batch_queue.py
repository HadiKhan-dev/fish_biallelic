"""Opt-in shared-filesystem queue for independent scientific batch bundles.

Only the chromosome coordinator publishes ordinary pipeline checkpoints. Helpers
return immutable results under attempt-specific names; a renewable claim fences
late writers after a worker interruption. No scheduler submission happens here.

Lifecycle: map_batches publishes inputs -> serve_one claims and executes a
bundle -> the coordinator gathers results and publishes scientific checkpoints.
batch_tasks owns array transport and the stage-specific callback adapters.
A stage is active, complete or stopped; its bundles are ready, running, done or
failed. Only the current claim token may publish a bundle's terminal result.
"""
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import traceback
import uuid

from . import checkpoints

LEASE_SECONDS = 600


def configured():
    """Return the opt-in shared transport directory, or None for local work."""
    value = os.environ.get('HAPLOTYPES_BATCH_QUEUE')
    return Path(value) if value else None


@lru_cache(maxsize=1)
def source_identity():
    """Require helpers and coordinators to use the same frozen package source."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_key(store, phase):
    """Reuse queue results only under an existing exact scientific identity."""
    if store is None or not hasattr(store, '_identity_json'):
        return None
    value = (store._identity_json, store.work_stage, store.contig, phase)
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def _write_json(path, data):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def _state(root):
    """Serialize short metadata updates; numerical work never holds this lock."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = root / 'queue.json'
        data = json.loads(path.read_text()) if path.exists() else dict(stages={}, bundles={})
        try:
            yield data
        finally:
            _write_json(path, data)
            fcntl.flock(handle, fcntl.LOCK_UN)


def _claim(root, stage=None):
    """Claim ready work, or reclaim a bundle whose worker stopped heartbeating."""
    with _state(root) as state:
        now = time.time()
        for key, record in state['bundles'].items():
            if stage is not None and record['stage'] != stage:
                continue
            parent = state['stages'][record['stage']]
            if parent.get('owner_expiry') and now > parent['owner_expiry']:
                parent['state'] = 'stopped'
            if parent['state'] != 'active':
                continue
            if record['state'] == 'running' and now - record['heartbeat'] > LEASE_SECONDS:
                record['state'] = 'ready'
            if record['state'] != 'ready':
                continue
            if parent['source'] != source_identity():
                raise RuntimeError('Batch worker source differs from its coordinator')
            token = uuid.uuid4().hex
            record.update(
                state='running', token=token, heartbeat=now,
                host=socket.gethostname(), pid=os.getpid(),
                job=os.environ.get('SLURM_JOB_ID'), started=now,
            )
            return key, dict(record)
    return None


def serve_one(root, cores, *, stage=None):
    """Use this allocation for one ready bundle; return False when none exist."""
    root = Path(root)
    claim = _claim(root, stage)
    if claim is None:
        return False
    key, record = claim
    token = record['token']
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(20):
            with _state(root) as state:
                current = state['bundles'][key]
                if current.get('token') != token:
                    return
                current['heartbeat'] = time.time()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    result_path = root / record['stage'] / f'{record["number"]:06d}.{token}.result.p5.b2'
    started = time.monotonic()
    try:
        from .batch_tasks import execute
        payload = checkpoints.read(str(root / record['input']), nthreads=cores)
        results = execute(payload, cores)
        checkpoints.write(str(result_path), results, nthreads=cores)
        with _state(root) as state:
            current = state['bundles'][key]
            if current.get('token') == token:
                current.update(
                    state='done', result=str(result_path.relative_to(root)),
                    seconds=time.monotonic() - started, cores=cores,
                    finished=time.time(),
                )
    except BaseException:
        with _state(root) as state:
            current = state['bundles'][key]
            if current.get('token') == token:
                current.update(
                    state='failed', error=traceback.format_exc(),
                    seconds=time.monotonic() - started, cores=cores,
                )
        raise
    finally:
        stop.set()
        thread.join()
    return True


def map_batches(kind, tasks, arrays, cores, *, key=None, worker_memory_gb=4.):
    """Yield original task results, with the coordinator helping its own queue.

    Inputs are shipped as contiguous genomic slices, never reread as a whole
    chromosome per task. Bundle sizing is purely operational: task boundaries,
    observation masks, positions, model parameters and result IDs are preserved.
    """
    from .batch_tasks import bundles
    root = configured()
    if root is None:
        raise RuntimeError('No shared batch queue configured')
    stage = key or uuid.uuid4().hex
    directory = root / stage
    directory.mkdir(parents=True, exist_ok=True)
    with _state(root) as state:
        existing = state['stages'].get(stage)
        if existing and existing['source'] != source_identity():
            raise RuntimeError('Queued stage belongs to different source code')
        state['stages'][stage] = dict(
            source=source_identity(), state='active', kind=kind,
            owner_job=os.environ.get('SLURM_JOB_ID'),
            owner_expiry=int(os.environ.get('SLURM_JOB_END_TIME', '0')),
            host=socket.gethostname(), pid=os.getpid(),
        )
    keys = []
    try:
        for number, payload in enumerate(bundles(kind, tasks, arrays, worker_memory_gb)):
            task_key = f'{stage}/{number:06d}'
            keys.append(task_key)
            with _state(root) as state:
                exists = task_key in state['bundles']
            if exists:
                continue
            filename = directory / f'{number:06d}.input.p5.b2'
            checkpoints.write(str(filename), payload, nthreads=cores)
            with _state(root) as state:
                state['bundles'][task_key] = dict(
                    stage=stage, number=number, state='ready',
                    input=str(filename.relative_to(root)), tasks=len(payload['tasks']),
                )
        print(f'[Cross-node {kind}] published {len(keys)} bundles / {len(tasks)} tasks', flush=True)
        pending = set(keys)
        while pending:
            with _state(root) as state:
                records = {
                    item: dict(state['bundles'][item])
                    for item in keys if item in pending
                }
            for item, record in records.items():
                if record['state'] == 'failed':
                    raise RuntimeError(f'Cross-node batch failed: {item}\n{record["error"]}')
                if record['state'] == 'done':
                    for result in checkpoints.read(str(root / record['result']), nthreads=cores):
                        yield result
                    pending.remove(item)
            if pending and not serve_one(root, cores, stage=stage):
                # The coordinator can help another chromosome while its own
                # final bundles are already running on other allocations.
                if not serve_one(root, cores):
                    time.sleep(2)
        with _state(root) as state:
            state['stages'][stage]['state'] = 'complete'
    except BaseException:
        with _state(root) as state:
            state['stages'][stage]['state'] = 'stopped'
        raise


def worker(root, cores, idle_seconds=120):
    """Helper CLI: reuse one allocation, then exit after bounded queue idleness."""
    from .runtime import available_cpu_count
    if not 1 <= cores <= available_cpu_count():
        raise ValueError('Batch worker cores must fit its CPU affinity')
    last_work = time.monotonic()
    while time.monotonic() - last_work < idle_seconds:
        expiry = int(os.environ.get('SLURM_JOB_END_TIME', '0'))
        if expiry and time.time() > expiry - 120:
            break
        if serve_one(root, cores):
            last_work = time.monotonic()
        else:
            time.sleep(2)
