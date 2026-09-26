"""Independent pedigree preparation, genetic-score and decision checkpoints.

Versioned products are immutable. The conventional pedigree/_global is a
published current view; an older view is archived before replacement. These
are cache identities for trusted pipeline files, not integrity guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import os

from haplotype_reconstruction import PACKAGE_ROOT
from ..core import checkpoints
from ..core import chromosome_parallel as scheduling
from . import components, eligibility, execution


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def versioned(stage, identity):
    return f"{stage}/versions/{digest(identity)}"


def source_files(store, stages, contigs):
    """Bind normal atomic checkpoint replacement without reading large arrays."""
    result = []
    for stage in dict.fromkeys(stages):
        for contig in contigs:
            path = Path(checkpoints.contig_path(store.root, stage, contig)).resolve()
            stat = path.stat()
            result.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return result


def preparation_code_identity():
    # Version covers the small preparation adapter in pipeline.py. Decision
    # code/configuration must not invalidate this expensive product.
    names = ("pedigree/components.py", "pedigree/sources.py",
             "pedigree/transmission.py", "pedigree/models.py", "pedigree/candidates.py",
             "painting/checkpoints.py", "painting/evidence.py", "painting/model.py",
             "core/genetic_map.py", "core/raw_evidence.py", "core/runtime.py")
    return dict(adapter_version=1, files={
        name: hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest() for name in names})


def scoring_identity(preparation_identity, settings, parent_eligibility, sample_ids, panel):
    resolved = eligibility._resolve_parent_eligibility(parent_eligibility, sample_ids)
    return dict(schema="pedigree_result-genetic-cache-v1", preparation=preparation_identity,
                config=components._score_config_identity(settings),
                eligibility=components._eligibility_score_identity(resolved),
                panel=dict(top_k=panel.top_k, anchor_k=panel.anchor_k,
                           use_anchor_union=panel.use_anchor_union,
                           mismatch_penalty=panel.mismatch_penalty),
                code=components.pedigree_evidence_scoring_code_identity(),
                adapter_files={name: hashlib.sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
                               for name in ("pedigree/components.py", "pedigree/likelihoods.py",
                                            "pedigree/execution.py", "core/chromosome_parallel.py",
                                            "pedigree/candidates.py", "pedigree/evidence.py")})


@dataclass(frozen=True)
class ChromosomeScoreCache:
    """Picklable per-chromosome cache; workers never publish global completion."""

    root: str
    stage: str

    def __call__(self, request, producer):
        from ..core.runtime import CheckpointStore
        store = CheckpointStore(self.root, nthreads=scheduling.current_threads())
        stage = f"{self.stage}/chromosomes/{request.chromosome_score_identity_sha256}"
        if store.contig_done(stage, request.contig):
            return store.load_contig(stage, request.contig)
        value = producer()
        store.nthreads = scheduling.current_threads()
        store.save_contig(stage, request.contig, value)
        return value


def load_or_score(store, identity, prepare, settings, parent_eligibility, panel, *, n_workers=None):
    """On replay, neither large sample paintings nor preparation tensors are read."""
    stage = versioned("pedigree_scores", identity)
    store.bind_stage_identity(stage, identity)
    if store.global_done(stage):
        payload = store.load_global(stage)
        return payload["scored"], payload["source_identities"], True
    source = prepare()
    scored, sources = execution.score_sources(
        source.layout, source.chromosomes, parent_eligibility=parent_eligibility, settings=settings,
        top_k=panel.top_k, anchor_k=panel.anchor_k, use_anchor_union=panel.use_anchor_union,
        mismatch_penalty=panel.mismatch_penalty,
        callback=ChromosomeScoreCache(store.root, stage), n_workers=n_workers, retain_runtime=False)
    store.save_global(stage, dict(scored=scored, source_identities=sources))
    store.mark_stage_complete(stage)
    return scored, sources, False


def publish_decision(store, stage, payload):
    """Publish a current view while retaining any previous scientific result."""
    replace_payload = True
    if store.global_done(stage):
        previous = store.load_global(stage)
        replace_payload = previous.get("identity") != payload["identity"]
        if replace_payload:
            archive = versioned(f"{stage}/previous", previous["identity"])
            if not store.global_done(archive):
                store.save_global(archive, previous)
                store.mark_stage_complete(archive)
    if replace_payload:
        store.save_global(stage, payload)
    target = Path(store.stage_dir(stage)) / "_identity.json"
    descriptor, temporary = tempfile.mkstemp(prefix="_identity.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload["identity"], handle, sort_keys=True)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    store.mark_stage_complete(stage)
