"""Independent pedigree preparation, genetic-score and decision checkpoints.

Versioned products are immutable. The conventional pedigree/_global is a
published current view; an older view is archived before replacement. These
are cache identities for trusted pipeline files, not integrity guarantees.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import os

from haplotype_reconstruction import PACKAGE_ROOT
from ..core import checkpoints
from . import components, eligibility, likelihoods


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
                                            "pedigree/candidates.py", "pedigree/evidence.py")})


def load_or_score(store, identity, prepare, settings, parent_eligibility, panel):
    """On replay, neither large sample paintings nor preparation tensors are read."""
    stage = versioned("pedigree_scores", identity)
    store.bind_stage_identity(stage, identity)
    if store.global_done(stage):
        payload = store.load_global(stage)
        return payload["scored"], payload["source_identities"], True
    prepared, sources = prepare()

    def chromosome_cache(request, producer):
        chrom_stage = f"{stage}/chromosomes/{request.chromosome_score_identity_sha256}"
        if store.contig_done(chrom_stage, request.contig):
            return store.load_contig(chrom_stage, request.contig)
        value = producer()
        store.save_contig(chrom_stage, request.contig, value)
        return value

    scored = likelihoods.score_prepared_parent_state_evidence(
        prepared, parent_eligibility=parent_eligibility, config=settings,
        top_k=panel.top_k, anchor_k=panel.anchor_k, use_anchor_union=panel.use_anchor_union,
        mismatch_penalty=panel.mismatch_penalty,
        candidate_source_mode=settings.parent_state_candidate_source_mode,
        chromosome_evidence_callback=chromosome_cache)
    # Runtime scorer tensors are not consumed by decision replay.
    scored = replace(scored, runtime_chromosome_results=None)
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
