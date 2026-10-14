"""Versioned pedigree preparation, genetic-score and decision records.

Every product is stored under a key versioned by the digest of its identity
and is never rewritten. The run file's ``pedigree`` record names the current
decision (core.products.published_pedigree); earlier decisions stay readable
under their own versioned keys.
These are cache identities for trusted pipeline records, not integrity
guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from haplotype_reconstruction import PACKAGE_ROOT
from ..core import chromosome_parallel as scheduling
from ..core import products
from . import components, eligibility, execution


EVIDENCE = "pedigree/evidence"
SCORES = "pedigree/scores"
DECISION = "pedigree/decision"


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def versioned(section, identity):
    return f"{section}/{digest(identity)}"


def preparation_code_identity():
    # Version covers the small preparation adapter in pipeline.py. Decision
    # code/configuration must not invalidate this expensive product.
    names = ("pedigree/components.py", "pedigree/sources.py",
             "pedigree/transmission.py", "pedigree/models.py", "pedigree/candidates.py",
             "painting/checkpoints.py", "painting/evidence.py", "painting/model.py",
             "core/genetic_map.py", "core/runtime.py")
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
    section: str

    def __call__(self, request, producer):
        from ..core.runtime import CheckpointStore
        file = CheckpointStore(self.root).contig(request.contig)
        key = f"{self.section}/{request.chromosome_score_identity_sha256}"
        if file.has(key):
            return file.get(key, nthreads=scheduling.current_threads())
        value = producer()
        file.put(key, value, nthreads=scheduling.current_threads())
        return value


def load_or_score(store, identity, prepare, settings, parent_eligibility, panel, *, n_workers=None):
    """On replay, neither large sample paintings nor preparation tensors are read."""
    section = versioned(SCORES, identity)
    store.bind_identity(store.run, section, identity)
    if store.run.has(section):
        payload = store.run.get(section, nthreads=store.nthreads)
        return payload["scored"], payload["source_identities"], True
    source = prepare()
    scored, sources = execution.score_sources(
        source.layout, source.chromosomes, parent_eligibility=parent_eligibility, settings=settings,
        top_k=panel.top_k, anchor_k=panel.anchor_k, use_anchor_union=panel.use_anchor_union,
        mismatch_penalty=panel.mismatch_penalty,
        callback=ChromosomeScoreCache(store.root, section), n_workers=n_workers, retain_runtime=False)
    store.run.put(section, dict(scored=scored, source_identities=sources), nthreads=store.nthreads)
    return scored, sources, False


def publish_decision(store, key, payload):
    """Name ``key`` (a stored decision) as the run's current genome-wide pedigree.

    Read it back with core.products.published_pedigree.
    """
    current = dict(decision=key, identity_sha256=digest(payload["identity"]))
    if not store.run.has(products.PEDIGREE) or store.run.get(products.PEDIGREE) != current:
        store.run.put(products.PEDIGREE, current)
    store.mark_stage_complete(products.PEDIGREE)
