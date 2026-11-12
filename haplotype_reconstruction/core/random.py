"""Name random streams with explicit seeds, generators and draw order.

Count bootstrap keys use default JSON spacing and SHA-256 first eight bytes,
not the canonical digest encoding. Refinement's key is
[first_position, last_position, B, incoming_K, label, kind]; joins use
('join_count', kind, sorted(seed_panel)). Completion folds use a deterministic
content-hash rule, with no random stream. NumPy streams are PCG64. The simulation structure
uses Python's Mersenne Twister seeded by a draw from its master. No name is hashed
into a seed: doing so would change the science. Callers retain each generator across
its sequence of draws; retries reconstruct it from the same semantic key.
"""

from __future__ import annotations

import hashlib
import json
import os
import random as _stdlib_random
from typing import Any

import numpy as np

from .parameters import PEDIGREE_BOOTSTRAP_SEED

__all__ = [
    "STREAMS",
    "bootstrap_seed",
    "realise_seed",
    "stream",
    "structure_stream",
    "child_seeds",
    "read_seeds",
    "resampled_individuals",
    "bootstrap_multiplicities",
]

_UNSET = object()

# Supported stream names; deterministic content-hash folds do not use a stream.
# There are no RandomState or explicitly constructed PCG64 streams.
STREAMS = (
    "assembly.count.bootstrap",  # count_rule/criterion.py: JSON decision key
    "pedigree.bootstrap",  # pedigree/resampling.py: replicate then contig
    "pedigree.relationship.reference",  # component then triple, on sorted sample identifiers
    "simulation.structure",  # Python MT via structure_stream(master), not stream()
    "simulation.master",  # pedigree.py: cohort/depth then contig then child
    "simulation.child",  # pedigree.py: mother's gamete then father's
    "simulation.founders",  # source.py: SeedSequence child per contig
    "simulation.reads",  # source.py: master producing one seed per contig
    "simulation.contig_reads",  # source.py: worker uses the already drawn seed
    "simulation.sample_reads",  # reads.py: one stream per sample from a seed drawn on the contig stream
    "simulation.sample_region",  # reads.py: one stream per sample for its regional depth factors
    "simulation.observation",  # design.py: subsampling after pedigree creation
    "templates.example",  # templates/example.py: SeedSequence child per contig
)


def bootstrap_seed(seed_key: Any) -> int:
    """Return the unsigned little-endian 64-bit seed of a JSON decision key.

    Refinement passes [first_position, last_position, B, incoming_K, label, kind];
    joins pass ('join_count', kind, sorted(seed_panel)) through the same stream.
    The same key must be reused by every member/refit/shadow of a count decision.
    """
    text = json.dumps(seed_key, sort_keys=True, default=str)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def stream(name: str, *, seed: int | None = None, key: Any = _UNSET, index: int = 0) -> np.random.Generator:
    """Construct a named PCG64 stream, without consuming any of its draws.

    ``seed`` is the original integer seed; pedigree defaults to 20260725. Count
    bootstrap accepts its JSON ``key`` or an already derived ``seed``. Founders
    and templates take the zero-based contig ``index`` in the bound contig order.
    Relationship references use seed + contig index and the component index in ``key``.
    Missing seeds and unknown names raise ValueError. No global state is mutated.
    """
    if name == "simulation.structure":
        raise ValueError("simulation.structure uses structure_stream(master), advancing the original master once")
    if name not in STREAMS:
        raise ValueError(f"unknown random stream {name!r}")
    if name == "assembly.count.bootstrap" and key is not _UNSET:
        if seed is not None:
            raise ValueError("count bootstrap requires either key or seed, not both")
        seed = bootstrap_seed(key)
    if name in ("pedigree.bootstrap", "pedigree.relationship.reference") and seed is None:
        seed = PEDIGREE_BOOTSTRAP_SEED
    if seed is None:
        raise ValueError(f"{name}: a realised seed is required")
    if name == "pedigree.relationship.reference":
        component = 0 if key is _UNSET else int(key)
        return np.random.default_rng([seed + int(index), component])
    if name == "simulation.reads":
        return np.random.default_rng(seed + 1_000_000)
    if name == "simulation.sample_reads":
        return np.random.default_rng(np.random.SeedSequence([int(seed), 4_000_000, int(index)]))
    if name == "simulation.sample_region":
        return np.random.default_rng(np.random.SeedSequence([int(seed), 5_000_000, int(index)]))
    if name == "simulation.observation":
        return np.random.default_rng(np.random.SeedSequence([int(seed), 3_000_000]))
    if name in ("simulation.founders", "templates.example"):
        entropy = [seed, 2_000_000] if name == "simulation.founders" else seed
        child = np.random.SeedSequence(entropy).spawn(index + 1)[index]
        return np.random.default_rng(child)
    return np.random.default_rng(seed)


def structure_stream(master: np.random.Generator) -> _stdlib_random.Random:
    """Draw simulation.structure's MT seed from the master, advancing it exactly once.

    Use for a generated pedigree only. An explicit pedigree starts drawing child
    seeds from the untouched master.
    """
    return _stdlib_random.Random(int(master.integers(0, 2**31)))


def child_seeds(master: np.random.Generator, contigs: int, children: int) -> np.ndarray:
    """Draw int64[C, children] seeds in contig-major order for one cohort/depth."""
    return master.integers(0, 2**63, size=(contigs, children), dtype=np.int64)


def read_seeds(seed: int, contigs: int) -> tuple[int, ...]:
    """Draw one simulation read seed per contig, including already completed contigs."""
    rng = stream("simulation.reads", seed=seed)
    return tuple(int(rng.integers(0, 2**63)) for _ in range(contigs))


def resampled_individuals(seed: int, resamples: int, samples: int) -> np.ndarray:
    """Return int64[R, N] count-bootstrap sample indices in the original draw order."""
    return stream("assembly.count.bootstrap", seed=seed).integers(0, samples, size=(resamples, samples))


def bootstrap_multiplicities(n_contigs: int, replicates: int = 1000, seed: int = PEDIGREE_BOOTSTRAP_SEED) -> np.ndarray:
    """Return float64[R, C] pedigree bootstrap counts; each row draws C contigs.

    Consume one vector per replicate so the generator's buffering and draw order
    are explicit. Inputs are not mutated.
    """
    rng = stream("pedigree.bootstrap", seed=seed)
    multiplicities = np.empty((replicates, n_contigs), dtype=np.float64)
    for replicate in range(replicates):
        draws = rng.integers(0, n_contigs, size=n_contigs)
        multiplicities[replicate] = np.bincount(draws, minlength=n_contigs).astype(np.float64)
    return multiplicities


def realise_seed(seed: int | None) -> int:
    """Resolve simulation's optional seed once at init from a supplied value or OS entropy.

    init calls realise_seed(configured_seed) once, persists the returned integer in
    the run manifest, and passes it to all science streams and retries. On resume,
    init reads that persisted seed rather than realizing another. Science never
    calls this boundary helper.

    A supplied integer is unchanged; otherwise use eight OS-random bytes interpreted
    little-endian. The stage must bind this value before any work or retry.
    """
    return seed if seed is not None else int.from_bytes(os.urandom(8), "little")
