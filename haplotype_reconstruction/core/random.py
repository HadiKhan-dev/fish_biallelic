"""Name the 3.5 random streams without changing their seeds, generators or draw order.

Count bootstrap keys use the original JSON spacing and SHA-256 first eight bytes,
not the canonical digest encoding. Refinement's key is
[first_position, last_position, B, incoming_K, label, kind]; joins use
('join_count', kind, sorted(seed_panel)). Completion folds are AS1's deterministic
content-hash rule, with no random stream. NumPy streams are PCG64. The simulation structure
uses Python's Mersenne Twister seeded by a draw from its master. No name is hashed
into a seed: doing so would change the science. Callers retain each generator across
its original sequence of draws; retries reconstruct it from the same semantic key.
"""

from __future__ import annotations

import hashlib
import json
import os
import random as _stdlib_random
from typing import Any

import numpy as np

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

# Live constructors in B; the uncalled deterministic_two_folds helper is excluded.
# There are no RandomState or explicitly constructed PCG64 streams.
STREAMS = (
    "assembly.count.bootstrap",  # count_rule/criterion.py: JSON decision key
    "pedigree.bootstrap",  # pedigree/resampling.py: replicate then contig
    "simulation.structure",  # Python MT via structure_stream(master), not stream()
    "simulation.master",  # pedigree.py: cohort/depth then contig then child
    "simulation.child",  # pedigree.py: mother's gamete then father's
    "simulation.founders",  # source.py: SeedSequence child per contig
    "simulation.reads",  # source.py: master producing one seed per contig
    "simulation.contig_reads",  # source.py: worker uses the already drawn seed
    "simulation.observation",  # design.py: subsampling after pedigree creation
    "templates.example",  # templates/example.py: SeedSequence child per contig
)


def bootstrap_seed(seed_key: Any) -> int:
    """Return the original unsigned little-endian 64-bit seed of a JSON decision key.

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
    if name == "pedigree.bootstrap" and seed is None:
        seed = 20260725
    if seed is None:
        raise ValueError(f"{name}: a realised seed is required")
    if name == "simulation.reads":
        return np.random.default_rng(seed + 1_000_000)
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
    seeds from the untouched master, as in 3.5 simulation/pedigree.py.
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


def bootstrap_multiplicities(n_contigs: int, replicates: int = 1000, seed: int = 20260725) -> np.ndarray:
    """Return float64[R, C] pedigree bootstrap counts; each row draws C contigs.

    Keep the replicate loop: consuming one vector per replicate is the documented
    3.5 protocol, including its generator buffering. Inputs are not mutated.
    """
    rng = stream("pedigree.bootstrap", seed=seed)
    multiplicities = np.empty((replicates, n_contigs), dtype=np.float64)
    for replicate in range(replicates):
        draws = rng.integers(0, n_contigs, size=n_contigs)
        multiplicities[replicate] = np.bincount(draws, minlength=n_contigs).astype(np.float64)
    return multiplicities


def realise_seed(seed: int | None) -> int:
    """Resolve simulation's optional seed once at init, retaining the 3.5 OS-entropy form.

    init calls realise_seed(configured_seed) once, persists the returned integer in
    the run manifest, and passes it to all science streams and retries. On resume,
    init reads that persisted seed rather than realizing another. Science never
    calls this boundary helper (ARCHITECTURE_40 §3.3).

    A supplied integer is unchanged; otherwise use eight OS-random bytes interpreted
    little-endian. The stage must bind this value before any work or retry.
    """
    return seed if seed is not None else int.from_bytes(os.urandom(8), "little")
