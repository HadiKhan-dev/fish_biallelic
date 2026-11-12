"""Stock pedigrees and mutation-free label meioses on the full contig extent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from ..contracts.roles import freeze_record_arrays, record_field, schema_field
from ..core.errors import InputError
from .haplotypes import Mosaic, founder_mosaic, transmit
from .pedigree import _map_crossovers


def stock_stream(name: str, seed: int, *key) -> np.random.Generator:
    """Independent PCG64 streams with stable UTF-8 semantic keys.

    A SHA-256 JSON key supplies eight uint32 entropy words after the realised seed.
    Existing released stream names and their draw order are untouched.
    """
    if name not in (
        "simulation.stock",
        "simulation.stock.gamete",
        "simulation.source",
        "simulation.sites",
        "simulation.profile",
    ):
        raise ValueError(f"unknown generating stream {name!r}")
    digest = hashlib.sha256(json.dumps([name, *key], ensure_ascii=True, separators=(",", ":")).encode()).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, len(digest), 4)]
    return np.random.default_rng(np.random.SeedSequence([int(seed), *words]))


@dataclass(frozen=True)
class StockRecord:
    """Parents index the previous generation; taken indexes the last generation."""

    name: str = record_field("obs", dtype="str")
    population: str = record_field("obs", dtype="str")
    sizes: np.ndarray = record_field("obs", axes=("generation",), units="individuals", dtype="int32")
    parents: np.ndarray = record_field("obs", axes=("child", "2"), units="index", dtype="int32")
    first_source: int = record_field("obs", units="index", dtype="int")
    taken: np.ndarray = record_field("obs", axes=("founder",), units="index", dtype="int32")

    schema: int = schema_field(1)

    def __post_init__(self):
        for name in ("sizes", "parents", "taken"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), np.int32))
        if self.sizes.ndim != 1 or not len(self.sizes) or np.any(self.sizes < 2):
            raise InputError("stock generation sizes must be at least two individuals")
        if self.parents.shape != (int(self.sizes[1:].sum()), 2):
            raise InputError("stock parent axis does not match its generation sizes")
        first = 0
        for previous, size in zip(self.sizes[:-1], self.sizes[1:], strict=True):
            pairs = self.parents[first : first + size]
            if np.any(pairs < 0) or np.any(pairs >= previous) or np.any(pairs[:, 0] == pairs[:, 1]):
                raise InputError("stock children require two distinct previous-generation parents")
            first += size
        if (
            self.taken.ndim != 1
            or len(np.unique(self.taken)) != len(self.taken)
            or np.any(self.taken < 0)
            or np.any(self.taken >= self.sizes[-1])
        ):
            raise InputError("stock founders must be distinct last-generation individuals")
        freeze_record_arrays(self)

    def record(self):
        return dict(
            schema=self.schema,
            name=self.name,
            population=self.population,
            sizes=self.sizes,
            parents=self.parents,
            first_source=self.first_source,
            taken=self.taken,
        )


def stock_record(name, specification, taken: int, first_source: int, seed: int) -> StockRecord:
    """Draw only the stock structure and selection, once for all contigs."""
    available = specification.size if specification.generations else specification.founders
    if taken < 0 or taken > available:
        raise InputError(f"stock {name}: requested {taken} individuals, available {available}")
    rng = stock_stream("simulation.stock", seed, name)
    sizes = np.asarray([specification.founders] + [specification.size] * specification.generations, np.int32)
    parents = np.empty((int(sizes[1:].sum()), 2), np.int32)
    first = 0
    for previous, size in zip(sizes[:-1], sizes[1:], strict=True):
        for child in range(int(size)):
            parents[first + child] = rng.choice(int(previous), 2, replace=False)
        first += int(size)
    selected = rng.choice(available, taken, replace=False).astype(np.int32)
    return StockRecord(name, specification.population, sizes, parents, first_source, selected)


def stock_gamete(parent, extent, chromosome_map, rng) -> Mosaic:
    """Released crossover draws, without mutation draws, on labels."""
    first = int(rng.choice([0, 1]))
    sites = np.array([extent.start_bp, extent.end_bp], np.float64)
    if chromosome_map.has_map:
        events = _map_crossovers(sites, chromosome_map, rng)
    else:
        events = []
        rate = chromosome_map.fallback_rate_per_bp
        current = extent.start_bp
        if rate > 0:
            while current <= extent.end_bp:
                current += np.ceil(rng.exponential(1.0 / rate))
                if current <= extent.end_bp:
                    events.append(float(current))
        events = np.asarray(events, np.float64)
    return transmit(parent, first, events, np.empty(0, np.int64), None)


def stock_founders(record: StockRecord, extent, chromosome_map, seed: int, contig: str):
    """Retain only two successive generations of mosaics; return taken individuals."""
    previous = [
        (
            founder_mosaic(extent.start_bp, record.first_source + 2 * i, -1),
            founder_mosaic(extent.start_bp, record.first_source + 2 * i + 1, -1),
        )
        for i in range(int(record.sizes[0]))
    ]
    offset = 0
    for generation, size in enumerate(record.sizes[1:], 1):
        current = []
        for child in range(int(size)):
            rng = stock_stream("simulation.stock.gamete", seed, record.name, generation, child, contig)
            parents = record.parents[offset + child]
            current.append(tuple(stock_gamete(previous[p], extent, chromosome_map, rng) for p in parents))
        previous = current
        offset += int(size)
    return [previous[int(i)] for i in record.taken]
