"""Pure call-set transformations of sampled REF/ALT counts, in generating-model order.

Heterozygote balance, shared systematic errors, independent miss causes, then
reference blocks. No read sampler, global random state or truth store is touched.
The complete contig site axis is required; any partition/order of the sample
axis gives identical results when stable sample identifiers are retained.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..contracts.roles import freeze_record_arrays, record_field, schema_field
from ..core.errors import InputError, SettingsError
from . import parameters as p

__all__ = [
    "ArtefactRegion", "ArtefactModel", "ArtefactSites", "ArtefactResult", "artefact_stream",
    "draw_site_artefacts", "draw_stretches", "miss_probabilities", "reference_blocks", "apply_artefacts",
]


@dataclass(frozen=True)
class ArtefactRegion:
    """A half-open bp interval [start, end) multiplying the three miss terms.

    Regions on the same contig must not overlap; no overlap-combination law is
    specified. Factors may exceed one; each resulting miss term is clipped.
    """

    contig: str = record_field("obs", dtype="str")
    start: int = record_field("obs", units="bp", dtype="int64")
    end: int = record_field("obs", units="bp", dtype="int64")
    factor: float = record_field("obs", dtype="float64")
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        if not isinstance(self.contig, str) or not self.contig:
            raise SettingsError("artefact region contig must be a nonempty identifier")
        if not all(isinstance(v, (int, np.integer)) for v in (self.start, self.end)) or not 0 <= self.start < self.end:
            raise SettingsError("artefact region needs integer bp coordinates with 0 <= start < end")
        if not math.isfinite(self.factor) or self.factor < 0:
            raise SettingsError("artefact region factor must be finite and nonnegative")


@dataclass(frozen=True)
class ArtefactModel:
    """Generating call-set parameters; numerical defaults are the reference_blocks preset.

    Disabled by default. Probabilities/fractions and CVs are dimensionless;
    stretch_sites is a geometric mean in sites. heterozygote_alt_probability
    belongs to the generating read model and is supplied to apply_artefacts.
    """

    enabled: bool = record_field("obs", dtype="bool", default=p.ARTEFACT_ENABLED)
    hard_sites: float = record_field("obs", dtype="float64", default=p.ARTEFACT_HARD_SITES)
    hard_miss: float = record_field("obs", dtype="float64", default=p.ARTEFACT_HARD_MISS)
    stretch_cover: float = record_field("obs", dtype="float64", default=p.ARTEFACT_STRETCH_COVER)
    stretch_sites: float = record_field("obs", units="sites", dtype="float64", default=p.ARTEFACT_STRETCH_SITES)
    stretch_miss: float = record_field("obs", dtype="float64", default=p.ARTEFACT_STRETCH_MISS)
    miss: float = record_field("obs", dtype="float64", default=p.ARTEFACT_MISS)
    miss_sample_cv: float = record_field("obs", dtype="float64", default=p.ARTEFACT_MISS_SAMPLE_CV)
    call_min_alt_fraction: float = record_field("obs", dtype="float64", default=p.ARTEFACT_CALL_MIN_ALT_FRACTION)
    block_band: float = record_field("obs", dtype="float64", default=p.ARTEFACT_BLOCK_BAND)
    balance_dispersion: float = record_field("obs", dtype="float64", default=p.ARTEFACT_BALANCE_DISPERSION)
    systematic_sites: float = record_field("obs", dtype="float64", default=p.ARTEFACT_SYSTEMATIC_SITES)
    systematic_lo: float = record_field("obs", dtype="float64", default=p.ARTEFACT_SYSTEMATIC_LO)
    systematic_hi: float = record_field("obs", dtype="float64", default=p.ARTEFACT_SYSTEMATIC_HI)
    regions: tuple[ArtefactRegion, ...] = record_field("obs", dtype="tuple[ArtefactRegion, ...]", default=())
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise SettingsError("artefact enabled must be a boolean")
        for name in (
            "hard_sites", "hard_miss", "stretch_cover", "stretch_miss", "miss", "call_min_alt_fraction",
            "systematic_sites", "systematic_lo", "systematic_hi",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise SettingsError(f"artefact {name} must be between zero and one")
        if not 0 <= self.balance_dispersion < 1:
            raise SettingsError("artefact balance_dispersion must be in [0, 1)")
        if not math.isfinite(self.miss_sample_cv) or self.miss_sample_cv < 0:
            raise SettingsError("artefact miss_sample_cv must be finite and nonnegative")
        if not math.isfinite(self.block_band) or self.block_band <= 0:
            raise SettingsError("artefact block_band must be finite and positive")
        if not math.isfinite(self.stretch_sites) or self.stretch_sites < 1:
            raise SettingsError("artefact stretch_sites must be a finite mean of at least one site")
        if self.stretch_cover and (
            self.stretch_cover == 1 or self.stretch_sites * (1 - self.stretch_cover) / self.stretch_cover < 1
        ):
            raise SettingsError("artefact stretch_cover and stretch_sites require a geometric gap mean >= 1")
        if self.systematic_lo > self.systematic_hi:
            raise SettingsError("artefact systematic_lo must not exceed systematic_hi")
        regions = tuple(r if isinstance(r, ArtefactRegion) else ArtefactRegion(*r) for r in self.regions)
        ordered = sorted(regions, key=lambda r: (r.contig, r.start))
        if any(a.contig == b.contig and a.end > b.start for a, b in zip(ordered, ordered[1:])):
            raise SettingsError("artefact regions on the same contig must not overlap")
        object.__setattr__(self, "regions", regions)

    @classmethod
    def preset(cls, name: str = p.ARTEFACT_PRESET) -> ArtefactModel:
        """Enable the named preset, without changing the generating read model."""
        if name != p.ARTEFACT_PRESET:
            raise SettingsError(f"unknown artefact preset {name!r}")
        return cls(enabled=True)


@dataclass(frozen=True)
class ArtefactSites:
    """Contig-shared causes and rates, independent of sample membership."""

    hard: NDArray[np.bool_] = record_field("obs", axes=("L",), dtype="bool")
    systematic_site: NDArray[np.bool_] = record_field("obs", axes=("L",), dtype="bool")
    systematic_rate: NDArray[np.float64] = record_field("obs", axes=("L",), dtype="float64")
    region_factor: NDArray[np.float64] = record_field("obs", axes=("L",), dtype="float64")
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


@dataclass(frozen=True)
class ArtefactResult:
    """Transformed counts and the three truth masks (all absent when disabled)."""

    counts: NDArray[np.int32] = record_field("obs", axes=("N", "L", "2"), units="reads", dtype="int32")
    missed: NDArray[np.bool_] | None = record_field("obs", axes=("N", "L"), dtype="bool | None")
    block: NDArray[np.bool_] | None = record_field("obs", axes=("N", "L"), dtype="bool | None")
    systematic_site: NDArray[np.bool_] | None = record_field("obs", axes=("L",), dtype="bool | None")
    schema: int = schema_field(1)

    def __post_init__(self) -> None:
        freeze_record_arrays(self)


def artefact_stream(name: str, seed: int, contig: str, sample: str | None = None) -> np.random.Generator:
    """PCG64 stream keyed by stable identifiers, never by row or Python hash().

    The SHA-256 digest of a UTF-8 JSON array [name, contig] or [name, sample,
    contig] supplies eight little-endian uint32 entropy words after the realised
    seed. This local catalogue adds no names or draws to existing read streams.
    """
    if name not in (p.ARTEFACT_SITE_STREAM, p.ARTEFACT_SAMPLE_STREAM):
        raise ValueError(f"unknown artefact random stream {name!r}")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise InputError("artefacts require a realised nonnegative integer seed")
    if not isinstance(contig, str) or not contig:
        raise InputError("artefacts require a nonempty contig identifier")
    if name == p.ARTEFACT_SAMPLE_STREAM:
        if not isinstance(sample, str) or not sample:
            raise InputError("artefact sample stream requires a nonempty sample identifier")
        key = [name, sample, contig]
    else:
        if sample is not None:
            raise InputError("artefact site stream does not take a sample identifier")
        key = [name, contig]
    digest = hashlib.sha256(json.dumps(key, ensure_ascii=True, separators=(",", ":")).encode("utf-8")).digest()
    entropy = [int(seed), *[int.from_bytes(digest[i:i + 4], "little") for i in range(0, len(digest), 4)]]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def draw_site_artefacts(positions: NDArray[np.int64], seed: int, contig: str, model: ArtefactModel) -> ArtefactSites:
    """Draw independent hard/systematic sites and shared uniform systematic rates."""
    rng = artefact_stream(p.ARTEFACT_SITE_STREAM, seed, contig)
    sites = len(positions)
    hard = rng.random(sites) < model.hard_sites
    systematic = rng.random(sites) < model.systematic_sites
    rate = rng.uniform(model.systematic_lo, model.systematic_hi, sites)
    factor = np.ones(sites, dtype=np.float64)
    for region in model.regions:
        if region.contig == contig:
            factor[(positions >= region.start) & (positions < region.end)] = region.factor
    return ArtefactSites(hard, systematic, rate, factor)


def draw_stretches(sites: int, model: ArtefactModel, rng: np.random.Generator) -> NDArray[np.bool_]:
    """Alternating geometric gaps/lengths, starting with a gap at site zero.

    Both geometric laws have support 1, 2, ... . The terminal interval is
    censored at the contig boundary, without imposing stationarity there.
    """
    covered = np.zeros(sites, dtype=bool)
    if not model.stretch_cover or not sites:
        return covered
    gap_probability = model.stretch_cover / model.stretch_sites / (1 - model.stretch_cover)
    if gap_probability == 0:
        # A probability below the floating-point range cannot open a stretch
        # within the finite, representable site axis.
        return covered
    offset = 0
    while offset < sites:
        gaps = rng.geometric(gap_probability, p.ARTEFACT_RENEWAL_BATCH)
        lengths = rng.geometric(1 / model.stretch_sites, p.ARTEFACT_RENEWAL_BATCH)
        # Only the finite contig is observed. Censor waits beyond its length;
        # NumPy encodes geometric draws beyond int64 as a negative integer.
        gaps = np.where((gaps < 1) | (gaps > sites), sites, gaps)
        lengths = np.where((lengths < 1) | (lengths > sites), sites, lengths)
        ends = offset + np.cumsum(gaps + lengths)
        starts = ends - lengths
        for start, end in zip(starts[starts < sites], ends[starts < sites]):
            covered[start:min(end, sites)] = True
        offset = int(ends[-1])
    return covered


def miss_probabilities(hard, stretch, region_factor, model: ArtefactModel, sample_factor: float) -> NDArray[np.float64]:
    """Combine independently acting causes, clipping each region-scaled term first."""
    h = np.clip(model.hard_miss * np.asarray(hard) * region_factor, 0, 1)
    t = np.clip(model.stretch_miss * np.asarray(stretch) * region_factor, 0, 1)
    b = np.clip(model.miss * sample_factor * region_factor, 0, 1)
    return 1 - (1 - h) * (1 - t) * (1 - b)


def reference_blocks(counts: NDArray[np.int32], uncalled: NDArray[np.bool_], block_band: float) -> NDArray[np.int32]:
    """One sample's maximal equal-band uncalled runs report (minimum depth, 0).

    Input counts are copied. Called cells, including zero-depth non-reference
    cells that were not missed, cut a block and retain their counts.
    """
    if not math.isfinite(block_band) or block_band <= 0:
        raise SettingsError("artefact block_band must be finite and positive")
    result = counts.copy()
    sites = len(counts)
    if not sites:
        return result
    depth = counts.sum(axis=1, dtype=np.int64)
    band = np.full(sites, -1, dtype=np.int64)
    positive = depth > 0
    if block_band < 1 / max(1, int(depth.max())):
        # At this width distinct integer depths necessarily have different
        # bands. Comparing depths avoids overflowing unbounded band indices.
        band[positive] = depth[positive]
    else:
        band[positive] = np.floor(np.log(depth[positive]) / np.log1p(block_band)).astype(np.int64)
    boundaries = np.ones(sites, dtype=bool)
    boundaries[1:] = ~(uncalled[1:] & uncalled[:-1] & (band[1:] == band[:-1]))
    starts = np.flatnonzero(boundaries)
    widths = np.diff(np.append(starts, sites))
    minimum = np.repeat(np.minimum.reduceat(depth, starts), widths)
    result[uncalled, 0] = minimum[uncalled]
    result[uncalled, 1] = 0
    return result


def apply_artefacts(
    dosage: NDArray[np.int8],
    counts: NDArray[np.int32],
    positions: NDArray[np.int64],
    seed: int,
    contig: str,
    sample_ids: Sequence[str],
    model: ArtefactModel,
    *,
    heterozygote_alt_probability: float,
) -> ArtefactResult:
    """Apply the four frozen equations on a full contig and any sample subset.

    Inputs are never written; outputs are frozen. pi is explicitly supplied by
    the generating read model. Off state returns the original count values and
    absent masks without constructing or consuming any artefact stream.
    """
    if dosage.dtype != np.int8 or dosage.ndim != 2:
        raise InputError("artefact dosage must be int8[N, L]")
    samples, sites = dosage.shape
    if counts.dtype != np.int32 or counts.shape != (samples, sites, 2):
        raise InputError("artefact counts must be int32[N, L, 2] aligned with dosage")
    if positions.dtype != np.int64 or positions.shape != (sites,) or np.any(positions[1:] <= positions[:-1]):
        raise InputError("artefact positions must be strictly increasing int64[L]")
    if np.any((dosage < 0) | (dosage > 2)) or np.any(counts < 0):
        raise InputError("artefact dosage must be in {0, 1, 2} and counts must be nonnegative")
    if len(sample_ids) != samples or any(not isinstance(s, str) or not s for s in sample_ids):
        raise InputError("artefact sample identifiers must be nonempty and aligned with dosage")
    if len(set(sample_ids)) != samples:
        raise InputError("artefact sample identifiers must be unique")
    if not 0 < heterozygote_alt_probability < 1:
        raise InputError("artefact heterozygote_alt_probability must be in (0, 1)")
    if not model.enabled:
        return ArtefactResult(counts.copy(), None, None, None)
    shared = draw_site_artefacts(positions, seed, contig, model)
    result = counts.copy()
    missed = np.zeros((samples, sites), dtype=bool)
    block = np.zeros_like(missed)
    for s, sample in enumerate(sample_ids):
        rng = artefact_stream(p.ARTEFACT_SAMPLE_STREAM, seed, contig, sample)
        depth = counts[s].sum(axis=1, dtype=np.int64)
        if np.any(depth > np.iinfo(np.int32).max):
            raise InputError("artefact total cell depth must fit int32 reported counts")
        if model.balance_dispersion:
            cells = (dosage[s] == 1) & (depth > 0)
            concentration = (1 - model.balance_dispersion) / model.balance_dispersion
            probability = rng.beta(
                heterozygote_alt_probability * concentration,
                (1 - heterozygote_alt_probability) * concentration,
                int(cells.sum()),
            )
            alt = rng.binomial(depth[cells], probability)
            result[s, cells, 1], result[s, cells, 0] = alt, depth[cells] - alt
        cells = shared.systematic_site & (dosage[s] == 0) & (depth > 0)
        alt = rng.binomial(depth[cells], shared.systematic_rate[cells])
        result[s, cells, 1], result[s, cells, 0] = alt, depth[cells] - alt
        sample_factor = 1.0
        if model.miss_sample_cv:
            cv = float(model.miss_sample_cv)
            sigma2 = math.log1p(cv * cv)
            if not math.isfinite(sigma2):
                sigma2 = 2 * math.log(cv)
            sample_factor = float(rng.lognormal(-sigma2 / 2, math.sqrt(sigma2)))
        stretch = draw_stretches(sites, model, rng)
        probability = miss_probabilities(shared.hard, stretch, shared.region_factor, model, sample_factor)
        missed[s] = (dosage[s] >= 1) & (rng.random(sites) < probability)
        called_ref = (depth > 0) & (result[s, :, 1] >= model.call_min_alt_fraction * depth)
        block[s] = missed[s] | ((dosage[s] == 0) & ~called_ref)
        result[s] = reference_blocks(result[s], block[s], model.block_band)
    return ArtefactResult(result, missed, block, shared.systematic_site)
