"""Relatedness records: the run product ``relatedness`` and its identity.

Run:
  relatedness   RelatednessProduct   pairwise IBD summaries and full-sib families from genotype likelihoods

Axes: N samples in run order; P = N (N - 1) / 2 pairs (a < b, row-major); F families; H half-sib links.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..evidence.records import EVIDENCE as EVIDENCE_KEY
from ..store.identity import stage_identity

STAGE = "relatedness"
PRODUCT = "relatedness"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RelatednessProduct:
    """Genome-wide relatedness (record ``relatedness``); read only from genotype likelihoods.

    pairs:      int64 [P, 2] sample pairs (a < b).
    ibd:        float64 [P, 3] posterior mean share of the pair's jointly observed IBD sites in IBD0, IBD1, IBD2.
    ibd_sites:  float64 [P] jointly observed IBD sites.
    lr_ibd0:    float64 [P] log L(IBD0, 1, 2) - log L(IBD1, 2): evidence for stretches without IBD (a parent and
                its child share none; full sibs share about a quarter of the genome).
    lr_ibd2:    float64 [P] log L(IBD0, 1, 2) - log L(IBD0, 1): evidence for stretches IBD on both homologues.
    family:     int64 [N] each sample's family (index into ``parents``).
    parents:    int64 [F, 2] each family's sequenced parent per slot, -1 latent.
    member_support: float64 [N] share of support units (map units of RelatednessConfig.support_unit_morgans; a short
                chromosome is one unit) on which the sample's family is its best placement.
    parent_support: float64 [F, 2] share of support units on which the sequenced parent beats a latent one (NaN for
                a latent slot).
    half_sib:   int64 [H, 2] latent families better explained as sharing one latent parent; half_sib_gain [H].
    loglik, sweeps, converged: the family fit's objective and search.
    family_sites, chromosome_sites: thinned family-model sites (total, per contig); support_units: units scored.
    """

    sample_ids: tuple[str, ...]
    contigs: tuple[str, ...]
    pairs: np.ndarray
    ibd: np.ndarray
    ibd_sites: np.ndarray
    lr_ibd0: np.ndarray
    lr_ibd2: np.ndarray
    family: np.ndarray
    parents: np.ndarray
    member_support: np.ndarray
    parent_support: np.ndarray
    half_sib: np.ndarray
    half_sib_gain: np.ndarray
    loglik: float
    sweeps: int
    converged: bool
    family_sites: int
    chromosome_sites: tuple[int, ...]
    support_units: int = 0

    def pair_matrix(self, values) -> np.ndarray:
        """float64 [N, N] symmetric matrix of a per-pair value (NaN on the diagonal)."""
        n = len(self.sample_ids)
        out = np.full((n, n), np.nan)
        out[self.pairs[:, 0], self.pairs[:, 1]] = values
        out[self.pairs[:, 1], self.pairs[:, 0]] = values
        return out

    def supported_family(self, sample: int, minimum_support: float) -> np.ndarray:
        """The members of ``sample``'s family whose support reaches ``minimum_support`` (sample indices), or just
        the sample when it does not reach it itself."""
        if not self.member_support[sample] >= minimum_support:
            return np.asarray([sample], dtype=np.int64)
        members = np.flatnonzero(self.family == self.family[sample])
        return members[self.member_support[members] >= minimum_support]


def product_identity(store, contigs, config, genetic_maps) -> dict:
    """The product <- every contig's evidence record, its genetic map and the settings."""
    upstream = {}
    for contig in contigs:
        upstream[f"evidence/{contig}"] = store.contig(contig).info(EVIDENCE_KEY).identity
        upstream[f"genetic_map/{contig}"] = genetic_maps.identity(contig)
    return stage_identity(STAGE, schema=SCHEMA_VERSION, config={"relatedness": config}, upstream=upstream)


def encode_product(value: RelatednessProduct) -> dict:
    return dict(schema=SCHEMA_VERSION, sample_ids=list(value.sample_ids), contigs=list(value.contigs),
                pairs=value.pairs, ibd=value.ibd, ibd_sites=value.ibd_sites, lr_ibd0=value.lr_ibd0,
                lr_ibd2=value.lr_ibd2, family=value.family, parents=value.parents,
                member_support=value.member_support, parent_support=value.parent_support, half_sib=value.half_sib,
                half_sib_gain=value.half_sib_gain, loglik=float(value.loglik), sweeps=int(value.sweeps),
                converged=bool(value.converged), family_sites=int(value.family_sites),
                chromosome_sites=[int(n) for n in value.chromosome_sites], support_units=int(value.support_units))


def decode_product(record: dict) -> RelatednessProduct:
    record = dict(record)
    record.pop("schema")
    record["sample_ids"] = tuple(record["sample_ids"])
    record["contigs"] = tuple(record["contigs"])
    record["chromosome_sites"] = tuple(record["chromosome_sites"])
    return RelatednessProduct(**record)


def complete(store) -> bool:
    return store.run.has(PRODUCT)


def load_product(store) -> RelatednessProduct:
    return decode_product(store.run.get(PRODUCT))


def identity(store) -> str:
    """The recorded product's identity (pedigree binds it)."""
    return store.run.info(PRODUCT).identity
