"""Settings of the pipeline's one founder-count rule (the assembly join)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountRuleConfig:
    """Constants of the effective-carrier count rule, used by the assembly join.

    bootstrap_resamples, bootstrap_level: the sample bootstrap behind the ambiguity band of a
        count decision (percentile interval of min(m_d, m_q)); a call is ambiguous when the band
        contains zero in its interior. The point decision stands either way.
    ranked_refits: at most this many duplicate-seed birth candidates are refitted per decision; above it the
        candidates are ranked by cheap screens. A K-sized budget: every candidate is refitted while
        K <= 9.
    shadow_combinations: evaluate a complete nearest-row sequence-shadow family up to this
        many configurations. Larger families retain the incumbent evaluated decision and
        record count_unverified; this is separate from the shared sixteen-leaf
        higher-level continuation allowance.
    near_identity: called-site mismatch rate at or below which the join treats two paths as the same
        sequence (a path that close to another owns no sequence: q = 0).
    """

    bootstrap_resamples: int = 2000
    bootstrap_level: float = 0.95
    ranked_refits: int = 9
    near_identity: float = 0.01
    shadow_combinations: int = 2048
