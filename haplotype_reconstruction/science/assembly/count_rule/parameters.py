"""Settings of the pipeline's one founder-count rule (the assembly join and founder refinement).

Owns: count-rule values, units and rationale.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: docs/methods/assembly.md#aj30.
Invariant: every eligible count hypothesis is refitted on its natural founder domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CountRuleParameters:
    """Constants of the effective-carrier count rule, shared by the join and founder refinement.

    bootstrap_resamples, bootstrap_level: the sample bootstrap behind the ambiguity band of a
        count decision (percentile interval of min(m_d, m_q)); a call is ambiguous when the band
        contains zero in its interior. The point decision stands either way.
    shadow_combinations: evaluate a complete nearest-row sequence-shadow family up to this
        many configurations when the total test passes. Larger families with a passing
        total retain the conventional incumbent and record count_unverified; this is separate
        from the shared sixteen-leaf higher-level continuation allowance.
    near_identity: called-site mismatch rate at or below which the join treats two paths as the same
        sequence (a path that close to another owns no sequence: q = 0). Founder refinement does not
        use it: its sequence part is an exact shadow score.
    """

    bootstrap_resamples: int = field(
        default=2000,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True,
            "units": "replicates",
            "provenance": "Use 2000 sample bootstrap replicates for count-decision ambiguity intervals.",
        },
    )
    bootstrap_level: float = field(
        default=0.95,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "probability",
            "provenance": "Central bootstrap interval probability 0.95 for the count margin.",
        },
    )
    near_identity: float = field(
        default=0.01,
        metadata={
            "role": "state", "axes": (), "dtype": "float64", "stored": True,
            "units": "mismatch fraction",
            "provenance": "The join treats paths with called-site mismatch rate at most 0.01 as the same sequence.",
        },
    )
    shadow_combinations: int = field(
        default=2048,
        metadata={
            "role": "state", "axes": (), "dtype": "int64", "stored": True,
            "units": "configurations",
            ("provenance"): (
                "For a passing total test, evaluate complete nearest-row shadow families through 2048 "
                "configurations; larger families retain the conventional incumbent and are flagged."
            ),
        },
    )


# Report the count-change backstop at base + per-row times the largest input founder count.
COUNT_BACKSTOP_BASE = 50
COUNT_BACKSTOP_PER_ROW = 10
# Diagnostic sample ranks and the site-count normaliser in the founder cost.
TOP_SAMPLES = 5
REFERENCE_SITES = 200.0
# AJ23: cost-only number of distinct path-set birth-bound caches.
BIRTH_SETS = 4


__all__ = [
    "CountRuleParameters",
    "COUNT_BACKSTOP_BASE",
    "COUNT_BACKSTOP_PER_ROW",
    "TOP_SAMPLES",
    "REFERENCE_SITES",
    "BIRTH_SETS",
]
