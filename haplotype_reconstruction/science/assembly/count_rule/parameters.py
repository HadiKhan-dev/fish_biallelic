"""Settings of the pipeline's one founder-count rule (the assembly join and founder refinement).

Owns: count-rule constants with native provenance.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: WP_AS5 §7; AJ23, AJ25, AJ30–AJ35.
Identity: every eligible count hypothesis is refitted on its natural founder domain (4.2).
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
        many configurations. Larger families retain the incumbent evaluated decision and
        record count_unverified; this is separate from the shared sixteen-leaf
        higher-level continuation allowance.
    near_identity: called-site mismatch rate at or below which the join treats two paths as the same
        sequence (a path that close to another owns no sequence: q = 0). Founder refinement does not
        use it: its sequence part is an exact shadow score.
    """

    bootstrap_resamples: int = field(
        default=2000,
        metadata={"units": "replicates", "provenance": "AJ31: 3.5 count_rule/config.py bootstrap_resamples"},
    )
    bootstrap_level: float = field(
        default=0.95, metadata={"units": "probability", "provenance": "AJ31: 3.5 count_rule/config.py bootstrap_level"}
    )
    near_identity: float = field(
        default=0.01,
        metadata={"units": "mismatch fraction", "provenance": "AJ32/AJ33: 3.5 count_rule/config.py near_identity"},
    )
    shadow_combinations: int = field(
        default=2048,
        metadata={"units": "configurations", "provenance": "AJ32/AJ33: 3.5 count_rule/config.py shadow_combinations"},
    )


# AJ25, 3.5 join/search.py: reported guard after this many count changes.
COUNT_BACKSTOP_BASE = 50
COUNT_BACKSTOP_PER_ROW = 10
# AJ30, criterion.py: diagnostic ranks and original founder-cost site normalizer.
TOP_SAMPLES = 5
REFERENCE_SITES = 200.0
# AJ35: proof-only dimension budget and cost-only allocation accounting slack.
SHADOW_WORK_RATIO = 4
PROOF_SLACK_BYTES = 1 << 16
# AJ23: cost-only number of distinct path-set birth-bound caches.
BIRTH_SETS = 4


__all__ = [
    "CountRuleParameters",
    "COUNT_BACKSTOP_BASE",
    "COUNT_BACKSTOP_PER_ROW",
    "TOP_SAMPLES",
    "REFERENCE_SITES",
    "SHADOW_WORK_RATIO",
    "PROOF_SLACK_BYTES",
    "BIRTH_SETS",
]
