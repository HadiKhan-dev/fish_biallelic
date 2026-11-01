"""Name the optional pairwise IBD chain parameters.

Owns the relocated ibd_parameters units. No storage, environment or runtime policy.
Method: docs/methods/relatedness.md; WP_RL1. Legacy arithmetic retained for relocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PairwiseIbdParameters:
    ibd_site_step: int = field(default=10, metadata={"provenance": "3.5 config.py:39"})
    ibd_switch_rate: float = field(default=4.0, metadata={"provenance": "3.5 config.py:40"})
