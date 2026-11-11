"""One ordered selection of configured contigs for the final analysis.

Evidence, relatedness and reconstruction retain the configured axis. Only the
final analysis uses the included axis; exclusions never alter sample eligibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..core.digest import json_digest
from ..core.errors import SettingsError


@dataclass(frozen=True)
class AnalysisContigs:
    configured: tuple[str, ...]
    excluded: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.excluded) - set(self.configured)
        if unknown:
            raise SettingsError(
                f"unknown excluded contigs: {', '.join(sorted(unknown))}; "
                f"configured contigs: {', '.join(self.configured)}"
            )
        if self.configured and set(self.excluded) == set(self.configured):
            raise SettingsError("exclude_contigs excludes every configured contig; at least one contig must remain")
        # Exclusion is a set; retain the configured order for axes and identities.
        object.__setattr__(self, "excluded", tuple(c for c in self.configured if c in self.excluded))

    @property
    def included(self) -> tuple[str, ...]:
        return tuple(c for c in self.configured if c not in self.excluded)

    @property
    def namespace(self) -> str:
        return "" if not self.excluded else json_digest(dict(included_contigs=self.included))

    @property
    def identity_inputs(self) -> dict:
        """Empty selection adds no scientific parameter to existing derivations."""
        return {} if not self.excluded else dict(included_contigs=self.included)

    def document(self) -> dict:
        return dict(configured_contigs=self.configured, included_contigs=self.included, excluded_contigs=self.excluded)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> AnalysisContigs:
        return cls(tuple(manifest["contigs"]), tuple(manifest.get("analysis", {}).get("excluded_contigs", ())))
