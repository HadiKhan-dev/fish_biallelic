"""Publish complete pedigree records as plain stored-field dictionaries and lists.

The record codec owns recursive field/schema/type validation. This pedigree adapter owns only
whole-product completeness of schema-8 products: composed reports and views, a search row for
every sample and a summary for every configured chromosome. A cold reader never repairs reports
or reruns science.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...contracts import pedigree as records
from ...contracts.payload import from_payload, to_payload
from ...contracts.roles import require_schema
from ...core.errors import InputError

__all__ = ["PedigreeAdapter"]


class PedigreeAdapter:
    """Lossless schema-8 fields, decoded to read-only arrays by the shared record codec codec."""

    schema = 8

    def encode(self, product: records.Pedigree) -> Mapping[str, Any]:
        require_schema({"schema": product.schema}, self.schema)
        _complete(product)
        return to_payload(product)

    def decode(self, payload: Mapping[str, Any]) -> records.Pedigree:
        require_schema(payload, self.schema)
        product = from_payload(records.Pedigree, payload)
        _complete(product)
        return product


def _complete(product: records.Pedigree) -> None:
    """Refuse an incomplete product: a reader never repairs absent report/scientific state."""
    if not product.reports_complete:
        raise InputError("pedigree report inputs are incomplete; finish PE2/PE3 report composition before publication")
    if product.regime == "not_assessable":
        if any(value is not None for value in (product.panel, product.support, product.family_solve, product.retained)):
            raise InputError("not-assessable pedigree must have absent fit diagnostics")
        if len(product.edges.parents) or np.any(product.calls.state != -1):
            raise InputError("not-assessable pedigree must release nothing")
        if tuple(row.contig for row in product.views.chromosome_summary) != product.contigs:
            raise InputError("not-assessable pedigree lacks prepared chromosome summaries")
        return
    if product.views.relative_contrasts is None or product.views.sibling_guard is None:
        raise InputError("pedigree views lack relative or guard composition")
    if len(product.views.search_diagnostics) != len(product.sample_ids):
        raise InputError("pedigree views lack a search row for every sample")
    if tuple(row.contig for row in product.views.chromosome_summary) != product.contigs:
        raise InputError("pedigree views lack configured chromosome summaries")
