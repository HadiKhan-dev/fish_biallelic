"""Bind output formatting to the exact selected products and export options."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.errors import SettingsError

if TYPE_CHECKING:
    from ...results.reader import Results
    from ...outputs.export.records import ExportConfig


def expected(reader: Results, contig: str, config: ExportConfig) -> str:
    """Bind formatting to its checked source products and exact selected options."""
    from ...outputs.export.records import output_identity
    from ...store.identity import code_dependency_identity

    upstream: dict[str, str] = {}
    products = set(config.products)
    names = []
    if products & {"founders", "samples"}:
        names.append("sites")
    if products & {"founders", "tracks"}:
        names.append("release")
    if "samples" in products:
        names.append("phase")
    for name in names:
        identity = reader.product_identity(name, contig)
        if identity is None:
            raise SettingsError(f"{contig}: export requires published {name}")
        upstream[name] = identity
    code = code_dependency_identity(("stages.export.tasks", "stages.export.records", "outputs.export.task", "outputs.export.run")).digest
    return output_identity(upstream, config, code)
