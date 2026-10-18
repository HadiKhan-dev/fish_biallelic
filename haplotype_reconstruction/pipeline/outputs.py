"""Published tables (OUTPUT/<stage>/...): the reports layer writes them from records and products at fixed points
of the run (after each contig's evidence and painting, after discovery, after each genome-wide stage)."""
from __future__ import annotations


def publish(writer: str, *args, **keywords) -> list:
    """Call ``reports.<writer>(*args)``; return the paths written."""
    from .. import reports
    return getattr(reports, writer)(*args, **keywords)
