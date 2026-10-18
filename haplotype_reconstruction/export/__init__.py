"""The run's final products: per chromosome a founder VCF (one haploid column per founder row of every component)
and the final sample phase (GT:PS), full length by default, plus optional per-component founder-track TSVs, with a
manifest. Entry points: :func:`export_run`, :func:`export_command` (CLI `haplotypes export`), and the export stage
``export.stage.run`` (the pipeline's last stage, OUTPUT/export); settings :class:`ExportConfig`."""
from .config import ExportConfig
from .run import export_command, export_run

__all__ = ["ExportConfig", "export_command", "export_run"]
