"""Record of the export stage: one contig record ``export`` = {schema, identity, entry}.

``identity`` = stage_identity("export", config = ExportConfig, upstream = the RecordInfo identities of the records
the export read: evidence, final assembly release alternatives, the founder panels' source (founder_refit, else
painting) and family_phase when the
samples are exported); ``entry`` = the contig's manifest entry (its files under OUTPUT/export). The record marks the
contig's files complete. The export's products are derived files, so a different identity (changed [export]
settings or export code) re-exports the contig instead of refusing the run; no other stage reads this record.
"""
from __future__ import annotations

from ..store.identity import stage_identity
from .config import ExportConfig

STAGE = "export"
KEY = "export"
# 3: release 2.3 founder segments and ambiguity masks; final assembly is an explicit input.
SCHEMA = 3


def identity(file, config: ExportConfig) -> dict:
    """The identity of a contig's export under ``config`` (from the store index; no payload decoding)."""
    from ..evidence import records as evidence_records
    from ..family_phase import records as phase_records
    from .run import founder_source_key
    from ..painting import records as painting_records
    from ..assembly import records as assembly_records
    upstream = {"evidence": file.info(evidence_records.EVIDENCE).identity}
    if "founders" in config.products or "tracks" in config.products:
        upstream["founders"] = file.info(founder_source_key(file)).identity
        upstream["painting"] = file.info(painting_records.KEY).identity
        upstream["assembly_release"] = file.info(f"{assembly_records.FINAL_PREFIX}/{assembly_records.FINAL}").identity
    if "samples" in config.products:
        upstream["family_phase"] = file.info(phase_records.KEY).identity
    packages = ("export", "assembly") if "assembly_release" in upstream else ("export",)
    return stage_identity(STAGE, schema=SCHEMA, config=config, upstream=upstream, packages=packages)


def complete(file) -> bool:
    return file.has(KEY)


def progress(file) -> dict:
    return {"stage": STAGE, "complete": file.has(KEY)}


def load(file) -> dict | None:
    """The contig's export record ({schema, identity, entry}), or None."""
    return file.get(KEY) if file.has(KEY) else None


def write(file, identity_: dict, entry: dict):
    return file.put(KEY, dict(schema=SCHEMA, identity=identity_, entry=entry))
