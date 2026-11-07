"""Report run progress by checking record headers without decoding scientific arrays.

Owns status formatting and run-record inspection (ENG-30). Completion certificates
are checked against committed container headers; workflow markers alone never
certify completion. Stage progress adapters supply index-only summaries.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.errors import InputError, SettingsError
from ..engine.records import task_summaries
from ..results.paths import RunPaths

__all__ = ["run_status", "format_status"]

_CONTIG_STAGES = ("evidence", "discovery", "selection", "assembly", "painting", "refit", "phase", "recombination")


def _natural_key(name: str) -> list[Any]:
    return [int(part) if index % 2 else part for index, part in enumerate(re.split(r"(\d+)", name))]


def _records(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((directory / "run_records").glob("*.json")):
        data = json.loads(path.read_text())
        stages = data.get("stages", [])
        summaries = task_summaries(data)
        rows.append(
            dict(
                record=str(path),
                command=data.get("command"),
                state=data.get("state"),
                host=data.get("host"),
                pid=data.get("pid"),
                wall_seconds=data.get("wall_seconds"),
                peak_memory_bytes=data.get("peak_memory_bytes", data.get("peak_rss_bytes")),
                memory=data.get("memory", {}),
                peak_live_use_bytes=data.get("memory", {}).get("peak_live_use_bytes"),
                task_count=data.get("task_count", sum(row["tasks"] for row in summaries.values())),
                task_summaries=summaries,
                task_wall_seconds={name: row["wall_seconds"] for name, row in sorted(summaries.items())},
                updated_utc=data.get("updated_utc"),
                active_stages=data.get("active_stages", []),
                error=data.get("error"),
                exclusive_stage_seconds={
                    name: sum(row.get("exclusive_seconds", 0) for row in stages if row.get("stage") == name)
                    for name in sorted({row["stage"] for row in stages if "stage" in row})
                },
            )
        )
    return rows


def _completion(file: Any) -> list[str]:
    # mark_complete validates the derivations before appending its certificate.
    # In an append-only container, referenced records at or before that sequence
    # are the same committed versions it checked. A later replacement invalidates
    # this status certificate conservatively; scientific resume still checks the
    # independently expected derivations. Snapshot once, avoiding a stat per key.
    index = file._records()
    complete = []
    for key, info in sorted(index.items()):
        if not key.startswith("complete/"):
            continue
        hint = file._read(info, 1, items=("records",))
        records = hint.get("records", {})
        if records and all(name in index and index[name].seq <= info.seq for name in records):
            complete.append(key)
    return complete


def _progress(stage: str, file: Any) -> Any:
    module = importlib.import_module(f"haplotype_reconstruction.stages.{stage}.records")
    return module.progress(file)


def _configuration(directory: Path, manifest: dict[str, Any], store: Any) -> None:
    """Compare a present recorded settings file without reopening observation payloads."""
    path = directory / "config.toml"
    if not path.is_file():
        return  # An interrupted init can publish its manifest before config.toml.
    from ..core.digest import canonical_json
    from ..stages.configure import resolve
    from .run import _match_manifest
    from .settings.load import load_settings

    settings = load_settings(path)
    resolved = resolve(settings, sample_ids=manifest["sample_ids"], bound_seeds=manifest["realised_seeds"])
    candidate = resolved.manifest()
    _match_manifest(dict(manifest, **candidate), manifest, recombination=True)
    requested = canonical_json(candidate["parameters"]["recombination"])
    initial = canonical_json(manifest["parameters"]["recombination"])
    from ..contracts.roles import require_schema
    from ..store.protocol import publication_metadata

    # The immutable configuration record is a small control binding, like the
    # run manifest. Status compares that choice without reconstructing products.
    # Results independently derives RC's scientific identity before decoding it.
    for contig in resolved.contigs:
        file = store.contig(contig)
        if file.has("configuration/recombination"):
            metadata = publication_metadata(file, "configuration/recombination", ("schema", "parameters"))
            if metadata is None:
                raise SettingsError(f"{contig}: recombination parameter binding disappeared")
            require_schema(metadata.fields, 1)
            if canonical_json(metadata.fields["parameters"]) != requested:
                raise SettingsError(f"{contig}: recorded settings differ from the recombination parameter binding")
        elif requested != initial:
            raise SettingsError(f"{contig}: changed recombination settings have no stage parameter binding")


def _error_code(error: Exception) -> int:
    from ..store.container import CheckpointMismatch

    if isinstance(error, (CheckpointMismatch, SettingsError)):
        return 2
    return 3 if isinstance(error, InputError) else 1


def run_status(
    output: str | Path, *, checkpoints: str | Path | None = None, contigs: Sequence[str] | None = None
) -> dict[str, Any]:
    """Return layout, manifest, checked progress, invocation records and workflow marker paths.

    An unreadable or uninitialised run is represented in errors (CLI exit 1).
    No record, directory or marker is written. Partial runs retain available progress.
    """
    from ..contracts.roles import require_schema
    from ..store.store import CheckpointStore

    directory = Path(output).expanduser().resolve()
    report: dict[str, Any] = dict(
        output=str(directory), records=[], checkpoints=None, run=None, contigs={}, markers=[], errors=[], exit_code=0
    )
    try:
        paths = RunPaths.from_directory(directory)
        root = paths.checkpoints if checkpoints is None else Path(checkpoints).expanduser().resolve()
        report["checkpoints"] = str(root)
        report["records"] = _records(directory)
        if not directory.is_dir() or not root.is_dir():
            raise OSError(f"{directory}: no initialized run directory")
        store = CheckpointStore(root)
        from ..store.status_index import load_index

        load_index(store.run)
        for name in store.contig_names():
            load_index(store.contig(name))
        store.check_layout()
        manifest = store.manifest("run")
        if manifest is None:
            raise OSError(f"{directory}: run has not been initialized; use haplotypes init")
        require_schema(manifest, 1)
        _configuration(directory, manifest, store)
        report["run"] = dict(complete=_completion(store.run), contigs=manifest.get("contigs"), manifest=manifest)
        report["markers"] = [str(path.relative_to(root)) for path in sorted((root / "markers").glob("*.done"))]
        names = (
            list(contigs)
            if contigs is not None
            else list(dict.fromkeys([*manifest.get("contigs", ()), *sorted(store.contig_names(), key=_natural_key)]))
        )
        for name in names:
            file = store.contig(name)
            row: dict[str, Any] = dict(size_bytes=file.size(), stages={}, complete=[], work=[])
            try:
                row["complete"] = _completion(file)
                for stage in _CONTIG_STAGES:
                    prefix = {"refit": "founder_refit"}.get(stage, stage)
                    if file.keys(prefix + "/") or file.has(prefix):
                        row["stages"][stage] = _progress(stage, file)
            except (OSError, ValueError, RuntimeError, ImportError, AttributeError) as error:
                report["errors"].append(f"{name}: {error}")
                report["exit_code"] = _error_code(error)
                row["error"] = str(error)
            report["contigs"][name] = row
    except (OSError, ValueError, RuntimeError) as error:
        report["errors"].append(str(error))
        report["exit_code"] = _error_code(error)
    return report


def _describe(stage: str, progress: Any) -> str:
    if progress is None:
        return "-"
    if isinstance(progress, dict):
        if progress.get("complete") is True or progress.get("done") is True or progress.get(stage) is True:
            return "done"
        parts = []
        for key, value in progress.items():
            if key in ("stage", "complete", "done", stage, "decoded_bytes") or value is None or value is False:
                continue
            if isinstance(value, (list, tuple)):
                if value and len(value) <= 12 and all(isinstance(item, (str, int)) for item in value):
                    parts.append(f"{key} {','.join(map(str, value))}")
            elif not isinstance(value, dict):
                parts.append(f"{key} {value}")
        return " ".join(parts) or "-"
    return str(progress)


def format_status(report: dict[str, Any]) -> str:
    """Render the compact 3.5 status vocabulary plus record-checked completion."""
    lines = [f"Run records: {len(report['records'])} in {report['output']}/run_records"]
    for row in report["records"]:
        lines.append(
            f"  {Path(row['record']).stem}  {row['command']} {row['state']}  {row['host']} pid "
            f"{row['pid']}  wall {row['wall_seconds'] or 0:.0f} s  updated {row['updated_utc']}"
        )
        if row.get("peak_live_use_bytes") is not None:
            lines.append(f"    peak live memory: {row['peak_live_use_bytes'] / 2**30:.2f} GiB")
        if row.get("task_wall_seconds"):
            lines.append(
                "    task wall: "
                + ", ".join(f"{stage} {seconds:.2f} s" for stage, seconds in row["task_wall_seconds"].items())
            )
        if row["error"]:
            lines.append(f"    error: {row['error']}")
        if row["active_stages"]:
            lines.append(
                "    active: "
                + " > ".join(f"{stage['stage']} {stage.get('contig') or ''}".strip() for stage in row["active_stages"])
            )
    lines.append(f"Checkpoints: {report['checkpoints']}")
    if report["run"] is not None:
        lines.append(
            "  complete: " + (" ".join(name.removeprefix("complete/") for name in report["run"]["complete"]) or "-")
        )
    for name, row in report["contigs"].items():
        if "error" in row:
            lines.append(f"  {name}: unreadable ({row['error']})")
            continue
        text = " | ".join(f"{stage} {_describe(stage, progress)}" for stage, progress in row["stages"].items())
        lines.append(f"  {name}: {text or 'not started'}")
    lines.extend(f"ERROR: {message}" for message in report["errors"])
    return "\n".join(lines)
