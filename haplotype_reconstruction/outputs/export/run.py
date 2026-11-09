"""Format released contract records into per-contig files and the export manifest.

Owns P0 OU04/OU05/OU08 formatting; never computes release or reads source observations.
Method: docs/methods/export.md. The 3.5 formatting order and manifest text are retained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from ...core.concurrency import live_threads
from ...core.errors import InvariantError
from ...core.files import staged_path
from ..reports.tables import record_mapping, write_json
from .founders import write_identity_flags
from .records import project_sites

if TYPE_CHECKING:
    from ...contracts.assembly import FounderRelease
    from ...contracts.phase import PhasedSamples
    from ...contracts.release import ReleasedSites
import numpy as np

from ...contracts.samples import input_rows
from .founders import component_record, write_founder_tracks, write_founder_vcf
from .records import COMMON_ALLELE_RULE, RULE_DESCRIPTIONS, ExportConfig
from .samples import write_sample_phase
from .vcf import place, suffix

SCHEMA = "released-allele-export-v4"


def _writer_threads() -> int:
    """The CPUs this writer may use now (release 4.10: the export task's map gives each writer a live share; its
    BGZF compression and index threads come from it)."""
    return live_threads()


def _released_alleles(calls: np.ndarray) -> bool:
    """Every entry is -1 (unknown), 0 or 1 (min/max: no temporaries the size of the calls)."""
    return calls.size == 0 or (int(calls.min()) >= -1 and int(calls.max()) <= 1)


def export_contig(
    destination: Path,
    contig: str,
    *,
    products,
    founders=None,
    phase=None,
    reference=None,
    synthetic=False,
    file_format="bcf",
    sites=None,
    index=False,
    input_sample_ids=None,
    withheld=None,
    phase_ambiguity=None,
    founder_release=None,
    map_work=None,
) -> dict:
    """Write one chromosome's files; returns its manifest entry.

    ``founders``: (stage, [(positions int64 [L], released int8 [K, L])]) in component order; ``phase``:
    (sample_ids, positions, component_ids, calls int8 [N, L, 2]) as family_phase.records.load_calls returns;
    ``sites``: the export.sites table of the VCF rows (default: the phase positions as kept sites, REF/ALT from
    ``reference``); ``index``: write each VCF/BCF's index (.tbi / .csi); ``input_sample_ids``: the run's input
    sample order (the sample columns are written in it; the records hold the canonical order, release 2.1), None
    to write the records' order; ``withheld``: founder alleles withheld per component as unobserved
    (unobserved_founder_alleles; recorded in the manifest entry and the founder VCF header); ``founder_release``:
    assembly release metadata per segment (source axes, ambiguity and r5 exclusions); ``phase_ambiguity``:
    family phase's (phase breaks per row, frame_undetermined) (phase_ambiguity; None = component phase sets only).
    """
    destination = Path(destination)
    stem = quote(contig, safe="")
    products = tuple(products)
    files, components = [], []
    summarise_sites = sites is not None  # a caller's site table (CLI, export stage) enters the manifest entry
    if sites is None and ({"samples", "founders"} & set(products)):
        raise InvariantError(f"{contig}: VCF rows require a ReleasedSites projection")
    stage: str = ""
    panels = None
    if "founders" in products or "tracks" in products:
        stage, raw = founders
        panels = []
        for positions, calls in raw:
            positions = np.asarray(positions, dtype=np.int64)
            calls = np.asarray(calls, dtype=np.int8)
            if calls.ndim != 2 or calls.shape[1] != len(positions) or not _released_alleles(calls):
                raise InvariantError(f"{contig}: founder call/position axes are inconsistent")
            panels.append((positions, calls))
    vcf_name = stem + ".founders" + suffix(file_format) if "founders" in products else None
    track_name = f"{stem}.tracks.csv.gz" if "tracks" in products else None

    def write_founders():
        threads = _writer_threads()
        with staged_path(destination / vcf_name) as temporary:
            write_founder_vcf(
                temporary,
                sites,
                panels,
                stage,
                synthetic=synthetic,
                file_format=file_format,
                withheld=withheld,
                release_segments=founder_release,
                threads=threads,
            )
            return place(
                temporary,
                destination / vcf_name,
                file_format=file_format,
                last_position=int(sites.positions[-1]),
                index=index,
                threads=_writer_threads(),  # re-read: the share may have grown while writing
            )

    def write_tracks():
        write_founder_tracks(destination / track_name, contig, panels)
        return [track_name]

    for component, (positions, calls) in enumerate(panels or ()):
        filename = track_name
        components.append(
            component_record(
                contig,
                component,
                positions,
                calls,
                stage,
                filename,
                vcf_name,
                None if withheld is None else withheld[component],
                None if founder_release is None else founder_release[component],
            )
        )

    def write_samples():
        sample_ids, positions, component_ids, calls = phase
        calls = np.asarray(calls)
        if not _released_alleles(calls):
            raise InvariantError("unexpected final allele encoding")
        filename = stem + ".samples" + suffix(file_format)
        threads = _writer_threads()
        with staged_path(destination / filename) as temporary:
            rows = input_rows(sample_ids, input_sample_ids)
            names = tuple(sample_ids) if rows is None else tuple(map(str, input_sample_ids))
            breaks, undetermined = (None, None) if phase_ambiguity is None else phase_ambiguity
            write_sample_phase(
                temporary,
                sites,
                np.asarray(positions),
                calls,
                np.asarray(component_ids),
                names,
                synthetic=synthetic,
                file_format=file_format,
                sample_rows=rows,
                phase_breaks=breaks,
                frame_undetermined=undetermined,
                threads=threads,
            )
            return place(
                temporary,
                destination / filename,
                file_format=file_format,
                last_position=int(sites.positions[-1]),
                index=index,
                threads=_writer_threads(),  # re-read: the share may have grown while writing
            )

    # Each callback owns a distinct data/index pair. The engine bounds their
    # combined CPU use (each writer's threads are its share); the ordered map preserves manifest file order.
    writers = [
        writer
        for product, writer in (("founders", write_founders), ("tracks", write_tracks), ("samples", write_samples))
        if product in products
    ]
    if map_work is None:
        parts = [writer() for writer in writers]
    else:
        parts = map_work(lambda writer: writer(), writers)
    files = [name for names in parts for name in names]
    entry: dict = dict(contig=contig, files=files, founder_components=components)
    if withheld is not None:
        entry["unobserved_founder_alleles"] = int(sum(withheld))
    if summarise_sites:
        entry["sites"] = sites.summary()
    return entry


def write_manifest(
    destination: Path,
    checkpoint_root,
    *,
    synthetic: bool,
    reference: str | None,
    contigs: list[dict],
    settings: ExportConfig | None = None,
) -> dict:
    """DEST/manifest.json (written last, atomically); returns the manifest. ``settings`` (the CLI and the export
    stage) describe the VCF products (schema v3); without them the manifest is the v1 one of founder tracks and
    kept-site sample phase."""
    manifest = dict(
        schema=SCHEMA,
        checkpoint_root=str(Path(checkpoint_root).resolve()),
        coordinate_system="1-based input SNP positions",
        unknown_allele=".",
        founder_identity="chromosome AND release-segment local; H0 across segments is not an identity",
        unobserved_founder_alleles=(
            "released founder alleles that no painted carrier observes (no "
            "sample painted with the founder there has read depth) are '.'; "
            "counts per contig (unobserved_founder_alleles) and component "
            "(withheld_unobserved_alleles)"
        ),
        sample_phase=(
            "PS resets at supported component boundaries and after each undetermined phase "
            "boundary of the sample (family phase); a site whose displayed frame is "
            "undetermined is unphased; no invented phase qualities"
        ),
        synthetic_alleles=synthetic,
        reference_vcf=reference,
        contigs=contigs,
    )
    if settings is not None:
        manifest.update(
            products=list(settings.products),
            format=settings.file_format,
            site_set=(
                "full length: every input biallelic SNP that reached the site-quality filter (FILTER PASS = "
                "kept, else the applied rules that removed the site)"
                if settings.full_length
                else "the sites the site-quality filter kept (FILTER '.')"
            ),
            omitted_records=(
                "records the loader drops never reach the site filter and are not written: "
                "non-biallelic records (r0) and biallelic SNPs sharing a position"
            ),
            founder_columns=(
                "cC_Hk = founder row k of release segment C of the chromosome; source_component "
                "and segment_index identify its inference component and local segment. A column "
                "has calls only inside its segment's span and is '.' outside it"
            ),
            founder_identity_flags=(
                "founder_identity.csv: junctions where a founder row copies other rows' "
                "prepared local rows on both sides with disjoint partner sets "
                "(local_duplicate_partner_change): no local content of the row's own carries "
                "it across. Diagnostic only: no routing, call, column or break changes; not a "
                "calibrated identity confidence, and absence is no identity certificate"
            ),
            founder_continuity=(
                "live assembly alternatives split founder columns/tracks at unsupported "
                "continuity; unmatched evaluated final local content is withheld. "
                "Computational count overflow is count_unverified; unperformed source continuation is "
                "continuation_unverified. Both preserve calls and record the unperformed scope. "
                "Uncertainty intervals use observed 1-based SNP positions; old assembly "
                "records without alternatives have legacy_unknown coverage"
            ),
        )
        if settings.full_length:
            manifest["filters"] = dict(PASS="kept by the site-quality filter", **RULE_DESCRIPTIONS)
            manifest["common_allele"] = (
                COMMON_ALLELE_RULE + "; assembly ambiguity additionally withholds "
                "unmatched rows and unexplored source spans, including r5 sites"
            )
    write_json(Path(destination) / "manifest.json", manifest)
    return manifest


SEGMENT_METADATA = (
    "source_component",
    "segment_index",
    "block_start",
    "block_stop",
    "source_founder_rows",
    "ambiguity",
    "continuation_unverified",
    "count_unverified",
    "withheld_ambiguous_alleles",
    "r5_withheld_rows",
)


def export_records(
    destination: Path,
    contig: str,
    *,
    settings: ExportConfig,
    sites: ReleasedSites | None = None,
    founders: FounderRelease | None = None,
    phase: PhasedSamples | None = None,
    input_sample_ids: Sequence[str] | None = None,
    synthetic: bool = False,
    index: bool = True,
    release_phase_seconds: Mapping[str, float] | None = None,
    map_work=None,
) -> dict:
    """Render contract records and the product-dependent manifest entry.

    All release calls, identity flags and site decisions already exist. Kept-only
    projection changes only formatting. Raises InvariantError for a missing required
    product or an inconsistent axis. Writes only files under destination.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    products = settings.products
    need_founders = bool({"founders", "tracks"} & set(products))
    need_sites = bool({"founders", "samples"} & set(products))
    if (
        (need_founders and founders is None)
        or (need_sites and sites is None)
        or ("samples" in products and phase is None)
    ):
        raise InvariantError(f"{contig}: missing released export record")
    panels = metadata = withheld = None
    if need_founders:
        assert founders is not None
        panels = [(segment.positions, segment.calls) for segment in founders.segments]
        metadata = [
            {name: record_mapping(getattr(segment, name)) for name in SEGMENT_METADATA} for segment in founders.segments
        ]
        withheld = [segment.withheld_unobserved_alleles for segment in founders.segments]
    sample_phase = ambiguity = None
    if "samples" in products:
        assert phase is not None
        sample_phase = (phase.sample_ids, phase.positions, phase.component_ids, phase.calls)
        offsets, breaks = phase.phase_break_offsets, phase.phase_breaks
        ambiguity = (
            [breaks[offsets[row] : offsets[row + 1]] for row in range(len(offsets) - 1)],
            phase.frame_undetermined,
        )
    projected = None
    if need_sites:
        assert sites is not None  # refused above when a needed released site record is missing
        projected = project_sites(sites, full_length=settings.full_length)
    entry = export_contig(
        destination,
        contig,
        products=products,
        founders=("founder_refit", panels) if need_founders else None,
        phase=sample_phase,
        synthetic=synthetic,
        file_format=settings.file_format,
        sites=projected,
        index=index,
        input_sample_ids=input_sample_ids,
        withheld=withheld,
        phase_ambiguity=ambiguity,
        founder_release=metadata,
        map_work=map_work,
    )
    if need_founders:
        assert founders is not None
        entry["refit_ambiguous_alleles"] = int(founders.refit_ambiguous_alleles)
        entry["files"].append("founder_identity.csv")
        entry["founder_identity_flags"] = len(founders.identity_flags)
        entry.update(
            assembly_release_phase_seconds=dict(release_phase_seconds or {}),
            assembly_release_work_counters=dict(founders.work_counters),
            founder_continuity_breaks=int(founders.founder_continuity_breaks),
            assembly_ambiguous_alleles=int(founders.assembly_ambiguous_alleles),
            assembly_ambiguity_metadata=founders.assembly_ambiguity_metadata,
            assembly_continuation_unverified_records=int(founders.assembly_continuation_unverified_records),
            assembly_count_unverified_records=int(founders.assembly_count_unverified_records),
        )
    return entry


def write_founder_identity(destination: Path, results, contigs, products, *, collected=None) -> Path | None:
    """Gather the checked release flags for both pipeline export and standalone re-export."""
    if not {"founders", "tracks"}.intersection(products):
        return None

    def record(contig):
        release = results.founder_release(contig)
        if release is None:
            raise InvariantError(f"{contig}: missing released founder identity flags")
        return contig, "founder_refit", release.identity_flags

    records = collected if collected is not None else map(record, contigs)

    return write_identity_flags(Path(destination) / "founder_identity.csv", records)
