"""`haplotypes export RUN DEST`: the founder VCF and phased sample VCF of a finished run, one chromosome per task.

The same files as the run's export stage (OUTPUT/export, export.stage), written to DEST, which must be new or empty
(an export never overwrites). Per chromosome, by default: the founder VCF (one haploid column c<C>_H<k> per founder
row k of every component C; founder-refit panels when recorded, else the painted panels) and the family-phase
calls, full length (every input biallelic SNP that reached the site filter, FILTER PASS or the rules that removed
it), each with its index; optionally the per-component founder-track TSVs. DEST/manifest.json describes every file.
The products are read through their stages' records APIs; nothing is recomputed or recorded.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from ..evidence.sample_order import input_rows, recorded_input_order
from .config import FORMATS, PRODUCTS, ExportConfig
from .founders import component_record, write_founder_tracks, write_founder_vcf
from .samples import write_sample_phase
from .sites import COMMON_ALLELE_RULE, RULE_DESCRIPTIONS, R5_ONLY, build_site_table, evidence_sites, kept_site_table
from .vcf import place, suffix

DEFAULT_PRODUCTS = ExportConfig().products
SCHEMA_V1 = "released-allele-export-v1"     # founder tracks and kept-site sample phase only (export_contig API)
SCHEMA = "released-allele-export-v2"
# Per-task allowance besides the decoded records: htslib and gzip buffers, record formatting and the fixed-size
# text chunks (founders.CHUNK_SITES, samples.CHUNK_SITES). With SITE_ROW_BYTES, calibrated on the a90 integration
# runs (18 Oct): measured task peak / estimate 0.74-0.79 (reconstruct_actm3 0.14 of 0.18 GiB, std_n80 0.19 of
# 0.27 GiB, full length or kept sites); TroMau chr7 (422,133 rows, 116 samples) 0.24 GiB process RSS of 0.25.
TASK_ALLOWANCE_BYTES = 128 << 20
# Per written VCF row: decoded input/kept positions, masks, rule bits, REF/ALT, read totals, the row and FILTER
# lists the writers hold (Python ints and references) and the row -> kept-site index.
SITE_ROW_BYTES = 128


def _released_alleles(calls: np.ndarray) -> bool:
    """Every entry is -1 (unknown), 0 or 1 (min/max: no temporaries the size of the calls)."""
    return calls.size == 0 or (int(calls.min()) >= -1 and int(calls.max()) <= 1)


def founder_panels(file, *, withhold_unobserved: bool = True) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """(founder stage, [(positions, released calls)]) of a contig: founder refit when recorded, else painting.

    ``withhold_unobserved`` (release 2.2, :func:`unobserved_founder_alleles`): founder alleles that no painted
    carrier observes are released as -1 ('.')."""
    from ..founder_refit import records as refit_records
    from ..painting import records as painting_records
    if refit_records.complete(file):
        stage, panels = "founder_refit", refit_records.founder_panels(file)
    elif painting_records.complete(file):
        stage, panels = "painting", painting_records.founder_panels(file)
    else:
        raise FileNotFoundError(f"{file.label}: no painted founder panels")
    panels = [(panel.positions, panel.released) for panel in panels]
    if withhold_unobserved:
        panels, _ = unobserved_founder_alleles(file, panels)
    return stage, panels


def refit_ambiguous_alleles(file) -> int | None:
    """Founder alleles the founder refit released unknown because two tied refit moves exclude each other (2.2;
    observed but not attributable to one row; written '.' like every unknown), or None without a refit record."""
    from ..founder_refit import records as refit_records
    if not refit_records.complete(file):
        return None
    refit = refit_records.load(file)
    return int(sum(int(np.sum(component.panel.released[component.changed_rows, component.changed_sites] < 0))
                   for component in refit.components))


def phase_ambiguity(file):
    """(phase breaks per call row, frame_undetermined int8 [N, L]) of the contig's family phase, or None when its
    record does not carry the released phase ambiguity (release 2.2; records before it)."""
    from ..family_phase import records as phase_records
    names = ("frame_undetermined", "phase_breaks", "phase_break_offsets")
    if not all(name in getattr(phase_records, "ARRAYS", ()) for name in names):
        return None
    try:
        arrays = phase_records.load_arrays(file, names)
    except KeyError:
        return None
    offsets = np.asarray(arrays["phase_break_offsets"], dtype=np.int64)
    breaks = np.asarray(arrays["phase_breaks"], dtype=np.int64)
    return ([breaks[offsets[row]:offsets[row + 1]] for row in range(len(offsets) - 1)],
            np.asarray(arrays["frame_undetermined"]))


def unobserved_founder_alleles(file, panels) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[int]]:
    """(panels with unobserved alleles set to -1, withheld alleles per component) of a contig (release 2.2).

    A released founder allele (row k, site j) is kept only if some sample whose released painting names k at j
    (a track's member set: named, pooled, local equivalence or switch interval; painting
    ComponentPainting.carrier_observed) has read depth at j (the evidence observed mask). Elsewhere no carrier
    observes it and the allele is the upstream fit's convention (e.g. a seed allele kept by keep-current rules),
    so it is withheld. Panels are the painting's components in order (the founder-refit panels keep its rows)."""
    from ..evidence import records as evidence_records
    from ..painting import records as painting_records
    painting = painting_records.load(file, panels=False, emissions=False)
    if len(painting.components) != len(panels):
        raise ValueError(f"{file.label}: founder panels and painting components differ")
    evidence = evidence_records.load_arrays(file, ("positions", "observed"))
    sites, observed = np.asarray(evidence["positions"]), np.asarray(evidence["observed"], dtype=bool)
    result, withheld = [], []
    for component, (positions, calls) in zip(painting.components, panels):
        positions = np.asarray(positions, dtype=np.int64)
        calls = np.asarray(calls, dtype=np.int8)
        columns = np.searchsorted(sites, positions)
        if np.any(columns >= len(sites)) or not np.array_equal(sites[np.minimum(columns, len(sites) - 1)],
                                                                 positions):
            raise ValueError(f"{file.label}: founder panel sites are missing from the evidence record")
        seen = component.carrier_observed(positions, columns, observed, len(calls))
        hidden = (calls >= 0) & ~seen
        withheld.append(int(np.count_nonzero(hidden)))
        if withheld[-1]:
            calls = calls.copy()
            calls[hidden] = -1
        result.append((positions, calls))
    return result, withheld


def founder_source_key(file) -> str:
    """The record the founder panels are read from (founder refit when recorded, else painting)."""
    from ..founder_refit import records as refit_records
    from ..painting import records as painting_records
    if refit_records.complete(file):
        return refit_records.KEY
    if painting_records.complete(file):
        return painting_records.KEY
    raise FileNotFoundError(f"{file.label}: no painted founder panels")


def contig_bytes(file, products, *, full_length: bool = False, synthetic: bool = False, evidence=None) -> int:
    """Planned peak bytes of exporting one contig, from the store index's decoded record sizes: the founder
    panels' source (the founder-refit record; else the painting record and, twice, the final assembly's prepared
    panels the painted panels are materialised from) and, for the founder VCF, its allele blocks (the panels
    widened to the written rows); the family-phase record (the calls, positions and components decoded are about
    two thirds of it); SITE_ROW_BYTES per written row of a VCF product (``evidence``: evidence_sites(file), decoded
    when not given); a simulation's reads record when sites removed by r5 alone need read totals; and
    TASK_ALLOWANCE_BYTES."""
    from ..assembly import records as assembly_records
    from ..family_phase import records as phase_records
    from ..founder_refit import records as refit_records
    from ..painting import records as painting_records
    from ..simulation import truth
    total = TASK_ALLOWANCE_BYTES
    source = 0
    if "founders" in products or "tracks" in products:
        if refit_records.complete(file):
            source = file.info(refit_records.KEY).decoded_bytes
        elif painting_records.complete(file):
            completion = file.info(f"{assembly_records.FINAL_PREFIX}/{assembly_records.COMPLETION}")
            source = file.info(painting_records.KEY).decoded_bytes + 2 * completion.decoded_bytes
        # Unobserved-allele withholding (release 2.2): the painting's release grids (labels, classes, status,
        # members, direct: 28 bytes per sample, track pair and bin, twice for decoding) and the evidence observed
        # mask with its positions (one byte per sample and site, 8 per site).
        evidence = evidence_sites(file) if evidence is None else evidence
        summary = painting_records.load_summary(file)
        bins = sum(shape.bins for shape in summary.components)
        total += 2 * 28 * summary.samples * bins + (summary.samples + 8) * len(evidence["kept_positions"])
    total += source
    if "samples" in products and phase_records.complete(file):
        total += file.info(phase_records.KEY).decoded_bytes
    if "founders" in products or "samples" in products:
        evidence = evidence_sites(file) if evidence is None else evidence
        rows = len(evidence["input_positions"] if full_length else evidence["kept_positions"])
        total += SITE_ROW_BYTES * rows
        if "founders" in products:
            total += int(source * max(1.0, rows / max(1, len(evidence["kept_positions"]))))
        if (synthetic and full_length and file.has(truth.READS)
                and bool(np.any(evidence["removed_bits"] == R5_ONLY))):
            total += file.info(truth.READS).decoded_bytes
    return total


def export_contig(destination: Path, contig: str, *, products, founders=None, phase=None, reference=None,
                  synthetic=False, file_format="bcf", sites=None, index=False, input_sample_ids=None,
                  withheld=None, phase_ambiguity=None) -> dict:
    """Write one chromosome's files; returns its manifest entry.

    ``founders``: (stage, [(positions int64 [L], released int8 [K, L])]) in component order; ``phase``:
    (sample_ids, positions, component_ids, calls int8 [N, L, 2]) as family_phase.records.load_calls returns;
    ``sites``: the export.sites table of the VCF rows (default: the phase positions as kept sites, REF/ALT from
    ``reference``); ``index``: write each VCF/BCF's index (.tbi / .csi); ``input_sample_ids``: the run's input
    sample order (the sample columns are written in it; the records hold the canonical order, release 2.1), None
    to write the records' order; ``withheld``: founder alleles withheld per component as unobserved
    (unobserved_founder_alleles; recorded in the manifest entry and the founder VCF header); ``phase_ambiguity``:
    family phase's (phase breaks per row, frame_undetermined) (phase_ambiguity; None = component phase sets only).
    """
    destination = Path(destination)
    stem = quote(contig, safe="")
    products = tuple(products)
    files, components = [], []
    summarise_sites = sites is not None      # a caller's site table (CLI, export stage) enters the manifest entry
    if sites is None and ("samples" in products or "founders" in products):
        if phase is None:
            raise ValueError(f"{contig}: VCF rows need the export site table or the final phase sites")
        sites = kept_site_table(contig, phase[1], reference=reference, synthetic=synthetic,
                                sample_ids=tuple(phase[0]))
    stage = panels = None
    if "founders" in products or "tracks" in products:
        stage, raw = founders
        panels = []
        for positions, calls in raw:
            positions = np.asarray(positions, dtype=np.int64)
            calls = np.asarray(calls, dtype=np.int8)
            if calls.ndim != 2 or calls.shape[1] != len(positions) or not _released_alleles(calls):
                raise ValueError(f"{contig}: founder call/position axes are inconsistent")
            panels.append((positions, calls))
    vcf_name = None
    if "founders" in products:
        vcf_name = stem + ".founders" + suffix(file_format)
        temporary = destination / f".{vcf_name}.tmp"
        write_founder_vcf(temporary, sites, panels, stage, synthetic=synthetic, file_format=file_format,
                          withheld=withheld)
        files.extend(place(temporary, destination / vcf_name, file_format=file_format,
                           last_position=int(sites.positions[-1]), index=index))
    for component, (positions, calls) in enumerate(panels or ()):
        filename = None
        if "tracks" in products:
            filename = f"{stem}.component{component:04d}.founders.tsv.gz"
            write_founder_tracks(destination / filename, contig, positions, calls)
            files.append(filename)
        components.append(component_record(contig, component, positions, calls, stage, filename, vcf_name,
                                           None if withheld is None else withheld[component]))
    if "samples" in products:
        sample_ids, positions, component_ids, calls = phase
        calls = np.asarray(calls)
        if not _released_alleles(calls):
            raise ValueError("unexpected final allele encoding")
        filename = stem + ".samples" + suffix(file_format)
        temporary = destination / f".{filename}.tmp"
        rows = input_rows(sample_ids, input_sample_ids)
        names = tuple(sample_ids) if rows is None else tuple(map(str, input_sample_ids))
        breaks, undetermined = (None, None) if phase_ambiguity is None else phase_ambiguity
        write_sample_phase(temporary, sites, np.asarray(positions), calls, np.asarray(component_ids),
                           names, synthetic=synthetic, file_format=file_format, sample_rows=rows,
                           phase_breaks=breaks, frame_undetermined=undetermined)
        files.extend(place(temporary, destination / filename, file_format=file_format,
                           last_position=int(sites.positions[-1]), index=index))
    entry = dict(contig=contig, files=files, founder_components=components)
    if withheld is not None:
        entry["unobserved_founder_alleles"] = int(sum(withheld))
    if summarise_sites:
        entry["sites"] = sites.summary()
    return entry


def _export_task(task, shared, budget):
    """Pool task: read one chromosome's products and write its files (and, for the export stage, its record)."""
    from ..family_phase import records as phase_records
    from ..store import CheckpointStore
    file = CheckpointStore(task["checkpoints"], nthreads=max(1, int(budget.threads()))).contig(task["contig"])
    products = task["products"]
    if "samples" in products and not phase_records.complete(file):
        raise FileNotFoundError(f"{file.label}: no family phase (export --products founders for founders only)")
    sites = None
    if "founders" in products or "samples" in products:
        sites = build_site_table(file, task["contig"], full_length=task["full_length"],
                                 reference=task["reference"], synthetic=task["synthetic"])
    founders = withheld = ambiguous = None
    if "founders" in products or "tracks" in products:
        stage, panels = founder_panels(file, withhold_unobserved=False)
        panels, withheld = unobserved_founder_alleles(file, panels)
        founders = (stage, panels)
        ambiguous = refit_ambiguous_alleles(file)
        print(f"  [export] {task['contig']}: {sum(withheld)} founder alleles withheld (no observing painted "
              f"carrier)" + ("" if ambiguous is None else f"; {ambiguous} unknown from tied refit moves"), flush=True)
    phase = phase_records.load_calls(file) if "samples" in products else None
    ambiguity = phase_ambiguity(file) if "samples" in products else None
    destination = Path(task["destination"])
    entry = export_contig(destination, task["contig"], products=products, founders=founders, phase=phase,
                          reference=task["reference"], synthetic=task["synthetic"],
                          file_format=task["file_format"], sites=sites, index=True,
                          input_sample_ids=task.get("input_sample_ids"), withheld=withheld,
                          phase_ambiguity=ambiguity)
    if ambiguous is not None:
        entry["refit_ambiguous_alleles"] = ambiguous
    if task.get("record") is not None:
        from . import records
        for name in set(task.get("previous_files", ())) - set(entry["files"]):
            (destination / name).unlink(missing_ok=True)     # files of an earlier export of this contig
        records.write(file, task["record"], entry)
    return entry


def write_manifest(destination: Path, checkpoint_root, *, synthetic: bool, reference: str | None,
                   contigs: list[dict], settings: ExportConfig | None = None) -> dict:
    """DEST/manifest.json (written last, atomically); returns the manifest. ``settings`` (the CLI and the export
    stage) describe the VCF products (schema v2); without them the manifest is the v1 one of founder tracks and
    kept-site sample phase."""
    manifest = dict(schema=SCHEMA_V1 if settings is None else SCHEMA,
                    checkpoint_root=str(Path(checkpoint_root).resolve()),
                    coordinate_system="1-based input SNP positions", unknown_allele=".",
                    founder_identity="chromosome AND component local; H0 across components is not an identity",
                    unobserved_founder_alleles=("released founder alleles that no painted carrier observes (no "
                                                "sample painted with the founder there has read depth) are '.'; "
                                                "counts per contig (unobserved_founder_alleles) and component "
                                                "(withheld_unobserved_alleles)"),
                    sample_phase=("PS resets at supported component boundaries and after each undetermined phase "
                                  "boundary of the sample (family phase); a site whose displayed frame is "
                                  "undetermined is unphased; no invented phase qualities"),
                    synthetic_alleles=synthetic, reference_vcf=reference, contigs=contigs)
    if settings is not None:
        manifest.update(
            products=list(settings.products), format=settings.file_format,
            site_set=("full length: every input biallelic SNP that reached the site-quality filter (FILTER PASS = "
                      "kept, else the applied rules that removed the site)" if settings.full_length
                      else "the sites the site-quality filter kept (FILTER '.')"),
            omitted_records=("records the loader drops never reach the site filter and are not written: "
                             "non-biallelic records (r0) and biallelic SNPs sharing a position"),
            founder_columns=("cC_Hk = founder row k of component C of the chromosome (component-local labels); a "
                             "column has calls only inside its component's span and is '.' outside it"))
        if settings.full_length:
            manifest["filters"] = dict(PASS="kept by the site-quality filter", **RULE_DESCRIPTIONS)
            manifest["common_allele"] = COMMON_ALLELE_RULE
    temporary = Path(destination) / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(Path(destination) / "manifest.json")
    return manifest


def reference_vcf(store, run_directory, supplied) -> str:
    """The input VCF whose alleles name 0/1 at every exported site.

    The run's recorded source identity (path, size, modification time) when present: a supplied path must be
    that file, unchanged. Otherwise the supplied path, else [input] vcf of RUN/config.toml.
    """
    recorded = (store.manifest("run") or {}).get("source") or {}
    if recorded.get("vcf"):
        path = Path(supplied or recorded["vcf"]).resolve()
        status = path.stat()
        if (str(path) != recorded["vcf"] or status.st_size != recorded["size"]
                or status.st_mtime_ns != recorded["mtime_ns"]):
            raise ValueError("reference VCF does not match the recorded reconstruction input")
        return str(path)
    if supplied is None:
        config = Path(run_directory) / "config.toml"
        if config.is_file():
            import tomllib
            with open(config, "rb") as handle:
                supplied = tomllib.load(handle).get("input", {}).get("vcf") or None
    if supplied is None:
        raise ValueError("this run records no input VCF: give --vcf")
    return str(Path(supplied).resolve())


def is_simulation(store) -> bool:
    """A simulated run: its alleles are abstract 0/1, exported as A/C."""
    from ..simulation import truth as truth_records
    return store.run.has(truth_records.MANIFEST)


def run_tasks(tasks: list[dict], estimates: list[int], workers: int, *, label: str = "export") -> list[dict]:
    """Run export tasks with ``workers`` CPUs: min(workers, tasks) chromosome processes, each holding its planned
    peak of the run's memory while it runs (it waits until it fits); returns their manifest entries in order."""
    from .. import execution
    if not tasks:
        return []
    processes = min(workers, len(tasks))
    threads = max(1, workers // processes)
    print(f"Export: {len(tasks)} chromosomes, {processes} processes x {threads} I/O threads; record formatting is "
          "serial within a chromosome", flush=True)
    if processes == 1:
        budget = execution.CpuBudget.fixed(workers)
        results = []
        for task, estimate in zip(tasks, estimates):
            with execution.reserve_memory(estimate, label=label):
                results.append(_export_task(task, {}, budget))
        return results
    with execution.BudgetedPool(execution.CpuBudget.fixed(workers), processes=processes, label=label) as pool:
        return pool.map(_export_task, tasks, task_bytes=estimates)


def export_run(run_directory, destination, *, checkpoints=None, contigs=None, products=DEFAULT_PRODUCTS,
               file_format="vcf", vcf=None, full_length=True, cores=None) -> dict:
    """Export each chromosome independently (see the module docstring); returns the manifest.

    ``contigs`` default to the run's contigs; ``cores`` (default: the CPU affinity) bounds the chromosome
    processes and their checkpoint-decoding threads (record formatting is serial within a chromosome). A
    simulation's abstract alleles are written as A (0) and C (1).
    """
    from .. import execution
    from ..store import CheckpointStore
    available = execution.affinity_cpus()
    workers = available if cores is None else int(cores)
    if not 1 <= workers <= available:
        raise ValueError("export cores must fit the current CPU affinity")
    products = tuple(dict.fromkeys(products))
    if not products or set(products) - set(PRODUCTS):
        raise ValueError(f"export products must be among {', '.join(PRODUCTS)}")
    if file_format not in FORMATS:
        raise ValueError(f"export format must be one of {', '.join(FORMATS)}")
    settings = ExportConfig(full_length=bool(full_length), file_format=file_format, products=products)
    store = CheckpointStore(checkpoints or Path(run_directory) / "checkpoints")
    if contigs is None:
        contigs = (store.manifest("run") or {}).get("contigs")
        if not contigs:
            raise ValueError(f"{store.root}: no run manifest; give the contigs to export")
    contigs = tuple(contigs)
    if not contigs or len(set(contigs)) != len(contigs):
        raise ValueError("export requires unique contigs")
    synthetic = is_simulation(store)
    if synthetic and vcf is not None:
        raise ValueError("a simulation's alleles are abstract (written as A/C): --vcf is for VCF input runs")
    reference = (None if synthetic or not ({"founders", "samples"} & set(products))
                 else reference_vcf(store, run_directory, vcf))
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("export destination must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    input_ids = recorded_input_order(store.manifest("run"))
    tasks = [dict(checkpoints=str(store.root), destination=str(destination), contig=contig, products=products,
                  file_format=file_format, reference=reference, synthetic=synthetic, full_length=settings.full_length,
                  input_sample_ids=input_ids)
             for contig in contigs]
    estimates = [contig_bytes(store.contig(contig), products, full_length=settings.full_length, synthetic=synthetic)
                 for contig in contigs]
    results = run_tasks(tasks, estimates, workers)
    return write_manifest(destination, store.root, synthetic=synthetic, reference=reference, contigs=list(results),
                          settings=settings)


def export_command(args) -> int:
    """`haplotypes export RUN DEST [--products ...] [--format vcf|bcf] [--[no-]full-length] [--vcf FILE]
    [--contigs ...] [--cores N]`."""
    from .. import execution
    available = execution.affinity_cpus()
    cores = available if args.cores is None else int(args.cores)
    if not 1 <= cores <= available:
        raise SystemExit(f"export: --cores must be between 1 and the CPU affinity ({available})")
    from ..pipeline.driver import RunPaths
    execution.start_run(cores, preload=(__name__,))
    manifest = export_run(args.run_directory, args.destination,
                          checkpoints=RunPaths.from_directory(args.run_directory).checkpoints,
                          contigs=None if args.contigs is None else tuple(args.contigs),
                          products=tuple(args.products), file_format=args.format, vcf=args.vcf,
                          full_length=args.full_length, cores=cores)
    print(f"Export: {len(manifest['contigs'])} chromosomes written to {Path(args.destination).resolve()}",
          flush=True)
    for line in execution.memory_governor().summary_lines():
        print(line, flush=True)
    return 0
