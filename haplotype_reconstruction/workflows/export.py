"""Lossless founder-track and final-phase export, separate from inference."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
from cyvcf2 import VCF, Writer

from ..core import parallel, runtime
from ..core.products import load_panels, panel_alleles


def _reference_identity(store, supplied):
    """Check the general runner's original input, without inferring allele identities."""
    identity_path = Path(store.root)/"block_discovery"/"_identity.json"
    record = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    if "vcf" not in record:
        if supplied is None:
            raise ValueError("this dataset adapter needs --vcf identifying its original allele basis")
        return str(Path(supplied).resolve())
    supplied = Path(supplied or record["vcf"]).resolve()
    stat = supplied.stat()
    if (str(supplied) != record["vcf"] or stat.st_size != record["size"]
            or stat.st_mtime_ns != record["mtime_ns"]):
        raise ValueError("reference VCF does not match the recorded reconstruction input")
    return str(supplied)


def _sample_variants(path, contig, positions, calls, components, names,
                     reference, synthetic, file_format):
    """Write one chromosome, leaving unknown calls and unsupported phase unknown."""
    if calls.shape != (len(names), len(positions), 2):
        raise ValueError("final phase sample/site axes do not match")
    reader = None if synthetic else VCF(reference)
    length = int(positions[-1]) if synthetic else None
    if reader is not None:
        if set(names)-set(reader.samples) or contig not in reader.seqnames:
            reader.close()
            raise ValueError("export samples/contig are absent from the original VCF")
        try:
            length = int(reader.seqlens[reader.seqnames.index(contig)])
        except AttributeError:
            pass  # A source header without lengths stays unspecified.
    contig_line = f"##contig=<ID={contig}" + (f",length={length}" if length and length > 0 else "") + ">\n"
    header = ("##fileformat=VCFv4.3\n"
              "##source=haplotype-reconstruction\n"
              + contig_line +
              '##FORMAT=<ID=GT,Number=1,Type=String,Description="Released genotype; no imputation">\n'
              '##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Component-local phase set">\n')
    if synthetic:
        header += '##synthetic_alleles="A/C encode abstract simulated 0/1; not reference-genome bases"\n'
    header += "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"+"\t".join(names)+"\n"
    first = {}
    for index, component in enumerate(components):
        if component >= 0:
            first.setdefault(int(component), int(positions[index]))
    variants = None if synthetic else iter(reader(contig))
    record = None
    mode = "wb" if file_format == "bcf" else "wz"
    writer = Writer.from_string(str(path), header, mode=mode)
    lookup = np.array([".", "0", "1"])
    try:
        for site, position in enumerate(positions):
            if synthetic:
                ref, alt = "A", "C"
            else:
                while record is None or record.POS < position:
                    record = next(variants, None)
                    if record is None:
                        raise ValueError(f"input VCF lacks {contig}:{position}")
                if record.POS != position or len(record.ALT) != 1:
                    raise ValueError(f"input VCF does not match {contig}:{position}")
                ref, alt = record.REF, record.ALT[0]
                if len(ref) != 1 or len(alt) != 1:
                    raise ValueError("export requires the original biallelic SNP alleles")
            component = int(components[site])
            alleles = lookup[calls[:, site, :]+1]
            # Missing alleles remain dots; phase sets do not cross components.
            separator, ps = ("|", str(first[component])) if component >= 0 else ("/", ".")
            genotypes = [a+separator+b+":"+ps for a, b in alleles]
            line = (f"{contig}\t{int(position)}\t.\t{ref}\t{alt}\t.\t.\t.\tGT:PS\t"
                    +"\t".join(genotypes))
            writer.write_record(writer.variant_from_string(line))
    finally:
        writer.close()
        if reader is not None:
            reader.close()


def _export_contig(task):
    root, destination, contig, products, file_format, reference, synthetic, threads = task
    store = runtime.CheckpointStore(root, nthreads=threads)
    output = Path(destination)
    stem = quote(contig, safe="")
    files, panels = [], []
    if "founders" in products:
        blocks = load_panels(store, contig)
        for component, block in enumerate(blocks):
            positions, keys, calls = panel_alleles(block)
            filename = f"{stem}.component{component:04d}.founders.tsv.gz"
            path, temporary = output/filename, output/("."+filename+".tmp")
            with gzip.open(temporary, "wt", compresslevel=6) as handle:
                handle.write("CHROM\tPOS\t"+"\t".join(f"H{i}" for i in range(len(keys)))+"\n")
                text = np.array([".", "0", "1"])[calls+1]
                for i, position in enumerate(positions):
                    handle.write(f"{contig}\t{int(position)}\t"+"\t".join(text[:, i])+"\n")
            temporary.replace(path)
            files.append(filename)
            panels.append(dict(contig=contig, component=component,
                               local_columns={f"H{i}": str(key) for i, key in enumerate(keys)},
                               first_position=int(positions[0]), last_position=int(positions[-1]),
                               markers=len(positions), called_alleles=int((calls>=0).sum()),
                               missing_alleles=int((calls<0).sum()), file=filename))
        del blocks
    if "samples" in products:
        phase = store.load_contig("family_phase", contig)
        calls = np.asarray(phase["phase"].allele_calls)
        if np.any(~np.isin(calls, (-1, 0, 1))):
            raise ValueError("unexpected final allele encoding")
        names = tuple(phase["sample_ids"])
        suffix = ".bcf" if file_format == "bcf" else ".vcf.gz"
        filename = stem+".samples"+suffix
        path, temporary = output/filename, output/("."+filename+".tmp")
        _sample_variants(temporary, contig, np.asarray(phase["positions"]), calls,
                         np.asarray(phase["component_ids"]), names, reference, synthetic, file_format)
        temporary.replace(path)
        files.append(filename)
    return dict(contig=contig, files=files, founder_components=panels)


def export_run(output_dir, destination, *, checkpoints=None, contigs=None,
               products=("founders", "samples"), file_format="bcf", vcf=None,
               synthetic_alleles=False, cores=None):
    """Export each chromosome independently; never overwrite an existing export."""
    workers = runtime.available_cpu_count() if cores is None else int(cores)
    if not 1 <= workers <= runtime.available_cpu_count():
        raise ValueError("export cores must fit the current CPU affinity")
    store = runtime.CheckpointStore(checkpoints or Path(output_dir)/"checkpoints")
    if contigs is None:
        identity = json.loads((Path(store.root)/"painting"/"_identity.json").read_text())
        contigs = tuple(identity["ordered_contigs"])
    contigs = tuple(contigs)
    if not contigs or len(set(contigs)) != len(contigs):
        raise ValueError("export requires unique contigs")
    if synthetic_alleles and not store.global_done("simulated_reads"):
        raise ValueError("synthetic allele labels are only allowed for simulations")
    reference = (None if "samples" not in products or synthetic_alleles
                 else _reference_identity(store, vcf))
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("export destination must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    processes = min(workers, len(contigs))
    threads = max(1, workers//processes)
    print(f"Export: {processes} chromosome processes × {threads} I/O threads; "
          "record formatting is serial within a chromosome", flush=True)
    tasks = [(store.root, str(destination), c, tuple(products), file_format,
              reference, synthetic_alleles, threads) for c in contigs]
    if processes == 1:
        result = list(map(_export_contig, tasks))
    else:
        with parallel.ForkserverPool(processes=processes) as pool:
            result = list(pool.imap_unordered(_export_contig, tasks, chunksize=1))
    result.sort(key=lambda row: contigs.index(row["contig"]))
    manifest = dict(schema="released-allele-export-v1", checkpoint_root=str(Path(store.root).resolve()),
                    coordinate_system="1-based input SNP positions", unknown_allele=".",
                    founder_identity="chromosome AND component local; H0 across components is not an identity",
                    sample_phase="PS resets at supported component boundaries; no invented phase qualities",
                    synthetic_alleles=synthetic_alleles, reference_vcf=reference, contigs=result)
    temporary = destination/".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2)+"\n")
    temporary.replace(destination/"manifest.json")
    return manifest
