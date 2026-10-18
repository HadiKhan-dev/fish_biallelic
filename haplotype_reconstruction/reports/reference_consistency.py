"""OUTPUT/discovery/reference_consistency.csv: discovered blocks vs the reference-generation rows' genotypes.

A descriptive consistency check, not founder or pedigree truth, and never a gate: for every informative
discovery block it compares the released block haplotypes with the confident genotypes of the reference rows
([samples] reference_generation, e.g. TroMau's G0): each reference's homozygous calls against single haplotypes
(best haplotype, error %), every discovered haplotype against the references, and each reference's genotypes
against the best unordered pair of haplotypes (dosage, error %). A reference genotype is confident when its
normalised genotype likelihood (the run's read model, from the evidence record) reaches GENOTYPE_CONFIDENCE.
The references are analysed samples (they took part in discovery), so this is not independent validation; no
match can mean absent ancestry, incomplete reference sampling or a reconstruction error, and generation labels do
not establish anyone's parentage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

GENOTYPE_CONFIDENCE = 0.85      # minimum genotype probability of a confident reference call
MATCH_THRESHOLD_PCT = 2.0       # a match: error below this percentage of the compared sites
MIN_CONF_SITES = 10             # minimum compared sites for an error percentage


def reference_genotypes(likelihoods: np.ndarray, sites: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """int8 [R, n]: confident genotype dosage (0/1/2) of each reference at the block ``positions``, else -1.

    ``likelihoods`` float [R, L, 3] normalised genotype likelihoods of the references at the evidence ``sites``.
    """
    index = np.clip(np.searchsorted(sites, positions), 0, len(sites) - 1)
    matched = sites[index] == positions
    genotypes = np.full((likelihoods.shape[0], len(positions)), -1, dtype=np.int8)
    for reference in range(likelihoods.shape[0]):
        values = likelihoods[reference, index, :]
        dosage = np.argmax(values, axis=1)
        confident = (values[np.arange(len(positions)), dosage] >= GENOTYPE_CONFIDENCE) & matched
        genotypes[reference, confident] = dosage[confident].astype(np.int8)
    return genotypes


def _error_pct(reference, haplotype, mask) -> float:
    return float(np.mean(reference[mask] != haplotype[mask]) * 100.0)


def compare_blocks(panels, likelihoods, sites, names, *, contig: str, stage: str, label: str) -> list[dict]:
    """One row per block: ``panels`` [(positions int64 [n], calls int8 [K, n])] in block order; ``likelihoods``
    [R, L, 3] and ``names`` [R] of the reference rows; ``label`` prefixes the per-reference columns."""
    rows = []
    references = likelihoods.shape[0]
    matched_column = f"references_matched_hom_under_{MATCH_THRESHOLD_PCT:.0f}pct"
    for block_index, (positions, calls) in enumerate(panels):
        positions = np.asarray(positions)
        if len(positions) == 0:
            continue
        genotypes = reference_genotypes(likelihoods, sites, positions)
        homozygous = np.where(genotypes == 0, 0, np.where(genotypes == 2, 1, -1)).astype(np.int8)
        haplotypes = np.asarray(calls, dtype=np.int8)
        count = len(haplotypes)

        # Homozygous references against single haplotypes (only explicit calls are scored).
        best_for_reference = []
        for reference in range(references):
            valid = homozygous[reference] != -1
            if np.sum(valid) < MIN_CONF_SITES:
                best_for_reference.append((-1, float("nan"), int(np.sum(valid))))
                continue
            best_error, best_row = 101.0, -1
            for row in range(count):
                mask = valid & (haplotypes[row] >= 0)
                if np.sum(mask) < MIN_CONF_SITES:
                    continue
                error = _error_pct(homozygous[reference], haplotypes[row], mask)
                if error < best_error:
                    best_error, best_row = error, row
            best_for_reference.append((best_row, best_error, int(np.sum(valid))))

        # Every haplotype against the homozygous references.
        best_for_haplotype = []
        for row in range(count):
            called = haplotypes[row] >= 0
            best_error = 101.0
            for reference in range(references):
                mask = (homozygous[reference] != -1) & called
                if np.sum(mask) < MIN_CONF_SITES:
                    continue
                best_error = min(best_error, _error_pct(homozygous[reference], haplotypes[row], mask))
            best_for_haplotype.append(best_error)

        # Reference genotypes against the best unordered haplotype pair.
        pairs = []
        scorable = 0
        dosage_rows = haplotypes.astype(np.int16)
        for reference in range(references):
            confident = genotypes[reference] != -1
            sites_confident = int(confident.sum())
            if sites_confident < MIN_CONF_SITES:
                pairs.append(("", float("nan"), sites_confident))
                continue
            scorable += 1
            if count == 0:
                pairs.append(("", float("inf"), sites_confident))
                continue
            observed = genotypes[reference, confident].astype(np.int16)
            subset = dosage_rows[:, confident]
            valid = (subset[:, None, :] >= 0) & (subset[None, :, :] >= 0)
            dosage = subset[:, None, :] + subset[None, :, :]
            compared = np.sum(valid, axis=2)
            mismatches = np.sum((dosage != observed[None, None, :]) & valid, axis=2)
            error = np.divide(mismatches * 100.0, compared, out=np.full(compared.shape, np.inf, dtype=np.float64),
                              where=compared >= MIN_CONF_SITES)
            upper = np.triu_indices(count)
            best = int(np.argmin(error[upper]))
            first, second = int(upper[0][best]), int(upper[1][best])
            pairs.append((f"{first}+{second}", float(error[upper][best]), int(compared[first, second])))

        row = {
            "stage": stage, "contig": contig, "block": block_index, "n_sites": len(positions),
            "block_start": int(positions[0]), "block_end": int(positions[-1]),
            "n_g0_reference_samples": references,
            # The references are analysed samples: they took part in discovery.
            "g0_reference_is_independent": False,
            "n_scorable": scorable,
            "n_scorable_homonly": sum(1 for _, error, _ in best_for_reference if not np.isnan(error)),
            "n_discovered": count,
            matched_column: sum(1 for _, error, _ in best_for_reference
                                if not np.isnan(error) and error < MATCH_THRESHOLD_PCT),
            "references_matched_pair": sum(1 for _, error, _ in pairs
                                           if not np.isnan(error) and error < MATCH_THRESHOLD_PCT),
            "haplotypes_matching_reference": sum(1 for error in best_for_haplotype if error < MATCH_THRESHOLD_PCT),
            "haplotypes_without_reference_match": sum(1 for error in best_for_haplotype
                                                      if error >= MATCH_THRESHOLD_PCT),
        }
        for reference, (best_row, error, valid_sites) in enumerate(best_for_reference):
            prefix = f"{label}_{reference}_{names[reference]}"
            row[f"{prefix}_best_disc"] = best_row
            row[f"{prefix}_err_pct"] = error
            row[f"{prefix}_valid_sites"] = valid_sites
        for reference, (pair, error, confident_sites) in enumerate(pairs):
            prefix = f"{label}_{reference}_{names[reference]}"
            row[f"{prefix}_pair"] = pair
            row[f"{prefix}_pair_err_pct"] = error
            row[f"{prefix}_conf_sites"] = confident_sites
        rows.append(row)
    return rows


def contig_bytes(file) -> int:
    """Planned peak bytes of loading one contig's inputs: the evidence record (every sample's likelihoods
    dominate it; the load measured 0.96 of its decoded size on std_n80 chr1) and the discovery block records,
    decoded; 5/4 for allocator slack."""
    from ..discovery import records as discovery_records
    from ..evidence import records as evidence_records
    decoded = file.info(evidence_records.EVIDENCE).decoded_bytes + sum(
        info.decoded_bytes for info in file.infos(discovery_records.CHUNK_PREFIX))
    return decoded * 5 // 4


def write_reference_consistency(output, store, contigs, sample_ids, reference_rows, *,
                                label: str = "G0") -> Path | None:
    """Compare every discovered contig's blocks with the reference rows (indices into the run's sample axis
    ``sample_ids``); writes OUTPUT/discovery/reference_consistency.csv, or nothing without rows to report.
    ``label``: the reference generation, prefixing the per-reference columns (G0 for TroMau)."""
    import pandas as pd
    from ..discovery import records as discovery_records
    from ..evidence import records as evidence_records
    from ..execution.memory import reserve_memory
    from .tables import write_frame
    reference_rows = np.asarray(reference_rows, dtype=np.int64)
    names = [str(sample_ids[row]) for row in reference_rows]
    print(f"{label} genotype consistency (non-independent; references included in discovery); "
          "not founder or pedigree truth", flush=True)
    if not len(reference_rows):
        print("  No reference rows: no reference-consistency report.", flush=True)
        return None
    rows = []
    for contig in contigs:
        file = store.contig(contig)
        if not discovery_records.complete(file):
            print(f"  [skip] {contig}: no discovery record", flush=True)
            continue
        # Every sample's likelihoods are decoded to take the reference rows: the load holds a reservation of
        # the run's memory (contig_bytes; waits while it does not fit).
        with reserve_memory(contig_bytes(file), label="reference consistency"):
            evidence = evidence_records.load_arrays(file, ("positions", "likelihoods"))
            likelihoods = np.ascontiguousarray(evidence["likelihoods"][reference_rows])
            sites = evidence["positions"]
            del evidence
            panels = [(positions, calls) for positions, calls, _ in discovery_records.founder_panels(file)]
        current = compare_blocks(panels, likelihoods, sites, names, contig=contig, stage="discovery", label=label)
        rows.extend(current)
        print(f"  {contig}: {len(current)} blocks compared", flush=True)
    if not rows:
        print("  No reference-comparison rows available.", flush=True)
        return None
    path = write_frame(Path(output) / "discovery" / "reference_consistency.csv", pd.DataFrame(rows))
    print(f"Reference consistency: {len(rows)} blocks; {path}", flush=True)
    return path

