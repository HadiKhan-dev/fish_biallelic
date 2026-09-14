"""workflows / tropheops for the canonical reconstruction pipeline."""
from __future__ import annotations

import json


def configured_regions(default, *, template_regions=False):
    requested = os.environ.get("HAPLOTYPES_CONTIGS")
    if requested is None:
        return default
    names = json.loads(requested)
    if not names or len(names) != len(set(names)):
        raise ValueError("contigs must be a nonempty unique ordered list")
    return [dict(contig=str(name), **({'start': 0, 'end': 3000} if template_regions else {}))
            for name in names]


import os
import haplotype_reconstruction.assembly.pipeline as assembly_pipeline
import haplotype_reconstruction.core.environment as core_environment
import haplotype_reconstruction.core.genetic_map as core_genetic_map
import haplotype_reconstruction.core.haplotypes as core_haplotypes
import haplotype_reconstruction.core.numerics as core_numerics
import haplotype_reconstruction.core.parallel as core_parallel
import haplotype_reconstruction.core.runtime as core_runtime
import haplotype_reconstruction.core.variants as core_variants
import haplotype_reconstruction.discovery.blocks as discovery_blocks
import haplotype_reconstruction.discovery.search as discovery_search
import haplotype_reconstruction.pedigree.pipeline as pedigree_pipeline
import haplotype_reconstruction.recombination.model as module_recombination_model
import haplotype_reconstruction.recombination.pipeline as recombination_pipeline
import haplotype_reconstruction.refinement.pipeline as refinement_pipeline
import haplotype_reconstruction.refinement.model as refinement_model
import haplotype_reconstruction.workflows.design as workflows_design
import haplotype_reconstruction.workflows.reconstruction as workflows_reconstruction

INCLUDE_REFERENCE_SAMPLES = True


STAGE1_BACKEND = "reversible_cavity_depth_observation_v1"


_mode_label = "withFounders" if INCLUDE_REFERENCE_SAMPLES else "withoutFounders"


_run_label = f"{_mode_label}_{STAGE1_BACKEND}_stage2_component_v1"


CHECKPOINT_DIR = os.environ.get("HAPLOTYPES_CHECKPOINT_DIR", "work/runs/tropheops/checkpoints")


output_dir = os.environ.get("HAPLOTYPES_OUTPUT_DIR", "work/runs/tropheops")


def run():
    """Execute or resume the configured, chromosome-checkpointed workflow."""
    import os
    import sys
    from datetime import datetime

    # Enable faulthandler FIRST — catches C-level segfaults in numba-compiled
    # code, numpy, BLAS, etc. and prints a Python traceback to stderr before
    # the process dies.  Without this, such faults leave no trail (silent
    # worker death).  Writes to the parent's stderr so it also shows up in
    # the log file via TeeOutput below.
    import faulthandler
    faulthandler.enable()

    # FORCE NUMPY/BLAS TO USE 1 THREAD PER PROCESS
    core_environment.force_single_threaded_numeric_libraries()

    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    # INCLUDE_REFERENCE_SAMPLES, _mode_label, CHECKPOINT_DIR and output_dir are defined
    # at module top level. Edit the comparison flag there, not here.

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================

    os.makedirs(os.environ.get("HAPLOTYPES_LOG_DIR", "work/logs"), exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(os.environ.get("HAPLOTYPES_LOG_DIR", "work/logs"), f"run_{run_timestamp}.log")
    sys.stdout = core_runtime.TeeOutput(log_path, sys.stdout)
    print(f"Logging to: {log_path}")
    print(f"Run started: {run_timestamp}")
    print(f"INCLUDE_REFERENCE_SAMPLES = {INCLUDE_REFERENCE_SAMPLES}  (mode: {_mode_label})")
    print(f"STAGE1_BACKEND = {STAGE1_BACKEND}")
    print("STAGE2_ROUTE = component-local T01 -> T09")

    import numpy as np
    import pandas as pd
    import time
    import warnings
    import platform
    import gc
    from dataclasses import asdict
    from cyvcf2 import VCF

    np.seterr(divide='ignore', invalid='ignore')


    if STAGE1_BACKEND != discovery_blocks.STAGE1_BACKEND:
        raise RuntimeError("Stage-1 checkpoint identity mismatch")


    from dataclasses import replace

    rate_maps = core_genetic_map.load_genetic_maps_from_environment()
    inference_recombination_rate = rate_maps.default_rate_cm_per_mb / 1e8
    inference_genetic_maps = rate_maps if rate_maps.maps else None

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    if platform.system() != "Windows":
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

    n_processes = int(os.environ.get("BHD_NUM_PROCESSES", str(core_runtime.available_cpu_count())))
    # T01 uses the complete allocation: one block worker per Numba thread.
    # Dynamic reallocation gives the full budget to remaining stragglers.
    block_discovery_processes = n_processes
    block_discovery_numba_threads = n_processes
    # Reuse native code across four batches; trim between batches and retain
    # periodic process recycling to bound long-lived allocator fragmentation.
    WORKER_MAXTASKS = 4

    # Start forkserver before data loading
    _warmup_pool = core_parallel.NonDaemonicForkserverPool(1)
    _warmup_pool.terminate()
    _warmup_pool.join()
    del _warmup_pool
    print("Forkserver started (lightweight, pre-data).")
    print(f"Numba threading layer: {os.environ.get('NUMBA_THREADING_LAYER', 'not set')}")

    # =========================================================================
    # Paths & Regions (AcTm tropheops cross)
    # =========================================================================
    vcf_path = os.environ.get("HAPLOTYPES_VCF", "work/data/fish_vcf_restriped/AcTm.biallelic.bcf.gz")
    meta_path = os.environ.get("HAPLOTYPES_METADATA", "work/data/fish_vcf_restriped/X_AcTm_metadata.xlsx")

    # AcTm BCF covers the same reference as the AsAc files: chr1-chr20, chr22,
    # chr23 autosomes, plus chrM and U_scaffolds.  We only run the pipeline on
    # the 22 autosomes (chrM has no recombination; U_scaffolds are short/unplaced
    # and not useful for pedigree-scale linkage).
    regions_config = configured_regions([
        {"contig": "chr1"},  {"contig": "chr2"},  {"contig": "chr3"},
        {"contig": "chr4"},  {"contig": "chr5"},  {"contig": "chr6"},
        {"contig": "chr7"},  {"contig": "chr8"},  {"contig": "chr9"},
        {"contig": "chr10"}, {"contig": "chr11"}, {"contig": "chr12"},
        {"contig": "chr13"}, {"contig": "chr14"}, {"contig": "chr15"},
        {"contig": "chr16"}, {"contig": "chr17"}, {"contig": "chr18"},
        {"contig": "chr19"}, {"contig": "chr20"}, {"contig": "chr22"},
        {"contig": "chr23"},
    ], template_regions=False)

    # CHECKPOINT_DIR and output_dir are defined at module top level (above).
    # =========================================================================
    # Checkpoint Infrastructure (blosc2 via core/checkpoints — matches pipeline.py)
    # =========================================================================
    checkpoint_store = core_runtime.CheckpointStore(
        CHECKPOINT_DIR, nthreads=n_processes, global_log_indent="    "
    )
    os.makedirs(output_dir, exist_ok=True)
    stage_complete = checkpoint_store.stage_complete
    mark_stage_complete = checkpoint_store.mark_stage_complete
    contig_done = checkpoint_store.contig_done
    save_contig = checkpoint_store.save_contig
    load_contig = checkpoint_store.load_contig
    save_global = checkpoint_store.save_global


    # =========================================================================
    # VALIDATION HELPERS (module-level, shared across stages)
    # =========================================================================
    # T01 is compared with the four observed G0 genotype references stashed in
    # every per-contig checkpoint. This is a non-independent post-hoc check when
    # G0 rows participated in discovery and a held-out comparison otherwise.
    # Min argmax-prob to treat a G0 site as confidently homozygous.  Sites
    # below this confidence, or where the max state is heterozygous (state=1),
    # are masked out of the G0 reference comparison.
    HOM_CONFIDENCE = 0.85
    # A discovered haplotype is considered consistent with a homozygous G0
    # reference call if the allele-level disagreement rate is below this
    # threshold (in %).
    MATCH_THRESHOLD_PCT = 2.0
    MIN_CONF_SITES = 10   # min confident G0 sites to score a founder in a block

    def extract_g0_block_haps(g0_probs, g0_sites, block_positions):
        """Build observed G0 reference genotypes for one block.

        g0_probs has shape (n_g0, n_global_sites, 3) — genotype probabilities
        0=homref, 1=het, 2=homalt.  A site is kept when the max genotype
        probability >= HOM_CONFIDENCE; the stored value is then the genotype
        dosage 0/1/2 (hom-ref / het / hom-alt).  Low-confidence sites (and
        positions that fail to match the G0 site list) are masked to -1.

        Unlike the earlier hom-only version, HETEROZYGOUS sites are retained
        (dosage 1): a heterozygous founder carries two distinct founder
        haplotypes, and the pair-reconstruction recall uses exactly these het
        sites to require that BOTH are present in the discovered set.

        Returns: (n_g0, n_block_sites) int8 genotypes, values in {0, 1, 2, -1}
        """
        pos_idx = np.searchsorted(g0_sites, block_positions)
        pos_idx = np.clip(pos_idx, 0, len(g0_sites) - 1)
        matched = (g0_sites[pos_idx] == block_positions)

        n_g0 = g0_probs.shape[0]
        n_block = len(block_positions)
        g0_geno = np.full((n_g0, n_block), -1, dtype=np.int8)

        for g in range(n_g0):
            probs_g = g0_probs[g, pos_idx, :]
            argmax = np.argmax(probs_g, axis=1)          # 0/1/2 = dosage
            maxp = probs_g[np.arange(n_block), argmax]
            conf = (maxp >= HOM_CONFIDENCE) & matched
            g0_geno[g, conf] = argmax[conf].astype(np.int8)

        return g0_geno

    def validate_block_list_against_g0(blocks, g0_probs, g0_sites,
                                       g0_names, stage_label, contig_name):
        """Compare a list of blocks with observed G0 genotype references.

        The comparison is a non-independent post-hoc consistency check when
        G0 rows participated in discovery and a held-out reference comparison
        when those rows were excluded.

        Two recall metrics are reported per block, side by side:

          * hom-only single-haplotype (A/B reference): each founder's
            confident HOMOZYGOUS consensus is matched to the single best
            discovered haplotype.  Columns: founders_found_under_<X>pct and the
            per-G0 *_err_pct / *_valid_sites.  A founder with < MIN_CONF_SITES
            confident hom sites is unscored (NaN) — this is what previously
            depressed the count for heterozygous founders.

          * pair-reconstruction (genotype-aware, PRIMARY): each founder's
            confident GENOTYPE (dosage 0/1/2, het sites kept) is matched to the
            best PAIR of discovered haplotypes (i <= j) by summed dosage
            d_i + d_j.  A heterozygous founder is "found" only when BOTH of its
            haplotypes are present in the discovered set; a homozygous founder
            reduces to the hom-only case (best pair is one hap used twice).
            Columns: founders_found_pair, n_scorable, per-G0 *_pair /
            *_pair_err_pct / *_conf_sites.

        'All founders found' (pair) is judged relative to the SCORABLE founders
        (n_scorable = founders with >= MIN_CONF_SITES confident genotype sites),
        so a founder that cannot be scored in a block is not counted as a miss.

        good_haps / chimeras (precision) stay on the hom-site basis.
        """
        rows = []
        n_g0 = g0_probs.shape[0]
        ff_col = f'founders_found_under_{MATCH_THRESHOLD_PCT:.0f}pct'

        for block_idx, block in enumerate(blocks):
            positions = block.positions
            if len(positions) == 0:
                continue

            # Observed G0 reference genotypes {0,1,2,-1}; hom-only calls
            # {0,1,-1}.
            g0_geno = extract_g0_block_haps(g0_probs, g0_sites, positions)
            g0_hom = np.where(g0_geno == 0, 0,
                              np.where(g0_geno == 2, 1, -1)).astype(np.int8)

            # Score only explicit Stage-1 calls; unknown founder alleles are
            # excluded rather than hardened to the reference allele.
            discovered = [
                (
                    hid,
                    np.asarray(block.discrete_haps[hid], dtype=np.int8),
                )
                for hid in block.haplotypes
            ]
            n_disc = len(discovered)
            disc_ids = [hid for hid, _ in discovered]
            if n_disc > 0:
                D = np.vstack([h for _, h in discovered]).astype(np.int16)  # (n_disc, n_block)
            else:
                D = np.zeros((0, len(positions)), dtype=np.int16)

            # ---- hom-only single-haplotype recall ----
            g0_best_matches = []  # (g, best_disc_id, err_pct, n_valid_sites)
            for g in range(n_g0):
                g_valid = (g0_hom[g] != -1)
                if np.sum(g_valid) < MIN_CONF_SITES:
                    g0_best_matches.append((g, -1, float('nan'), int(np.sum(g_valid))))
                    continue
                best_err = 101.0
                best_id = -1
                for (hid, disc_h) in discovered:
                    disc_valid = (disc_h != -1) if -1 in disc_h else np.ones_like(disc_h, dtype=bool)
                    mask = g_valid & disc_valid
                    if np.sum(mask) < MIN_CONF_SITES:
                        continue
                    err = np.mean(g0_hom[g, mask] != disc_h[mask]) * 100.0
                    if err < best_err:
                        best_err = err
                        best_id = hid
                g0_best_matches.append((g, best_id, best_err, int(np.sum(g_valid))))

            # ---- precision: each discovered hap's best hom-only G0 match ----
            disc_best_matches = []
            for (hid, disc_h) in discovered:
                disc_valid = (disc_h != -1) if -1 in disc_h else np.ones_like(disc_h, dtype=bool)
                best_err = 101.0
                best_g = -1
                best_n = 0
                for g in range(n_g0):
                    g_valid = (g0_hom[g] != -1)
                    mask = g_valid & disc_valid
                    if np.sum(mask) < MIN_CONF_SITES:
                        continue
                    err = np.mean(g0_hom[g, mask] != disc_h[mask]) * 100.0
                    if err < best_err:
                        best_err = err
                        best_g = g
                        best_n = int(np.sum(mask))
                disc_best_matches.append((hid, best_g, best_err, best_n))

            # ---- pair-reconstruction (genotype-aware) recall ----
            g0_pair_matches = []  # (g, pair_str, err_pct, n_conf_sites)
            n_scorable = 0
            for g in range(n_g0):
                conf = (g0_geno[g] != -1)
                n_conf = int(conf.sum())
                if n_conf < MIN_CONF_SITES:
                    g0_pair_matches.append((g, '', float('nan'), n_conf))
                    continue
                n_scorable += 1
                if n_disc == 0:
                    g0_pair_matches.append((g, '', float('inf'), n_conf))
                    continue
                geno_m = g0_geno[g, conf].astype(np.int16)
                Dm = D[:, conf]
                valid = (
                    (Dm[:, None, :] >= 0)
                    & (Dm[None, :, :] >= 0)
                )
                dosage = Dm[:, None, :] + Dm[None, :, :]
                n_valid = np.sum(valid, axis=2)
                mismatch = np.sum(
                    (dosage != geno_m[None, None, :]) & valid,
                    axis=2,
                )
                error = np.divide(
                    mismatch * 100.0,
                    n_valid,
                    out=np.full(n_valid.shape, np.inf, dtype=np.float64),
                    where=n_valid >= MIN_CONF_SITES,
                )
                iu = np.triu_indices(n_disc)
                flat = error[iu]
                best = int(np.argmin(flat))
                best_err = float(flat[best])
                bi, bj = int(iu[0][best]), int(iu[1][best])
                evaluated = int(n_valid[bi, bj])
                g0_pair_matches.append((
                    g,
                    f"{disc_ids[bi]}+{disc_ids[bj]}",
                    best_err,
                    evaluated,
                ))

            # ---- block-level metrics ----
            founders_found = sum(
                1 for (_, _, err, _) in g0_best_matches
                if not np.isnan(err) and err < MATCH_THRESHOLD_PCT
            )
            # founders with >=MIN_CONF_SITES confident HOM sites (hom-only scorable set)
            n_scorable_homonly = sum(
                1 for (_, _, err, _) in g0_best_matches if not np.isnan(err)
            )
            founders_found_pair = sum(
                1 for (_, _, err, _) in g0_pair_matches
                if not np.isnan(err) and err < MATCH_THRESHOLD_PCT
            )
            chimera_count = sum(
                1 for (_, _, err, _) in disc_best_matches if err >= MATCH_THRESHOLD_PCT
            )
            good_count = sum(
                1 for (_, _, err, _) in disc_best_matches if err < MATCH_THRESHOLD_PCT
            )

            row = {
                'stage': stage_label,
                'contig': contig_name,
                'block': block_idx,
                'n_sites': len(positions),
                'block_start': int(positions[0]),
                'block_end': int(positions[-1]),
                # Retained for CSV compatibility; these are observed G0
                # reference rows, not independent truth when included above.
                'n_true_founders': n_g0,
                'n_g0_reference_samples': n_g0,
                'g0_reference_is_independent': not INCLUDE_REFERENCE_SAMPLES,
                'n_scorable': n_scorable,
                'n_scorable_homonly': n_scorable_homonly,
                'n_discovered': n_disc,
                ff_col: founders_found,
                'founders_found_pair': founders_found_pair,
                'good_haps': good_count,
                'chimeras': chimera_count,
            }
            for g, bid, err, nsites in g0_best_matches:
                row[f'G0_{g}_{g0_names[g]}_best_disc'] = bid
                row[f'G0_{g}_{g0_names[g]}_err_pct'] = err
                row[f'G0_{g}_{g0_names[g]}_valid_sites'] = nsites
            for g, pair_str, err, nconf in g0_pair_matches:
                row[f'G0_{g}_{g0_names[g]}_pair'] = pair_str
                row[f'G0_{g}_{g0_names[g]}_pair_err_pct'] = err
                row[f'G0_{g}_{g0_names[g]}_conf_sites'] = nconf
            rows.append(row)

        return rows

    def load_g0_from_t1(r_name):
        """Cheaply load only the G0-reference fields from T01 (skips the big
        global_probs / block_results / site priors). Used by every post-stage
        validation pass so we don't reload the full T01 pickle just to get
        g0_probs."""
        t1 = load_contig("T00_founder_templates", r_name)
        g0_probs = t1['g0_probs']
        g0_sites = t1['global_sites']
        g0_names = t1['g0_sample_names']
        del t1
        return g0_probs, g0_sites, g0_names

    def run_stage_validation(stage_label, stage_key, blocks_loader_fn, csv_filename):
        """Compare block haplotypes with observed G0 genotype references.

        This is a post-hoc consistency diagnostic when ``INCLUDE_REFERENCE_SAMPLES``
        is true because the same rows participated in reconstruction. It is
        an independent held-out comparison only when those rows were excluded.

        Args:
            stage_label: Human-readable tag written to the CSV stage column.
            stage_key: Checkpoint directory name used to check contig completion.
            blocks_loader_fn: Callable returning the block list for one contig.
            csv_filename: Filename under output_dir for the per-block CSV.

        Runs unconditionally on each pipeline invocation (the check is fast).
        """
        print(f"\n{'='*60}")
        validation_kind = (
            "post-hoc G0 consistency (non-independent)"
            if INCLUDE_REFERENCE_SAMPLES
            else "held-out G0 reference comparison"
        )
        print(f"VALIDATION: {stage_label} — {validation_kind}")
        print(f"{'='*60}")

        all_rows = []
        contigs_with_data = 0
        ff_col = f'founders_found_under_{MATCH_THRESHOLD_PCT:.0f}pct'

        for r_name in region_keys:
            if not contig_done(stage_key, r_name):
                print(f"  [skip] {r_name}: no checkpoint in {stage_key}")
                continue
            if not contig_done("T00_founder_templates", r_name):
                print(f"  [skip] {r_name}: no T01 checkpoint (needed for G0 reference)")
                continue

            g0_probs, g0_sites, g0_names = load_g0_from_t1(r_name)
            blocks = blocks_loader_fn(r_name)

            rows = validate_block_list_against_g0(
                blocks, g0_probs, g0_sites, g0_names,
                stage_label=stage_label, contig_name=r_name
            )
            all_rows.extend(rows)
            contigs_with_data += 1

            if rows:
                mean_haps = np.mean([r['n_discovered'] for r in rows])
                mean_good = np.mean([r['good_haps'] for r in rows])
                mean_chim = np.mean([r['chimeras'] for r in rows])
                all_found = sum(1 for r in rows if r[ff_col] == r['n_scorable_homonly'])
                all_found_pair = sum(1 for r in rows if r['founders_found_pair'] == r['n_scorable'])
                mean_scor = np.mean([r['n_scorable'] for r in rows])
                print(f"  {r_name}: {len(rows)} blocks, mean {mean_haps:.1f} haps/block, "
                      f"all-found pair: {all_found_pair}/{len(rows)} ({100*all_found_pair/len(rows):.1f}%), "
                      f"hom-only: {all_found}/{len(rows)} ({100*all_found/len(rows):.1f}%), "
                      f"scorable={mean_scor:.1f}/{rows[0]['n_true_founders']}, "
                      f"good={mean_good:.1f}, chim={mean_chim:.1f}")

            del g0_probs, blocks
            gc.collect()

        if all_rows:
            df = pd.DataFrame(all_rows)
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False)

            total_blocks = len(df)
            total_all_found = int((df[ff_col] == df['n_scorable_homonly']).sum())
            total_all_found_pair = int((df['founders_found_pair'] == df['n_scorable']).sum())
            overall_good = df['good_haps'].mean()
            overall_chim = df['chimeras'].mean()
            overall_disc = df['n_discovered'].mean()
            overall_scor = df['n_scorable'].mean()

            print(f"\n  Overall across {contigs_with_data} contigs:")
            print(f"    Total blocks: {total_blocks}")
            print(f"    Mean discovered haps per block: {overall_disc:.2f}")
            print(f"    Mean scorable founders per block: {overall_scor:.2f} / {df['n_true_founders'].iloc[0]}")
            print(f"    Blocks with ALL scorable founders recovered, PAIR "
                  f"(<{MATCH_THRESHOLD_PCT:.0f}% err): {total_all_found_pair} "
                  f"({100*total_all_found_pair/total_blocks:.1f}%)")
            print(f"    Blocks with ALL scorable founders recovered, hom-only "
                  f"(<{MATCH_THRESHOLD_PCT:.0f}% err): {total_all_found} "
                  f"({100*total_all_found/total_blocks:.1f}%)")
            print(f"    Mean good haps per block: {overall_good:.2f}")
            print(f"    Mean chimera haps per block: {overall_chim:.2f}")
            print(f"  CSV: {csv_path}")
        else:
            print(f"  WARNING: no validation rows produced for {stage_label}")


    region_keys = [r['contig'] for r in regions_config]

    # =========================================================================
    # SAMPLE IDENTIFICATION — match VCF samples to metafile, find G0 indices
    # =========================================================================
    # This runs before any stage so we always know:
    #   g0_vcf_indices      : positions of the 4 G0 samples in the VCF header
    #   active_vcf_indices  : positions of the samples the pipeline will see
    #                         (all 116 if INCLUDE_REFERENCE_SAMPLES, else 112 = no G0s)
    #   sample_names_active : VCF sample names the pipeline will see
    #                         (the ordered T01 and component-T09 sample axis)
    #   g0_sample_names     : the 4 G0 primary_IDs (for post-hoc validation)
    print(f"\n{'='*60}")
    print("Sample Identification (VCF <-> metafile)")
    print(f"{'='*60}")

    _vcf_tmp = VCF(vcf_path)
    sample_names = list(_vcf_tmp.samples)
    _vcf_tmp.close()
    n_samples_total = len(sample_names)
    print(f"VCF samples: {n_samples_total}")

    # Load metafile main_data sheet — contains generation column
    meta_df = pd.read_excel(meta_path, sheet_name=os.environ.get('HAPLOTYPES_METADATA_SHEET', 'main_data'))
    print(f"Metafile main_data rows: {len(meta_df)}")

    # Match BCF samples to metafile by primary_ID (user verified this is the
    # ID column with 116/116 matches).
    bcf_set = set(sample_names)
    matched_meta = meta_df[meta_df['primary_ID'].astype(str).isin(bcf_set)].copy()
    print(f"Matched {len(matched_meta)}/{n_samples_total} VCF samples via primary_ID")

    unmatched = bcf_set - set(matched_meta['primary_ID'].astype(str))
    if unmatched:
        print(f"WARNING: {len(unmatched)} VCF samples not in metafile:")
        for s in sorted(unmatched)[:5]:
            print(f"  {s}")
        # Do not hard-fail: unmatched samples remain active but cannot be
        # identified as G0 reference rows from metadata.

    # Build a primary_ID -> generation lookup
    id_to_gen = dict(zip(matched_meta['primary_ID'].astype(str),
                         matched_meta['generation'].astype(str)))

    # Identify G0 indices in the VCF sample list
    g0_vcf_indices = []
    g0_sample_names = []
    for i, s in enumerate(sample_names):
        if id_to_gen.get(s) == 'G0':
            g0_vcf_indices.append(i)
            g0_sample_names.append(s)

    if len(g0_vcf_indices) != 4:
        print(f"WARNING: Expected 4 G0 samples, found {len(g0_vcf_indices)}: "
              f"{g0_sample_names}")
    else:
        print(f"Identified 4 G0 samples at VCF indices {g0_vcf_indices}:")
        for idx, name in zip(g0_vcf_indices, g0_sample_names):
            print(f"  [{idx}] {name}")

    # Decide which samples the pipeline will see
    if INCLUDE_REFERENCE_SAMPLES:
        active_vcf_indices = np.arange(n_samples_total, dtype=np.int64)
        print(f"\nINCLUDE_REFERENCE_SAMPLES=True -> pipeline sees ALL {n_samples_total} samples "
              f"(G0 included)")
    else:
        g0_set = set(g0_vcf_indices)
        active_vcf_indices = np.array(
            [i for i in range(n_samples_total) if i not in g0_set],
            dtype=np.int64
        )
        print(f"\nINCLUDE_REFERENCE_SAMPLES=False -> pipeline sees {len(active_vcf_indices)} "
              f"samples (G0 removed)")

    sample_names_active = [sample_names[i] for i in active_vcf_indices]

    # Sanity-log generation composition of active samples
    gen_counts_active = pd.Series(
        [id_to_gen.get(s, '?') for s in sample_names_active]
    ).value_counts()
    print(f"Active sample generation breakdown:")
    for gen, count in gen_counts_active.items():
        print(f"  {gen}: {count}")

    print(f"Regions: {len(region_keys)}")

    # =========================================================================
    # STAGE T01: VCF Loading + Block Discovery + Global Probabilities
    # =========================================================================
    # Identical to pipeline_real.py STAGE R01, with ONE addition: we always
    # split out the G0 reads into a separate `g0_slice` that's stashed in the
    # checkpoint for T01 validation and missing-aware Stage 2.  When
    # INCLUDE_REFERENCE_SAMPLES=False, the main global_probs/global_sites/block_results
    # are computed from the 112 non-G0 samples only (the reads array is sliced
    # along the sample axis before reads_to_probabilities / block discovery).
    STAGE_T1 = "T00_founder_templates"
    discovery_config = discovery_search.ReversibleCavitySearchConfig()
    discovery_config_record = asdict(discovery_config)
    stage1_identity_record = {
        "backend": STAGE1_BACKEND,
        "config": discovery_config_record,
    }
    checkpoint_store.bind_stage_identity(
        STAGE_T1, stage1_identity_record
    )

    if stage_complete(STAGE_T1):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T01: VCF Loading + Block Haplotype Discovery")
        print(f"{'='*60}")
        start = time.time()
        print(
            "  Cap-free reversible cavity discovery: no explicit K grid or "
            "scientific K cap; "
            f"beam_width={discovery_config.beam_width}, "
            f"max_expansions={discovery_config.max_expansions}, "
            f"max_exact_scores={discovery_config.max_exact_scores}, "
            "max_proposals_per_expansion="
            f"{discovery_config.max_proposals_per_expansion}"
        )
        print(
            "  Block discovery parallelism: "
            f"workers={block_discovery_processes}, "
            f"Numba budget={block_discovery_numba_threads}"
        )

        with discovery_blocks.BlockDiscoveryPool(
            block_discovery_processes,
            block_discovery_numba_threads,
        ) as block_pool:
            for r_name in region_keys:
                if contig_done(STAGE_T1, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue
                print(f"\n  Processing {r_name}...")

                t0 = time.time()
                genomic_data = core_variants.cleanup_block_reads_list(
                    vcf_path, r_name,
                    use_snp_count=True, snps_per_block=200, snp_shift=200,
                    num_processes=n_processes
                )
                print(f"    [Loader] {len(genomic_data)} blocks in {time.time()-t0:.1f}s")

                # Full reads: (n_samples_total, n_sites, 2) — all 116 samples
                global_sites, global_reads_full = (
                    core_variants.concatenate_unique_block_reads(genomic_data)
                )
                if global_sites is None:
                    print(f"    WARNING: No data for {r_name}, skipping")
                    continue

                # ALWAYS extract G0 reads separately for post-hoc validation.
                # This slice is independent of the INCLUDE_REFERENCE_SAMPLES flag — we want ground
                # truth available regardless of what the pipeline sees.
                g0_reads = global_reads_full[g0_vcf_indices, :, :]
                (_, g0_probs) = core_numerics.reads_to_probabilities(
                    g0_reads,
                    use_hwe_prior=False,
                )
                # Downcast G0 probs to float32 — we only use argmax for validation,
                # so float64 precision is wasted.
                if g0_probs.dtype == np.float64:
                    g0_probs = g0_probs.astype(np.float32)

                # Select which samples the pipeline will see (116 or 112).
                # IMPORTANT: we also need to slice genomic_data.reads along the
                # sample axis so block_haplotypes.generate_all_block_haplotypes
                # operates on the filtered sample set.  The GenomicData container
                # stores per-block (samples, sites, 2) arrays.
                if INCLUDE_REFERENCE_SAMPLES:
                    active_reads_full = global_reads_full
                else:
                    active_reads_full = global_reads_full[active_vcf_indices, :, :]
                    # Also filter genomic_data in place so block discovery sees 112 samples
                    for bi in range(len(genomic_data.reads)):
                        if genomic_data.reads[bi].shape[0] == n_samples_total:
                            genomic_data.reads[bi] = genomic_data.reads[bi][active_vcf_indices, :, :]

                # Preserve the exact observation event before read counts are
                # released. Zero-depth cells are scientifically distinct from
                # uncertain observed genotypes and must remain state-neutral in
                # missing-aware Stage 2.
                global_observed_mask = (
                    workflows_reconstruction.observed_call_mask_from_read_counts(
                        active_reads_full
                    )
                )

                # Downstream linkage models require per-sample genotype
                # likelihoods, not the empirical HWE posterior used as an
                # optional regularizer within local haplotype discovery.
                (site_priors, global_probs) = core_numerics.reads_to_probabilities(
                    active_reads_full,
                    use_hwe_prior=False,
                )
                avg_depth = np.mean(np.sum(active_reads_full, axis=-1))
                print(f"    Sites: {len(global_sites)}, Samples (active): {global_probs.shape[0]}, "
                      f"Depth: {avg_depth:.1f}x")
                del global_reads_full, active_reads_full, g0_reads, site_priors

                t0 = time.time()
                block_results = discovery_blocks.generate_all_block_haplotypes(
                    genomic_data,
                    num_processes=block_discovery_processes,
                    discovery_config=discovery_config,
                    total_numba_threads=block_discovery_numba_threads,
                    block_pool=block_pool,
                )
                valid_blocks = [b for b in block_results if len(b.positions) > 0]
                block_results = core_haplotypes.BlockResults(valid_blocks)

                hap_counts = [len(b.haplotypes) for b in valid_blocks]
                print(f"    [Discovery] {len(valid_blocks)} blocks, haps/block: "
                      f"min={min(hap_counts)}, max={max(hap_counts)}, "
                      f"mean={np.mean(hap_counts):.1f} in {time.time()-t0:.1f}s")

                selected_k = np.asarray(
                    [int(block.K_final) for block in valid_blocks],
                    dtype=np.int64,
                )
                k_values, k_counts = np.unique(
                    selected_k, return_counts=True
                )
                k_distribution = {
                    int(k): int(count)
                    for k, count in zip(k_values, k_counts)
                }
                cavity_blocks = [
                    block
                    for block in valid_blocks
                    if hasattr(block, 'cavity_discovery_diagnostics')
                    and hasattr(block, 'cavity_selection')
                ]
                cavity_diagnostics = [
                    block.cavity_discovery_diagnostics
                    for block in cavity_blocks
                ]
                cavity_selections = [
                    block.cavity_selection for block in cavity_blocks
                ]
                boundary_count = sum(
                    bool(diagnostic['boundary_limited'])
                    for diagnostic in cavity_diagnostics
                )
                candidate_searches = [
                    diagnostic.get('candidate_search', {})
                    for diagnostic in cavity_diagnostics
                ]
                search_limited_count = sum(
                    bool(candidate_search.get('search_limited', False))
                    for candidate_search in candidate_searches
                )
                search_limit_reason_counts = {}
                for candidate_search in candidate_searches:
                    for reason in candidate_search.get(
                        'search_limit_reasons', ()
                    ):
                        search_limit_reason_counts[reason] = (
                            search_limit_reason_counts.get(reason, 0) + 1
                        )
                if len(cavity_blocks) != len(valid_blocks):
                    raise AssertionError(
                        "every informative Stage-1 block must use the "
                        "canonical cavity model"
                    )
                score_margins = np.asarray([
                    float(selection.log_score_by_k[selection.map_k])
                    - float(selection.log_score_by_k[selection.runner_up_k])
                    for selection in cavity_selections
                    if selection.runner_up_k is not None
                ], dtype=np.float64)
                mode_cap_count = sum(
                    bool(selection.mode_cap_applied)
                    for selection in cavity_selections
                )
                uncertainty_count = sum(
                    bool(block.uncertainty_flag)
                    for block in cavity_blocks
                )
                nonconverged_count = sum(
                    not bool(selection.all_mean_field_converged)
                    for selection in cavity_selections
                )
                shortlist_sizes = np.asarray([
                    len(selection.hybrid_diagnostic.shortlisted_k)
                    for selection in cavity_selections
                    if selection.hybrid_diagnostic is not None
                ], dtype=np.int64)
                uncertainty_reason_counts = {}
                for diagnostic in cavity_diagnostics:
                    for reason in diagnostic['uncertainty_reasons']:
                        uncertainty_reason_counts[reason] = (
                            uncertainty_reason_counts.get(reason, 0) + 1
                        )
                score_margin_summary = (
                    "unavailable"
                    if len(score_margins) == 0
                    else "min/median/max=" + "/".join(
                        f"{value:.6f}" for value in (
                            np.min(score_margins),
                            np.median(score_margins),
                            np.max(score_margins),
                        )
                    )
                )
                shortlist_size_summary = (
                    "unavailable"
                    if len(shortlist_sizes) == 0
                    else "min/median/max=" + "/".join(
                        f"{value:.0f}" for value in (
                            np.min(shortlist_sizes),
                            np.median(shortlist_sizes),
                            np.max(shortlist_sizes),
                        )
                    )
                )
                wildcard_mass = np.asarray(
                    [float(block.wildcard_mass) for block in valid_blocks],
                    dtype=np.float64,
                )
                wildcard_quartiles = np.quantile(
                    wildcard_mass, [0.0, 0.25, 0.5, 0.75, 1.0]
                )
                print(
                    "    [Cavity audit] selected K distribution="
                    f"{k_distribution}, mean={np.mean(selected_k):.3f}"
                )
                print(
                    "    [Cavity audit] search-boundary blocks="
                    f"{boundary_count}/{len(cavity_diagnostics)}; "
                    "operationally search-limited blocks="
                    f"{search_limited_count}/{len(candidate_searches)}"
                )
                print(
                    "    [Cavity audit] search-limit reasons="
                    f"{search_limit_reason_counts}"
                )
                print(
                    "    [Cavity audit] winner/runner-up log-score margin "
                    f"{score_margin_summary}; hybrid shortlist size "
                    f"{shortlist_size_summary}"
                )
                print(
                    "    [Cavity audit] mode-cap blocks="
                    f"{mode_cap_count}/{len(cavity_selections)}; "
                    "materialization uncertainty="
                    f"{uncertainty_count}/{len(cavity_blocks)}; "
                    "mean-field nonconverged="
                    f"{nonconverged_count}/{len(cavity_selections)}"
                )
                print(
                    "    [Cavity audit] uncertainty reasons="
                    f"{uncertainty_reason_counts}"
                )
                print(
                    "    [Cavity audit] wildcard nonzero="
                    f"{np.count_nonzero(wildcard_mass > 0.0)}/"
                    f"{len(wildcard_mass)}; "
                    "q0/q25/q50/q75/q100="
                    + "/".join(
                        f"{value:.6f}" for value in wildcard_quartiles
                    )
                )

                # G0 probabilities are retained as an explicit reference; they are
                # non-independent when G0 rows entered reconstruction above.
                save_contig(STAGE_T1, r_name, {
                    'global_probs': global_probs, 'global_sites': global_sites,
                    'global_observed_mask': global_observed_mask,
                    'observed_call_mask_mode': (
                        workflows_reconstruction.EXACT_OBSERVED_MASK_MODE),
                    'genotype_evidence_mode': (
                        'normalized_raw_linear_likelihood_v1'),
                    'block_results': block_results, 'avg_depth': avg_depth,
                    'g0_probs': g0_probs, 'g0_sample_names': g0_sample_names,
                    'active_vcf_indices': active_vcf_indices,
                    'stage1_backend': STAGE1_BACKEND,
                    'stage1_config': discovery_config_record,
                })
                del genomic_data, block_results, global_probs, global_sites
                del global_observed_mask, g0_probs
                gc.collect()

        save_global(STAGE_T1, {
            'sample_ids': sample_names_active,
            'sample_names_full': sample_names,
            'contigs': region_keys,
            'g0_vcf_indices': g0_vcf_indices,
            'g0_sample_names': g0_sample_names,
            'active_vcf_indices': active_vcf_indices,
            'use_known_founders': INCLUDE_REFERENCE_SAMPLES,
            'genotype_evidence_mode': 'normalized_raw_linear_likelihood_v1',
            'observed_call_mask_mode': (
                workflows_reconstruction.EXACT_OBSERVED_MASK_MODE
            ),
            'stage1_backend': STAGE1_BACKEND,
            'stage1_config': discovery_config_record,
        })
        print(f"\nVCF loading + discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T1)

    # =========================================================================
    # VALIDATION: After T01 Block Discovery
    # =========================================================================
    # Compare raw 200-SNP haplotypes with the four observed G0 references.
    # In withFounders mode this is explicitly post-hoc and non-independent.
    run_stage_validation(
        stage_label="T01_block_discovery",
        stage_key="T00_founder_templates",
        blocks_loader_fn=lambda r: load_contig("T00_founder_templates", r)['block_results'],
        csv_filename="validation_T01_block_discovery.csv"
    )


    # =====================================================================
    # STAGE 2 THROUGH TYPED COMPONENT T09
    # =====================================================================
    # The canonical release preprocesses the raw T01 blocks, assembles
    # rectangular-K phase components through L1-L4, and paints each component
    # in an independent founder namespace using the exact T01 observation mask.
    stage2_config = workflows_reconstruction.ReconstructionConfig(
        release_config=assembly_pipeline.AssemblyConfig(
            num_processes=n_processes,
            maxtasksperchild=WORKER_MAXTASKS,
            recombination_rate=inference_recombination_rate,
        ),
        paint_cores=n_processes,
        paint_recombination_rate=inference_recombination_rate,
    )
    print(f"\n{'='*60}")
    print("STAGE 2: T01 -> COMPONENT T09")
    print(f"{'='*60}")
    print(
        "  Sequential contigs; release and painting are non-overlapping "
        f"phases with a {n_processes}-core ceiling"
    )
    stage2_summaries = workflows_reconstruction.run_reconstruction(
        checkpoint_store,
        region_keys,
        sample_names_active,
        stage1_identity=stage1_identity_record,
        config=stage2_config,
        genetic_maps=inference_genetic_maps,
        source_stage=STAGE_T1,
    )
    for summary in stage2_summaries:
        status = "resumed" if summary.resumed else "completed"
        print(
            f"  [{status}] {summary.contig}: "
            f"components={summary.component_count}, "
            "evidence-eligible component-sample pairs="
            f"{summary.evidence_eligible_component_sample_pairs}/"
            f"{summary.total_component_sample_pairs}, "
            f"observation mask={summary.observed_mask_mode}"
        )
    # Existing F2 <- F1 design eligibility excludes G0 and outside-pedigree
    # reference samples; it does not assume an individual parental pair.
    parent_eligibility = workflows_design.build_tropheops_parent_eligibility(
        meta_df, sample_names_active, require_opposite_sex_pair=True,
    )
    _stage10_summaries, stage10_payload = pedigree_pipeline.run_pedigree(
        checkpoint_store, region_keys, sample_names_active,
        output_dir=output_dir, n_workers=n_processes,
        raw_gl_stage=STAGE_T1, raw_sites_stage=STAGE_T1,
        parent_eligibility=parent_eligibility,
        genetic_maps=inference_genetic_maps, recombination_rate=inference_recombination_rate,
    )
    print("STAGE 11: pedigree-conditioned refinement and canonical final phase polishing")
    refinement_pipeline.run_refinement(
        checkpoint_store, region_keys, sample_names_active,
        pedigree_payload=stage10_payload, output_dir=output_dir,
        raw_gl_stage=STAGE_T1, raw_sites_stage=STAGE_T1,
        n_workers=n_processes,
        genetic_maps=inference_genetic_maps,
        config=refinement_model.FamilyRefinementConfig(recombination_rate=inference_recombination_rate),
    )
    recombination_pipeline.run_recombination(
        checkpoint_store, region_keys, sample_names_active,
        pedigree_payload=stage10_payload, output_dir=output_dir,
        n_workers=n_processes,
        genetic_maps=inference_genetic_maps,
        config=module_recombination_model.RecombinationMapConfig(recombination_rate=inference_recombination_rate),
    )


if __name__ == "__main__":
    run()
