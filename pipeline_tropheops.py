#%%
# =============================================================================
# MODULE-LEVEL DEFINITIONS (config + PICKLABLE forkserver callbacks)
# =============================================================================
# Functions/closures defined inside `if __name__ == '__main__':` cannot be
# pickled by multiprocessing.  Forkserver workers (phase correction, greedy
# refinement) receive their callbacks via pickle, so any function that crosses
# the worker boundary MUST live at module top level here.  This mirrors the
# module-level section at the top of pipeline.py.
#
# The pipeline configuration (USE_KNOWN_FOUNDERS) and the checkpoint/output
# directories live here too, because the picklable loader below needs
# CHECKPOINT_DIR at import time.  The __main__ block reads these module-level
# values rather than redefining them.

import os
import pipeline_runtime

from thread_env import force_single_threaded_numeric_libraries
# -----------------------------------------------------------------------------
# CONFIGURATION — EDIT THIS to switch between the two comparison runs.
# -----------------------------------------------------------------------------
# USE_KNOWN_FOUNDERS controls whether the 4 G0 (parental) samples are fed into
# the reconstruction pipeline:
#   True  -> all 116 samples go through T01-T11 (easy mode: G0s are mostly
#            homozygous for distinct parental species and form clean founder
#            clusters on their own during block discovery)
#   False -> the 4 G0 sample rows are sliced out, so T01-T11 see only the 112
#            admixed F1+F2 samples (hard mode: parental haplotypes must be
#            reconstructed purely from offspring)
# In BOTH modes the 4 G0 reads are ALSO loaded separately and stashed in the
# T01 checkpoint as genotype-reference rows. Their comparison is independent
# only when those rows are excluded from reconstruction.
USE_KNOWN_FOUNDERS = True

_mode_label = "withFounders" if USE_KNOWN_FOUNDERS else "withoutFounders"
# This label is part of the checkpoint/output identity. Bump it whenever the
# T01 scientific backend or its routine configuration changes, so a run cannot
# silently resume legacy or otherwise incompatible T01 contigs.
BLOCK_DISCOVERY_BACKEND = "reversible_cavity_cap_free_v1"
_run_label = f"{_mode_label}_{BLOCK_DISCOVERY_BACKEND}"
CHECKPOINT_DIR = f".pipeline_checkpoints_tropheops_{_run_label}"
output_dir = f"results_tropheops_{_run_label}"


def _load_contig_for_phase_correction(r_name):
    """Load the atomic final-panel painting bundle for phase correction."""
    return pipeline_runtime.load_phase_correction_inputs(
        CHECKPOINT_DIR,
        r_name,
        tolerance_stage="T09_viterbi_painting",
        strip_founder_probs=True,
    )


#%%
if __name__ == '__main__':
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
    force_single_threaded_numeric_libraries()

    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    # USE_KNOWN_FOUNDERS, _mode_label, CHECKPOINT_DIR and output_dir are defined
    # at MODULE TOP LEVEL (see the top of this file).  The picklable forkserver
    # callback _load_contig_for_phase_correction needs CHECKPOINT_DIR at import
    # time, so the config must exist there.  EDIT THE FLAG THERE, not here.

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================

    os.makedirs("logs", exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", f"run_tropheops_{_run_label}_{run_timestamp}.log")
    sys.stdout = pipeline_runtime.TeeOutput(log_path, sys.stdout)
    print(f"Logging to: {log_path}")
    print(f"Run started: {run_timestamp}")
    print(f"USE_KNOWN_FOUNDERS = {USE_KNOWN_FOUNDERS}  (mode: {_mode_label})")
    print(f"BLOCK_DISCOVERY_BACKEND = {BLOCK_DISCOVERY_BACKEND}")

    import numpy as np
    import pandas as pd
    import time
    import warnings
    import platform
    import pickle
    import gc
    from dataclasses import asdict
    from cyvcf2 import VCF

    warnings.filterwarnings("ignore")
    np.seterr(divide='ignore', invalid='ignore')

    import thread_config
    import vcf_data_loader
    import block_haplotypes
    from bhd_reversible_cavity import ReversibleCavitySearchConfig
    import small_block_refine
    import residual_discovery
    import hierarchical_assembly
    from founder_alleles import hard_alleles
    import paint_samples
    import pedigree_inference
    import phase_correction
    import analysis_utils
    import terminal_cavity_refinement

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    if platform.system() != "Windows":
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

    n_processes = 112
    # T01 uses the complete allocation: one block worker per Numba thread.
    # Dynamic reallocation gives the full budget to remaining stragglers.
    block_discovery_processes = n_processes
    block_discovery_numba_threads = n_processes
    # Recycle workers after each batch to prevent memory accumulation
    # from glibc malloc fragmentation (Python doesn't return freed pages to OS).
    WORKER_MAXTASKS = 1

    # Start forkserver before data loading
    _warmup_pool = hierarchical_assembly.NoDaemonPool(1)
    _warmup_pool.terminate()
    _warmup_pool.join()
    del _warmup_pool
    print("Forkserver started (lightweight, pre-data).")
    print(f"Numba threading layer: {os.environ.get('NUMBA_THREADING_LAYER', 'not set')}")

    # =========================================================================
    # Paths & Regions (AcTm tropheops cross)
    # =========================================================================
    vcf_path = "./fish_vcf_restriped/AcTm.biallelic.bcf.gz"
    meta_path = "./fish_vcf_restriped/X_AcTm_metadata.xlsx"

    # AcTm BCF covers the same reference as the AsAc files: chr1-chr20, chr22,
    # chr23 autosomes, plus chrM and U_scaffolds.  We only run the pipeline on
    # the 22 autosomes (chrM has no recombination; U_scaffolds are short/unplaced
    # and not useful for pedigree-scale linkage).
    regions_config = [
        {"contig": "chr1"},  {"contig": "chr2"},  {"contig": "chr3"},
        {"contig": "chr4"},  {"contig": "chr5"},  {"contig": "chr6"},
        {"contig": "chr7"},  {"contig": "chr8"},  {"contig": "chr9"},
        {"contig": "chr10"}, {"contig": "chr11"}, {"contig": "chr12"},
        {"contig": "chr13"}, {"contig": "chr14"}, {"contig": "chr15"},
        {"contig": "chr16"}, {"contig": "chr17"}, {"contig": "chr18"},
        {"contig": "chr19"}, {"contig": "chr20"}, {"contig": "chr22"},
        {"contig": "chr23"},
    ]

    # CHECKPOINT_DIR and output_dir are defined at module top level (above).
    # =========================================================================
    # Checkpoint Infrastructure (blosc2 via checkpoint_io — matches pipeline.py)
    # =========================================================================
    checkpoint_store = pipeline_runtime.CheckpointStore(
        CHECKPOINT_DIR, nthreads=n_processes, global_log_indent="    "
    )
    os.makedirs(output_dir, exist_ok=True)
    stage_complete = checkpoint_store.stage_complete
    mark_stage_complete = checkpoint_store.mark_stage_complete
    contig_done = checkpoint_store.contig_done
    save_contig = checkpoint_store.save_contig
    load_contig = checkpoint_store.load_contig
    save_global = checkpoint_store.save_global
    load_global = checkpoint_store.load_global

    strip_block_probs = pipeline_runtime.strip_block_probs

    def load_global_arrays(r_name):
        return pipeline_runtime.load_global_arrays(
            checkpoint_store, STAGE_T1, r_name
        )

    # =========================================================================
    # VALIDATION HELPERS (module-level, shared across stages)
    # =========================================================================
    # Each pipeline stage that produces a block-shaped output (T01-T08) calls
    # `run_stage_validation` at the end to compare those blocks with the four
    # observed G0 genotype references stashed in every T01 per-contig checkpoint
    # regardless of USE_KNOWN_FOUNDERS.  This is a non-independent post-hoc
    # consistency check when G0 rows participated in discovery, and a held-out
    # reference comparison when they were excluded.  T10 calls
    # `run_pedigree_validation` on the inferred pedigree_df to cross-check
    # it against the metafile's biological generation column.
    #
    # Semantics identical to the old monolithic validation stage, just factored
    # out and called at each stage boundary so the user sees quality progression
    # as the pipeline runs, not only at the very end.

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

          * hom-only single-haplotype (legacy, A/B reference): each founder's
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
            reduces to the legacy check (best pair is one hap used twice).
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

            # Discovered haplotypes -> concrete {0,1}
            discovered = []
            for hid, h_arr in block.haplotypes.items():
                discovered.append((hid, hard_alleles(h_arr)))
            n_disc = len(discovered)
            disc_ids = [hid for hid, _ in discovered]
            if n_disc > 0:
                D = np.vstack([h for _, h in discovered]).astype(np.int16)  # (n_disc, n_block)
            else:
                D = np.zeros((0, len(positions)), dtype=np.int16)

            # ---- legacy hom-only single-haplotype recall ----
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
                geno_m = g0_geno[g, conf].astype(np.int16)            # (n_conf,)
                Dm = D[:, conf]                                       # (n_disc, n_conf)
                S = Dm[:, None, :] + Dm[None, :, :]                   # (n_disc, n_disc, n_conf)
                mm = (S != geno_m[None, None, :]).mean(axis=2) * 100.0  # (n_disc, n_disc)
                iu = np.triu_indices(n_disc)                         # i <= j
                flat = mm[iu]
                k = int(np.argmin(flat))
                best_err = float(flat[k])
                bi, bj = int(iu[0][k]), int(iu[1][k])
                g0_pair_matches.append((g, f"{disc_ids[bi]}+{disc_ids[bj]}", best_err, n_conf))

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
                'g0_reference_is_independent': not USE_KNOWN_FOUNDERS,
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
        t1 = load_contig("T01_vcf_discovery", r_name)
        g0_probs = t1['g0_probs']
        g0_sites = t1['global_sites']
        g0_names = t1['g0_sample_names']
        del t1
        return g0_probs, g0_sites, g0_names

    def run_stage_validation(stage_label, stage_key, blocks_loader_fn, csv_filename):
        """Compare block haplotypes with observed G0 genotype references.

        This is a post-hoc consistency diagnostic when ``USE_KNOWN_FOUNDERS``
        is true because the same rows participated in reconstruction. It is
        an independent held-out comparison only when those rows were excluded.

        Args:
            stage_label: human-readable tag that goes into the CSV 'stage' column
                         (e.g. "T01_block_discovery", "T04_L1_assembly").
            stage_key: checkpoint dir name, used to check contig_done
                       (e.g. "T01_vcf_discovery", "T04_assembly_L1").
            blocks_loader_fn: callable(r_name) -> list of blocks for that contig.
                              Knows how to extract the right block list from
                              this stage's checkpoint (block_results vs
                              super_blocks_L1 vs super_blocks_L4, etc.).
            csv_filename: filename under output_dir to write the per-block CSV.

        Runs unconditionally each pipeline invocation (no checkpointing — fast).
        Produces one CSV with every block from every contig tagged with its
        stage, so per-stage CSVs can be concatenated for quality-progression
        analysis.
        """
        print(f"\n{'='*60}")
        validation_kind = (
            "post-hoc G0 consistency (non-independent)"
            if USE_KNOWN_FOUNDERS
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
            if not contig_done("T01_vcf_discovery", r_name):
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

    def run_pedigree_validation(pedigree_df):
        """Validate the inferred pedigree structure against the metafile.

        Writes:
          - validation_T10_pedigree_confusion.csv (structural x biological crosstab)
          - validation_T10_pedigree_per_sample.csv (per-sample audit with
            inferred Generation, Parent1, Parent2, and true_generation columns)

        Returns (n_correct, n_samples_audit, pedigree_accuracy, expected_mapping)
        for use in the final summary.
        """
        print(f"\n{'='*60}")
        print(f"VALIDATION: T10 Pedigree Structure vs Metafile")
        print(f"{'='*60}")

        pedigree_augmented = pedigree_df.copy()
        pedigree_augmented['true_generation'] = pedigree_augmented['Sample'].map(id_to_gen)

        confusion = pd.crosstab(
            pedigree_augmented['Generation'].fillna('(unassigned)'),
            pedigree_augmented['true_generation'].fillna('(no metadata)'),
            margins=True
        )
        print("\nInferred Generation × True Generation confusion matrix:")
        print(confusion)

        confusion_csv = os.path.join(output_dir, "validation_T10_pedigree_confusion.csv")
        confusion.to_csv(confusion_csv)
        print(f"\nConfusion matrix saved to: {confusion_csv}")

        audit_csv = os.path.join(output_dir, "validation_T10_pedigree_per_sample.csv")
        pedigree_augmented.to_csv(audit_csv, index=False)
        print(f"Per-sample pedigree audit saved to: {audit_csv}")

        # ---------------------------------------------------------------------
        # Generation-label accuracy
        # ---------------------------------------------------------------------
        # IMPORTANT — the pipeline's `Generation` column is a STRUCTURAL label,
        # not a biological one.  pedigree_inference assigns "F1" to any sample
        # with no inferable parents (i.e. a root node in the inferred pedigree
        # graph) and increments the label by one for each descendant
        # generation.  So "F1" in pipeline output means "root of the graph",
        # NOT "biologically an F1".  True biological generations come from the
        # metafile's `generation` column (stored here as `true_generation`).
        # G0 samples are NEVER relabeled as F1 biologically in either mode —
        # in withFounders mode the pipeline's STRUCTURAL "F1" happens to be
        # biologically G0 (because G0s are the roots the pipeline sees); in
        # withoutFounders mode G0s are absent from the pipeline entirely.
        #
        # `expected_mapping` translates the pipeline's structural label into
        # the biological label that sample should have, given what the
        # pipeline saw as input:
        #
        #   withFounders (G0s fed in): G0s are the structural roots, so the
        #     pipeline's "F1" should map to true G0; its "F2" to true F1; its
        #     "F3" to true F2.  Labels are shifted by one vs biology because
        #     the pipeline has no way to know its roots are biologically G0s.
        #
        #   withoutFounders (G0s excluded): biological F1s become the
        #     structural roots (their G0 parents aren't in the data), so the
        #     pipeline's "F1" maps to true F1 and its "F2" to true F2.
        #     Labels coincide with biology by accident — the pipeline isn't
        #     "recognising" F1 biology, its root-label convention just happens
        #     to start at the same generation biology does in this mode.
        if USE_KNOWN_FOUNDERS:
            # Structural pipeline label -> expected biological truth label
            expected_mapping = {'F1': 'G0', 'F2': 'F1', 'F3': 'F2'}
        else:
            # Structural pipeline label -> expected biological truth label
            expected_mapping = {'F1': 'F1', 'F2': 'F2'}

        n_samples_audit = len(pedigree_augmented)
        n_correct = 0
        for _, row in pedigree_augmented.iterrows():
            inf = row['Generation']
            tru = row['true_generation']
            if expected_mapping.get(inf) == tru:
                n_correct += 1
        pedigree_accuracy = 100.0 * n_correct / max(1, n_samples_audit)
        print(f"\nStructural->biological label translation: {expected_mapping}")
        print(f"  (pipeline's 'Generation' is a graph-position label, not a")
        print(f"   biological generation — 'F1' in this column means 'graph root')")
        print(f"Samples whose structural label matches the expected biological truth: "
              f"{n_correct}/{n_samples_audit} = {pedigree_accuracy:.1f}%")

        # If USE_KNOWN_FOUNDERS: check that G0 samples are inferred as roots
        if USE_KNOWN_FOUNDERS:
            g0_rows = pedigree_augmented[pedigree_augmented['true_generation'] == 'G0']
            g0_as_roots = int(g0_rows['Parent1'].isna().sum())
            print(f"G0 samples correctly inferred as roots (Parent1 NaN): "
                  f"{g0_as_roots}/{len(g0_rows)}")
        else:
            # In withoutFounders mode, G0s were not in the pipeline input, so
            # they shouldn't appear in pedigree_df at all.  Sanity-check:
            g0_in_pedigree = pedigree_augmented[pedigree_augmented['true_generation'] == 'G0']
            if len(g0_in_pedigree) > 0:
                print(f"WARNING: {len(g0_in_pedigree)} G0 samples in pedigree_df "
                      f"despite USE_KNOWN_FOUNDERS=False (should be 0)")
            else:
                print("Confirmed: no G0 samples in pedigree (as expected for withoutFounders).")

        return n_correct, n_samples_audit, pedigree_accuracy, expected_mapping

    region_keys = [r['contig'] for r in regions_config]

    # =========================================================================
    # SAMPLE IDENTIFICATION — match VCF samples to metafile, find G0 indices
    # =========================================================================
    # This runs before any stage so we always know:
    #   g0_vcf_indices      : positions of the 4 G0 samples in the VCF header
    #   active_vcf_indices  : positions of the samples the pipeline will see
    #                         (all 116 if USE_KNOWN_FOUNDERS, else 112 = no G0s)
    #   sample_names_active : VCF sample names the pipeline will see
    #                         (the ordered T01-T11 active sample axis)
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
    meta_df = pd.read_excel(meta_path, sheet_name='main_data')
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
        # Don't hard-fail — downstream logic tolerates it, but pedigree validation
        # will ignore those samples.

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
    if USE_KNOWN_FOUNDERS:
        active_vcf_indices = np.arange(n_samples_total, dtype=np.int64)
        print(f"\nUSE_KNOWN_FOUNDERS=True -> pipeline sees ALL {n_samples_total} samples "
              f"(G0 included)")
    else:
        g0_set = set(g0_vcf_indices)
        active_vcf_indices = np.array(
            [i for i in range(n_samples_total) if i not in g0_set],
            dtype=np.int64
        )
        print(f"\nUSE_KNOWN_FOUNDERS=False -> pipeline sees {len(active_vcf_indices)} "
              f"samples (G0 removed)")

    sample_names_active = [sample_names[i] for i in active_vcf_indices]
    n_samples = len(sample_names_active)

    # Sanity-log generation composition of active samples
    gen_counts_active = pd.Series(
        [id_to_gen.get(s, '?') for s in sample_names_active]
    ).value_counts()
    print(f"Active sample generation breakdown:")
    for gen, count in gen_counts_active.items():
        print(f"  {gen}: {count}")

    print(f"Regions: {len(region_keys)}")

    total_pipeline_start = time.time()

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T01: VCF Loading + Block Discovery + Global Probabilities
    # =========================================================================
    # Identical to pipeline_real.py STAGE R01, with ONE addition: we always
    # split out the G0 reads into a separate `g0_slice` that's stashed in the
    # checkpoint for the interleaved T01-T08 and final validations.  When
    # USE_KNOWN_FOUNDERS=False, the main global_probs/global_sites/block_results
    # are computed from the 112 non-G0 samples only (the reads array is sliced
    # along the sample axis before reads_to_probabilities / block discovery).
    STAGE_T1 = "T01_vcf_discovery"

    if stage_complete(STAGE_T1):
        print(f"\n[RESUME] Skipping VCF loading + discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T01: VCF Loading + Block Haplotype Discovery")
        print(f"{'='*60}")
        start = time.time()
        reversible_cavity_config = ReversibleCavitySearchConfig()
        reversible_cavity_config_record = asdict(reversible_cavity_config)
        print(
            "  Cap-free reversible cavity discovery: no explicit K grid or "
            "scientific K cap; "
            f"beam_width={reversible_cavity_config.beam_width}, "
            f"max_expansions={reversible_cavity_config.max_expansions}, "
            f"max_exact_scores={reversible_cavity_config.max_exact_scores}, "
            "max_proposals_per_expansion="
            f"{reversible_cavity_config.max_proposals_per_expansion}"
        )
        print(
            "  Block discovery parallelism: "
            f"workers={block_discovery_processes}, "
            f"Numba budget={block_discovery_numba_threads}"
        )

        with block_haplotypes.BlockDiscoveryPool(
            block_discovery_processes,
            block_discovery_numba_threads,
        ) as block_pool:
            for r_name in region_keys:
                if contig_done(STAGE_T1, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue
                print(f"\n  Processing {r_name}...")

                t0 = time.time()
                genomic_data = vcf_data_loader.cleanup_block_reads_list(
                    vcf_path, r_name,
                    use_snp_count=True, snps_per_block=200, snp_shift=200,
                    num_processes=16
                )
                print(f"    [Loader] {len(genomic_data)} blocks in {time.time()-t0:.1f}s")

                # Full reads: (n_samples_total, n_sites, 2) — all 116 samples
                global_sites, global_reads_full = (
                    vcf_data_loader.concatenate_unique_block_reads(genomic_data)
                )
                if global_sites is None:
                    print(f"    WARNING: No data for {r_name}, skipping")
                    continue

                # ALWAYS extract G0 reads separately for post-hoc validation.
                # This slice is independent of the USE_KNOWN_FOUNDERS flag — we want ground
                # truth available regardless of what the pipeline sees.
                g0_reads = global_reads_full[g0_vcf_indices, :, :]
                (_, g0_probs) = analysis_utils.reads_to_probabilities(
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
                if USE_KNOWN_FOUNDERS:
                    active_reads_full = global_reads_full
                else:
                    active_reads_full = global_reads_full[active_vcf_indices, :, :]
                    # Also filter genomic_data in place so block discovery sees 112 samples
                    for bi in range(len(genomic_data.reads)):
                        if genomic_data.reads[bi].shape[0] == n_samples_total:
                            genomic_data.reads[bi] = genomic_data.reads[bi][active_vcf_indices, :, :]

                # Downstream linkage models require per-sample genotype
                # likelihoods, not the empirical HWE posterior used as an
                # optional regularizer within local haplotype discovery.
                (site_priors, global_probs) = analysis_utils.reads_to_probabilities(
                    active_reads_full,
                    use_hwe_prior=False,
                )
                avg_depth = np.mean(np.sum(active_reads_full, axis=-1))
                print(f"    Sites: {len(global_sites)}, Samples (active): {global_probs.shape[0]}, "
                      f"Depth: {avg_depth:.1f}x")
                del global_reads_full, active_reads_full, g0_reads, site_priors

                t0 = time.time()
                block_results = block_haplotypes.generate_all_block_haplotypes(
                    genomic_data,
                    uniqueness_threshold_percent=1.0,
                    diff_threshold_percent=0.5,
                    wrongness_threshold=1.0,
                    num_processes=block_discovery_processes,
                    reversible_cavity_config=reversible_cavity_config,
                    total_numba_threads=block_discovery_numba_threads,
                    block_pool=block_pool,
                )
                valid_blocks = [b for b in block_results if len(b.positions) > 0]
                block_results = block_haplotypes.BlockResults(valid_blocks)

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
                legacy_fallback_count = (
                    len(valid_blocks) - len(cavity_blocks)
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
                    f"{search_limited_count}/{len(candidate_searches)}; "
                    "legacy empty/no-kept fallbacks="
                    f"{legacy_fallback_count}"
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
                    'block_results': block_results, 'avg_depth': avg_depth,
                    'g0_probs': g0_probs, 'g0_sample_names': g0_sample_names,
                    'active_vcf_indices': active_vcf_indices,
                    'block_discovery_backend': BLOCK_DISCOVERY_BACKEND,
                    'reversible_cavity_config': (
                        reversible_cavity_config_record),
                })
                del genomic_data, block_results, global_probs, global_sites, g0_probs
                gc.collect()

        save_global(STAGE_T1, {
            'sample_names_active': sample_names_active,
            'sample_names_full': sample_names,
            'region_keys': region_keys,
            'g0_vcf_indices': g0_vcf_indices,
            'g0_sample_names': g0_sample_names,
            'active_vcf_indices': active_vcf_indices,
            'use_known_founders': USE_KNOWN_FOUNDERS,
            'block_discovery_backend': BLOCK_DISCOVERY_BACKEND,
            'reversible_cavity_config': reversible_cavity_config_record,
        })
        print(f"\nVCF loading + discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T1)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T01 Block Discovery
    # =========================================================================
    # Compare raw 200-SNP haplotypes with the four observed G0 references.
    # In withFounders mode this is explicitly post-hoc and non-independent.
    run_stage_validation(
        stage_label="T01_block_discovery",
        stage_key="T01_vcf_discovery",
        blocks_loader_fn=lambda r: load_contig("T01_vcf_discovery", r)['block_results'],
        csv_filename="validation_T01_block_discovery.csv"
    )

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T02: Refinement (if avg depth < 100x)
    # =========================================================================
    # Threshold raised from 10x to 100x so that refinement runs for every
    # contig regardless of depth variation across chromosomes.  Guarantees
    # uniform treatment — no chr gets refined while another skips it due to
    # crossing the threshold from below.  AcTm is well below this limit
    # everywhere (mean ~9.3x, max per-sample ~45x, per-contig means all in
    # single digits to low tens), so the conditional is effectively
    # "always run refinement" for this cross.
    STAGE_T2 = "T02_refinement"

    if stage_complete(STAGE_T2):
        print(f"\n[RESUME] Skipping refinement (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T02: Checking Read Depth for Refinement")
        print(f"{'='*60}")

        REFINEMENT_DEPTH_THRESHOLD = 100.0
        REFINEMENT_BATCH_SIZE = 10
        REFINEMENT_PENALTY_SCALE = 20.0
        RECOMB_RATE = 5e-8
        N_GENERATIONS = 3

        import chimera_resolution
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_T2, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue

            t1 = load_contig(STAGE_T1, r_name)
            avg_depth = t1['avg_depth']
            global_probs = t1['global_probs']
            global_sites = t1['global_sites']
            block_results = strip_block_probs(t1['block_results'])
            del t1
            # Downcast: float64 only needed for HDBSCAN (T01)
            if global_probs.dtype == np.float64:
                global_probs = global_probs.astype(np.float32)

            print(f"\n{'='*60}")
            print(f"{r_name}: average read depth = {avg_depth:.1f}x")
            print(f"{'='*60}")
            
            if avg_depth < REFINEMENT_DEPTH_THRESHOLD:
                print(f"  Depth < {REFINEMENT_DEPTH_THRESHOLD}x -> Running L1+L2 refinement")
                num_samples = global_probs.shape[0]
                chimera_resolution.warmup_jit(num_samples)

                # L1/L2 step functions mirror pipeline.py STAGE 4 exactly:
                # cc_scale=0.5; maxtasksperchild=WORKER_MAXTASKS (recycle workers
                # to bound glibc-malloc fragmentation); refine_after_stitch=False
                # (the refinement pipeline below runs its OWN refinement, so the
                # per-level post-stitch refinement is opted out here to avoid
                # doing it twice).  verbose=False on L2 to match pipeline.py.
                l1_fn, l2_fn = pipeline_runtime.make_refinement_assembly_functions(
                    hierarchical_assembly.run_hierarchical_step,
                    global_probs,
                    global_sites,
                    batch_size=REFINEMENT_BATCH_SIZE,
                    recomb_rate=RECOMB_RATE,
                    n_generations=N_GENERATIONS,
                    beam_width=200,
                    max_founders=12,
                    cc_scale=0.5,
                    num_processes=n_processes,
                    maxtasksperchild=WORKER_MAXTASKS,
                )

                t0 = time.time()
                refinement_results = small_block_refine.run_refinement_pipeline(
                    raw_blocks=block_results, global_probs=global_probs,
                    global_sites=global_sites, num_samples=num_samples,
                    run_l1_assembly_fn=l1_fn,
                    run_l2_assembly_fn=l2_fn,
                    batch_size=REFINEMENT_BATCH_SIZE, penalty_scale=REFINEMENT_PENALTY_SCALE,
                    recomb_rate=RECOMB_RATE, n_generations=N_GENERATIONS, verbose=True)
                print(f"\n  Refinement complete in {time.time()-t0:.0f}s")

                l2_refined = refinement_results['l2_refined']
                l2_refined_dd = small_block_refine.dedup_blocks(l2_refined, verbose=True)
                save_contig(STAGE_T2, r_name, {'block_results': l2_refined_dd})
                del refinement_results, l2_refined, l2_refined_dd
            else:
                print(f"  Depth >= {REFINEMENT_DEPTH_THRESHOLD}x -> Skipping refinement")
                save_contig(STAGE_T2, r_name, {'block_results': block_results})

            del block_results, global_probs, global_sites
            gc.collect()

        print(f"\nRefinement stage complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T2)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T02 Refinement
    # =========================================================================
    # Same block layout as T01 but with L1+L2 refinement applied.  Should show
    # improved founder recovery if refinement was triggered (depth < 10x).
    # In contigs where refinement was skipped (depth >= 10x) this matches T01.
    run_stage_validation(
        stage_label="T02_refinement",
        stage_key="T02_refinement",
        blocks_loader_fn=lambda r: load_contig("T02_refinement", r)['block_results'],
        csv_filename="validation_T02_refinement.csv"
    )

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T03: Residual Discovery (Missing Founder Recovery)
    # =========================================================================
    STAGE_T3 = "T03_residual_discovery"

    if stage_complete(STAGE_T3):
        print(f"\n[RESUME] Skipping residual discovery (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T03: Residual Discovery (Missing Founder Recovery)")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_T3, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            t2 = load_contig(STAGE_T2, r_name)
            blocks = strip_block_probs(t2['block_results'])
            del t2

            global_probs, global_sites = load_global_arrays(r_name)

            print(f"    Input: {len(blocks)} blocks, "
                  f"avg haps: {np.mean([len(b.haplotypes) for b in blocks]):.1f}")

            blocks_out = residual_discovery.discover_missing_haplotypes(
                blocks, global_probs, global_sites,
                min_residual_reduction=0.10,
                num_processes=n_processes,
                verbose=True
            )

            print(f"    Output: {len(blocks_out)} blocks, "
                  f"avg haps: {np.mean([len(b.haplotypes) for b in blocks_out]):.1f}")

            pipeline_runtime.strip_block_evidence(blocks_out)
            save_contig(STAGE_T3, r_name, {'block_results': blocks_out})
            del blocks, blocks_out, global_probs, global_sites
            gc.collect()

        print(f"\nResidual discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T3)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T03 Residual Discovery
    # =========================================================================
    # Block layout unchanged but some blocks now have extra haplotypes added by
    # the residual-discovery pass.  We expect 'founders_found' to go up where
    # HDBSCAN missed a founder at this block.
    run_stage_validation(
        stage_label="T03_residual_discovery",
        stage_key="T03_residual_discovery",
        blocks_loader_fn=lambda r: load_contig("T03_residual_discovery", r)['block_results'],
        csv_filename="validation_T03_residual_discovery.csv"
    )

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T04: L1 Assembly
    # =========================================================================
    STAGE_T4 = "T04_assembly_L1"

    if stage_complete(STAGE_T4):
        print(f"\n[RESUME] Skipping L1 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T04: Level 1 Hierarchical Assembly")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_T4, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            t3 = load_contig(STAGE_T3, r_name)
            block_results = strip_block_probs(t3['block_results'])
            del t3

            global_probs, global_sites = load_global_arrays(r_name)

            print(f"    Input: {len(block_results)} blocks")

            # cc_scale=0.5 everywhere per user direction.
            super_blocks = hierarchical_assembly.run_hierarchical_step(
                block_results, global_probs, global_sites,
                batch_size=10, use_hmm_linking=False, beam_width=200,
                max_founders=12, max_sites_for_linking=2000, cc_scale=0.5,
                num_processes=n_processes, maxtasksperchild=WORKER_MAXTASKS,
                verbose=False)

            hap_counts = [len(b.haplotypes) for b in super_blocks]
            print(f"    Output: {len(super_blocks)} L1 super-blocks, "
                  f"haps: min={min(hap_counts)}, max={max(hap_counts)}, "
                  f"mean={np.mean(hap_counts):.1f}")

            pipeline_runtime.strip_block_evidence(super_blocks)
            save_contig(STAGE_T4, r_name, {'super_blocks_L1': super_blocks})
            del block_results, global_probs, super_blocks
            gc.collect()

        print(f"\nL1 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T4)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T04 L1 Assembly
    # =========================================================================
    # Blocks are now L1 super-blocks (longer, fewer).  Harder test: assembly
    # has to correctly link founder haps across the input 200-SNP blocks,
    # which gives more room to accumulate error but also averages-out noise.
    run_stage_validation(
        stage_label="T04_L1_assembly",
        stage_key="T04_assembly_L1",
        blocks_loader_fn=lambda r: load_contig("T04_assembly_L1", r)['super_blocks_L1'],
        csv_filename="validation_T04_L1_assembly.csv"
    )

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T05: L2 Assembly
    # =========================================================================
    STAGE_T5 = "T05_assembly_L2"

    if stage_complete(STAGE_T5):
        print(f"\n[RESUME] Skipping L2 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T05: Level 2 Hierarchical Assembly")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_T5, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            t4 = load_contig(STAGE_T4, r_name)
            l1_blocks = strip_block_probs(t4['super_blocks_L1'])
            del t4

            global_probs, global_sites = load_global_arrays(r_name)

            print(f"    Input: {len(l1_blocks)} L1 super-blocks")

            l2_blocks = hierarchical_assembly.run_hierarchical_step(
                l1_blocks, global_probs, global_sites,
                batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                beam_width=200, max_founders=12, cc_scale=0.5,
                num_processes=n_processes, maxtasksperchild=WORKER_MAXTASKS,
                n_generations=3, verbose=False)

            hap_counts = [len(b.haplotypes) for b in l2_blocks]
            print(f"    Output: {len(l2_blocks)} L2 super-blocks, haps: {hap_counts}")

            pipeline_runtime.strip_block_evidence(l2_blocks)
            save_contig(STAGE_T5, r_name, {'super_blocks_L2': l2_blocks})
            del l1_blocks, global_probs, l2_blocks
            gc.collect()

        print(f"\nL2 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T5)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T05 L2 Assembly
    # =========================================================================
    run_stage_validation(
        stage_label="T05_L2_assembly",
        stage_key="T05_assembly_L2",
        blocks_loader_fn=lambda r: load_contig("T05_assembly_L2", r)['super_blocks_L2'],
        csv_filename="validation_T05_L2_assembly.csv"
    )

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T06: L3 Assembly
    # =========================================================================
    STAGE_T6 = "T06_assembly_L3"

    if stage_complete(STAGE_T6):
        print(f"\n[RESUME] Skipping L3 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T06: Level 3 Hierarchical Assembly")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_T6, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            t5 = load_contig(STAGE_T5, r_name)
            l2_blocks = strip_block_probs(t5['super_blocks_L2'])
            del t5

            global_probs, global_sites = load_global_arrays(r_name)

            print(f"    Input: {len(l2_blocks)} L2 super-blocks")

            l3_blocks = hierarchical_assembly.run_hierarchical_step(
                l2_blocks, global_probs, global_sites,
                batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                beam_width=200, max_founders=12, cc_scale=0.5,
                num_processes=n_processes, maxtasksperchild=WORKER_MAXTASKS,
                n_generations=3, verbose=False)

            hap_counts = [len(b.haplotypes) for b in l3_blocks]
            print(f"    Output: {len(l3_blocks)} L3 super-blocks, haps: {hap_counts}")

            pipeline_runtime.strip_block_evidence(l3_blocks)
            save_contig(STAGE_T6, r_name, {'super_blocks_L3': l3_blocks})
            del l2_blocks, global_probs, l3_blocks
            gc.collect()

        print(f"\nL3 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T6)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T06 L3 Assembly
    # =========================================================================
    run_stage_validation(
        stage_label="T06_L3_assembly",
        stage_key="T06_assembly_L3",
        blocks_loader_fn=lambda r: load_contig("T06_assembly_L3", r)['super_blocks_L3'],
        csv_filename="validation_T06_L3_assembly.csv"
    )

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T07: L4 Assembly
    # =========================================================================
    STAGE_T7 = "T07_assembly_L4"

    if stage_complete(STAGE_T7):
        print(f"\n[RESUME] Skipping L4 assembly (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T07: Level 4 Hierarchical Assembly")
        print(f"{'='*60}")
        start = time.time()

        for r_name in region_keys:
            if contig_done(STAGE_T7, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  Processing {r_name}...")

            t6 = load_contig(STAGE_T6, r_name)
            l3_blocks = strip_block_probs(t6['super_blocks_L3'])
            del t6

            if len(l3_blocks) < 2:
                print("    Only 1 L3 block -- no L4 needed.")
                l4_blocks = l3_blocks
            else:
                global_probs, global_sites = load_global_arrays(r_name)

                print(f"    Input: {len(l3_blocks)} L3 super-blocks")

                l4_blocks = hierarchical_assembly.run_hierarchical_step(
                    l3_blocks, global_probs, global_sites,
                    batch_size=10, use_hmm_linking=True, recomb_rate=5e-8,
                    beam_width=200, max_founders=12, cc_scale=0.5,
                    num_processes=n_processes, maxtasksperchild=WORKER_MAXTASKS,
                    n_generations=3, verbose=False)
                del global_probs

            hap_counts = [len(b.haplotypes) for b in l4_blocks]
            print(f"    Output: {len(l4_blocks)} L4 super-blocks, haps: {hap_counts}")

            pipeline_runtime.strip_block_evidence(l4_blocks)
            save_contig(STAGE_T7, r_name, {'super_blocks_L4': l4_blocks})
            del l3_blocks, l4_blocks
            gc.collect()

        print(f"\nL4 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T7)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T07 raw L4 Assembly
    # =========================================================================
    # T07 is the raw chromosome-scale input; retain it to measure T08's effect.
    run_stage_validation(
        stage_label="T07_L4_assembly",
        stage_key="T07_assembly_L4",
        blocks_loader_fn=lambda r: load_contig("T07_assembly_L4", r)['super_blocks_L4'],
        csv_filename="validation_T07_L4_assembly.csv"
    )
#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T08: Terminal whole-bin cavity refinement (canonical final panel)
    # =========================================================================
    # T07 is the raw L4 intermediate; T08 publishes the only downstream panel.
    STAGE_T8 = "T08_terminal_cavity"

    missing_terminal = [r for r in region_keys if not contig_done(STAGE_T8, r)]
    if stage_complete(STAGE_T8) and missing_terminal:
        raise RuntimeError(
            f"{STAGE_T8} is marked complete but lacks: {missing_terminal}"
        )
    if stage_complete(STAGE_T8):
        print("\n[RESUME] Skipping terminal cavity refinement "
              "(checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T08: Terminal Cavity Refinement (canonical final panel)")
        print(f"{'='*60}")
        start = time.time()
        terminal_threads = min(
            n_processes,
            pipeline_runtime.available_cpu_count(),
        )
        print(f"  Sequential contigs; {terminal_threads} Numba threads/contig")

        for r_name in region_keys:
            if contig_done(STAGE_T8, r_name):
                print(f"  [RESUME] {r_name} already done")
                continue
            print(f"\n  [Terminal] Processing {r_name}...")

            t7 = load_contig(STAGE_T7, r_name)
            l4_blocks = strip_block_probs(t7['super_blocks_L4'])
            del t7
            if len(l4_blocks) != 1:
                raise RuntimeError(
                    f"{r_name}: terminal refinement requires exactly one "
                    f"chromosome-length L4 block; found {len(l4_blocks)}"
                )
            global_probs, global_sites = load_global_arrays(r_name)

            final_blocks, diagnostics = (
                terminal_cavity_refinement.refine_terminal_cavity_blocks(
                    l4_blocks,
                    global_sites,
                    global_probs,
                    return_diagnostics=True,
                    num_threads=terminal_threads,
                )
            )
            strip_block_probs(final_blocks)
            summary = (
                terminal_cavity_refinement.summarize_terminal_cavity_results(
                    diagnostics
                )
            )
            pipeline_runtime.strip_block_evidence(final_blocks)
            save_contig(STAGE_T8, r_name, {
                'super_blocks_L4': final_blocks,
                'terminal_cavity_summary': summary,
            })
            if not contig_done(STAGE_T8, r_name):
                raise OSError(f"Failed to checkpoint {STAGE_T8}/{r_name}")
            print(
                f"    Changed {summary['changed_founder_cells']} founder "
                f"cells at {summary['changed_sites']} sites"
            )
            del l4_blocks, global_probs, global_sites, final_blocks, diagnostics
            gc.collect()

        mark_stage_complete(STAGE_T8)
        print(f"Terminal refinement complete in {time.time()-start:.1f}s")

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T08 canonical terminal panel
    # =========================================================================
    # This final panel feeds painting and every later stage.  The G0 comparison
    # is non-independent when G0s participated in discovery and held-out only
    # when USE_KNOWN_FOUNDERS is false.
    run_stage_validation(
        stage_label="T08_terminal_cavity",
        stage_key="T08_terminal_cavity",
        blocks_loader_fn=lambda r: load_contig(STAGE_T8, r)['super_blocks_L4'],
        csv_filename="validation_T08_terminal_cavity.csv"
    )


#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T09: Viterbi Painting
    # =========================================================================
    STAGE_T9 = "T09_viterbi_painting"

    missing_painting = [
        r for r in region_keys if not contig_done(STAGE_T9, r)
    ]
    if stage_complete(STAGE_T9) and missing_painting:
        raise RuntimeError(
            f"{STAGE_T9} is marked complete but lacks: "
            f"{missing_painting}"
        )

    if stage_complete(STAGE_T9):
        print(f"\n[RESUME] Skipping Viterbi painting (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T09: Viterbi Painting (Tropheops)")
        print(f"{'='*60}")
        start = time.time()

        with paint_samples.PaintingPoolManager(num_processes=n_processes) as painter:
            for r_name in region_keys:
                if contig_done(STAGE_T9, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue

                print(f"\n  [Viterbi Painting] Processing Region: {r_name}")

                terminal_payload = load_contig(STAGE_T8, r_name)
                final_blocks = terminal_payload['super_blocks_L4']
                if len(final_blocks) != 1:
                    raise RuntimeError(
                        f"{r_name}: painting requires exactly one final L4 "
                        f"block; found {len(final_blocks)}"
                    )
                discovered_block = final_blocks[0]
                del terminal_payload, final_blocks

                global_probs, global_sites = load_global_arrays(r_name)

                painting_result = painter.paint_chromosome(
                    discovered_block, global_probs, global_sites,
                    recomb_rate=5e-8, switch_penalty_per_snp=1.0, batch_size=1)

                # Population painting visualization — uses the ACTIVE sample
                # names (116 in withFounders mode, 112 in withoutFounders mode)
                # so row labels match the sample axis of painting_result.
                print(f"  Generating Population Painting Plot...")
                plot_filename = os.path.join(output_dir, f"{r_name}_viterbi_population.png")
                paint_samples.plot_population_painting(
                    painting_result, output_file=plot_filename,
                    title=f"Viterbi Painting - {r_name} ({_run_label})",
                    sample_names=sample_names_active, figsize_width=20,
                    row_height_per_sample=0.25)

                founder_block = pipeline_runtime.compact_founder_block(
                    discovered_block
                )
                save_contig(STAGE_T9, r_name, {
                    'tolerance_result': painting_result,
                    pipeline_runtime.FOUNDER_BLOCK_KEY: founder_block,
                    pipeline_runtime.SAMPLE_IDS_KEY: tuple(
                        str(value) for value in sample_names_active
                    ),
                })
                del discovered_block, founder_block, global_probs, painting_result
                gc.collect()

        missing_painting = [
            r for r in region_keys if not contig_done(STAGE_T9, r)
        ]
        if missing_painting:
            raise OSError(f"Failed to checkpoint {STAGE_T9}: {missing_painting}")
        print(f"\nViterbi painting complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T9)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T10: Pedigree Inference
    # =========================================================================
    STAGE_T10 = "T10_pedigree_inference"

    if stage_complete(STAGE_T10) and not checkpoint_store.global_done(STAGE_T10):
        raise RuntimeError(f"{STAGE_T10} is complete but lacks _global")
    if stage_complete(STAGE_T10):
        print(f"\n[RESUME] Skipping pedigree inference (checkpoint found)")
        pedigree_df = load_global(STAGE_T10)['pedigree_df']
    else:
        print(f"\n{'='*60}")
        print("STAGE T10: Multi-Contig Pedigree Inference (Tropheops)")
        print(f"{'='*60}")

        contig_inputs = []
        for r_name in region_keys:
            painting_payload = load_contig(STAGE_T9, r_name)
            pipeline_runtime.validate_painting_bundle(
                painting_payload,
                expected_sample_ids=sample_names_active,
                context=f"{STAGE_T9}/{r_name}",
            )
            founder_block = pipeline_runtime.compact_founder_block(
                painting_payload[pipeline_runtime.FOUNDER_BLOCK_KEY]
            )
            entry = {
                'tolerance_painting': painting_payload['tolerance_result'],
                'founder_block': founder_block
            }
            contig_inputs.append(entry)
            del painting_payload

        start = time.time()
        pedigree_result = pedigree_inference.infer_pedigree_multi_contig_tolerance(
            contig_inputs, sample_ids=sample_names_active, top_k=20,
            n_workers=n_processes)
        print(f"\nPedigree inference time: {time.time()-start:.1f}s")

        pedigree_df = pedigree_result.relationships

        gen_counts = pedigree_df['Generation'].value_counts()
        print(f"\n--- Pedigree Summary ---")
        print(f"Generations: {gen_counts.to_dict()}")
        n_with_parents = pedigree_df['Parent1'].notna().sum()
        print(f"Individuals with parents: {n_with_parents}/{len(pedigree_df)}")

        output_csv = os.path.join(output_dir, "pedigree_inference_tropheops.csv")
        pedigree_df.to_csv(output_csv, index=False)
        print(f"Pedigree saved to: {output_csv}")

        output_tree = os.path.join(output_dir, "pedigree_tree_tropheops.png")
        pedigree_inference.draw_pedigree_tree(pedigree_df, output_file=output_tree)

        save_global(STAGE_T10, {'pedigree_df': pedigree_df})
        if not checkpoint_store.global_done(STAGE_T10):
            raise OSError(f"Failed to checkpoint {STAGE_T10}/_global")
        del contig_inputs
        gc.collect()
        mark_stage_complete(STAGE_T10)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T10 Pedigree Inference
    # =========================================================================
    # Cross-checks the inferred pedigree_df against the metafile's biological
    # generation column.  Writes a confusion matrix and per-sample audit CSV,
    # and computes the structural->biological label translation accuracy.
    # Runs unconditionally at each invocation (cheap, no checkpointing).
    if 'pedigree_df' not in dir():
        pedigree_df = load_global("T10_pedigree_inference")['pedigree_df']
    _t10_val_result = run_pedigree_validation(pedigree_df)
    # Keep the tuple for the final report: (n_correct, n_samples_audit,
    # pedigree_accuracy, expected_mapping)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T11: Phase Correction + Greedy Refinement + F1 Recoloring + Propagation
    # =========================================================================
    STAGE_T11 = "T11_phase_correction"

    missing_phase = [
        r for r in region_keys if not contig_done(STAGE_T11, r)
    ]
    if stage_complete(STAGE_T11) and missing_phase:
        raise RuntimeError(
            f"{STAGE_T11} is marked complete but lacks: "
            f"{missing_phase}"
        )

    if stage_complete(STAGE_T11):
        print(f"\n[RESUME] Skipping phase correction (checkpoint found)")
    else:
        print("\n" + "="*60)
        print("STAGE T11: Phase Correction (Tropheops)")
        print("="*60)

        if 'pedigree_df' not in dir():
            pedigree_df = load_global(STAGE_T10)['pedigree_df']

        # _load_contig_for_phase_correction is defined at MODULE TOP LEVEL
        # (picklable for forkserver workers — a closure here could not be
        # pickled into the worker initargs under the forkserver start method).
        # Workers load their own tolerance_result + founder_block from disk.

        # Lightweight dict — just contig names; workers load their own data.
        mcr = {r_name: {} for r_name in region_keys}

        # Step 1: Viterbi phase correction.  num_rounds=6 (was 3) to match
        # pipeline.py — Jacobi within-contig threading needs more rounds to
        # converge than Gauss-Seidel did, and 3 left contigs short.  max_workers
        # is REQUIRED by the current phase_correction API (it raises on None) so
        # the stage respects the pipeline's machine-wide core budget.
        start = time.time()
        mcr = phase_correction.correct_phase_all_contigs(
            mcr, pedigree_df, sample_names_active, num_rounds=6, verbose=True,
            max_workers=n_processes, load_fn=_load_contig_for_phase_correction)
        print(f"Phase correction time: {time.time()-start:.1f}s")

        # Step 2: Greedy phase refinement
        print("\n" + "="*60)
        print("Greedy Phase Refinement (HOM->HET boundary flips)")
        print("="*60)

        start_refine = time.time()
        mcr = phase_correction.post_process_phase_greedy_all_contigs(
            mcr, pedigree_df, sample_names_active,
            snps_per_bin=100, recomb_rate=5e-8, mismatch_cost=4.6,
            max_workers=n_processes, load_fn=_load_contig_for_phase_correction,
            verbose=True)
        print(f"Greedy refinement time: {time.time()-start_refine:.1f}s")

        # Pre-load founder_blocks for the F1 recoloring + propagation loops,
        # which run in the MAIN process and need founder_block in mcr.  The
        # phase-correction / greedy workers above no longer return founder_block
        # (the IPC-cost fix in phase_correction.py), so load it here in parallel
        # across contigs (parallel disk I/O).
        print("\n" + "="*60)
        print("Loading founder_blocks for F1 recoloring + propagation")
        print("="*60)
        _t0 = time.time()
        founder_blocks = pipeline_runtime.load_founder_blocks_parallel(
            checkpoint_store,
            region_keys,
            ((STAGE_T9, pipeline_runtime.FOUNDER_BLOCK_KEY),),
            max_workers=n_processes,
            require_all=True,
        )
        for r_name, founder_block in founder_blocks.items():
            mcr.setdefault(r_name, {})['founder_block'] = founder_block
        del founder_blocks
        print(f"Founder block parallel load: {time.time()-_t0:.1f}s")

        # Step 3: Parsimonious F1 recoloring
        print("\n" + "="*60)
        print("Parsimonious F1 Recoloring")
        print("="*60)

        for r_name in region_keys:
            if r_name not in mcr:
                continue
            data = mcr[r_name]
            painting_key = 'refined_painting' if 'refined_painting' in data else 'corrected_painting'
            if painting_key not in data or 'founder_block' not in data:
                continue

            recolored = phase_correction.apply_parsimonious_f1_recoloring(
                data[painting_key], data['founder_block'],
                pedigree_df, sample_names_active,
                max_workers=n_processes, max_mismatch_rate=0.02, verbose=True)
            data['final_painting'] = recolored

        # Step 4: Propagate recoloring to offspring
        print("\n" + "="*60)
        print("Propagate Recoloring to Offspring")
        print("="*60)

        for r_name in region_keys:
            if r_name not in mcr:
                continue
            data = mcr[r_name]
            if 'final_painting' not in data or 'founder_block' not in data:
                continue

            propagated = phase_correction.propagate_recoloring_to_offspring(
                data['final_painting'], data['founder_block'],
                pedigree_df, sample_names_active,
                max_workers=n_processes, max_mismatch_rate=0.02, verbose=True)
            data['final_painting'] = propagated

        # Save per-contig results
        for r_name in region_keys:
            if r_name in mcr:
                d = {k: mcr[r_name][k]
                     for k in ('corrected_painting', 'refined_painting',
                               'final_painting', 'founder_block')
                     if k in mcr[r_name]}
                save_contig(STAGE_T11, r_name, d)

        missing_phase = [
            r for r in region_keys if not contig_done(STAGE_T11, r)
        ]
        if missing_phase:
            raise OSError(f"Failed to checkpoint {STAGE_T11}: {missing_phase}")

        del mcr
        gc.collect()
        mark_stage_complete(STAGE_T11)

#%%
if __name__ == '__main__':
    # =========================================================================
    # FINAL REPORT: Aggregate all per-stage validation CSVs
    # =========================================================================
    # Each T01-T07 stage wrote its own per-block validation CSV as soon as it
    # finished (see the VALIDATION cells interleaved between stages).  T10
    # wrote its pedigree confusion + per-sample audit CSVs.  This final cell
    # just aggregates those into a cross-stage summary so you can see
    # reconstruction quality progression end-to-end without digging through
    # 9 separate files.
    #
    # Produces:
    #   - validation_all_stages_per_block.csv  (every block from every stage,
    #     stacked into one long table for downstream plotting/analysis)
    #   - validation_all_stages_summary.csv    (one row per stage with
    #     aggregate metrics: block count, mean discovered haps, % of blocks
    #     matching all four observed G0 genotype references, mean good/chimera
    #     haps)
    #   - validation_summary.txt               (human-readable overview
    #     including the pedigree validation result from T10)
    # Runs unconditionally at each invocation — no checkpointing, cheap.
    print(f"\n{'='*60}")
    print("FINAL REPORT: Cross-stage validation summary")
    print(f"{'='*60}")

    # Per-stage CSVs in pipeline order (label, filename)
    block_stage_csvs = [
        ('T01_block_discovery',     'validation_T01_block_discovery.csv'),
        ('T02_refinement',          'validation_T02_refinement.csv'),
        ('T03_residual_discovery',  'validation_T03_residual_discovery.csv'),
        ('T04_L1_assembly',         'validation_T04_L1_assembly.csv'),
        ('T05_L2_assembly',         'validation_T05_L2_assembly.csv'),
        ('T06_L3_assembly',         'validation_T06_L3_assembly.csv'),
        ('T07_L4_assembly',         'validation_T07_L4_assembly.csv'),
        ('T08_terminal_cavity',     'validation_T08_terminal_cavity.csv'),
    ]

    ff_col = f'founders_found_under_{MATCH_THRESHOLD_PCT:.0f}pct'
    stage_summary_rows = []
    all_block_dfs = []

    for stage_label, csv_name in block_stage_csvs:
        path = os.path.join(output_dir, csv_name)
        if not os.path.exists(path):
            print(f"  [skip] {stage_label}: {csv_name} not found")
            continue
        df_stage = pd.read_csv(path)
        if len(df_stage) == 0:
            print(f"  [skip] {stage_label}: {csv_name} empty")
            continue

        n_blocks = len(df_stage)
        if 'n_scorable_homonly' in df_stage.columns:
            n_all_found = int((df_stage[ff_col] == df_stage['n_scorable_homonly']).sum())
        else:
            n_all_found = int((df_stage[ff_col] == df_stage['n_true_founders']).sum())
        pct_all_found = 100.0 * n_all_found / n_blocks
        if 'founders_found_pair' in df_stage.columns and 'n_scorable' in df_stage.columns:
            n_all_found_pair = int((df_stage['founders_found_pair'] == df_stage['n_scorable']).sum())
            pct_all_found_pair = round(100.0 * n_all_found_pair / n_blocks, 2)
            mean_scor = round(df_stage['n_scorable'].mean(), 3)
        else:
            pct_all_found_pair = float('nan')
            mean_scor = float('nan')
        mean_disc = df_stage['n_discovered'].mean()
        mean_good = df_stage['good_haps'].mean()
        mean_chim = df_stage['chimeras'].mean()

        stage_summary_rows.append({
            'stage': stage_label,
            'g0_reference_is_independent': not USE_KNOWN_FOUNDERS,
            'total_blocks': n_blocks,
            'mean_discovered_haps': round(mean_disc, 3),
            'mean_scorable_founders': mean_scor,
            'all_found_pair_pct': pct_all_found_pair,
            'all_found_homonly_pct': round(pct_all_found, 2),
            'mean_good_haps': round(mean_good, 3),
            'mean_chimera_haps': round(mean_chim, 3),
        })
        all_block_dfs.append(df_stage)

    if stage_summary_rows:
        summary_df = pd.DataFrame(stage_summary_rows)
        summary_csv_path = os.path.join(output_dir, "validation_all_stages_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nQuality progression across stages:")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_csv_path}")
    else:
        summary_df = pd.DataFrame()
        print("WARNING: No per-stage validation CSVs found — did any stages run?")

    if all_block_dfs:
        combined = pd.concat(all_block_dfs, ignore_index=True, sort=False)
        combined_csv_path = os.path.join(output_dir, "validation_all_stages_per_block.csv")
        combined.to_csv(combined_csv_path, index=False)
        print(f"Combined per-block CSV saved to: {combined_csv_path}")

    # Pull in the T10 pedigree validation result if it ran earlier this session
    if '_t10_val_result' in dir():
        n_correct, n_samples_audit, pedigree_accuracy, expected_mapping = _t10_val_result
    else:
        n_correct, n_samples_audit, pedigree_accuracy, expected_mapping = (
            None, None, float('nan'), None
        )

    # Human-readable summary.txt
    summary_lines = []
    summary_lines.append(f"Tropheops Pipeline Validation Summary")
    summary_lines.append(f"Mode: {_mode_label} (USE_KNOWN_FOUNDERS={USE_KNOWN_FOUNDERS})")
    summary_lines.append(f"Block discovery: {BLOCK_DISCOVERY_BACKEND}")
    summary_lines.append(f"Timestamp: {datetime.now().isoformat()}")
    summary_lines.append(f"")
    summary_lines.append(f"Input:")
    summary_lines.append(f"  VCF: {vcf_path}")
    summary_lines.append(f"  Metafile: {meta_path}")
    summary_lines.append(f"  Total VCF samples: {n_samples_total}")
    summary_lines.append(f"  Active (pipeline-visible) samples: {n_samples}")
    summary_lines.append(f"  G0 genotype-reference samples: {len(g0_sample_names)}")
    for nm in g0_sample_names:
        summary_lines.append(f"    - {nm}")
    summary_lines.append(f"  Contigs processed: {len(region_keys)}")
    summary_lines.append(f"")
    summary_lines.append(f"Post-hoc G0 Consistency Progression")
    summary_lines.append(
        "  (non-independent when USE_KNOWN_FOUNDERS=True; legacy "
        f"match threshold: <{MATCH_THRESHOLD_PCT:.0f}% allele error)")
    if len(summary_df) > 0:
        summary_lines.append(summary_df.to_string(index=False))
    else:
        summary_lines.append(f"  (no per-stage CSVs found)")
    summary_lines.append(f"")
    summary_lines.append(f"Pedigree Structure (T10):")
    if n_samples_audit is not None:
        summary_lines.append(f"  Pipeline 'Generation' is a STRUCTURAL label")
        summary_lines.append(f"    ('F1' = graph root; not a biological generation)")
        summary_lines.append(f"  Structural->biological translation: {expected_mapping}")
        summary_lines.append(f"  Samples matching translation: "
                             f"{n_correct}/{n_samples_audit} = {pedigree_accuracy:.1f}%")
    else:
        summary_lines.append(f"  (T10 validation did not run — pedigree not available)")
    summary_lines.append(f"")
    summary_lines.append(f"Artefacts in {output_dir}/:")
    summary_lines.append(f"  validation_T01_block_discovery.csv")
    summary_lines.append(f"  validation_T02_refinement.csv")
    summary_lines.append(f"  validation_T03_residual_discovery.csv")
    summary_lines.append(f"  validation_T04_L1_assembly.csv")
    summary_lines.append(f"  validation_T05_L2_assembly.csv")
    summary_lines.append(f"  validation_T06_L3_assembly.csv")
    summary_lines.append(f"  validation_T07_L4_assembly.csv")
    summary_lines.append(f"  validation_T08_terminal_cavity.csv")
    summary_lines.append(f"  validation_T10_pedigree_confusion.csv")
    summary_lines.append(f"  validation_T10_pedigree_per_sample.csv")
    summary_lines.append(f"  validation_all_stages_summary.csv")
    summary_lines.append(f"  validation_all_stages_per_block.csv")
    summary_lines.append(f"  validation_summary.txt (this file)")

    summary_text = "\n".join(summary_lines)
    summary_txt_path = os.path.join(output_dir, "validation_summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write(summary_text + "\n")

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(summary_text)
    print(f"\nSummary saved to: {summary_txt_path}")

#%%
if __name__ == '__main__':
    elapsed = time.time() - total_pipeline_start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\n{'='*60}")
    print("TROPHEOPS PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Mode: {_mode_label} (USE_KNOWN_FOUNDERS={USE_KNOWN_FOUNDERS})")
    print(f"Block discovery: {BLOCK_DISCOVERY_BACKEND}")
    print(f"Total time: {hours}h {minutes}m ({elapsed:.0f}s)")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print(f"Results: {output_dir}/")
    print(f"Regions processed: {len(region_keys)}")
    print(f"Active samples: {n_samples} (of {n_samples_total} in VCF)")