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
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # =============================================================================
    # CONFIGURATION — EDIT THIS TO SWITCH BETWEEN THE TWO COMPARISON RUNS
    # =============================================================================
    # USE_KNOWN_FOUNDERS controls whether the 4 G0 (parental) samples are fed
    # into the reconstruction pipeline:
    #   True  -> all 116 samples go through T01-T10 (easy mode: G0s are
    #            mostly homozygous for distinct parental species and will form
    #            clean founder clusters on their own during block discovery)
    #   False -> the 4 G0 sample rows are sliced out of the input, so T01-T10
    #            only see the 112 admixed F1+F2 samples (hard mode: the pipeline
    #            must reconstruct parental haplotypes purely from offspring)
    # In BOTH modes, the 4 G0 sample reads are ALSO loaded separately and
    # stashed in the T01 checkpoint for use by the T11 validation stage, so
    # validation-against-ground-truth works regardless of the flag.
    USE_KNOWN_FOUNDERS = True

    # =============================================================================
    # DUAL LOGGING: Console + File
    # =============================================================================
    class TeeOutput:
        """Writes to both the original stdout and a log file."""
        def __init__(self, log_path, original_stdout):
            object.__setattr__(self, '_log_file', open(log_path, 'a', buffering=1))
            object.__setattr__(self, '_original', original_stdout)
        def write(self, message):
            self._original.write(message)
            try: self._log_file.write(message)
            except (ValueError, OSError): pass
            return None
        def flush(self):
            self._original.flush()
            try: self._log_file.flush()
            except (ValueError, OSError): pass
        def close(self):
            self._log_file.close()
        def __getattr__(self, name):
            return getattr(self._original, name)

    _mode_label = "withFounders" if USE_KNOWN_FOUNDERS else "withoutFounders"
    os.makedirs("logs", exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", f"run_tropheops_{_mode_label}_{run_timestamp}.log")
    sys.stdout = TeeOutput(log_path, sys.stdout)
    print(f"Logging to: {log_path}")
    print(f"Run started: {run_timestamp}")
    print(f"USE_KNOWN_FOUNDERS = {USE_KNOWN_FOUNDERS}  (mode: {_mode_label})")

    import numpy as np
    import pandas as pd
    import time
    import warnings
    import platform
    import pickle
    import gc
    from cyvcf2 import VCF

    warnings.filterwarnings("ignore")
    np.seterr(divide='ignore', invalid='ignore')

    import thread_config
    import vcf_data_loader
    import block_haplotypes
    import block_haplotype_refinement
    import residual_discovery
    import hierarchical_assembly
    import paint_samples
    import pedigree_inference
    import phase_correction
    import analysis_utils

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    if platform.system() != "Windows":
        print(f"Main process ({os.getpid()}) niceness set to: {os.nice(0)}")

    n_processes = 112
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

    # Separate checkpoint + results directories per flag setting so a run with
    # USE_KNOWN_FOUNDERS=True does not clobber one with USE_KNOWN_FOUNDERS=False
    # (and vice versa).  Lets the user run both back-to-back and compare outputs.
    CHECKPOINT_DIR = f".pipeline_checkpoints_tropheops_{_mode_label}"
    output_dir = f"results_tropheops_{_mode_label}"

    # =========================================================================
    # Checkpoint Infrastructure (identical to pipeline_real.py)
    # =========================================================================
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    def _stage_dir(stage):
        d = os.path.join(CHECKPOINT_DIR, stage)
        os.makedirs(d, exist_ok=True)
        return d

    def stage_complete(stage):
        return os.path.exists(os.path.join(_stage_dir(stage), '_done'))

    def mark_stage_complete(stage):
        with open(os.path.join(_stage_dir(stage), '_done'), 'w') as f:
            f.write(datetime.now().isoformat())
        print(f"  [Checkpoint] Stage '{stage}' marked complete")

    def contig_done(stage, r_name):
        return os.path.exists(os.path.join(_stage_dir(stage), f'{r_name}.pkl'))

    def save_contig(stage, r_name, data_dict):
        path = os.path.join(_stage_dir(stage), f'{r_name}.pkl')
        tmp = path + ".tmp"
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            sz = os.path.getsize(path) / (1024*1024)
            print(f"    [Checkpoint] {stage}/{r_name} ({sz:.1f} MB)")
        except OSError as e:
            print(f"    [Checkpoint] WARNING: {stage}/{r_name}: {e}")
            try: os.unlink(tmp)
            except OSError: pass

    def load_contig(stage, r_name):
        path = os.path.join(_stage_dir(stage), f'{r_name}.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_global(stage, data_dict):
        path = os.path.join(_stage_dir(stage), '_global.pkl')
        tmp = path + ".tmp"
        try:
            with open(tmp, 'wb') as f:
                pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            sz = os.path.getsize(path) / (1024*1024)
            print(f"    [Checkpoint] {stage}/_global ({sz:.1f} MB)")
        except OSError as e:
            print(f"    [Checkpoint] WARNING: {stage}/_global: {e}")
            try: os.unlink(tmp)
            except OSError: pass

    def load_global(stage):
        path = os.path.join(_stage_dir(stage), '_global.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def strip_block_probs(blocks):
        """Strip redundant probs_array from blocks to save memory.
        Workers access sample data via global_probs in shared memory."""
        for block in blocks:
            if hasattr(block, 'probs_array') and block.probs_array is not None:
                block.probs_array = None
        return blocks

    def load_global_arrays(r_name):
        """Load only global_probs and global_sites from T01, freeing block_results immediately.
        T01 checkpoints are huge (many GB per contig) because they contain block_results
        with probs_array. This extracts just what's needed and frees the rest.
        Downcasts global_probs to float32 (float64 only needed for HDBSCAN in T01)."""
        import ctypes
        t1 = load_contig(STAGE_T1, r_name)
        global_probs = t1['global_probs']
        global_sites = t1['global_sites']
        # Drop the heavy block_results (with redundant probs_array)
        del t1
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        # Downcast: float64 only needed for HDBSCAN (T01). Assembly, painting,
        # pedigree, phase correction all work fine with float32 precision.
        if global_probs.dtype == np.float64:
            global_probs = global_probs.astype(np.float32)
        return global_probs, global_sites

    # =========================================================================
    # VALIDATION HELPERS (module-level, shared across stages)
    # =========================================================================
    # Each pipeline stage that produces a block-shaped output (T01-T07) calls
    # `run_stage_validation` at the end to compare those blocks against the
    # 4 G0 ground-truth founder haplotypes (which are stashed in every T01
    # per-contig checkpoint regardless of USE_KNOWN_FOUNDERS).  T09 calls
    # `run_pedigree_validation` on the inferred pedigree_df to cross-check
    # it against the metafile's biological generation column.
    #
    # Semantics identical to the old monolithic T11 stage — just factored out
    # and called at each stage boundary so the user sees quality progression
    # as the pipeline runs, not only at the very end.

    # Min argmax-prob to treat a G0 site as confidently homozygous.  Sites
    # below this confidence, or where the max state is heterozygous (state=1),
    # are masked out of the G0 ground-truth comparison.
    HOM_CONFIDENCE = 0.85
    # A discovered haplotype "matches" a G0 ground-truth haplotype if the
    # allele-level disagreement rate is below this threshold (in %).
    MATCH_THRESHOLD_PCT = 2.0

    def extract_g0_block_haps(g0_probs, g0_sites, block_positions):
        """Build ground-truth founder haplotypes for one block.

        g0_probs has shape (n_g0, n_global_sites, 3) — genotype probabilities
        0=homref, 1=het, 2=homalt.  For an inbred founder we expect either
        the 0 or 2 state to dominate at most sites.  A site is "confidently
        homozygous" if the max genotype probability > HOM_CONFIDENCE and
        that max is at state 0 or state 2.  At such sites the consensus
        allele is 0 or 1 respectively.  Heterozygous-dominant or
        low-confidence sites are masked to -1.

        Returns: (n_g0, n_block_sites) int8, values in {0, 1, -1}
        """
        # Index of each block position within g0_sites
        pos_idx = np.searchsorted(g0_sites, block_positions)
        # Guard against off-by-one at the ends
        pos_idx = np.clip(pos_idx, 0, len(g0_sites) - 1)
        # Confirm positions actually match (after de-dup sort in T01 they should)
        matched = (g0_sites[pos_idx] == block_positions)

        n_g0 = g0_probs.shape[0]
        n_block = len(block_positions)
        g0_haps = np.full((n_g0, n_block), -1, dtype=np.int8)

        for g in range(n_g0):
            # probs for this G0 at block sites: (n_block, 3)
            probs_g = g0_probs[g, pos_idx, :]
            argmax = np.argmax(probs_g, axis=1)
            maxp = probs_g[np.arange(n_block), argmax]
            # Confident homozygous ref (state 0) -> allele 0
            # Confident homozygous alt (state 2) -> allele 1
            # Anything else (het, or low confidence) -> -1 (masked)
            conf = (maxp >= HOM_CONFIDENCE) & matched
            hom_ref = conf & (argmax == 0)
            hom_alt = conf & (argmax == 2)
            g0_haps[g, hom_ref] = 0
            g0_haps[g, hom_alt] = 1

        return g0_haps

    def validate_block_list_against_g0(blocks, g0_probs, g0_sites,
                                       g0_names, stage_label, contig_name):
        """Validate a list of blocks (raw, refined, L1/L2/L3/L4 super-blocks, ...)
        against G0 ground-truth founder haplotypes.

        Returns a list of dicts, one per non-empty block.  Each dict has:
          stage, contig, block, n_sites, block_start, block_end,
          n_true_founders, n_discovered,
          founders_found_under_<X>pct, good_haps, chimeras,
          and per-G0 best-match columns (discovered-hap id, error %, valid site count).
        """
        rows = []
        n_g0 = g0_probs.shape[0]
        ff_col = f'founders_found_under_{MATCH_THRESHOLD_PCT:.0f}pct'

        for block_idx, block in enumerate(blocks):
            positions = block.positions
            if len(positions) == 0:
                continue

            # Ground-truth G0 haplotypes at this block's positions
            g0_block_haps = extract_g0_block_haps(g0_probs, g0_sites, positions)

            # Discovered haplotypes (convert probabilistic to concrete via argmax)
            discovered = []
            for hid, h_arr in block.haplotypes.items():
                if h_arr.ndim == 2:
                    concrete = np.argmax(h_arr, axis=1).astype(np.int8)
                else:
                    concrete = h_arr.astype(np.int8)
                discovered.append((hid, concrete))
            n_disc = len(discovered)

            # For each G0, find the best-matching discovered haplotype.  Only
            # sites where the G0 ground truth is NOT masked (-1) are counted.
            g0_best_matches = []  # (g, best_disc_id, err_pct, n_valid_sites)
            for g in range(n_g0):
                g_valid = (g0_block_haps[g] != -1)
                if np.sum(g_valid) < 10:
                    # Not enough confident G0 sites — skip this G0 for this block
                    g0_best_matches.append((g, -1, float('nan'), int(np.sum(g_valid))))
                    continue
                best_err = 101.0
                best_id = -1
                for (hid, disc_h) in discovered:
                    disc_valid = (disc_h != -1) if -1 in disc_h else np.ones_like(disc_h, dtype=bool)
                    mask = g_valid & disc_valid
                    if np.sum(mask) < 10:
                        continue
                    err = np.mean(g0_block_haps[g, mask] != disc_h[mask]) * 100.0
                    if err < best_err:
                        best_err = err
                        best_id = hid
                g0_best_matches.append((g, best_id, best_err, int(np.sum(g_valid))))

            # For each discovered haplotype, find its best-matching G0
            disc_best_matches = []
            for (hid, disc_h) in discovered:
                disc_valid = (disc_h != -1) if -1 in disc_h else np.ones_like(disc_h, dtype=bool)
                best_err = 101.0
                best_g = -1
                best_n = 0
                for g in range(n_g0):
                    g_valid = (g0_block_haps[g] != -1)
                    mask = g_valid & disc_valid
                    if np.sum(mask) < 10:
                        continue
                    err = np.mean(g0_block_haps[g, mask] != disc_h[mask]) * 100.0
                    if err < best_err:
                        best_err = err
                        best_g = g
                        best_n = int(np.sum(mask))
                disc_best_matches.append((hid, best_g, best_err, best_n))

            # Block-level summary metrics
            founders_found = sum(
                1 for (_, _, err, _) in g0_best_matches
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
                'n_true_founders': n_g0,
                'n_discovered': n_disc,
                ff_col: founders_found,
                'good_haps': good_count,
                'chimeras': chimera_count,
            }
            for g, bid, err, nsites in g0_best_matches:
                row[f'G0_{g}_{g0_names[g]}_best_disc'] = bid
                row[f'G0_{g}_{g0_names[g]}_err_pct'] = err
                row[f'G0_{g}_{g0_names[g]}_valid_sites'] = nsites
            rows.append(row)

        return rows

    def load_g0_from_t1(r_name):
        """Cheaply load only the ground-truth fields from T01 (skips the big
        global_probs / block_results / site priors).  Used by every post-stage
        validation pass so we don't reload the full T01 pickle just to get
        g0_probs."""
        t1 = load_contig("T01_vcf_discovery", r_name)
        g0_probs = t1['g0_probs']
        g0_sites = t1['global_sites']
        g0_names = t1['g0_sample_names']
        del t1
        return g0_probs, g0_sites, g0_names

    def run_stage_validation(stage_label, stage_key, blocks_loader_fn, csv_filename):
        """Run block-level validation against G0 truth for one completed stage.

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
        print(f"VALIDATION: {stage_label} vs G0 Ground Truth")
        print(f"{'='*60}")

        all_rows = []
        contigs_with_data = 0
        ff_col = f'founders_found_under_{MATCH_THRESHOLD_PCT:.0f}pct'

        for r_name in region_keys:
            if not contig_done(stage_key, r_name):
                print(f"  [skip] {r_name}: no checkpoint in {stage_key}")
                continue
            if not contig_done("T01_vcf_discovery", r_name):
                print(f"  [skip] {r_name}: no T01 checkpoint (needed for G0 truth)")
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
                all_found = sum(1 for r in rows if r[ff_col] == r['n_true_founders'])
                print(f"  {r_name}: {len(rows)} blocks, mean {mean_haps:.1f} haps/block, "
                      f"all-4-G0s: {all_found}/{len(rows)} ({100*all_found/len(rows):.1f}%), "
                      f"mean good={mean_good:.1f}, chim={mean_chim:.1f}")

            del g0_probs, blocks
            gc.collect()

        if all_rows:
            df = pd.DataFrame(all_rows)
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False)

            total_blocks = len(df)
            total_all_found = int((df[ff_col] == df['n_true_founders']).sum())
            overall_good = df['good_haps'].mean()
            overall_chim = df['chimeras'].mean()
            overall_disc = df['n_discovered'].mean()

            print(f"\n  Overall across {contigs_with_data} contigs:")
            print(f"    Total blocks: {total_blocks}")
            print(f"    Mean discovered haps per block: {overall_disc:.2f}")
            print(f"    Blocks with ALL {df['n_true_founders'].iloc[0]} G0s recovered "
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
          - validation_T09_pedigree_confusion.csv (structural x biological crosstab)
          - validation_T09_pedigree_per_sample.csv (per-sample audit with
            inferred Generation, Parent1, Parent2, and true_generation columns)

        Returns (n_correct, n_samples_audit, pedigree_accuracy, expected_mapping)
        for use in the final summary.
        """
        print(f"\n{'='*60}")
        print(f"VALIDATION: T09 Pedigree Structure vs Metafile")
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

        confusion_csv = os.path.join(output_dir, "validation_T09_pedigree_confusion.csv")
        confusion.to_csv(confusion_csv)
        print(f"\nConfusion matrix saved to: {confusion_csv}")

        audit_csv = os.path.join(output_dir, "validation_T09_pedigree_per_sample.csv")
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
    #                         (what T08/T09/T10 receive as sample_ids)
    #   g0_sample_names     : the 4 G0 primary_IDs (for T11 validation)
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
    # checkpoint so Stage T11 can use it for validation.  When
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

            all_positions, all_reads = [], []
            for i in range(len(genomic_data)):
                pos_i = genomic_data.positions[i]
                reads_i = genomic_data.reads[i]
                if len(pos_i) > 0:
                    all_positions.append(pos_i)
                    all_reads.append(reads_i)

            if not all_positions:
                print(f"    WARNING: No data for {r_name}, skipping")
                continue

            global_sites = np.concatenate(all_positions)
            # Full reads: (n_samples_total, n_sites, 2) — all 116 samples
            global_reads_full = np.concatenate(all_reads, axis=1)

            # De-duplicate sites (overlapping blocks can repeat positions)
            _, unique_idx = np.unique(global_sites, return_index=True)
            unique_idx = np.sort(unique_idx)
            global_sites = global_sites[unique_idx]
            global_reads_full = global_reads_full[:, unique_idx, :]

            # ALWAYS extract G0 reads separately for T11 validation.  This slice
            # is independent of the USE_KNOWN_FOUNDERS flag — we want ground
            # truth available regardless of what the pipeline sees.
            g0_reads = global_reads_full[g0_vcf_indices, :, :]
            (_, g0_probs) = analysis_utils.reads_to_probabilities(g0_reads)
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

            (site_priors, global_probs) = analysis_utils.reads_to_probabilities(active_reads_full)
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
                num_processes=n_processes
            )
            valid_blocks = [b for b in block_results if len(b.positions) > 0]
            block_results = block_haplotypes.BlockResults(valid_blocks)

            hap_counts = [len(b.haplotypes) for b in valid_blocks]
            print(f"    [Discovery] {len(valid_blocks)} blocks, haps/block: "
                  f"min={min(hap_counts)}, max={max(hap_counts)}, "
                  f"mean={np.mean(hap_counts):.1f} in {time.time()-t0:.1f}s")

            # Checkpoint bundle.  g0_probs is always saved (T11 ground truth).
            save_contig(STAGE_T1, r_name, {
                'global_probs': global_probs, 'global_sites': global_sites,
                'block_results': block_results, 'avg_depth': avg_depth,
                'g0_probs': g0_probs, 'g0_sample_names': g0_sample_names,
                'active_vcf_indices': active_vcf_indices,
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
        })
        print(f"\nVCF loading + discovery complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T1)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T01 Block Discovery
    # =========================================================================
    # Compare the raw 200-SNP block haplotypes against the 4 G0 founders.
    # This is the first-stage quality gate: "did HDBSCAN on the reads produce
    # clusters that correspond to the true founder haplotypes?"
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

            print(f"\n  {r_name}: average read depth = {avg_depth:.1f}x")

            if avg_depth < REFINEMENT_DEPTH_THRESHOLD:
                print(f"  Depth < {REFINEMENT_DEPTH_THRESHOLD}x -> Running L1+L2 refinement")
                num_samples = global_probs.shape[0]
                chimera_resolution.warmup_jit(num_samples)

                # NOTE: cc_scale=0.5 here (not 0.2 as in pipeline_real.py) per
                # user direction — 0.5 is the correct value for this assembler
                # and should be used everywhere.
                #
                # DIAGNOSTIC CHANGES (refinement hang investigation):
                #   - verbose=True on L2 so we can see which block / gap the
                #     HMM linking is working on when it stalls (was False).
                #   - maxtasksperchild removed from BOTH L1 and L2 calls so the
                #     pool uses its default (workers not recycled after every
                #     batch).  With hundreds of refinement batches, the
                #     maxtasksperchild=1 setting caused hundreds of forkserver
                #     spawns which can wedge under high spawn rate; removing
                #     it trades the glibc-malloc-fragmentation protection for
                #     pool stability.  Node has 486 GB free so fragmentation
                #     is not an immediate concern here.
                def make_l1_fn(gp, gs):
                    def l1_fn(input_blocks):
                        return hierarchical_assembly.run_hierarchical_step(
                            input_blocks=input_blocks, global_probs=gp, global_sites=gs,
                            batch_size=REFINEMENT_BATCH_SIZE, use_hmm_linking=False,
                            beam_width=200, max_founders=12, max_sites_for_linking=2000,
                            cc_scale=0.5, num_processes=n_processes)
                    return l1_fn

                def make_l2_fn(gp, gs):
                    def l2_fn(input_blocks):
                        return hierarchical_assembly.run_hierarchical_step(
                            input_blocks=input_blocks, global_probs=gp, global_sites=gs,
                            batch_size=REFINEMENT_BATCH_SIZE, use_hmm_linking=True,
                            recomb_rate=RECOMB_RATE, beam_width=200, max_founders=12,
                            cc_scale=0.5, num_processes=n_processes,
                            n_generations=N_GENERATIONS, verbose=True)
                    return l2_fn

                t0 = time.time()
                refinement_results = block_haplotype_refinement.run_refinement_pipeline(
                    raw_blocks=block_results, global_probs=global_probs,
                    global_sites=global_sites, num_samples=num_samples,
                    run_l1_assembly_fn=make_l1_fn(global_probs, global_sites),
                    run_l2_assembly_fn=make_l2_fn(global_probs, global_sites),
                    batch_size=REFINEMENT_BATCH_SIZE, penalty_scale=REFINEMENT_PENALTY_SCALE,
                    recomb_rate=RECOMB_RATE, n_generations=N_GENERATIONS, verbose=True)
                print(f"\n  Refinement complete in {time.time()-t0:.0f}s")

                l2_refined = refinement_results['l2_refined']
                l2_refined_dd = block_haplotype_refinement.dedup_blocks(l2_refined, verbose=True)
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

            save_contig(STAGE_T7, r_name, {'super_blocks_L4': l4_blocks})
            del l3_blocks, l4_blocks
            gc.collect()

        print(f"\nL4 assembly complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T7)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T07 L4 Assembly (final founder-block validation)
    # =========================================================================
    # L4 is the final assembly level.  Ideally one chromosome-scale super-block
    # per contig containing all 4 founder haplotypes at low error.
    run_stage_validation(
        stage_label="T07_L4_assembly",
        stage_key="T07_assembly_L4",
        blocks_loader_fn=lambda r: load_contig("T07_assembly_L4", r)['super_blocks_L4'],
        csv_filename="validation_T07_L4_assembly.csv"
    )

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T08: Viterbi Painting
    # =========================================================================
    STAGE_T8 = "T08_viterbi_painting"

    if stage_complete(STAGE_T8):
        print(f"\n[RESUME] Skipping Viterbi painting (checkpoint found)")
    else:
        print(f"\n{'='*60}")
        print("STAGE T08: Viterbi Painting (Tropheops)")
        print(f"{'='*60}")
        start = time.time()

        with paint_samples.PaintingPoolManager(num_processes=n_processes) as painter:
            for r_name in region_keys:
                if contig_done(STAGE_T8, r_name):
                    print(f"  [RESUME] {r_name} already done")
                    continue

                print(f"\n  [Viterbi Painting] Processing Region: {r_name}")

                t7 = load_contig(STAGE_T7, r_name)
                discovered_block = t7['super_blocks_L4'][0]
                discovered_block.probs_array = None  # reconstructible from global_probs
                del t7

                global_probs, global_sites = load_global_arrays(r_name)

                painting_result = painter.paint_chromosome(
                    discovered_block, global_probs, global_sites,
                    recomb_rate=5e-8, switch_penalty=10.0, batch_size=1)

                # Population painting visualization — uses the ACTIVE sample
                # names (116 in withFounders mode, 112 in withoutFounders mode)
                # so row labels match the sample axis of painting_result.
                print(f"  Generating Population Painting Plot...")
                plot_filename = os.path.join(output_dir, f"{r_name}_viterbi_population.png")
                paint_samples.plot_population_painting(
                    painting_result, output_file=plot_filename,
                    title=f"Viterbi Painting - {r_name} ({_mode_label})",
                    sample_names=sample_names_active, figsize_width=20,
                    row_height_per_sample=0.25)

                save_contig(STAGE_T8, r_name, {'tolerance_result': painting_result})
                del discovered_block, global_probs, painting_result
                gc.collect()

        print(f"\nViterbi painting complete in {time.time()-start:.1f}s")
        mark_stage_complete(STAGE_T8)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T09: Pedigree Inference
    # =========================================================================
    STAGE_T9 = "T09_pedigree_inference"

    if stage_complete(STAGE_T9):
        print(f"\n[RESUME] Skipping pedigree inference (checkpoint found)")
        pedigree_df = load_global(STAGE_T9)['pedigree_df']
    else:
        print(f"\n{'='*60}")
        print("STAGE T09: Multi-Contig Pedigree Inference (Tropheops)")
        print(f"{'='*60}")

        contig_inputs = []
        for r_name in region_keys:
            t8 = load_contig(STAGE_T8, r_name)
            t7 = load_contig(STAGE_T7, r_name)
            founder_block = t7['super_blocks_L4'][0]
            founder_block.probs_array = None  # not needed for pedigree inference
            entry = {
                'tolerance_painting': t8['tolerance_result'],
                'founder_block': founder_block
            }
            contig_inputs.append(entry)
            del t8, t7

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

        save_global(STAGE_T9, {'pedigree_df': pedigree_df})
        del contig_inputs
        gc.collect()
        mark_stage_complete(STAGE_T9)

#%%
if __name__ == '__main__':
    # =========================================================================
    # VALIDATION: After T09 Pedigree Inference
    # =========================================================================
    # Cross-checks the inferred pedigree_df against the metafile's biological
    # generation column.  Writes a confusion matrix and per-sample audit CSV,
    # and computes the structural->biological label translation accuracy.
    # Runs unconditionally at each invocation (cheap, no checkpointing).
    if 'pedigree_df' not in dir():
        pedigree_df = load_global("T09_pedigree_inference")['pedigree_df']
    _t9_val_result = run_pedigree_validation(pedigree_df)
    # Keep the tuple for the final report: (n_correct, n_samples_audit,
    # pedigree_accuracy, expected_mapping)

#%%
if __name__ == '__main__':
    # =========================================================================
    # STAGE T10: Phase Correction + Greedy Refinement + F1 Recoloring + Propagation
    # =========================================================================
    STAGE_T10 = "T10_phase_correction"

    if stage_complete(STAGE_T10):
        print(f"\n[RESUME] Skipping phase correction (checkpoint found)")
    else:
        print("\n" + "="*60)
        print("STAGE T10: Phase Correction (Tropheops)")
        print("="*60)

        if 'pedigree_df' not in dir():
            pedigree_df = load_global(STAGE_T9)['pedigree_df']

        # Define loader callback — workers load their own contig data
        def _load_contig_for_phase_correction(r_name):
            t7 = load_contig(STAGE_T7, r_name)
            t8 = load_contig(STAGE_T8, r_name)
            founder_block = t7['super_blocks_L4'][0]
            founder_block.probs_array = None  # not needed for phase correction
            data = {
                'tolerance_result': t8['tolerance_result'],
                'founder_block': founder_block,
            }
            del t7, t8
            return data

        # Lightweight dict — just contig names, workers load their own data
        mcr = {r_name: {} for r_name in region_keys}

        # Step 1: Viterbi phase correction (3 rounds, workers load via load_fn)
        start = time.time()
        mcr = phase_correction.correct_phase_all_contigs(
            mcr, pedigree_df, sample_names_active, num_rounds=3, verbose=True,
            load_fn=_load_contig_for_phase_correction)
        print(f"Phase correction time: {time.time()-start:.1f}s")

        # Step 2: Greedy phase refinement
        print("\n" + "="*60)
        print("Greedy Phase Refinement (HOM->HET boundary flips)")
        print("="*60)

        start_refine = time.time()
        mcr = phase_correction.post_process_phase_greedy_all_contigs(
            mcr, pedigree_df, sample_names_active,
            snps_per_bin=100, recomb_rate=5e-8, mismatch_cost=4.6, verbose=True)
        print(f"Greedy refinement time: {time.time()-start_refine:.1f}s")

        # Step 3: Parsimonious F1 recoloring
        print("\n" + "="*60)
        print("Parsimonious F1 Recoloring")
        print("="*60)

        for r_name in region_keys:
            if r_name not in mcr:
                continue
            data = mcr[r_name]
            painting_key = 'refined_painting' if 'refined_painting' in data else 'corrected_painting'
            if painting_key not in data:
                continue

            recolored = phase_correction.apply_parsimonious_f1_recoloring(
                data[painting_key], data['founder_block'],
                pedigree_df, sample_names_active,
                max_mismatch_rate=0.02, verbose=True)
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
                max_mismatch_rate=0.02, verbose=True)
            data['final_painting'] = propagated

        # Save per-contig results
        for r_name in region_keys:
            if r_name in mcr:
                d = {k: mcr[r_name][k]
                     for k in ('corrected_painting', 'refined_painting',
                               'final_painting', 'founder_block')
                     if k in mcr[r_name]}
                save_contig(STAGE_T10, r_name, d)

        del mcr
        gc.collect()
        mark_stage_complete(STAGE_T10)

#%%
if __name__ == '__main__':
    # =========================================================================
    # FINAL REPORT: Aggregate all per-stage validation CSVs
    # =========================================================================
    # Each T01-T07 stage wrote its own per-block validation CSV as soon as it
    # finished (see the VALIDATION cells interleaved between stages).  T09
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
    #     where all 4 G0s were recovered, mean good/chimera haps)
    #   - validation_summary.txt               (human-readable overview
    #     including the pedigree validation result from T09)
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
        n_all_found = int((df_stage[ff_col] == df_stage['n_true_founders']).sum())
        pct_all_found = 100.0 * n_all_found / n_blocks
        mean_disc = df_stage['n_discovered'].mean()
        mean_good = df_stage['good_haps'].mean()
        mean_chim = df_stage['chimeras'].mean()

        stage_summary_rows.append({
            'stage': stage_label,
            'total_blocks': n_blocks,
            'mean_discovered_haps': round(mean_disc, 3),
            'all_4_G0s_found_pct': round(pct_all_found, 2),
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

    # Pull in the T09 pedigree validation result if it ran earlier this session
    if '_t9_val_result' in dir():
        n_correct, n_samples_audit, pedigree_accuracy, expected_mapping = _t9_val_result
    else:
        n_correct, n_samples_audit, pedigree_accuracy, expected_mapping = (
            None, None, float('nan'), None
        )

    # Human-readable summary.txt
    summary_lines = []
    summary_lines.append(f"Tropheops Pipeline Validation Summary")
    summary_lines.append(f"Mode: {_mode_label} (USE_KNOWN_FOUNDERS={USE_KNOWN_FOUNDERS})")
    summary_lines.append(f"Timestamp: {datetime.now().isoformat()}")
    summary_lines.append(f"")
    summary_lines.append(f"Input:")
    summary_lines.append(f"  VCF: {vcf_path}")
    summary_lines.append(f"  Metafile: {meta_path}")
    summary_lines.append(f"  Total VCF samples: {n_samples_total}")
    summary_lines.append(f"  Active (pipeline-visible) samples: {n_samples}")
    summary_lines.append(f"  G0 samples (ground truth): {len(g0_sample_names)}")
    for nm in g0_sample_names:
        summary_lines.append(f"    - {nm}")
    summary_lines.append(f"  Contigs processed: {len(region_keys)}")
    summary_lines.append(f"")
    summary_lines.append(f"Block-level Founder Recovery Progression")
    summary_lines.append(f"  (G0 match threshold: <{MATCH_THRESHOLD_PCT:.0f}% allele error)")
    if len(summary_df) > 0:
        summary_lines.append(summary_df.to_string(index=False))
    else:
        summary_lines.append(f"  (no per-stage CSVs found)")
    summary_lines.append(f"")
    summary_lines.append(f"Pedigree Structure (T09):")
    if n_samples_audit is not None:
        summary_lines.append(f"  Pipeline 'Generation' is a STRUCTURAL label")
        summary_lines.append(f"    ('F1' = graph root; not a biological generation)")
        summary_lines.append(f"  Structural->biological translation: {expected_mapping}")
        summary_lines.append(f"  Samples matching translation: "
                             f"{n_correct}/{n_samples_audit} = {pedigree_accuracy:.1f}%")
    else:
        summary_lines.append(f"  (T09 validation did not run — pedigree not available)")
    summary_lines.append(f"")
    summary_lines.append(f"Artefacts in {output_dir}/:")
    summary_lines.append(f"  validation_T01_block_discovery.csv")
    summary_lines.append(f"  validation_T02_refinement.csv")
    summary_lines.append(f"  validation_T03_residual_discovery.csv")
    summary_lines.append(f"  validation_T04_L1_assembly.csv")
    summary_lines.append(f"  validation_T05_L2_assembly.csv")
    summary_lines.append(f"  validation_T06_L3_assembly.csv")
    summary_lines.append(f"  validation_T07_L4_assembly.csv")
    summary_lines.append(f"  validation_T09_pedigree_confusion.csv")
    summary_lines.append(f"  validation_T09_pedigree_per_sample.csv")
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
    print(f"Total time: {hours}h {minutes}m ({elapsed:.0f}s)")
    print(f"Checkpoints: {CHECKPOINT_DIR}/")
    print(f"Results: {output_dir}/")
    print(f"Regions processed: {len(region_keys)}")
    print(f"Active samples: {n_samples} (of {n_samples_total} in VCF)")