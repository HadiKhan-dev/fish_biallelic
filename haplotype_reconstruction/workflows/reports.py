"""Readable workflow diagnostics, separate from inference decisions.

Reference comparisons are descriptive consistency checks, not pedigree truth.
These reports never select haplotypes, alter missingness, or gate reconstruction.
"""
from pathlib import Path

import numpy as np
import pandas as pd

GENOTYPE_CONFIDENCE = 0.85
MATCH_THRESHOLD_PCT = 2.0
MIN_CONF_SITES = 10


def extract_g0_block_haps(g0_probs, g0_sites, block_positions):
    """Call reference dosages (0/1/2), retaining -1 for uncertain sites.

    These are observed genotype references, not founder truth or phased
    chromosomes. Both homozygous and heterozygous genotype calls are retained.
    """
    pos_idx = np.searchsorted(g0_sites, block_positions)
    pos_idx = np.clip(pos_idx, 0, len(g0_sites) - 1)
    matched = (g0_sites[pos_idx] == block_positions)

    n_g0 = g0_probs.shape[0]
    n_block = len(block_positions)
    g0_geno = np.full((n_g0, n_block), -1, dtype=np.int8)

    for g in range(n_g0):
        probs_g = g0_probs[g, pos_idx,:]
        argmax = np.argmax(probs_g, axis=1)  # 0/1/2 = dosage
        maxp = probs_g[np.arange(n_block), argmax]
        conf = (maxp >= GENOTYPE_CONFIDENCE) & matched
        g0_geno[g, conf] = argmax[conf].astype(np.int8)

    return g0_geno

def validate_block_list_against_g0(blocks, g0_probs, g0_sites,
                                   g0_names, stage_label, contig_name, *,
                                   include_reference_samples):
    """Measure local compatibility with sequenced G0 genotype references.

    Compare each reference with both a single haplotype at homozygous sites
    and a pair at all confident genotype sites. A matching pair need not be
    uniquely phased. No match can mean absent ancestry, incomplete reference
    sampling, or reconstruction error; it does not establish a chimera.
    When the references participated in discovery this is not independent
    validation. Cohort labels do not establish any individual's parentage.
    """
    rows = []
    n_g0 = g0_probs.shape[0]
    ff_col = f'references_matched_hom_under_{MATCH_THRESHOLD_PCT:.0f}pct'

    for block_idx, block in enumerate(blocks):
        positions = block.positions
        if len(positions) == 0:
            continue

        # Observed G0 reference genotypes {0,1,2,-1}; hom-only calls
        # {0,1,-1}.
        g0_geno = extract_g0_block_haps(g0_probs, g0_sites, positions)
        g0_hom = np.where(g0_geno == 0, 0,
                          np.where(g0_geno == 2, 1, -1)).astype(np.int8)

        # Score only explicit block discovery calls; unknown founder alleles are
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

        # Compare each discovered hap with available homozygous reference calls.
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
                (Dm[:, None,:] >= 0)
                & (Dm[None,:,:] >= 0)
            )
            dosage = Dm[:, None,:] + Dm[None,:,:]
            n_valid = np.sum(valid, axis=2)
            mismatch = np.sum(
                (dosage != geno_m[None, None,:]) & valid,
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
        references_matched = sum(
            1 for (_, _, err, _) in g0_best_matches
            if not np.isnan(err) and err < MATCH_THRESHOLD_PCT
        )
        # References with enough confident homozygous sites to be evaluated.
        n_scorable_homonly = sum(
            1 for (_, _, err, _) in g0_best_matches if not np.isnan(err)
        )
        references_matched_pair = sum(
            1 for (_, _, err, _) in g0_pair_matches
            if not np.isnan(err) and err < MATCH_THRESHOLD_PCT
        )
        unmatched_count = sum(
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
            'n_g0_reference_samples': n_g0,
            'g0_reference_is_independent': not include_reference_samples,
            'n_scorable': n_scorable,
            'n_scorable_homonly': n_scorable_homonly,
            'n_discovered': n_disc,
            ff_col: references_matched,
            'references_matched_pair': references_matched_pair,
            'haplotypes_matching_reference': good_count,
            'haplotypes_without_reference_match': unmatched_count,
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


def write_reference_comparison(
    store, contigs, output_dir, *, include_reference_samples,
    source_stage="block_discovery",
):
    """Read each block discovery chromosome once and export G0 consistency metrics."""
    kind = ("non-independent; references included in discovery"
            if include_reference_samples else "held-out reference comparison")
    print(f"G0 genotype consistency ({kind}); not founder or pedigree truth")
    rows = []
    for contig in contigs:
        if not store.contig_done(source_stage, contig):
            print(f"  [skip] {contig}: no discovery checkpoint")
            continue
        payload = store.load_contig(source_stage, contig)
        current = validate_block_list_against_g0(
            payload["block_results"], payload["g0_probs"], payload["global_sites"],
            payload["g0_sample_names"], "block_discovery", contig,
            include_reference_samples=include_reference_samples,
        )
        rows.extend(current)
        print(f"  {contig}: {len(current)} blocks compared")
        del payload
    if not rows:
        print("  No reference-comparison rows available.")
        return
    frame = pd.DataFrame(rows)
    path = Path(output_dir) / "reference_consistency.csv"
    frame.to_csv(path, index=False)
    print(f"Reference consistency: {len(frame)} blocks; {path}")


def report_discovery(valid_blocks):
    """Summarize local cavity-search limits, uncertainty and wildcard mass."""
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
            "every informative block discovery block must use the "
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
