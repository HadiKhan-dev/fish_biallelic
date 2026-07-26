"""Command-line entry point for metadata-aware V7-margin pedigree inference."""

import argparse
import json
import os
import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from . import aggregation, model
from .design import (
    COMPATIBILITY_DESIGN,
    CONTIGS,
    candidate_sets,
    load_g0_seed_assignments,
)
from .io import atomic_csv, atomic_json, atomic_text, load_metadata
from .pedigree import build_pedigree, pedigree_edges, pedigree_fam
from .provenance import (
    build_cache_manifest,
    manifest_sha256,
    prepare_internal_cache,
    validate_cache_manifest,
)


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _prepare_output(path, settings, resume):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    settings_path = path / "settings.json"
    if settings_path.exists():
        if _read_json(settings_path) != settings:
            raise RuntimeError(
                "existing settings differ, including normalized seeds or cache "
                "provenance; choose a new output directory"
            )
        if not resume:
            raise RuntimeError("output exists; pass --resume")
    elif any(path.iterdir()):
        raise RuntimeError("refusing to use a non-empty output directory")
    else:
        atomic_json(settings_path, settings)
    return path


def _write_fam(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", header=False, index=False)
    os.replace(temporary, path)


def _score_caches(args, cache_dir, settings, candidates, selected_g0_pairs):
    requested = list(args.contigs) if args.contigs else list(CONTIGS)
    rows = []
    for contig in requested:
        result = model._score_contig(
            contig,
            cache_dir,
            settings,
            candidates,
            selected_g0_pairs,
            args.resume,
        )
        rows.append(result)
        print(
            f"{contig}: {result['source']} ({result['markers']:,} markers, "
            f"{result['seconds']:.1f}s)",
            flush=True,
        )
    atomic_csv(cache_dir.parent / "contig_run_status.csv", pd.DataFrame(rows))


def _write_outputs(output_dir, metadata, seeds, result):
    assignments = result.assignments
    atomic_csv(output_dir / "F1_G0_seed_assignments.csv", seeds)
    atomic_csv(
        output_dir / "F2_parent_assignments_v7_margin.csv", assignments
    )
    atomic_csv(
        output_dir / "F2_candidate_pair_evidence_v7_margin.csv",
        result.candidate_evidence,
    )
    atomic_csv(
        output_dir / "F2_parent_count_states_v7.csv", result.parent_states
    )
    policies = {
        "pedigree_v7_margin_best_estimate": "leading_hypothesis",
        "pedigree_v7_margin_tier_A": "tier_A",
        "pedigree_v7_margin_tier_B": "tier_B_or_better",
    }
    pedigrees = {}
    for stem, policy in policies.items():
        pedigree = build_pedigree(metadata, seeds, assignments, policy)
        pedigrees[policy] = pedigree
        atomic_csv(output_dir / f"{stem}.csv", pedigree)
        atomic_csv(
            output_dir / f"{stem}_edges.csv", pedigree_edges(pedigree)
        )
        _write_fam(output_dir / f"{stem}.fam", pedigree_fam(pedigree))
    return pedigrees


def _summary(args, result, pedigrees, cache_validation, elapsed):
    assignments = result.assignments
    return {
        "schema_version": 7,
        "model_revision": aggregation.MODEL_REVISION,
        "compatibility_mode": "exact_selected_tropheops_v7_margin",
        "children": int(len(assignments)),
        "candidate_pairs": int(len(result.candidate_evidence) / len(assignments)),
        "retained_informative_markers": int(np.sum(result.marker_counts)),
        "tier_A_exact_pairs": int(assignments["tier_A_exact_pair"].sum()),
        "tier_A_fathers": int(assignments["tier_A_father"].sum()),
        "tier_A_mothers": int(assignments["tier_A_mother"].sum()),
        "tier_B_or_better_exact_pairs": int(
            assignments["tier_B_or_better_exact_pair"].sum()
        ),
        "tier_B_or_better_fathers": int(
            assignments["tier_B_or_better_father"].sum()
        ),
        "tier_B_or_better_mothers": int(
            assignments["tier_B_or_better_mother"].sum()
        ),
        "leading_hypothesis_F2_edges": int(
            pedigrees["leading_hypothesis"].loc[
                lambda frame: frame["MetadataGeneration"].eq("F2"),
                ["Parent1", "Parent2"],
            ].notna().sum().sum()
        ),
        "cache_provenance_validation": cache_validation,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": model.BOOTSTRAP_SEED,
        "threads": int(args.threads),
        "elapsed_seconds": float(elapsed),
        "scientific_caveat": (
            "There is no independent individual-level parentage truth. Tier "
            "labels are internal stability policies, not calibrated error "
            "probabilities. G0-to-F1 pairs are explicit computational seeds "
            "and must not be represented as breeding-record truth unless their "
            "seed provenance independently establishes that status."
        ),
    }


def _scientific_parameters(args):
    return {
        "checkpoint_painting_stage": "T08_viterbi_painting",
        "checkpoint_founder_stage": "T10_phase_correction",
        "uses_T10_painting": False,
        "marker_selection": "all_retained_founder_informative_with_PL",
        "markers_per_contig": 0,
        "bcf_threads": int(args.bcf_threads),
        "recombination_rate": float(args.recombination_rate),
        "equivalence_window_bp": int(args.equivalence_window_bp),
        "max_diff_fraction": float(args.max_diff_fraction),
        "min_diff_sites": int(args.min_diff_sites),
        "variant_labels": list(model.VARIANT_LABELS),
        "variant_specifications": [
            {
                "error_rate": float(error_rate),
                "markers_per_block": int(markers_per_block),
                "unsmoothed": bool(unsmoothed),
            }
            for error_rate, markers_per_block, unsmoothed
            in model._variant_specifications()
        ],
        "primary_variant": int(model.PRIMARY_VARIANT),
        "effective_markers_per_block": 1.0,
        "parent_state_names": list(model.PARENT_STATE_NAMES),
        "parent_state_priors": {
            name: list(values) for name, values in model.PARENT_STATE_PRIORS.items()
        },
        "g0_linked_hmm_role": "diagnostic_only",
        "f1_reconstruction": "sitewise_phase_invariant_unphased_Mendelian",
        "unknown_gamete_model": "sex_specific_empirical_F1_homolog_mixture",
        "rank_weight": aggregation.RANK_WEIGHT,
        "margin_weight": aggregation.MARGIN_WEIGHT,
        "block_tempering_power": aggregation.BLOCK_TEMPERING_POWER,
        "chromosome_contamination": aggregation.CHROMOSOME_CONTAMINATION,
        "tier_B_minimum_variants": aggregation.TIER_B_VARIANT_MINIMUM,
        "tier_B_minimum_LOCO_chromosomes": aggregation.TIER_B_LOCO_MINIMUM,
        "bootstrap_seed": model.BOOTSTRAP_SEED,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone metadata-aware Tropheops V7-margin pedigree "
            "analysis. Inputs and outputs are never inferred from old result "
            "directory names."
        )
    )
    parser.add_argument("--bcf", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help=(
            "pipeline checkpoint directory; also required with --cache-dir "
            "to validate the 44 source checkpoint identities"
        ),
    )
    parser.add_argument(
        "--g0-seeds",
        type=Path,
        required=True,
        help="explicit F1-to-G0 pair seed CSV; see README.md",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "read a complete, provenance-manifested set of v7_<contig>.npz "
            "caches and skip checkpoint/BCF scoring"
        ),
    )
    parser.add_argument(
        "--allow-legacy-cache-without-manifest",
        action="store_true",
        help=(
            "UNSAFE compatibility override for pre-package exploratory caches; "
            "sample/design/seed/source provenance cannot be verified"
        ),
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bcf-threads", type=int, default=4)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=model.DEFAULT_BOOTSTRAPS
    )
    parser.add_argument("--recombination-rate", type=float, default=5e-8)
    parser.add_argument("--equivalence-window-bp", type=int, default=10_000)
    parser.add_argument("--max-diff-fraction", type=float, default=0.02)
    parser.add_argument("--min-diff-sites", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contigs", nargs="*", choices=CONTIGS)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="score requested contigs without requiring all 22 caches",
    )
    return parser


def run(args):
    started = time.monotonic()
    if args.threads < 1 or args.threads > numba.config.NUMBA_NUM_THREADS:
        raise ValueError("threads is outside the visible Numba thread range")
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    if args.bcf_threads < 1:
        raise ValueError("bcf-threads must be positive")
    if not np.isfinite(args.recombination_rate) or args.recombination_rate < 0.0:
        raise ValueError("recombination-rate must be finite and non-negative")
    if args.equivalence_window_bp < 1:
        raise ValueError("equivalence-window-bp must be positive")
    if (
        not np.isfinite(args.max_diff_fraction)
        or not 0.0 <= args.max_diff_fraction <= 1.0
    ):
        raise ValueError("max-diff-fraction must lie in [0, 1]")
    if args.min_diff_sites < 0:
        raise ValueError("min-diff-sites must be non-negative")
    if args.allow_legacy_cache_without_manifest and args.cache_dir is None:
        raise ValueError(
            "legacy cache override is meaningful only with --cache-dir"
        )
    numba.set_num_threads(args.threads)

    bcf = args.bcf.resolve()
    metadata_path = args.metadata.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    g0_seeds = args.g0_seeds.resolve()
    for path in (bcf, metadata_path, checkpoint_dir, g0_seeds):
        if not path.exists():
            raise FileNotFoundError(path)
    _, contig_lengths, metadata = load_metadata(bcf, metadata_path)
    missing_contigs = [contig for contig in CONTIGS if contig not in contig_lengths]
    if missing_contigs:
        raise RuntimeError(f"BCF is missing compatibility contigs: {missing_contigs}")
    candidates = candidate_sets(metadata)
    selected_g0_pairs, seed_audit = load_g0_seed_assignments(
        g0_seeds, candidates, metadata
    )
    scientific_parameters = _scientific_parameters(args)
    cache_manifest = build_cache_manifest(
        scoring_model_revision=model.MODEL_REVISION,
        contigs=CONTIGS,
        bcf_path=bcf,
        metadata_path=metadata_path,
        checkpoint_dir=checkpoint_dir,
        metadata=metadata,
        candidates=candidates,
        selected_g0_pairs_local=selected_g0_pairs,
        seed_audit=seed_audit,
        scientific_parameters=scientific_parameters,
    )

    external_cache = args.cache_dir.resolve() if args.cache_dir else None
    if external_cache is not None:
        cache_validation = validate_cache_manifest(
            external_cache,
            cache_manifest,
            allow_legacy_cache_without_manifest=(
                args.allow_legacy_cache_without_manifest
            ),
        )
        if cache_validation.startswith("unsafe_"):
            print(
                "WARNING: using a legacy cache without verifiable BCF/sample/"
                "design/seed/checkpoint provenance",
                flush=True,
            )
    else:
        cache_validation = "validated_package_owned_cache"

    settings = {
        "schema_version": 7,
        "model_revision": aggregation.MODEL_REVISION,
        "compatibility_mode": "exact_selected_tropheops_v7_margin",
        "bcf": str(bcf),
        "metadata": str(metadata_path),
        "checkpoint_dir": str(checkpoint_dir),
        "g0_seeds": str(g0_seeds),
        "normalized_g0_seed_sha256": cache_manifest["g0_seed_sha256"],
        "cache_manifest_sha256": manifest_sha256(cache_manifest),
        "external_cache_dir": str(external_cache) if external_cache else None,
        "allow_legacy_cache_without_manifest": bool(
            args.allow_legacy_cache_without_manifest
        ),
        "candidate_design": COMPATIBILITY_DESIGN.as_dict(),
        "contigs": list(CONTIGS),
        "threads": int(args.threads),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "scientific_parameters": scientific_parameters,
    }
    output_dir = _prepare_output(args.output_dir, settings, args.resume)
    atomic_json(
        output_dir / "expected_cache_manifest.json", cache_manifest
    )
    atomic_csv(output_dir / "F1_G0_seed_assignments.csv", seed_audit)

    if external_cache is None:
        cache_dir = prepare_internal_cache(
            output_dir / "contig_cache", cache_manifest, args.resume
        )
        _score_caches(
            args, cache_dir, scientific_parameters | {
                "bcf": str(bcf),
                "checkpoint_dir": str(checkpoint_dir),
            }, candidates, selected_g0_pairs
        )
    else:
        cache_dir = external_cache
    if args.diagnostics_only:
        return None
    missing = [
        contig for contig in CONTIGS
        if not (cache_dir / f"v7_{contig}.npz").exists()
    ]
    if missing:
        raise RuntimeError(f"missing V7 contig caches: {missing}")

    result = aggregation.aggregate_v7_margin(
        cache_dir,
        metadata,
        candidates,
        cache_manifest,
        bootstrap_replicates=args.bootstrap_replicates,
        allow_legacy_cache_without_manifest=(
            args.allow_legacy_cache_without_manifest
        ),
    )
    pedigrees = _write_outputs(output_dir, metadata, seed_audit, result)
    summary = _summary(
        args, result, pedigrees, cache_validation, time.monotonic() - started
    )
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(
        output_dir / "SCIENTIFIC_CAVEATS.txt",
        summary["scientific_caveat"] + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
