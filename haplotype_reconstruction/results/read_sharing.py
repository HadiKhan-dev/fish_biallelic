"""Derive the report-only pooled read-sharing scan from checked evidence and fixed calibration memberships."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import numpy as np

from ..contracts.contigs import AnalysisContigs
from ..core.concurrency import threaded_each
from ..core.errors import InvariantError
from ..stages.evidence import records
from ..stages.painting import read_library_records
from ..store.identity import code_dependency_identity
from ..store.protocol import Derivation, NumericalPolicy
from ..store.store import CheckpointStore

SCHEMA = 1
CAVEAT = ("The approximate conditional p-value is invalid against regional depth-intensity variation "
          "(independent-read simulations with regional depth variation gave p = .0047 and .0066).")


@dataclass(frozen=True)
class SharingProjection:
    """Sufficient statistics for one fixed membership on one contig's occupied, measurable lag bins."""

    grid_bp: np.ndarray
    numerator: np.ndarray  # float64[J], the signed curve projected onto pair-weighted overlap templates
    deletions: np.ndarray  # float64[N, J], centred jackknife scores in the input's canonical sample order
    active: np.ndarray  # bool[N], samples with depth here; absent rows contribute zero


def project(sharing, parameters):
    """Project one library's signed estimates and paired delete-one rows onto its configured survival grid."""
    jackknife = sharing.jackknife
    if jackknife is None:
        return None
    grid, weights = read_library_records.template_weights(parameters, sharing.edges, sharing.pairs, jackknife.bins)
    n = int(jackknife.active.sum())
    contributions = -math.sqrt((n - 1) / n) * (jackknife.centered @ weights)
    return SharingProjection(grid, sharing.excess[jackknife.bins] @ weights, contributions, jackknife.active)


MULTIPLIER_DRAWS = 9_999
"""Number of sign-flip maxima in the approximate conditional read-sharing diagnostic."""


def _template_maximum(score, contributions):
    scale = np.sqrt(np.einsum("ij,ij->j", contributions, contributions))
    usable = scale > 0
    if not np.any(usable):
        return None, None, usable, scale
    ratios = np.divide(score, scale, out=np.full_like(score, -np.inf), where=usable)
    winner = int(np.argmax(ratios))
    return float(ratios[winner]), winner, usable, scale


def pooled_sharing_scan(projections, sample_ids, seed, *, all_samples=False):
    """Pool a fixed membership, then studentise. Each sign belongs to one sample across every contig and length."""
    from ..core.random import bootstrap_seed

    # The all-sample stream at seed 421 reproduces the reference replay. Split streams name the fixed membership.
    names = tuple(sample_ids)
    draw_seed = int(seed) if all_samples else bootstrap_seed(("read sharing", int(seed), sorted(names)))
    used = [(c, p) for c, p in projections.items() if p is not None]
    row = dict(samples=len(names), active_samples=0, contigs=len(used), used_contigs=[c for c, _ in used],
               draw_seed=draw_seed, T=None, approximate_conditional_p=None, winning_length_bp=None,
               leave_one_contig_out_T={c: None for c, _ in used})
    if not used:
        return row
    grid = used[0][1].grid_bp
    score = np.zeros_like(grid, dtype=np.float64)
    contributions = np.zeros((len(names), len(grid)), dtype=np.float64)
    active = np.zeros(len(names), dtype=bool)
    for _, part in used:
        if not np.array_equal(part.grid_bp, grid):
            raise ValueError("read-sharing projection grids differ across contigs")
        score += part.numerator
        contributions += part.deletions
        active |= part.active
    row["active_samples"] = int(active.sum())
    for contig, part in used:
        row["leave_one_contig_out_T"][contig] = _template_maximum(
            score - part.numerator, contributions - part.deletions)[0]
    statistic, winner, usable, scale = _template_maximum(score, contributions)
    if winner is None:
        return row
    standardized = contributions[:, usable] / scale[usable]
    rng = np.random.default_rng(draw_seed)
    exceedances = 0
    # Batching bounds the number of live sign vectors. The draw order is independent of the CPU budget.
    for start in range(0, MULTIPLIER_DRAWS, 256):
        size = min(256, MULTIPLIER_DRAWS - start)
        signs = rng.integers(0, 2, size=(size, len(names))) * 2 - 1
        maxima = (signs @ standardized).max(axis=1)
        exceedances += int(np.count_nonzero(maxima >= statistic))
    row.update(T=statistic, winning_length_bp=float(grid[winner]),
               approximate_conditional_p=(1 + exceedances) / (1 + MULTIPLIER_DRAWS))
    return row


def scan_libraries(contigs, sample_ids, published, parameters, seed, load_sharing, load_depth, map_work=map):
    """Each distinct accepted membership is fixed across the analysed contigs, including its discovery contig."""
    by_contig = defaultdict(list)
    for row in published:
        if row["contig"] in contigs:
            by_contig[row["contig"]].append(row)

    def recover(contig):
        rows = sorted(by_contig[contig], key=lambda row: row["library"])
        depth, positions = load_depth(contig)
        sizes = [row["samples"] for row in rows]
        groups = read_library_records.published_groups(depth, positions, sizes, parameters)
        return [(members, (contig, row["library"])) for members, row in zip(groups, rows, strict=True)]

    origins = defaultdict(list)
    for leaves in map_work(recover, [c for c in contigs if len(by_contig[c]) > 1]):
        for members, origin in leaves:
            origins[members].append(origin)

    def project_contig(contig):
        parts = [project(load_sharing(contig), parameters)]
        if origins:
            depth, positions = load_depth(contig)
            curves = read_library_records.split_curves(depth, positions, origins)
            parts.extend(project(curve, parameters) for curve in curves)
        return parts

    parts = list(map_work(project_contig, contigs))
    rows = []
    for index, members in enumerate([tuple(range(len(sample_ids))), *origins]):
        ids = [sample_ids[i] for i in members]
        projections = {c: part[index] for c, part in zip(contigs, parts, strict=True)}
        row = pooled_sharing_scan(projections, ids, seed, all_samples=index == 0)
        row.update(library="all" if index == 0 else f"split-{index}", member_indices=list(members),
                   published_at=[dict(contig=c, library=k) for c, k in origins.get(members, ())])
        rows.append(row)
    return dict(schema=SCHEMA, label="read sharing (diagnostic)", interpretation=CAVEAT, run_seed=seed,
                multiplier_draws=MULTIPLIER_DRAWS, sample_ids=list(sample_ids), analysed_contigs=list(contigs),
                libraries=rows)


def identity(manifest, calibration_identity):
    """Bind the report record to its producers, run seed, checked calibration and selected axis."""
    return Derivation(
        code_dependency_identity(("results.read_sharing",)),
        dict(read_state=manifest["parameters"]["painting"]["read_state"],
             seed=manifest["realised_seeds"].get("simulation", 421),
             contigs=AnalysisContigs.from_manifest(manifest).included),
        NumericalPolicy(**manifest["numerical_policy"]), {"calibration": calibration_identity}, schema=SCHEMA,
    ).identity


def diagnostic(store, manifest, calibration, *, map_work=threaded_each):
    """Compute the run-report payload from immutable checked inputs. The report writer owns its publication."""
    contigs = AnalysisContigs.from_manifest(manifest).included
    seed = int(manifest["realised_seeds"].get("simulation", 421))
    parameters = read_library_records.read_state_parameters(manifest)
    identities = {c: records.expected(store, manifest, product="evidence", contig=c).identity for c in contigs}

    def sharing(contig):
        curve = records.load_read_sharing(store.contig(contig), identities[contig])
        if curve is None:
            raise InvariantError(f"{contig}: read-sharing evidence is missing")
        return curve

    def depth(contig):
        arrays = records.load_arrays(store.contig(contig), ("counts", "positions"),
                                    expected_identity=identities[contig])
        return np.sum(arrays["counts"], axis=2), arrays["positions"]

    return scan_libraries(contigs, manifest["sample_ids"], calibration["read_libraries"], parameters, seed,
                          sharing, depth, map_work)


def calibration_report_inputs(results):
    """Checked calibration and the identity and lazy computation for its independent diagnostic report."""
    request = results.stage_request("calibration")
    if request is None:
        return None
    calibration = results.require(request)
    store = CheckpointStore(results.checkpoint_location)
    manifest = results.manifest
    return calibration, identity(manifest, request.identity), partial(diagnostic, store, manifest, calibration)
