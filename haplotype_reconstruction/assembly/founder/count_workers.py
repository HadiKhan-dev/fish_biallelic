"""Independent deletion repairs sharing immutable evidence in one process.

Each candidate owns its score/painting workspace. The existing candidate thread
allocator grows survivors at numerical boundaries, bounded by the caller's CPU
budget. Results are applied in deletion order; checkpoints are saved by the
controller as each candidate finishes. No process startup or shared-memory copy
is needed, and native repair kernels release the GIL.
"""
import resource
import time

import numpy as np

from ...core import parallel
from.workspace import component_workspace, resolve_threads
from.candidates import completed_candidates
from.import scoring as founder_scoring
from ...discovery.objectives import compute_outer_bic_from_log_likelihood as bic


def fit_deletion(dropped, batch, selected, neutral, sites, config, threads,
                 cost, workspaces, prepared_arrays=None):
    started, cpu_started = time.perf_counter(), time.process_time()
    rows = np.delete(selected, int(dropped), axis=0)
    workspace = component_workspace(workspaces, batch, neutral, sites,
                                    config.proposal_max_bins, threads, prepared_arrays,
                                    minimum_bin_size=config.proposal_min_sites_per_bin)
    likelihood = workspace.evaluate(rows)
    history = []
    for iteration in range(config.count_repair_sweeps):
        painting = workspace.evaluate(rows, paint=True)
        with parallel.numba_thread_scope(resolve_threads(threads)):
            replacement, gains = founder_scoring.fixed_path_proposals(
                workspace.leaves, workspace.offsets, rows, workspace.evidence,
                workspace.complete, painting, workspace.logs)
        focal = int(np.argmax(gains))
        single = rows.copy()
        single[focal] = replacement[focal]
        best_rows, best_score = rows, likelihood
        # Two repairs per deletion, not K full rescans inside K deletion starts.
        for proposal in (replacement, single):
            if np.array_equal(proposal, rows):
                continue
            value = workspace.evaluate(proposal)
            if value > best_score + 1e-6:
                best_rows, best_score = proposal, value
        history.append(dict(iteration=iteration, gain=best_score - likelihood))
        if best_score <= likelihood + 1e-6:
            break
        rows, likelihood = best_rows, best_score
    found = dict(selected=rows, score=likelihood,
                 bic=float(bic(len(rows), likelihood, cost)), paired_changed=False,
                 repair_history=history,
                 runtime=dict(seconds=time.perf_counter() - started,
                              shared_process_cpu_seconds=time.process_time() - cpu_started,
                              maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return found


def run(tasks, *, batch, selected, neutral, sites,
        config, threads, cost, workspaces):
    """Return canonical deletion order, irrespective of completion order."""
    if not tasks:
        return []
    total = int(resolve_threads(threads))
    workspace = component_workspace(workspaces, batch, neutral, sites,
                                    config.proposal_max_bins, threads,
                                    minimum_bin_size=config.proposal_min_sites_per_bin)
    arrays = workspace.evidence, workspace.complete, workspace.logs
    functions = []
    for dropped, _ in tasks:
        def compute(budget, dropped=dropped):
            return fit_deletion(dropped, batch, selected, neutral, sites, config,
                threads if budget is None else budget, cost, {}, prepared_arrays=arrays)
        functions.append(compute)
    samples, length = workspace.evidence.shape[:2]
    founders = len(selected) - 1
    states = founders * (founders + 1) // 2
    # Private painting/allele arrays plus traceback scratch. The shared evidence
    # is already resident. Allow a full tail budget when bounding scratch RAM.
    per_candidate = (2 * workspace.evidence.nbytes + samples * length * 4
                     + min(total, samples) * length * (states + 4)
                     + 512 * 1024 ** 2)
    answers = [None] * len(tasks)
    for index, result in completed_candidates(functions, threads, per_candidate):
        answers[index] = result
        tasks[index][1].save("result", result)
    return answers
