"""Read-only evidence and batched leaf emissions for final founder refinement.

The common full-chromosome component shares its existing contiguous float32
evidence; ragged components gather into private contiguous arrays. No consumer
mutates these likelihoods. Batching preserves each pair's site/bin accumulation
order, the robust floor, neutral observations, and shared unknown-founder
identity. It changes preparation only, never the candidate panels or masks.
"""
import math
import numpy as np
from numba import njit, prange
from numba.typed import List
from..import observations, partial_emissions
from.packing import PreparedModels

@njit(cache=True, parallel=True, nogil=True)
def _gather_evidence(neutral, indices):
    out = np.empty((len(neutral), len(indices), 3), np.float32)
    chunks = (len(indices) + 4095) // 4096
    for task in prange(len(neutral) * chunks):
        sample, chunk = (task // chunks, task % chunks)
        for local in range(chunk * 4096, min(len(indices), (chunk + 1) * 4096)):
            for genotype in range(3):
                out[sample, local, genotype] = neutral[sample, indices[local], genotype]
    return out


def gather_evidence(neutral, indices):
    if len(indices) and indices[-1] - indices[0] + 1 == len(indices) and np.all(np.diff(indices) == 1):
        view = neutral[:, indices[0]:indices[-1] + 1,:]
        if view.dtype == np.float32 and view.flags.c_contiguous:
            return view
    return _gather_evidence(neutral, indices)


@njit(cache=True, parallel=True, nogil=True)
def fill(evidence, offsets, alleles, source, keeps, full, bin_size, outputs):
    samples = len(evidence)
    center = math.log(1.0 / 3.0)
    for task in prange(len(alleles) * samples):
        block, sample = (task // samples, task % samples)
        local = alleles[block]
        rows = source[block]
        kept = keeps[block]
        out = outputs[block]
        logs = np.empty(7, np.float64)
        for site in range(local.shape[1]):
            if not kept[site]:
                continue
            index = offsets[block] + site
            p0, p1, p2 = (evidence[sample, index, 0], evidence[sample, index, 1], evidence[sample, index, 2])
            total = p0 + p1 + p2
            if total <= 0.0 or (p0 == p1 and p1 == p2):
                continue
            p0, p1, p2 = (p0 / total, p1 / total, p2 / total)
            if full[block]:
                logs[0] = max(math.log(p0 * 0.99 + 0.01 / 3.0), -2.0) - center
                logs[1] = max(math.log(p1 * 0.99 + 0.01 / 3.0), -2.0) - center
                logs[2] = max(math.log(p2 * 0.99 + 0.01 / 3.0), -2.0) - center
            else:
                for code in range(7):
                    value = 0.99 * partial_emissions.predictive_likelihood(p0, p1, p2, code) + 0.01 / 3.0
                    logs[code] = max(math.log(value), -2.0) - center
            for first in range(len(local)):
                for second in range(first, len(local)):
                    if full[block]:
                        code = local[first, site] + local[second, site]
                    else:
                        code = partial_emissions.pair_code(
                            local[first, site],
                            local[second, site],
                            rows[first, site] == rows[second, site]
                        )
                    value = logs[code]
                    bin_index = site // bin_size
                    out[sample, first, second, bin_index] += value
                    if first != second:
                        out[sample, second, first, bin_index] += value


def build_models(workspace):
    alleles = List()
    source = List()
    keeps = List()
    outputs = List()
    full = []
    keys = []
    for block in workspace.batch:
        panel = observations.founder_inference_panel_from_block_result(block)
        flags = np.ones(len(block.positions), bool) if block.keep_flags is None else np.asarray(block.keep_flags, bool)
        complete = bool(np.all(panel.called[:, flags]))
        full.append(complete)
        usable = np.ascontiguousarray(flags & np.any(panel.called, axis=0))
        keeps.append(usable)
        alleles.append(np.ascontiguousarray(np.where(panel.called, panel.q, -1), np.int8))
        source.append(np.empty((0, 0), np.int32) if complete else partial_emissions.source_row_ids(block))
        bins = (len(block.positions) + workspace.bin_size - 1) // workspace.bin_size
        outputs.append(
            np.zeros((len(workspace.evidence), len(panel.keys), len(panel.keys), bins), np.float64)
        )
        keys.append(list(panel.keys))
    fill(
        workspace.evidence,
        workspace.offsets,
        alleles,
        source,
        keeps,
        np.asarray(full, bool),
        workspace.bin_size,
        outputs
    )
    return PreparedModels(
        [dict(hap_keys=k, bin_emissions=e, n_bins=e.shape[3], key_to_local_idx={key: i for i, key in enumerate(k)}) for k, e in zip(keys, outputs)]
    )
