"""The distance-aware sum-product linker: dense haploid founder transitions between the blocks of one batch.

Modules (import them directly; this package imports nothing, so the join can import ``emissions`` alone):
    em         ``fit_transitions(inputs, genetic_map, *, max_gap, config, budget) -> Transitions`` (the EM)
    scan       within-block normal/error-tract scans (lane-vectorised, log-domain fallback), ``BlockScans``
    kernels    pair-prior propagation, posterior edge counts, log matmul, M-step regularisation
    emissions  predictive pair classes and log emissions (shared with the join's binned emissions)
    proxies    linking-proxy sites of a block (also used by the batch partition)
"""
