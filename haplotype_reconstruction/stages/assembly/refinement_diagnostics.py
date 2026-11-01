"""Report scientific cap diagnostics after a component leaves the pure solver."""

import warnings

from haplotype_reconstruction.contracts.assembly import ComponentRefinement


def report_refinement(result: ComponentRefinement) -> None:
    """Expose nonconvergence and approximate search caps without changing decisions."""
    for fit in result.fits:
        prefix = f"founder refinement component{result.component}.{fit.label}"
        if not fit.converged:
            warnings.warn(f"{prefix} reached its sweep limit without converging", RuntimeWarning, stacklevel=2)
        for key, description in (
            ("exchange_cap", "exchange round cap"),
            ("round_cap", "content round cap"),
            ("escape_beam_binding", "beam-escape width"),
            ("long_beam_binding", "long-repair width"),
            ("beam_fallbacks", "window beam fallback"),
            ("beam_truncations", "window beam truncation"),
        ):
            count = fit.work_counters.get(key, 0)
            if count:
                warnings.warn(
                    f"{prefix}: {description} bound {count} times; exact acceptance retained",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if fit.frontier_fallbacks:
            warnings.warn(
                f"{prefix}: {fit.frontier_fallbacks} programmes kept the incumbent at the frontier cap",
                RuntimeWarning,
                stacklevel=2,
            )
    if result.count_limit_reached:
        warnings.warn(
            f"founder refinement component{result.component}: additions reached count limit {result.count_limit}",
            RuntimeWarning,
            stacklevel=2,
        )
