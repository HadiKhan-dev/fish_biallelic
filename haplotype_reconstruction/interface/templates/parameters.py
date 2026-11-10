"""Template path-score scales, physical windows and working-set estimates.

Score scales determine the generated templates; byte estimates describe execution.
"""

NODE_PENALTY: float = 10.0  # Path-score units; penalise a used row and similar rows to spread template paths.
EDGE_PENALTY: float = 10.0  # Path-score units; penalise each used internal junction.
GAP_SCORE: float = 1.0  # Junction score assigned to window pairs with no physical overlap.
EDGE_DIVISOR: float = 3.0  # Dimensionless divisor of junction rewards relative to path penalties.
PATH_TOLERANCE: float = 1e-9  # Absolute path-score tolerance for reconstructing a best path.
CANCELLATION_TOLERANCE: float = 1e-9  # Dimensionless denominator tolerance for cancellation of opposing information.
BISECTION_STEPS: int = 60  # Iterations for inverting signed information to an allele probability.
SIMILARITY_CUTOFF: float = 50.0  # Percent mismatches; junction similarity is zero above this cutoff.
WINDOW_BP: int = 100_000  # bp; physical span of each template-discovery window.
WINDOW_SHIFT_BP: int = 50_000  # bp; stride between overlapping template-discovery windows.
TEMPLATE_HAPLOTYPES: int = 6  # Maximum paths extracted for each contig template panel.
UNDECLARED_LENGTH_BP: int = 300_000_000  # bp; assumed contig length when the input header omits it.
WINDOW_EVIDENCE_FACTOR: int = 8  # Dimensionless multiplier of the reported dense evidence and site-axis bytes.
WINDOW_ALLOWANCE_BYTES: int = 1280 << 20  # Bytes of fixed allowance in the reported window working set.
CONTIG_CELL_BYTES: int = 32  # Bytes per sample-site in the declared contig working-set estimate.

__all__ = [
    "NODE_PENALTY",
    "EDGE_PENALTY",
    "GAP_SCORE",
    "EDGE_DIVISOR",
    "PATH_TOLERANCE",
    "CANCELLATION_TOLERANCE",
    "BISECTION_STEPS",
    "SIMILARITY_CUTOFF",
    "WINDOW_BP",
    "WINDOW_SHIFT_BP",
    "TEMPLATE_HAPLOTYPES",
    "UNDECLARED_LENGTH_BP",
    "WINDOW_EVIDENCE_FACTOR",
    "WINDOW_ALLOWANCE_BYTES",
    "CONTIG_CELL_BYTES",
]
