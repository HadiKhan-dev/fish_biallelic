"""Name carried scientific definitions and reported working-set estimates.

Each value cites its P0 unit and frozen 3.5 source. No values are inferred
from environment, worker count, clocks or filesystem state.
"""

NODE_PENALTY: float = 10.0  # OU21: B templates/linker.py:28; every similar row is penalized.
EDGE_PENALTY: float = 10.0  # OU21: B templates/linker.py:29; internal edges only.
GAP_SCORE: float = 1.0  # OU21: B templates/linker.py:30; no physical overlap.
EDGE_DIVISOR: float = 3.0  # OU21: B templates/linker.py:33; carried unused sample-usage term.
PATH_TOLERANCE: float = 1e-9  # OU21: B templates/linker.py:121; strict reconstruction test.
CANCELLATION_TOLERANCE: float = 1e-9  # OU21: B templates/linker.py:173; strict information cancellation.
BISECTION_STEPS: int = 60  # OU21: B templates/linker.py:185; fixed inversion budget.
SIMILARITY_CUTOFF: float = 50.0  # OU21: B templates/linker.py:69; mismatch percentage.
WINDOW_BP: int = 100_000  # OU20: B templates/builder.py:20; physical window length.
WINDOW_SHIFT_BP: int = 50_000  # OU20: B templates/builder.py:21; physical stride.
TEMPLATE_HAPLOTYPES: int = 6  # OU20: B templates/builder.py:22; path budget.
UNDECLARED_LENGTH_BP: int = 300_000_000  # OU20: B templates/builder.py:33; missing header length.
WINDOW_EVIDENCE_FACTOR: int = 8  # OU20: B templates/builder.py:27; working-set report.
WINDOW_ALLOWANCE_BYTES: int = 1280 << 20  # OU20: B templates/builder.py:28; working-set report.
CONTIG_CELL_BYTES: int = 32  # OU20: B templates/builder.py:31; working-set report.

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
