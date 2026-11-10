"""Truth-evaluation metric definitions and execution estimates.

Count thresholds define the reported metric; tiles and byte allowances only
partition its computation. None of these values feeds inference.
"""

LONG_SWITCH_MIN_RUN: int = 20  # Scored sites; minimum orientation run separating two long-switch errors.
SEGMENT_SWITCH_COST: float = 20.0  # Allele-error units charged per founder-label switch and per split row.
SITE_TILE: int = 65_536  # Sites per tile; bounds temporary storage for integer error counting.
IDENTITY_WINDOW_MARKERS: int = 100  # Truth markers per window for copy-stratified identity evaluation.
LABEL_MIN_CALLS: int = 16  # Called sites required to assign a reconstructed row a truth-founder label.
LABEL_MARGIN: int = 2  # Error-count margin required between the best and next truth-founder labels.
LABEL_ERROR_FRACTION: float = 0.05  # Fraction of called sites allowed as errors for a window label.
LABEL_MIN_ERRORS: int = 1  # Errors; minimum allowance in the window-label error cutoff.
COVERAGE_MIN_DISTANCE: int = 3  # Marker mismatches from other founders required for an eligible coverage window.
COVERAGE_MAX_GAP: int = 3  # Windows; maximum gap bridged by continuity evaluation.
TASK_ALLOWANCE_BYTES: int = 256 << 20  # Bytes of fixed allowance in the evaluation task working-set estimate.
PHASE_METRIC: str = "released_phase_v3"  # Published identifier of the released-phase metric definition.

__all__ = [
    "LONG_SWITCH_MIN_RUN",
    "SEGMENT_SWITCH_COST",
    "SITE_TILE",
    "IDENTITY_WINDOW_MARKERS",
    "LABEL_MIN_CALLS",
    "LABEL_MARGIN",
    "LABEL_ERROR_FRACTION",
    "LABEL_MIN_ERRORS",
    "COVERAGE_MIN_DISTANCE",
    "COVERAGE_MAX_GAP",
    "TASK_ALLOWANCE_BYTES",
    "PHASE_METRIC",
]
