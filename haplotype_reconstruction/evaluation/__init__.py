"""Simulation evaluation against the known truth (`haplotypes evaluate RUN`): truth is read only here, after
inference, and nothing is fed back. evaluate.py: the command, stage vocabulary and tables; founders.py: founder-panel
metrics; metrics.py: phase metrics."""
from .evaluate import EVALUATION_STAGES, FOUNDER_STAGES, evaluate_command, evaluate_run

__all__ = ["EVALUATION_STAGES", "FOUNDER_STAGES", "evaluate_command", "evaluate_run"]
