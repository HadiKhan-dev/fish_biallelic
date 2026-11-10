"""Nearest-shadow family policy shared by join and refinement.

Callers supply their exact scorer and bootstrap stream. The total test is a
necessary acceptance condition, so a rejected total needs only its conventional
shadow. Computational overflow never supplies biological count evidence.
Method: docs/methods/assembly.md#aj33.
"""

from __future__ import annotations

import itertools
import math

from .criterion import passes

SHADOW_PRIOR = "smaller_k+nearest_shadow"


def shadow_family_decision(
    choices, larger, smaller, cost, config, evaluate, *, reverse, withhold, decision_options=None
):
    """Evaluate an admitted family or its conventional incumbent; reduce one rule.

    ``evaluate(paths, multiple)`` returns decisions in Cartesian order. Withhold
    preserves a decision's bootstrap ambiguity, including the join's lazy band.
    Score in canonical Cartesian order to align prefix-shared member vectors.
    Reversal changes the single incumbent and the reduction traversal: exact
    (verdict, margin, sequence LOSO margin) ties choose the last canonical member.
    Family membership and the ordering of diagnostic arrays stay unchanged.
    """
    combinations = math.prod(map(len, choices))
    total_passes = passes(larger - smaller, cost, len(larger))
    admitted = combinations <= int(config.shadow_combinations)
    enumerate_family = admitted and total_passes
    incumbent = tuple(options[-1] if reverse else options[0] for options in choices)
    paths = itertools.product(*choices) if enumerate_family else [incumbent]
    decisions = evaluate(paths, enumerate_family and combinations > 1)
    if decision_options is not None:
        decision_options.extend(decisions)
    unresolved = total_passes and len({d.accept_larger for d in decisions}) > 1
    # The caller's score table follows canonical paths; reverse only the reduction.
    # A strictly smaller key always wins; the convention decides exact key ties.
    decision = min(
        reversed(decisions) if reverse else decisions,
        key=lambda d: (d.accept_larger, d.margin, d.sequence.loso_margin),
    )
    if unresolved:
        decision = withhold(decision)
    details = dict(
        nearest_choices=choices,
        combinations=combinations,
        evaluated=len(decisions),
        complete=len(decisions) == combinations,
        resolved_by_total=not total_passes,
        unresolved=unresolved,
        count_unverified=total_passes and not admitted,
        incumbent_shadow=incumbent,
        verdicts=[d.accept_larger for d in decisions],
        sequence_margins=[d.sequence.margin for d in decisions],
        sequence_loso_margins=[d.sequence.loso_margin for d in decisions],
    )
    return decision, details
