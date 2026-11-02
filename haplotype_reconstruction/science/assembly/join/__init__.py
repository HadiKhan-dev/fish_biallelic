"""One joined batch: binned emissions, the join objective (one Viterbi recursion kernel), the
endpoint beam, tier-1 proposals, certification bounds, moves, the count step and the search.

Owns: the join science namespace.
Must not: read files, environment, live memory or clocks, or emit logs/warnings.
Method: WP_AS4 §6; AJ01–AJ29.
Identity: Public numerical entry points live in the named modules; orchestration belongs to AS6.
"""

__all__ = []
