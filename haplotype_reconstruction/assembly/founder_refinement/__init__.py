"""Final founder-path refinement of the assembly (one entry: ``refine_level``).

Called by the assembly level loop after every level of the final L1-L4 assembly. Each phase
component's founders are re-chosen as paths of original prepared local rows to maximise the
full-site objective F (objective.py) and its founder count is decided between converged fits by
the shared count rule (count.py). Modules: stage (entry, scheduling, census), workspace
(component inputs, binned surrogate emissions), objective (F and its inline site steps), messages
(block messages), gains (exact move gains), guard (coarse-move guard), acceptance (compiled
Gauss-Seidel pass), proposals (window programmes), search (fixed-K fit), count (count pass),
records (checkpoint records and readers).
"""
from .config import FounderRefinementConfig
from .count import BestFReplacement, CountDecisionRecord
from .records import ComponentRefinement, LevelSummary
from .search import FitSummary, WeakJunction
from .stage import LevelInput, LevelResult, refine_level

__all__ = ["FounderRefinementConfig", "LevelInput", "LevelResult", "LevelSummary",
           "ComponentRefinement", "CountDecisionRecord", "BestFReplacement", "FitSummary",
           "WeakJunction", "refine_level"]
