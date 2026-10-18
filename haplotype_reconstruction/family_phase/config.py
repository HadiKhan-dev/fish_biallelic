"""Family-phase model and schedule constants (defaults = the production values of base 92b3daf).

Only ``protected_intervals`` is user-visible (``[family_phase] protected_intervals``, a breakpoint TSV read by
``family_phase.breakpoints.read_breakpoint_table``). Recombination rates come from the chromosome's genetic map.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyPhaseConfig:
    """Pedigree-conditioned phase of one chromosome.

    Family model (belief propagation over ordered genotypes, four states per sample and site):
      transmission_error: probability that a copy factor transmits the other parental allele.
      scaffold_phase_error: probability that a root's scaffold orientation is wrong at a heterozygous site.
      phase_switch_probability: probability of a frame switch at a painting-bin edge inside a component (the
        family phase chain and the polish's coarse bins); component edges switch with probability 1/2 and the
        family chain never switches phase inside a painting bin (the polish's refinement: polish_window_sites).
    Belief-propagation schedule:
      damping: weight of the new message (probability space); tolerance: largest message change that counts
        as converged; max_iterations: iterations run before the phase is released when not converged earlier
        (base released after 20 initial iterations and 5 stability checks, which can never fail under the
        scaffold polish, see tickets/T8/questions.md Q1); gauge_move_interval: iterations between joint
        root/children gauge moves; minimum_gauge_support: children whose selector changes at a boundary needed
        for a gauge move (one child cannot distinguish a root frame change from its own crossover).
    Root-frame repair (frames.py), on the final family state:
      root_frame_window_bp: flank reach W around a bin edge; root_frame_location_bp: location tolerance (net
      votes ignore +- this around the edge); root_frame_minimum_children: children needed to move a frame (also
      the junction guard's parent size); root_frame_informative_nats: per-child vote threshold tau;
      root_frame_max_rounds: repair rounds; protected_intervals: (contig, start, end) closed bp intervals where
      no frame repair is made (reference-order breakpoints).
    Genotype-preserving polish (polish.py), from the (repaired) scaffold:
      polish_copy_error: copy-consistency error of called scaffold alleles; polish_window_sites: fine phase
      boundaries allowed within this many sites of a current phase or selector change, inside the bin to the
      right of a bin edge (that edge's flip probability is split over them; the first painting bin of a
      chromosome or component has no such edge and stays rigid); polish_max_sweeps: colour sweeps before the
      polish must have converged;
      polish_minimum_gain: smallest accepted score gain (nats); polish_minimum_biological_gain: smallest
      copy-consistency gain for an interval that changes an observable phase.
    """

    transmission_error: float = 0.001
    scaffold_phase_error: float = 0.01
    phase_switch_probability: float = 0.005
    damping: float = 0.5
    tolerance: float = 1e-4
    max_iterations: int = 25
    gauge_move_interval: int = 5
    minimum_gauge_support: int = 2
    root_frame_window_bp: float = 5e5
    root_frame_location_bp: float = 2.5e5
    root_frame_minimum_children: int = 3
    root_frame_informative_nats: float = 2.0
    root_frame_max_rounds: int = 8
    protected_intervals: tuple[tuple[str, int, int], ...] = ()
    polish_copy_error: float = 0.01
    polish_window_sites: int = 100
    polish_max_sweeps: int = 6
    polish_minimum_gain: float = 1e-6
    polish_minimum_biological_gain: float = 0.0

    def __post_init__(self):
        for name in ("transmission_error", "scaffold_phase_error", "phase_switch_probability",
                     "polish_copy_error"):
            if not 0 < getattr(self, name) < 0.5:
                raise ValueError(f"family_phase.{name} must lie strictly between 0 and 0.5")
        if not 0 < self.damping <= 1:
            raise ValueError("family_phase.damping must lie in (0, 1]")
        if not self.tolerance > 0:
            raise ValueError("family_phase.tolerance must be positive")
        for name in ("max_iterations", "gauge_move_interval", "minimum_gauge_support",
                     "root_frame_minimum_children", "root_frame_max_rounds", "polish_max_sweeps"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"family_phase.{name} must be a positive integer")
        if not 0 <= self.root_frame_location_bp < self.root_frame_window_bp:
            raise ValueError("family_phase.root_frame_location_bp must lie in [0, root_frame_window_bp)")
        if not self.root_frame_informative_nats > 0 or self.polish_window_sites < 0:
            raise ValueError("invalid family_phase root-frame threshold or polish window")

    def protected_for(self, contig):
        """Closed bp intervals of ``contig`` where no root frame is repaired, as an int64 (n, 2) list."""
        return [(int(start), int(end)) for name, start, end in self.protected_intervals if name == contig]
