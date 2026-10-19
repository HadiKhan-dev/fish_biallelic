"""Family-phase model and schedule constants (defaults = the production values of base 92b3daf).

Only ``protected_intervals`` is user-visible (``[family_phase] protected_intervals``, a breakpoint TSV read by
``family_phase.breakpoints.read_breakpoint_table``). Recombination rates come from the chromosome's genetic map.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyPhaseConfig:
    """Pedigree-conditioned phase of one chromosome.

    Family model (ordered genotypes, four states per sample and site; its posterior mode, the family MAP, is
    computed by map_solver.py):
      transmission_error: probability that a copy factor transmits the other parental allele.
      scaffold_phase_error: probability that a root's scaffold orientation is wrong at a heterozygous site.
      phase_switch_probability: probability of a frame switch at a painting-bin edge inside a component (the
        family phase chain and the polish's coarse bins); component edges switch with probability 1/2 and the
        family chain never switches phase inside a painting bin (the polish's refinement: polish_window_sites).
    Family MAP solver (map_solver.py, release 2.1):
      map_max_sweeps: safety cap on the sweeps of one ascent (release 2.2: both sweep tiers count; the two-schedule
        estimator runs up to three ascents). Every accepted step raises the posterior by more than a tie quantum, so
        the solver terminates by construction; reaching the cap is an error. Measured (19 Oct 2026): at most 9 sweeps
        (stress_n80), TroMau and std_n80 at most 4.
      map_long_moves: run the long compound moves (change pairs longer than map_solver.LONG_SPAN sites and the
        terminal segments, sweep tier 1) in the final ascent of the two-schedule MAP. Off by default (release 2.2):
        they cost 4-6x the family-phase time of 2.1 on dense data (one pass ~200 s on reallike6 chr13) and changed
        about 10 of 63 validated reference releases.
        2.0's belief-propagation settings (damping, tolerance, max_iterations, gauge_move_interval,
        minimum_gauge_support) are gone.
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
      polish_minimum_gain: the polish's tie quantum q (nats): an interval is accepted only when its exact gain
      exceeds q, inside q the current configuration is kept (the score reads called scaffold alleles only, so q only
      has to exceed rounding); polish_minimum_biological_gain: smallest copy-consistency gain for an interval that
      changes an observable phase, compared with a band of q.
    """

    transmission_error: float = 0.001
    scaffold_phase_error: float = 0.01
    phase_switch_probability: float = 0.005
    map_max_sweeps: int = 60
    map_long_moves: bool = False
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
        for name in ("map_max_sweeps", "root_frame_minimum_children", "root_frame_max_rounds",
                     "polish_max_sweeps"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"family_phase.{name} must be a positive integer")
        if not 0 <= self.root_frame_location_bp < self.root_frame_window_bp:
            raise ValueError("family_phase.root_frame_location_bp must lie in [0, root_frame_window_bp)")
        if not self.root_frame_informative_nats > 0 or self.polish_window_sites < 0:
            raise ValueError("invalid family_phase root-frame threshold or polish window")

    def protected_for(self, contig):
        """Closed bp intervals of ``contig`` where no root frame is repaired, as an int64 (n, 2) list."""
        return [(int(start), int(end)) for name, start, end in self.protected_intervals if name == contig]
