"""Family-phase model, solver and release constants (release 3.0 rebuild).

Only ``protected_intervals`` is user-visible (``[family_phase] protected_intervals``, a breakpoint TSV read by
``family_phase.breakpoints.read_breakpoint_table``). Recombination rates come from the chromosome's genetic map; the
founder label process comes from the painting's estimated copying model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyPhaseConfig:
    """Pedigree founder-mosaic phase of one chromosome (docs/methods/family_phase.md).

    Model:
      not_copy_rate_per_bp: rate per bp at which a homologue linked to a parent enters the not-a-copy state (a
        painting or panel error, conversion or mutation, or a false edge); its label is then free.
      not_copy_mean_bp: mean length of a not-a-copy run.
    Solver (exact block coordinate ascent):
      max_sweeps: work cap on block sweeps. Every applied block raises the objective by more than
        ``gain_quantum`` nats, so the ascent terminates without it; a component that reaches the cap keeps its
        (valid) current state and is recorded as not settled.
      gain_quantum: smallest objective gain (nats) for which a block path is applied.
      tie_quantum: Viterbi scores within this many nats are tied; ties keep the current state, then the lowest
        index (the highest under the convention reversal, maths.ties.reverse_conventions).
    Identity junctions (bin edges where panel rows may have exchanged founder identity):
      junction_min_samples: distinct samples whose free or not-a-copy tracks must switch between the same two
        labels at one bin edge for the edge to be a junction candidate.
      junction_pin_nats: a carrier of both junction labels keeps its phase join across a candidate only when the
        parental homologue it copies there is pinned by the parent's own reads: on each flank the parent's pair beats
        every pair putting the other junction label on that homologue, whatever the partner, by at least this many
        nats.
      junction_pin_bins: the flank window of that test (bins, cut at the relative's own label changes).
    Release (release.py):
      site_deviation: mixture mass of the release's site model: given its label pair, a sample's ordered alleles at a
        site are the pair's panel alleles, or with this probability redrawn from a uniform genotype (a panel allele
        error, a label error shorter than a bin, a conversion or mutation; the genotype then changes with probability
        2/3 of it). Weighed by the sample's own reads at the site, reads that contradict the labels' genotype lower,
        withdraw or move the call. A working value chosen on the validation simulations (0.01 unphases 12-13% of
        phased joins; 3e-4 keeps more genotype errors), not an estimate of an error rate.
      genotype_posterior: a genotype is called where its posterior reaches this value (the painting's tau).
      phase_posterior: a heterozygous call is released phased where its ordered allele posterior reaches this
        value.
      set_posterior: a gauge sample's phase set continues while the orientation of its alleles, given the set's
        anchor, stays at or above this value at its heterozygous bins (the median over a bin's confidently
        heterozygous sites of max(P01, P10) / (P01 + P10)); uncertainty about which labels it carries does not
        break a set. Stricter than phase_posterior because a wrong continuation misphases the rest of the set,
        whereas a wrong site call misphases one site. A working value chosen on the validation simulations: at 0.99,
        near-ties decided only by the label-switch prior added released long switches in partial-genome designs,
        which 0.999 avoided. Not an estimate of an error rate, nor a calibrated confidence for a whole set; longer
        sets still carry an upstream identity error over more sites. It must lie below the orientation certainty of a fully determined
        heterozygous site under the site model, (1 - 5 site_deviation / 6) / (1 - 2 site_deviation / 3)
        (0.99983 at the default site_deviation), or every bin would end a set.
        Phase sets also break at junction candidates for samples carrying both labels unless a parental homologue
        pins them, and, for samples without a linked parent, at every painting tie boundary (release.py).
      protected_intervals: (contig, start, end) closed bp intervals across which meiosis chains reset (the switch
        probability is 1/2): reference-order breakpoints, and both ends of reversed segments.
    """

    not_copy_rate_per_bp: float = 1e-8
    not_copy_mean_bp: float = 5e4
    max_sweeps: int = 30
    gain_quantum: float = 1e-6
    tie_quantum: float = 1e-9
    junction_min_samples: int = 2
    junction_pin_nats: float = 10.0
    junction_pin_bins: int = 10
    site_deviation: float = 1e-3
    genotype_posterior: float = 0.9
    phase_posterior: float = 0.99
    set_posterior: float = 0.999
    protected_intervals: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self):
        if not 0 < self.not_copy_rate_per_bp < 1e-3 or not self.not_copy_mean_bp > 0:
            raise ValueError("family_phase not-a-copy rates must be positive (rate per bp below 1e-3)")
        for name in ("max_sweeps", "junction_min_samples", "junction_pin_bins"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f"family_phase.{name} must be a positive integer")
        for name in ("genotype_posterior", "phase_posterior", "set_posterior"):
            if not 0.5 < getattr(self, name) < 1:
                raise ValueError(f"family_phase.{name} must lie strictly between 0.5 and 1")
        if self.set_posterior < self.phase_posterior:
            raise ValueError("family_phase.set_posterior must be at least phase_posterior")
        ceiling = (1 - 5 * self.site_deviation / 6) / (1 - 2 * self.site_deviation / 3)
        if not self.set_posterior < ceiling:
            raise ValueError(f"family_phase.set_posterior must lie below {ceiling:.6f}, the orientation certainty of a "
                             f"fully determined heterozygous site at site_deviation {self.site_deviation}")
        if not 0 < self.site_deviation < 0.5:
            raise ValueError("family_phase.site_deviation must lie strictly between 0 and 0.5")
        if not self.junction_pin_nats > 0:
            raise ValueError("family_phase.junction_pin_nats must be positive")
        if not self.gain_quantum > 0 or not self.tie_quantum >= 0:
            raise ValueError("family_phase quanta must be positive")

    def protected_for(self, contig):
        """Closed bp intervals of ``contig`` where meiosis chains reset (reference-order breakpoints)."""
        return [(int(start), int(end)) for name, start, end in self.protected_intervals if name == contig]
