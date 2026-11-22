"""Parameters for material units and regime cut-offs."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.errors import SettingsError


@dataclass(frozen=True)
class MaterialParameters:
    """The genome's material: equal-material units of the working map and the regime cut-offs (4.18, DESIGN_417).

    Every run resamples consecutive units of ``unit_morgans`` of the genome-ordered working map (configured contig
    order, then painting components in map order; the per-bin Haldane coordinates of the selector switch
    probabilities, the configured map whether or not it is declared measured, gaps between components contributing
    nothing). A final piece shorter than ``tail_morgans`` joins the unit before it, so a run holds round(U) units,
    where U is the working Morgans of its informative components. The regime reads U alone; no count of chromosomes,
    contigs or components enters it. Distances are Morgans of the configured working map: a wrong configured rate
    scales U, and with it every cut below, by the same factor.
    """

    unit_morgans: float = field(
        default=1.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "DESIGN_417 D1; one unit per working Morgan, about one meiosis's correlation length (Haldane "
                "exp(-2d) = 0.14 at 1 M); adjacent 1-M units correlate 0.12-0.22 for relatives on one chromosome, "
                "lag 2 at most 0.04 (CALIB2 section 5)."
            ),
        },
    )
    tail_morgans: float = field(
        default=0.5,
        metadata={
            "units": "Morgan",
            "provenance": (
                "DESIGN_4154 A2 / DESIGN_417 D1; a final piece shorter than half a unit joins the unit before it."
            ),
        },
    )
    switch_clip: float = field(
        default=0.499999,
        metadata={
            "units": "probability",
            "provenance": "DESIGN_4154 A2; switch probabilities are clipped below 1/2 before the Haldane distance.",
        },
    )
    assessable_morgans: float = field(
        default=5.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "MCAL section 4.1 (60 simulated genomes, configured map = generating map, so U in true Morgans; "
                "single chromosomes 0.5-15 M; pre-registered one-sided exact tests, clustered by genome); below 5 "
                "Morgans nothing is assessed: the low-material rule released non-parents in mixed-family designs "
                "at 3.5-5 M (12 of 971 against 0 of 600 at 5-7 M) and in small families; from 5 M every design "
                "passes the tested criterion (8 M on its literal point reading). V5 (64 end-to-end cells, "
                "g4c5_12): no hold criterion fires below 12 M. Replaces DESIGN_417 rev 3's 3.5 (MCAL: 3.5-5 M in excess)."
            ),
        },
    )
    standard_morgans: float = field(
        default=12.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "MCAL section 4.2; the released Tier B thresholds apply from 12 working Morgans: below it they "
                "released more non-parents than their own reference (multi-chromosome genomes of >= 15 M) in "
                "small families (11-12 M: 10 of 486 against 10 of 1,379, p = 0.035) and in parent-free sib-rich "
                "designs (10-11 M: 10 releases per 3,808 parentless children against 2 per 6,720, p = 0.008); "
                "from 12 M no excess in any design (10 M on the cumulative test; per-design power 0.1-0.4 near "
                "the cut, so an empirical calibration, not demonstrated equivalence). V5: rev 3's 9.5 released "
                "4.46 edges per 100 parentless children on sib-rich cells read on a 2.5x long map (hold H2); "
                "12 returns the 9.5-12 M band to the low-material rule, as 4.16.1 (0.11 per 100), and passes the "
                "direction positive control (SCORE_g4c5_12, CONTROL_SCORE_g4c5_12)."
            ),
        },
    )
    tier_a_morgans: float = field(
        default=12.0,
        metadata={
            "units": "Morgan",
            "provenance": (
                "MCAL section 4.3; no Tier A non-parent in designs with sequenced parents from 8 M (74,337 Tier A "
                "edges at 8-15+ M), but Tier A is assessed only with the released Tier B thresholds, so its "
                "effective minimum is standard_morgans (12 M). Parent-free sib-rich designs release Tier A "
                "non-parents at 12-13, 13-15 and >= 15 M (2, 2, 1; their >= 15-M reference holds 1), so no "
                "higher cut removes them (documented limitation; V5 g4c5_12: 1 such edge in 448 parentless "
                "children at 13.7 M)."
            ),
        },
    )
    low_bootstrap: float = field(
        default=0.95,
        metadata={
            "units": "fraction",
            "provenance": (
                "DESIGN_4154 A4; Tier B configuration and parent bootstrap cut-off of the low-material band "
                "(assessable_morgans <= U < standard_morgans), where adjacent units of one chromosome are dependent."
            ),
        },
    )
    low_leave_one_out: float = field(
        default=1.0,
        metadata={
            "units": "fraction",
            "provenance": "DESIGN_4154 A4; every leave-one-unit-out fit must agree in the low-material band.",
        },
    )

    def __post_init__(self) -> None:
        if not 0 < self.tail_morgans <= self.unit_morgans:
            raise SettingsError("pedigree material tail must lie in (0, unit]")
        if not 0 < self.switch_clip < 0.5:
            raise SettingsError("pedigree material switch clip must lie in (0, 0.5)")
        if not 0 < self.assessable_morgans <= self.standard_morgans <= self.tier_a_morgans:
            raise SettingsError("pedigree material cuts must increase: assessable <= standard <= Tier A")
