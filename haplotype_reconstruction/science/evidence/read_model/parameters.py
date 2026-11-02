"""Declare unchanged evidence parameters with their 3.5 provenance (EC01-24)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReadModelParameters:
    """Held-out calibration of the observed-AD read model (``evidence/read_model``).

    Fitting uses every kept site, cells of depth 1..``maximum_fit_depth``, and four physical folds of
    ``fold_window_bp`` windows (folds 0/1 fit, folds 2/3 are held out). The candidates are nested:
    shared binomial (EM from ``em_starts``), pooled sample binomial and sample beta-binomial
    heterozygotes (joint concentration); homozygotes are binomial. A more complex candidate is
    selected when its held-out gain is at least ``selection_z`` standard errors over the held-out
    stretches. When disabled, or when the shared model has no finite held-out score, the fixed
    binomial model (``EvidenceParameters.fixed_error``) applies.
    """

    fixed_error: float = field(
        default=0.02, metadata={"provenance": "B/evidence/config.py: fixed_error; P0 evidence-core.parameters.csv"}
    )
    enabled: bool = field(
        default=True, metadata={"provenance": "B/evidence/config.py: enabled; P0 evidence-core.parameters.csv"}
    )
    # Selection: a more complex candidate is accepted when its held-out log-likelihood gain is at least
    # selection_z standard errors. The error comes from the gains of the held-out stretches (folds 2/3 of each
    # four-window cycle), since sites linked through ancestry or sharing reads are not independent observations.
    selection_z: float = field(
        default=2.0, metadata={"provenance": "B/evidence/config.py: selection_z; P0 evidence-core.parameters.csv"}
    )
    # With fewer held-out stretches than this the error cannot be estimated and the shared model is kept.
    selection_minimum_blocks: int = field(
        default=4,
        metadata={"provenance": "B/evidence/config.py: selection_minimum_blocks; P0 evidence-core.parameters.csv"},
    )
    maximum_fit_depth: int = field(
        default=48, metadata={"provenance": "B/evidence/config.py: maximum_fit_depth; P0 evidence-core.parameters.csv"}
    )
    fold_window_bp: int = field(
        default=1_000_000,
        metadata={"provenance": "B/evidence/config.py: fold_window_bp; P0 evidence-core.parameters.csv"},
    )
    em_starts: tuple[tuple[float, float], ...] = field(
        default=((0.02, 0.5), (0.01, 0.35), (0.08, 0.65)),
        metadata={"provenance": "B/evidence/config.py: em_starts; P0 evidence-core.parameters.csv"},
    )
    em_maximum_iterations: int = field(
        default=2000,
        metadata={"provenance": "B/evidence/config.py: em_maximum_iterations; P0 evidence-core.parameters.csv"},
    )
    em_tolerance_per_observation: float = field(
        default=1e-10,
        metadata={"provenance": "B/evidence/config.py: em_tolerance_per_observation; P0 evidence-core.parameters.csv"},
    )
    # Start of the joint fit's shared heterozygote concentration (kappa).
    initial_heterozygote_concentration: float = field(
        default=40.0,
        metadata={
            "provenance": "B/evidence/config.py: initial_heterozygote_concentration; P0 evidence-core.parameters.csv"
        },
    )
    # Likelihood application: cells up to this depth come from a per-sample lookup table,
    # deeper cells are evaluated directly (a performance setting, not a depth filter).
    lookup_depth: int = field(
        default=64, metadata={"provenance": "B/evidence/config.py: lookup_depth; P0 evidence-core.parameters.csv"}
    )


# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_ERROR_LOGIT_BOUNDS = (-16.0, -1e-08)

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_BALANCE_LOGIT_BOUNDS = (-12.0, 12.0)

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_LOG_CONCENTRATION_BOUNDS = (-8.0, 16.0)

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_MIXING_BOUNDS = (-25.0, 25.0)

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_SAMPLE_FIT_OPTIONS = dict(maxiter=400, ftol=1e-11, gtol=1e-07)

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_JOINT_FIT_OPTIONS = dict(maxiter=1200, ftol=1e-12, gtol=1e-09, maxcor=20)

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_HESSIAN_STEP = 0.0001

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_HESSIAN_EIGENVALUE_FLOOR = 1e-08

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_HYPERFIT_MINIMUM_SAMPLES = 3

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_POLISH_ITERATIONS = 40

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_POLISH_STEP = 1e-05

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_POLISH_TOLERANCE = 1e-13

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_POLISH_STAGNATION = 1e-08

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_POLISH_DECREASE = 1e-12

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_POLISH_CURVATURE_FLOOR = 1e-12

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_EM_START_BAND_PER_OBSERVATION = 1e-07

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_LIKELIHOOD_TIE_NATS = 1.0

# Provenance: B/evidence/read_model/fit.py; P0 evidence-core.parameters.csv.
FIT_HYPERFIT_BOUNDARY_BAND = 1e-10


# Provenance: B/evidence/read_model/calibrate.py; P0 evidence-core.parameters.csv.
CALIBRATION_TRAIN_FOLDS = 2

# Provenance: B/evidence/read_model/calibrate.py; P0 evidence-core.parameters.csv.
CALIBRATION_SELECTION_TIE_Z = 0.5


# Provenance: B/evidence/read_model; unchanged inline numerical setting (EC04-22).
EM_DECREASE_PER_OBSERVATION = 1e-07

# Provenance: B/evidence/read_model; unchanged inline numerical setting (EC04-22).
POSITIVE_WEIGHT_FLOOR = 1e-300

# Provenance: B/evidence/read_model; unchanged inline numerical setting (EC04-22).
PROFILE_BRACKET_INITIAL_WIDTH = 0.0001

# Provenance: B/evidence/read_model; unchanged inline numerical setting (EC04-22).
PROFILE_ROOT_XTOL = 1e-300

__all__ = ["ReadModelParameters"]


# Provenance: B/evidence/read_model/fit.py:202-268, unchanged polish backtracking and stagnation.
POLISH_BACKTRACK_TRIALS = 13
POLISH_STAGNATION_RATIO = 4.0
# Provenance: B/evidence/read_model/fit.py: profile-root bracket widening.
PROFILE_BRACKET_GROWTH = 4.0

# Provenance: B/evidence/read_model/fit.py:_normal_hyperfit and _profile_root, exact native budgets.
HYPERFIT_VARIANCE_CEILING_FACTOR = 4
PROFILE_BRACKET_TRIALS = 40
PROFILE_ROOT_ITERATIONS = 200
PROFILE_ROOT_EPS_FACTOR = 4
