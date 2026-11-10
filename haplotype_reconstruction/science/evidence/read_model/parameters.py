"""Read-model fit bounds, convergence scales and calibration controls."""

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
    binomial model (``ReadModelParameters.fixed_error``) applies.
    """

    fixed_error: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "Read error probability of the fixed binomial fallback (probability); applies "
                "when calibration is disabled or lacks a finite shared held-out score."
            ),
        },
    )
    enabled: bool = field(
        default=True,
        metadata={
            "provenance": (
                "Enable held-out read-model calibration (boolean); otherwise use the fixed "
                "binomial model."
            ),
        },
    )
    # Selection: a more complex candidate is accepted when its held-out log-likelihood gain is at least
    # selection_z standard errors. The error comes from the gains of the held-out stretches (folds 2/3 of each
    # four-window cycle), since sites linked through ancestry or sharing reads are not independent observations.
    selection_z: float = field(
        default=2.0,
        metadata={
            "provenance": (
                "Required held-out gain in standard-error units (dimensionless); favours a more "
                "complex model only with evidence across held-out stretches."
            ),
        },
    )
    # With fewer held-out stretches than this the error cannot be estimated and the shared model is kept.
    selection_minimum_blocks: int = field(
        default=4,
        metadata={
            "provenance": (
                "Minimum held-out stretches (count) for estimating gain uncertainty; fewer "
                "stretches retain the shared model."
            ),
        },
    )
    maximum_fit_depth: int = field(
        default=48,
        metadata={
            "provenance": (
                "Largest cell depth included in calibration (reads); bounds the fitted histograms "
                "without filtering deeper cells from likelihood application."
            ),
        },
    )
    fold_window_bp: int = field(
        default=1_000_000,
        metadata={
            "provenance": (
                "Physical window width for the four-fold calibration cycle (base pairs); keeps "
                "linked sites together in training or held-out stretches."
            ),
        },
    )
    em_starts: tuple[tuple[float, float], ...] = field(
        default=((0.02, 0.5), (0.01, 0.35), (0.08, 0.65)),
        metadata={
            "provenance": (
                "Initial shared-model pairs of error probability and heterozygote ALT fraction "
                "(probabilities); supplies distinct EM starts for stationary-fit comparison."
            ),
        },
    )
    em_maximum_iterations: int = field(
        default=2000,
        metadata={
            "provenance": (
                "Maximum EM iterations per shared-model start (iterations); bounds calibration "
                "work."
            ),
        },
    )
    em_tolerance_per_observation: float = field(
        default=1e-10,
        metadata={
            "provenance": (
                "EM log-likelihood gain stopping threshold (nats per observation); scales "
                "convergence to the fitted read count."
            ),
        },
    )
    # Start of the joint fit's shared heterozygote concentration (kappa).
    initial_heterozygote_concentration: float = field(
        default=40.0,
        metadata={
            "provenance": (
                "Starting beta-binomial heterozygote concentration (dimensionless); initializes "
                "the joint fit."
            ),
        },
    )
    # Likelihood application: cells up to this depth come from a per-sample lookup table,
    # deeper cells are evaluated directly (a performance setting, not a depth filter).
    lookup_depth: int = field(
        default=64,
        metadata={
            "provenance": (
                "Largest depth tabulated for likelihood application (reads); deeper cells are "
                "evaluated directly, so this is not a depth filter."
            ),
        },
    )


# Box for the shared/sample error logit (dimensionless); keeps error below one half in the transformed model.
FIT_ERROR_LOGIT_BOUNDS = (-16.0, -1e-08)

# Box for heterozygote ALT-balance logits (dimensionless).
FIT_BALANCE_LOGIT_BOUNDS = (-12.0, 12.0)

# Box for log heterozygote beta-binomial concentration (dimensionless).
FIT_LOG_CONCENTRATION_BOUNDS = (-8.0, 16.0)

# Box for genotype-mixture logits (dimensionless).
FIT_MIXING_BOUNDS = (-25.0, 25.0)

# Per-sample L-BFGS-B iteration and convergence controls (iterations and dimensionless tolerances).
FIT_SAMPLE_FIT_OPTIONS = dict(maxiter=400, ftol=1e-11, gtol=1e-07)

# Joint L-BFGS-B iteration, convergence and correction-history controls (iterations and dimensionless
# tolerances).
FIT_JOINT_FIT_OPTIONS = dict(maxiter=1200, ftol=1e-12, gtol=1e-09, maxcor=20)

# Absolute finite-difference step of the observed Hessian (transformed-coordinate units).
FIT_HESSIAN_STEP = 0.0001

# Minimum observed-Hessian eigenvalue (objective curvature); requires resolved positive curvature before empirical-Bayes fitting.
FIT_HESSIAN_EIGENVALUE_FLOOR = 1e-08

# Minimum identifiable sample effects (samples) for an empirical-Bayes hyperfit.
FIT_HYPERFIT_MINIMUM_SAMPLES = 3

# Maximum Newton polish steps (iterations). Polish resolves stationary points beyond L-BFGS-B stopping: a
# 465-parameter fit had logit offsets of 8e-3, while three Newton steps agreed to 1e-14.
FIT_POLISH_ITERATIONS = 40

# Relative central-difference step for analytic-gradient Hessian columns (dimensionless).
FIT_POLISH_STEP = 1e-05

# Maximum relative coordinate motion for Newton convergence (fraction of max(1, abs(x))).
FIT_POLISH_TOLERANCE = 1e-13

# Small-step scale for the Newton stagnation test (relative coordinate motion); flat directions can stop
# before full coordinate precision.
FIT_POLISH_STAGNATION = 1e-08

# Relative objective-rounding allowance (dimensionless); predicted larger improvements require a decreasing
# backtracked step.
FIT_POLISH_DECREASE = 1e-12

# Minimum Hessian diagonal relative to the largest free diagonal (fraction); holds coordinates without
# resolved curvature.
FIT_POLISH_CURVATURE_FLOOR = 1e-12

# Shared EM optimum band (nats per observation); groups near-identical starts for diagnostics after stationary
# polishing.
FIT_EM_START_BAND_PER_OBSERVATION = 1e-07

# Training log-likelihood tie band (nats) for distinct stationary models; also used by held-out selection.
FIT_LIKELIHOOD_TIE_NATS = 1.0

# Relative profile-value band (dimensionless); keeps zero hypervariance when the boundary and interior optimum
# tie.
FIT_HYPERFIT_BOUNDARY_BAND = 1e-10


# Training windows per four-window calibration cycle (windows); the remaining two windows are held out.
CALIBRATION_TRAIN_FOLDS = 2

# Distance from selection_z in standard-error units counted as a near-tie in the selection diagnostics.
CALIBRATION_SELECTION_TIE_Z = 0.5


# Allowed EM log-likelihood decrease (nats per observation); detects a non-monotone fit beyond numerical
# noise.
EM_DECREASE_PER_OBSERVATION = 1e-07

# Positive probability/weight floor (dimensionless); keeps logarithms and denominators finite.
POSITIVE_WEIGHT_FLOOR = 1e-300

# Initial profile-root bracket half-width (transformed-coordinate units).
PROFILE_BRACKET_INITIAL_WIDTH = 0.0001

# Absolute profile-root location tolerance (coordinate units); relative rounding tolerance governs ordinary
# roots.
PROFILE_ROOT_XTOL = 1e-300

__all__ = ["ReadModelParameters"]


# Maximum Newton step halvings (trials); bounds objective backtracking.
POLISH_BACKTRACK_TRIALS = 13
POLISH_STAGNATION_RATIO = 4.0
# Multiplicative profile-root bracket expansion (dimensionless).
PROFILE_BRACKET_GROWTH = 4.0

# Multiplier of sample-effect variance for the variance search ceiling (dimensionless).
HYPERFIT_VARIANCE_CEILING_FACTOR = 4
# Maximum bracket expansions (trials).
PROFILE_BRACKET_TRIALS = 40
# Maximum profile-root solver iterations.
PROFILE_ROOT_ITERATIONS = 200
# Relative profile-root tolerance in float64 epsilon units.
PROFILE_ROOT_EPS_FACTOR = 4
