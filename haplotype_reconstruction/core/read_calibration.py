"""Observed-AD model selection shared by discovery and downstream inference.

Physical folds separate parameter fitting, nested model selection and predictive
diagnostics. The selected fit is reused across the contig; genotype mixtures are
fitting nuisances, never downstream priors. No truth or pedigree enters fitting.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os

import numpy as np

from .config import DEFAULT_READ_ERROR_PROBABILITY, ReadCalibrationConfig
from .environment import boolean_setting
from .genotypes import allele_depths_to_raw_genotype_likelihoods
from . import read_model, read_homozygote_model
from .read_likelihoods import raw_likelihoods
from .read_kernels import binomial_expectation
from .run_record import timed_stage

MODEL_VERSION = "heldout-homozygote-read-model-v4"


def enabled(value=None):
    """Resolve the CLI/TOML/environment setting; calibration is on by default."""
    value = os.environ.get("HAPLOTYPES_READ_CALIBRATION", "on") if value is None else value
    return boolean_setting(value, "read_calibration")


def identity():
    return dict(model=MODEL_VERSION, enabled=enabled(),
                config=asdict(ReadCalibrationConfig()),
                fallback_error=DEFAULT_READ_ERROR_PROBABILITY,
                selection="maximum_fold2_prediction_among_converged_nested_models",
                homozygote_model="shared_symmetric_beta_binomial_rho_nested_v1",
                homozygote_starts=[0., .02, .1],
                priors_applied_to_output=False)


def discovery_identity(backend, discovery_config):
    """Bind discovery and all its consumers to the same observation settings."""
    key = hashlib.sha256(json.dumps(identity(), sort_keys=True).encode()).hexdigest()[:16]
    return dict(backend=f"{backend}+read-model-{key}", config=asdict(discovery_config))


def histograms(reads, positions, config):
    indices = np.arange(0, len(positions), config.marker_stride)
    counts = np.asarray(reads[:, indices], np.int64)
    depth = counts.sum(axis=2)
    alt = counts[:, :, 1]
    folds = (positions[indices] // config.fold_window_bp) % 4
    width = config.maximum_fit_depth + 1
    result = np.zeros((4, len(counts), width * width), np.float64)
    admitted = (depth > 0) & (depth <= config.maximum_fit_depth)
    for fold in range(4):
        for sample in range(len(counts)):
            mask = admitted[sample] & (folds == fold)
            result[fold, sample] = np.bincount(
                depth[sample, mask] * width + alt[sample, mask], minlength=width * width)
    d, a = np.divmod(np.arange(width * width), width)
    valid = (d > 0) & (a <= d) & (result.sum(axis=(0, 1)) > 0)
    return result[:, :, valid], d[valid], a[valid], dict(
        calibration_stride=config.marker_stride,
        calibration_depth_cap=config.maximum_fit_depth,
        observations_used=int(admitted.sum()), high_depth_excluded=int((depth > config.maximum_fit_depth).sum()))


def fit(hist, depth, alt, initial, config, *, fixed=False):
    """EM for P(ALT count | depth, sample); genotype order is constrained."""
    e, q = initial
    mix = np.full((len(hist), 3), 1 / 3.)
    sample_weight = hist.sum(axis=1)
    scale = max(1., float(sample_weight.sum()))
    old = -np.inf
    last_gain = None
    for iteration in range(config.maximum_iterations):
        values, mass, moments = binomial_expectation(hist, depth, alt, e, q, mix)
        value = float(values.sum())  # Binomial coefficient cancels in comparisons.
        if not np.isfinite(value) or value < old - 1e-7 * scale:
            raise RuntimeError("read-model EM likelihood decreased or became non-finite")
        if iteration:
            last_gain = value - old
            if last_gain <= config.likelihood_tolerance_per_observation * scale:
                break
        old = value
        mix = np.divide(mass, sample_weight[:, None], out=np.full_like(mass, 1 / 3.),
                        where=sample_weight[:, None] > 0)
        if not fixed:
            totals = moments.sum(axis=0)
            e = float(totals[0] / max(totals[1], 1e-300))
            q = float(totals[2] / max(totals[3], 1e-300))
            if not (0 < e < q < 1 - e < 1):
                raise ValueError("EM left the ordered-component domain")
    else:
        raise RuntimeError("read-model EM did not converge")
    return dict(error=e, heterozygote_alt=q, mixture=mix, log_likelihood=value,
                iterations=iteration + 1, last_gain=last_gain)



def select_model(hist, depth, alt, *, threads, config):
    """Fit on folds 0/1, select on fold 2; fold 3 is diagnostic only.

    The simpler converged model wins an exact predictive tie. A failed complex
    fit cannot displace the shared model. The diagnostic fold never selects a
    model or changes its parameters.
    """
    report = dict(truth_used=False, priors_applied_to_output=False,
                  cross_fitted=False, train_folds=[0, 1], selection_fold=2,
                  diagnostic_fold=3,
                  fold_observations=[int(fold.sum()) for fold in hist])
    if any(fold.sum() == 0 for fold in hist):
        report["fallback_reason"] = "all four physical folds need calibration observations"
        return None, report
    training = hist[0] + hist[1]
    from .parallel import numba_thread_scope
    try:
        with numba_thread_scope(threads):
            shared = read_model.fit_shared(training, depth, alt, config)
    except (ValueError, RuntimeError) as error:
        report["fallback_reason"] = str(error)
        return None, report
    models = dict(binomial=shared)
    try:
        sample, joint = read_model.fit_joint(training, depth, alt, shared, threads)
        models.update(sample_binomial=sample, sample_joint=joint)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        report["complex_fit_failure"] = str(error)
    joint = models.get("sample_joint")
    if joint is not None and joint["success"]:
        try:
            null, extended, optimization = read_homozygote_model.fit_nested(
                training, depth, alt, joint, threads=threads)
            models.update(sample_joint_refit=null, homo_beta_binomial=extended)
            report["homozygote_optimization"] = optimization
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            report["homozygote_fit_failure"] = str(error)
    scores = {}
    with numba_thread_scope(threads):
        for name, model in models.items():
            scorer = (read_homozygote_model.score if "homo_rho" in model else read_model.score)
            scores[name] = dict(selection=scorer(hist[2], depth, alt, model),
                                test=scorer(hist[3], depth, alt, model))
    eligible = [name for name, model in models.items()
                if model["success"] and np.isfinite(scores[name]["selection"])]
    if not eligible:
        report["fallback_reason"] = "no converged finite predictive fit"
        return None, report
    selected = max(eligible, key=lambda name: scores[name]["selection"])
    report.update(
        selected=selected, scores=scores,
        selection_gain_over_shared=scores[selected]["selection"]-scores["binomial"]["selection"],
        diagnostic_gain_over_shared=scores[selected]["test"]-scores["binomial"]["test"],
        sample_order="input_axis_0",
        models={name: {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in model.items()} for name, model in models.items()})
    return models[selected], report


def calibrate(reads, positions, *, threads, config):
    hist, depth, alt, meta = histograms(reads, positions, config)
    fitted, report = select_model(hist, depth, alt, threads=threads, config=config)
    return fitted, dict(meta, **report)


def likelihoods(reads, positions, fitted, *, threads, config):
    """Apply the selected observation model without genotype-mixture priors."""
    from .parallel import numba_thread_scope
    with numba_thread_scope(threads):
        return raw_likelihoods(reads, fitted)


@timed_stage("read_calibration")
def prepare_likelihoods(reads, positions, *, threads):
    """Fit once per contig; return raw GLs plus a checkpointable fit report.

    Insufficient or non-convergent fits fall back explicitly to the fixed read
    model. No genotype, pedigree, founder panel or simulation truth enters fitting.
    """
    from .runtime import available_cpu_count
    if not 1 <= threads <= available_cpu_count():
        raise ValueError("read calibration threads must fit CPU affinity")
    reads, positions = np.asarray(reads), np.asarray(positions)
    if reads.ndim != 3 or reads.shape[2] != 2 or positions.shape != (reads.shape[1],):
        raise ValueError("read calibration requires matching (sample, site, 2) counts and positions")
    if np.any(reads < 0) or not np.all(np.isfinite(reads)):
        raise ValueError("read calibration requires finite non-negative allele depths")
    if not np.issubdtype(reads.dtype, np.integer) and np.any(reads != np.floor(reads)):
        raise ValueError("read calibration requires integer allele depths")
    if np.any(~np.isfinite(positions)) or np.any(np.diff(positions) <= 0):
        raise ValueError("read calibration requires sorted unique positions")
    record = identity()
    fitted = None
    if record["enabled"]:
        fitted, report = calibrate(reads, positions, threads=threads, config=ReadCalibrationConfig())
    else:
        report = dict(fallback_reason="calibration disabled", folds=[])
    report.update(identity=record, applied=fitted is not None)
    if fitted is None:
        from .parallel import numba_thread_scope
        with numba_thread_scope(threads):
            gl = allele_depths_to_raw_genotype_likelihoods(reads)
        print(f"  Read calibration: fixed model ({report['fallback_reason']})", flush=True)
    else:
        gl = likelihoods(reads, positions, fitted, threads=threads, config=ReadCalibrationConfig())
        print(f"  Read calibration: {report['selected']}; "
              f"selection gain over shared={report['selection_gain_over_shared']:.3f}; "
              f"diagnostic gain={report['diagnostic_gain_over_shared']:.3f}", flush=True)
    return gl, report


def block_likelihoods(genomic_data, positions, gl):
    """Slice the one chromosome read model into discovery blocks, preserving order."""
    result = []
    for sites, reads, _keep in genomic_data:
        indices = np.searchsorted(positions, sites)
        if np.any(indices >= len(positions)) or not np.array_equal(positions[indices], sites):
            raise ValueError("discovery block sites do not match the calibrated chromosome")
        if len(reads) != len(gl):
            raise ValueError("discovery block samples do not match calibrated likelihoods")
        # Contiguous views avoid another chromosome-sized likelihood allocation.
        if len(indices) and indices[-1] - indices[0] == len(indices) - 1:
            result.append(gl[:, indices[0]:indices[-1] + 1])
        else:
            result.append(gl[:, indices])
    return result
