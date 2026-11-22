"""Interpret the painting calibration's read-length grid and published read-library leaves for the report."""

import numpy as np

from ...contracts.payload import from_payload
from ...science.evidence.read_sharing import read_sharing
from ...science.painting.parameters import ReadStateParameters
from ...science.shared.reads import _bin_average_overlap, _correlation_curves, _two_means, length_grid


def read_state_parameters(manifest):
    return from_payload(ReadStateParameters, manifest["parameters"]["painting"]["read_state"])


def template_weights(parameters, edges, pairs, bins):
    """The survival fit's configured lengths and their pair-weighted overlap with the measured lag bins."""
    grid = length_grid(parameters)
    overlap = _bin_average_overlap(grid, edges[:-1][bins], edges[1:][bins]) / grid
    return grid, pairs[bins, None] * overlap


def split_curves(depth, positions, memberships):
    """Measure each fixed library on its own canonical sample axis."""
    for members in memberships:
        yield read_sharing(depth[list(members)], positions)


def published_groups(depth, positions, leaf_sizes, parameters):
    """Recover the published leaves in library_groups' depth-first order, splitting only accepted parents."""
    groups = []
    lags = tuple(parameters.library_lags_bp)

    def visit(members):
        if len(groups) >= len(leaf_sizes) or len(members) < leaf_sizes[len(groups)]:
            raise ValueError("published read-library sizes disagree with the evidence partition")
        if len(members) == leaf_sizes[len(groups)]:
            groups.append(tuple(map(int, members)))
            return
        labels = _two_means(_correlation_curves(depth[members], positions, lags), lags,
                            parameters.library_seed_lag_bp, parameters.library_iterations)
        if np.all(labels == labels[0]):
            raise ValueError("published read-library split has an empty child")
        visit(members[labels == 0])
        visit(members[labels == 1])

    visit(np.arange(len(depth)))
    if len(groups) != len(leaf_sizes):
        raise ValueError("published read-library leaf count disagrees with the evidence partition")
    return groups
