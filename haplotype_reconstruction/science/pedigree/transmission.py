"""Joint likelihood of scheduled homologues under an ordered parent-offspring direction.

One child homologue is a meiotic copy of the parent's two; the other three homologues
are founder mosaics. Minimum-switch homologue alignments are averaged at each boundary.
Unknown painted labels suppress the bin's copy emission. Components have independent phase.

Method: docs/methods/pedigree.md#finite-continuous-direction
"""

from __future__ import annotations

import math

import numpy as np
from numba import prange

from ...core.kernels import kernel
from .junctions import ComponentSchedule
from .kernels.signatures import OVERLOADS
from .parameters import TRANSMISSION_SWITCH_FLOOR


@kernel(overloads=OVERLOADS["transmission._log_likelihood"], cache=True, parallel=True, nogil=True)
def _log_likelihood(
    dyads,
    scheduled_trajectories,
    local_classes,
    known_labels,
    component_bin_start,
    component_bin_stop,
    selector_switch_probability,
    component_switch_start,
    trajectory_count,
    mismatch_probability,
):
    """float64 [D] log joint likelihoods (nats), dyads int64 [D, 2] = (child, parent).

    scheduled_trajectories int64 [N, B, 2] index component trajectories; local_classes
    int64 [N, B, 2] are their bin-local classes; known_labels bool [N, B] requires
    both painted labels. component_bin_start/stop int64 [K] delimit components;
    selector_switch_probability float64 [sum(B_k - 1)] and component_switch_start
    int64 [K] delimit selector probabilities; trajectory_count int64 [K] is T.
    mismatch_probability is the dimensionless bin mismatch probability.

    Four states (child homologue, parental selector) suffice. Time O(dyads x bins),
    constant workspace per dyad, independent of T once schedules are constructed.
    Rescale each bin and accumulate log mass to avoid chromosome-length underflow.
    """
    output = np.zeros(len(dyads))
    for dyad in prange(len(dyads)):
        child = dyads[dyad, 0]
        parent = dyads[dyad, 1]
        joint_log_likelihood = 0.0
        mass = np.zeros((2, 2))
        next_mass = np.zeros((2, 2))
        for component in range(len(component_bin_start)):
            component_first_bin = component_bin_start[component]
            if component_first_bin == component_bin_stop[component]:
                continue
            informative = known_labels[child, component_first_bin] and known_labels[parent, component_first_bin]
            normalization = 0.0
            for child_homologue in range(2):
                for parent_homologue in range(2):
                    emission = 1.0
                    if informative:
                        emission = (
                            1.0 - mismatch_probability
                            if local_classes[child, component_first_bin, child_homologue]
                            == local_classes[parent, component_first_bin, parent_homologue]
                            else mismatch_probability
                        )
                    mass[child_homologue, parent_homologue] = 0.25 * emission
                    normalization += mass[child_homologue, parent_homologue]
            for child_homologue in range(2):
                for parent_homologue in range(2):
                    mass[child_homologue, parent_homologue] /= normalization
            joint_log_likelihood += math.log(normalization)
            # A mosaic changes uniformly to one of T - 1 other trajectories: this is derived,
            # not a separately fitted junction probability. T = 1 permits no changes.
            alternative_trajectory_weight = 1.0 / max(1, trajectory_count[component] - 1)
            for boundary in range(component_bin_stop[component] - component_first_bin - 1):
                left_bin = component_first_bin + boundary
                right_bin = left_bin + 1
                switch_probability = min(
                    0.5,
                    max(
                        TRANSMISSION_SWITCH_FLOOR,
                        selector_switch_probability[component_switch_start[component] + boundary],
                    ),
                )
                child_left_first, child_left_second = (
                    scheduled_trajectories[child, left_bin, 0],
                    scheduled_trajectories[child, left_bin, 1],
                )
                child_right_first, child_right_second = (
                    scheduled_trajectories[child, right_bin, 0],
                    scheduled_trajectories[child, right_bin, 1],
                )
                child_straight_changes = int(child_left_first != child_right_first) + int(
                    child_left_second != child_right_second
                )
                child_crossed_changes = int(child_left_first != child_right_second) + int(
                    child_left_second != child_right_first
                )
                child_minimum_changes = min(child_straight_changes, child_crossed_changes)
                child_alignment_count = int(child_straight_changes == child_minimum_changes) + int(
                    child_crossed_changes == child_minimum_changes
                )
                parent_left_first, parent_left_second = (
                    scheduled_trajectories[parent, left_bin, 0],
                    scheduled_trajectories[parent, left_bin, 1],
                )
                parent_right_first, parent_right_second = (
                    scheduled_trajectories[parent, right_bin, 0],
                    scheduled_trajectories[parent, right_bin, 1],
                )
                parent_straight_changes = int(parent_left_first != parent_right_first) + int(
                    parent_left_second != parent_right_second
                )
                parent_crossed_changes = int(parent_left_first != parent_right_second) + int(
                    parent_left_second != parent_right_first
                )
                parent_minimum_changes = min(parent_straight_changes, parent_crossed_changes)
                parent_alignment_count = int(parent_straight_changes == parent_minimum_changes) + int(
                    parent_crossed_changes == parent_minimum_changes
                )
                # Both parental homologues follow the founder-mosaic law.
                joint_log_likelihood += parent_minimum_changes * math.log(
                    switch_probability * alternative_trajectory_weight
                ) + (2 - parent_minimum_changes) * math.log(1.0 - switch_probability)
                informative = known_labels[child, right_bin] and known_labels[parent, right_bin]
                next_mass[:, :] = 0.0
                for child_crossed in range(2):
                    if (
                        child_straight_changes if child_crossed == 0 else child_crossed_changes
                    ) != child_minimum_changes:
                        continue
                    for child_homologue in range(2):
                        next_child_homologue = child_homologue ^ child_crossed
                        child_alignment_weight = 1.0 / child_alignment_count
                        # The child homologue left unexplained by transmission is a founder mosaic.
                        other_child_homologue_changed = (
                            scheduled_trajectories[child, left_bin, 1 - child_homologue]
                            != scheduled_trajectories[child, right_bin, 1 - next_child_homologue]
                        )
                        child_alignment_weight *= (
                            switch_probability * alternative_trajectory_weight
                            if other_child_homologue_changed
                            else 1.0 - switch_probability
                        )
                        for parent_crossed in range(2):
                            if (
                                parent_straight_changes if parent_crossed == 0 else parent_crossed_changes
                            ) != parent_minimum_changes:
                                continue
                            for parent_homologue in range(2):
                                path_mass = mass[child_homologue, parent_homologue]
                                if path_mass == 0.0:
                                    continue
                                # Step the meiotic selector, then compare the copied local classes.
                                for selector_switch in range(2):
                                    next_parent_homologue = (parent_homologue ^ selector_switch) ^ parent_crossed
                                    weight = (
                                        child_alignment_weight
                                        * (switch_probability if selector_switch == 1 else 1.0 - switch_probability)
                                        / parent_alignment_count
                                    )
                                    emission = 1.0
                                    if informative:
                                        emission = (
                                            1.0 - mismatch_probability
                                            if local_classes[child, right_bin, next_child_homologue]
                                            == local_classes[parent, right_bin, next_parent_homologue]
                                            else mismatch_probability
                                        )
                                    next_mass[next_child_homologue, next_parent_homologue] += (
                                        path_mass * weight * emission
                                    )
                normalization = next_mass[0, 0] + next_mass[0, 1] + next_mass[1, 0] + next_mass[1, 1]
                if normalization <= 0.0:
                    joint_log_likelihood = -np.inf
                    break
                for child_homologue in range(2):
                    for parent_homologue in range(2):
                        mass[child_homologue, parent_homologue] = (
                            next_mass[child_homologue, parent_homologue] / normalization
                        )
                joint_log_likelihood += math.log(normalization)
            if not np.isfinite(joint_log_likelihood):
                break
        output[dyad] = joint_log_likelihood
    return output


def log_likelihood(
    dyads: np.ndarray,
    components: list[ComponentSchedule],
    thetas: list[np.ndarray],
    mismatch_probability: float,
) -> np.ndarray:
    """float64 [D] joint transmission log likelihoods over independent chromosome components."""
    if not components:
        return np.zeros(len(dyads))
    sizes = np.asarray([part.pairs.shape[1] for part in components], dtype=np.int64)
    bin_stop = np.cumsum(sizes)
    bin_start = bin_stop - sizes
    theta_sizes = np.asarray([len(value) for value in thetas], dtype=np.int64)
    theta_start = np.cumsum(theta_sizes) - theta_sizes
    return _log_likelihood(
        np.ascontiguousarray(dyads, dtype=np.int64),
        np.ascontiguousarray(np.concatenate([part.pairs for part in components], axis=1), dtype=np.int64),
        np.ascontiguousarray(np.concatenate([part.classes for part in components], axis=1), dtype=np.int64),
        np.ascontiguousarray(np.concatenate([part.known for part in components], axis=1)),
        bin_start,
        bin_stop,
        np.ascontiguousarray(np.concatenate(thetas), dtype=np.float64),
        theta_start,
        np.asarray([part.n_trajectories for part in components], dtype=np.int64),
        float(mismatch_probability),
    )
