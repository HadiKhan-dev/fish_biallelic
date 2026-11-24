"""Score family transmission segments and apply the evidence to retained edges."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .family_models import Mapper, _maps, family_models

TESTED, KEPT, WITHHELD, UNTESTED = "tested", "kept", "withheld", "untested"
BASIS = "transmission"


def segments(theta: np.ndarray, unit_morgans: float, tail_morgans: float) -> tuple[np.ndarray, np.ndarray]:
    """Component-local evidence segments: int64 [B] segment of each bin and float64 [n] working Morgans.

    Consecutive pieces of ``unit_morgans`` of the component's working map (Haldane distances of its selector
    probabilities, as the material units); a final piece shorter than ``tail_morgans`` joins the previous one."""
    theta = np.asarray(theta, dtype=np.float64)
    distance = -0.5 * np.log1p(-2.0 * np.minimum(theta, 0.499999))
    position = np.concatenate(([0.0], np.cumsum(distance)))
    total = float(position[-1])
    count = max(1, int(math.floor(total / unit_morgans + (1.0 - tail_morgans / unit_morgans))))
    segment = np.minimum(np.floor(position / unit_morgans).astype(np.int64), count - 1)
    bounds = np.minimum(np.arange(count + 1, dtype=np.float64) * unit_morgans, total)
    bounds[-1] = total
    lengths = np.diff(bounds)
    used = np.zeros(count, dtype=np.bool_)
    used[segment] = True
    if not used.all():
        # one bin step longer than a unit leaves units without bins (unit 0 always holds bin 0): their Morgans join
        # the last unit before them that holds bins, so every segment has data and the total is kept
        renumber = np.cumsum(used) - 1
        owner = np.maximum.accumulate(np.where(used, np.arange(count), 0))
        merged = np.zeros(int(used.sum()))
        np.add.at(merged, renumber[owner], lengths)
        segment, lengths = renumber[segment], merged
    return segment, lengths


def segment_models(tc, mcls, mknown, cpairs, theta, floor, mdelta, ft, seed, width=1, mapper=None):
    """(PAR, REL, UNR) profile log likelihoods of one segment for members [C, S, witnesses...] (UNR includes C's own
    mosaic): the best, per model, of two searches from data-derived starts, the witnesses ordered by a digest of
    each witness's painted classes and known bits and in the reverse of that order. Each search only raises the
    profile, so the value is the better of two local optima; it depends neither on the input witness order nor on
    sample names (witnesses with identical data are interchangeable).

    ``width``, ``mapper``: threads for this segment and the caller's map factory (family_models). The two searches
    are independent and run at once, each with half of the threads for its own independent fits; the result does
    not depend on ``width``."""
    n = mcls.shape[0]
    # Witnesses with identical painted classes and known bits but different label allowances are ordered by their
    # allowances (a second key: the order of all others is that of the classes and known bits alone).
    order = sorted(range(2, n), key=lambda w: (hashlib.sha256(np.ascontiguousarray(mcls[w]).tobytes()
                                                              + np.ascontiguousarray(mknown[w]).tobytes()).digest(),
                                               np.ascontiguousarray(mdelta[w]).tobytes()))
    orders = (order, order[::-1]) if n > 3 else (order,)

    def search(witnesses):
        idx = np.asarray([0, 1] + witnesses, dtype=np.int64)
        res = family_models(tc, np.ascontiguousarray(mcls[idx]), np.ascontiguousarray(mknown[idx]), cpairs, theta,
                            floor, np.ascontiguousarray(mdelta[idx]), max_iter=ft.maximum_iterations, rounds=ft.start_rounds, rho=ft.role_prior,
                            n_random=ft.random_starts, seed=seed, width=max(1, width // len(orders)),
                            mapper=mapper)
        return (res["PAR"][0], res["REL"][0], res["C"][0] + res["UNR"][0])

    best = None
    for value in _maps(mapper, min(width, len(orders)))(search, orders):
        best = value if best is None else tuple(max(a, b) for a, b in zip(best, value, strict=True))
    return best


def member_allowances(schedule, members: np.ndarray, lo: int, hi: int, delta: float) -> np.ndarray:
    """float64 [M, hi - lo, 4]: each member's per-bin probabilities of its two painted labels' error pattern on bins
    lo:hi (PD29, 4.22 R4b): (both right, label 0 wrong and label 1 right, label 0 right and label 1 wrong, both wrong).
    A wrong label's whole mass goes to whichever class a hypothesis needs (the statistic keeps no class-specific mass),
    so the vote is never more decisive than this law about a wrong label, whatever the number of local classes.

    From the painting's untempered label-error statistic moved onto the scheduled homologues (junctions._label_errors):
    v_t, the probability that label t is wrong, and u, that the released pair is wrong. For two true labels "the pair
    is right" is "both released classes are carried", so u = P(label 0 wrong or label 1 wrong) and the posterior joint
    law is exact: both wrong v_0 + v_1 - u, one wrong v_t - (both), none 1 - u. The posterior is overconfident (4.22
    R5), so the 4.20 floor delta bounds the law: each label's marginal lies in [delta, 1/2]; both-wrong is clipped to
    what the marginals allow; and every error pattern is at least its independence value at the floor (one label wrong
    delta (1 - delta), both delta^2), the remainder being both right. No pattern is then zero: every member emission
    is positive, so a hypothesis that needs a wrong label is penalised, never excluded, and no member's selector states
    can all emit 0. Without label errors this is the floor's independence law of 4.21: (1 - delta)^2,
    delta (1 - delta), delta (1 - delta), delta^2."""
    n = hi - lo
    errors = getattr(schedule, "homologue_error", None)
    pairs = getattr(schedule, "pair_error", None)
    v = (np.zeros((len(members), n, 2)) if errors is None
         else np.asarray(errors, dtype=np.float64)[members, lo:hi])
    u = np.zeros((len(members), n)) if pairs is None else np.asarray(pairs, dtype=np.float64)[members, lo:hi]
    floor = float(delta)
    v0 = np.minimum(0.5, np.maximum(floor, v[..., 0]))
    v1 = np.minimum(0.5, np.maximum(floor, v[..., 1]))
    joint = np.minimum(np.maximum(v[..., 0] + v[..., 1] - u, 0.0), np.minimum(v0, v1))
    both = np.maximum(floor * floor, joint)
    single = floor * (1.0 - floor)
    first, second = np.maximum(single, v0 - both), np.maximum(single, v1 - both)
    return np.ascontiguousarray(np.stack((1.0 - first - second - both, first, second, both), axis=2))


def component_plan(schedule, theta: np.ndarray, parameters) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(trajectory classes, the segment of every bin, the Morgans of every segment) of one component, or None when
    the component has no bins. The plan does not depend on the edge: one per component serves every edge."""
    if schedule.classes.shape[1] == 0:
        return None
    m = parameters.material
    seg, lengths = segments(theta, m.unit_morgans, m.tail_morgans)
    return _trajectory_classes(schedule), seg, lengths


def segment_unit(schedule, plan, theta: np.ndarray, members: tuple[int, ...], u: int, parameters, delta: float,
                 floor: float, width: int = 1, mapper: Mapper | None = None) -> tuple[float, float, float]:
    """(PAR, REL, UNR) profile log likelihoods of segment ``u`` of one component for members (C, S, witnesses...).
    Every segment of every component and edge is fitted independently of the others (segment_models): a caller may
    fit them in any order, or at once on separate threads; ``width`` threads, mapped by the caller's ``mapper``,
    serve this one segment's search."""
    trajectories, seg, _ = plan
    idx = np.asarray(members, dtype=np.int64)
    bins = np.flatnonzero(seg == u)
    lo, hi = int(bins[0]), int(bins[-1]) + 1
    return segment_models(
        np.ascontiguousarray(trajectories[:, lo:hi], dtype=np.int64),
        np.ascontiguousarray(schedule.classes[idx, lo:hi], dtype=np.int64),
        np.ascontiguousarray(schedule.known[idx, lo:hi], dtype=np.bool_),
        np.ascontiguousarray(schedule.pairs[idx[0], lo:hi].T, dtype=np.int64),
        np.array(theta[lo:hi - 1], dtype=np.float64, order="C", copy=True),
        float(floor),
        member_allowances(schedule, idx, lo, hi, delta),
        parameters.family_transmission, parameters.resampling.bootstrap_seed, width=width, mapper=mapper,
    )


def score_component(schedule, theta: np.ndarray, members: tuple[int, ...], parameters, delta: float,
                    floor: float) -> tuple[np.ndarray, np.ndarray]:
    """(float64 [n, 3] PAR, REL, UNR profile log likelihoods, float64 [n] Morgans) of one component's segments for
    members (C, S, witnesses...): segment_unit of every segment, in sequence."""
    plan = component_plan(schedule, theta, parameters)
    if plan is None:
        return np.zeros((0, 3)), np.zeros(0)
    lengths = plan[2]
    out = np.zeros((len(lengths), 3))
    for u in range(len(lengths)):
        out[u] = segment_unit(schedule, plan, theta, members, u, parameters, delta, floor)
    return out, lengths


def _trajectory_classes(schedule) -> np.ndarray:
    """int64 [T, B] bin-local class of every whole-component trajectory, recovered from the schedule."""
    if getattr(schedule, "trajectory_classes", None) is not None:
        return np.asarray(schedule.trajectory_classes, dtype=np.int64)
    raise ValueError("family transmission needs the schedule's trajectory class table")


def request(sample_ids, contigs, before, guard_record, relatedness, relationship_request, relationship_records,
            observed_counts, parameters):
    """(TransmissionRequest, witnesses per guard vote) after the guard's fixed point (PD29).

    Witnesses of a retained edge: the guard's tested members of the child's relatedness family (linked members
    removed against the retained edges), minus those failing the guard's family check, minus those with a retained
    parent other than C or a retained parent of S; at most ``maximum_witnesses`` by relatedness member support, then
    observed-site counts in configured contig order, then the witness's own family-check margins (no sample names;
    complete ties keep the guard's member order). The guard's 'both hypotheses fail' set-aside is not applied: it
    depends on C, the hypothesis under test."""
    from ...contracts.pedigree import TransmissionRequest
    from . import guard

    ft = parameters.family_transmission
    rp = parameters.relationship
    withheld = guard_record.withheld
    retained = {(int(p), int(c)) for p, c in zip(before.parents.tolist(), before.children.tolist(), strict=True)}
    retained -= set(withheld)
    parents_of: dict[int, set[int]] = {}
    for p, c in retained:
        parents_of.setdefault(c, set()).add(p)
    trio = guard.summed_margins(relationship_request, relationship_records)[1] if relationship_records else {}
    support = np.nan_to_num(np.asarray(relatedness.member_support, dtype=np.float64), nan=0.0)
    counts = np.asarray(observed_counts, dtype=np.int64)
    witnesses: dict[tuple[int, int], int] = {}
    tested: list[tuple[int, int, list[int]]] = []
    for vote in guard_record.votes:
        f = vote.family
        if vote.call == guard.WITHHELD:
            continue
        c, p = int(f.child), int(f.parent)
        partners = guard.check_partner_map(f, rp.sibship_check_partners)
        keep = []
        content: dict[int, tuple[float, ...]] = {}
        for w in f.tested:
            low = min(support[c], support[w])
            band = int(low >= rp.support_bounds[0]) + int(low >= rp.support_bounds[1])
            price = max(0.0, math.log(rp.sibship_prior[band] / (1.0 - rp.sibship_prior[band])))
            others = partners[w]
            if others and float(np.median([trio[tuple(sorted((c, w, o)))] for o in others])) <= -price:
                continue
            if parents_of.get(int(w), set()) - {p} - parents_of.get(c, set()):
                continue
            keep.append(int(w))
            # content key: the witness's own family-check margins (data, independent of C and of sample names)
            content[int(w)] = tuple(sorted(-round(float(trio[tuple(sorted((c, w, o)))]), 9) for o in others))
        keep.sort(key=lambda w: (-float(support[w]), tuple(-int(x) for x in counts[:, w]), content[w]))
        keep = keep[: ft.maximum_witnesses]
        witnesses[(p, c)] = len(keep)
        if len(keep) >= ft.minimum_witnesses:
            tested.append((c, p, keep))
    tested.sort(key=lambda row: (row[0], row[1]))  # canonical edge order (child, parent): vote order is immaterial
    parents = [p for _, p, _ in tested]
    children = [c for c, _, _ in tested]
    members = [w for _, _, keep in tested for w in keep]
    offsets = np.concatenate(([0], np.cumsum([len(keep) for _, _, keep in tested]))).astype(np.int64)
    req = TransmissionRequest(
        tuple(sample_ids), tuple(contigs), np.asarray(parents, dtype=np.int64), np.asarray(children, dtype=np.int64),
        offsets, np.asarray(members, dtype=np.int64),
    )
    return req, witnesses


def robust_log_odds(difference: float, error_probability: float) -> float:
    """PD06's robust per-unit log odds of a log-likelihood difference (capped at ln((2 - eps) / eps))."""
    if difference == 0.0:
        return 0.0
    keep = 1.0 - error_probability
    floor = 0.5 * error_probability
    scaled = math.exp(-abs(difference))
    value = math.log(keep / (1.0 + scaled) + floor) - math.log(keep * scaled / (1.0 + scaled) + floor)
    return value if difference > 0.0 else -value


def apply(guard_record, req, witnesses, evidence, parameters):
    """GuardRecord with every retained edge's PD29 vote; a withheld edge is withdrawn one pass after the guard's
    fixed point (no iteration: the guard's cleaning is not recomputed)."""
    from dataclasses import replace

    from .orientation import unit_failure

    rp = parameters.relationship
    ft = parameters.family_transmission
    for record in evidence:
        if record.request_digest != req.digest or len(record.offsets) != len(req.parents) + 1:
            raise ValueError("family transmission evidence differs from its request")
    order = {(int(p), int(c)): e for e, (p, c) in enumerate(zip(req.parents.tolist(), req.children.tolist()))}
    votes = []
    for vote in guard_record.votes:
        f = vote.family
        if vote.call == "withheld":
            votes.append(vote)
            continue
        key = (int(f.parent), int(f.child))
        k = witnesses[key]
        if k < ft.minimum_witnesses or key not in order:
            votes.append(replace(vote, transmission_call=UNTESTED, transmission_witnesses=int(k)))
            continue
        e = order[key]
        ll = np.concatenate([r.log_likelihoods[r.offsets[e]:r.offsets[e + 1]] for r in evidence], axis=0)
        morgans = np.concatenate([r.morgans[r.offsets[e]:r.offsets[e + 1]] for r in evidence])
        total = ll.sum(axis=0)
        alternative = 1 if total[1] >= total[2] else 2
        eps = unit_failure(morgans, rp.failure_region_rate, rp.failure_region_length)
        score = float(sum(robust_log_odds(float(d), float(x)) for d, x in zip(ll[:, 0] - ll[:, alternative], eps)))
        changes = dict(transmission_witnesses=int(k), transmission_segments=int(len(morgans)),
                       transmission_score=score, transmission_alternative="REL" if alternative == 1 else "UNR")
        if score < -rp.withholding_score:
            votes.append(replace(vote, call="withheld", evidence_call="withheld", relationship_basis="",
                                 withholding_basis=BASIS, withdrawal_iteration=guard_record.cleaning_iterations + 1,
                                 transmission_call=WITHHELD, **changes))
        else:
            votes.append(replace(vote, transmission_call=KEPT, **changes))
    return replace(guard_record, votes=tuple(votes))
