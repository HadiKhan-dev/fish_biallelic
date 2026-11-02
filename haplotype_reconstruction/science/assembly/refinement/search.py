"""Run one fixed-K sweep without storage policy, clocks, environment or count continuation.

Owns the RS11 transition: messages, ordered proposals, oriented acceptance, optional
lexicographic retry, and exact rescoring. RS12 selection is a separate transition.
Must not choose K, polish, publish checkpoints, or treat non-completion as ambiguity.
Method: docs/methods/founder_refinement.md#fixed-k-search (RS01–RS12).

Identity: direction and convention are independent. Reversed passes use forward
messages clamped through the next informative site's transition, with reversed
forward normalisers. A plain optimistic pass aborts before a coarse move needs
switch costs; its partial edits and trace are discarded before the guarded retry.
"""

from __future__ import annotations

import numpy as np
from numba.typed import List

from haplotype_reconstruction.contracts.assembly import (
    AcceptanceCounters,
    ProposalCounters,
    RefinementDecisionTrace,
    RefinementProblem,
    RefinementProposal,
    RefinementSweep,
    RefinementWeakJunction,
)
from haplotype_reconstruction.core.errors import InvariantError

from . import checkpoints, messages
from .acceptance import MOVE_COARSE, MOVE_FINE, accept_pass
from .capabilities import RefinementExecution, SweepWorkspaceView
from .objective import score_panel, selected_alleles
from .parameters import EXACT_GAIN_ABSOLUTE, EXACT_GAIN_RELATIVE, PAIR_TABLE_BYTES, SweepControl, SweepParameters
from .proposals import WindowSolver
from .reductions import panel_objective
from .work_limit import WindowWorkLimit
from .workspace import sample_ranges

__all__ = ["sweep", "Orientation"]


class Orientation:
    """View coded leaves and offsets in one pass direction, retaining forward evidence.

    Reverse creates an int16[R_b,L_b] leaf copy and int64[B+1] offsets. haps returns
    a private int16[K,L] materialisation; window returns a workspace-owned view.
    """

    def __init__(self, workspace: SweepWorkspaceView, reverse: bool):
        sites = int(workspace.offsets[-1])
        self.reverse = reverse
        self.workspace = workspace
        if reverse:
            self.offsets = np.ascontiguousarray(sites - workspace.offsets[::-1], np.int64)
            self.leaves = List([np.ascontiguousarray(leaf[:, ::-1]) for leaf in list(workspace.leaves)[::-1]])
        else:
            self.offsets = np.ascontiguousarray(workspace.offsets, np.int64)
            self.leaves = workspace.leaves
        self.blocks = len(self.offsets) - 1

    def haps(self, panel):
        """Return private int16[K,L] coded alleles of int64[K,B] oriented founder rows."""
        return selected_alleles(self.leaves, self.offsets, np.ascontiguousarray(panel, np.int64))

    def window(self, founders):
        """Read K's evidence/cost capability; its view lifetime is the workspace lease."""
        return self.workspace.windows.view(founders, self.reverse)


def _scores(rows, workspace, team):
    with team.scope():
        alleles = selected_alleles(workspace.leaves, workspace.offsets, np.ascontiguousarray(rows, np.int64))
        values = score_panel(
            alleles, workspace.windows.view(len(rows)), sample_ranges(workspace.samples, team), PAIR_TABLE_BYTES, team
        )
    return np.asarray(values) + workspace.normalisers


def _messages(rows, orientation, workspace, lex, team):
    samples, offsets = workspace.samples, orientation.offsets
    with team.scope():
        haps = orientation.haps(rows)
    arrays = checkpoints.allocate(
        len(rows), orientation.blocks, samples, lex, haps, orientation, workspace.message_step
    )
    fwd, bwd, fcnt, bcnt, fscale, bscale = arrays
    win = orientation.window(len(rows))
    changed = np.ones(orientation.blocks, np.bool_)
    count3 = np.zeros((1, 1, 1))
    for s0, s1, n in sample_ranges(samples, team):
        with team.scope(n):
            if fwd.dense:
                if lex:
                    messages.messages_lex(haps, win, offsets, s0, s1, fwd.data, bwd.data, fcnt, bcnt, fscale, bscale)
                else:
                    messages.messages(haps, win, offsets, s0, s1, fwd.data, bwd.data, fscale, bscale)
            else:
                for msg, cnt, scale in ((fwd, fcnt, fscale), (bwd, bcnt, bscale)):
                    checkpoints.update(
                        haps,
                        win,
                        offsets,
                        msg.bounds,
                        changed,
                        msg.data,
                        cnt if lex else count3,
                        scale,
                        False,
                        msg.forward,
                        lex,
                        s0,
                        s1,
                    )
    return arrays


def _proposals(rows, fwd, bwd, solver, parameters, convention, team):
    with team.scope():
        fwd = fwd.flanks(a for a, _ in solver.spans)
        bwd = bwd.flanks(c for _, c in solver.spans)
    sizes = [solver.template_bytes(*template, solver.arrays[0].shape[0], len(rows)) for template in solver.templates]
    # Scheduling order is cost only; restore template/focal order before acceptance.
    order = sorted(range(len(solver.spans)), key=lambda i: (-sizes[i], i))

    ordinals = solver.work.bind((0, 0, 0), range(len(solver.templates)))

    def solve(index):
        return solver.solve(index, rows, fwd, bwd, parameters.frontier_bytes, convention, ordinal=ordinals[index])

    try:
        solved = team.map(solve, order)
    except WindowWorkLimit as error:
        # map joins all active templates before propagation. Preserve the complete
        # computational stop domain, without exposing partial founder results.
        solver.work.attach(error)
        raise
    results = [None] * len(order)
    for index, result in zip(order, solved, strict=True):
        results[index] = result
    proposals = []
    stats = dict(
        windows=0,
        certified=0,
        dp=0,
        fixing_dps=0,
        max_frontier=0,
        fallbacks=0,
        changed=0,
        templates_solved=len(order),
        templates_reused=0,
        beam_truncations=0,
        beam_fallbacks=0,
    )
    for result in results:
        for kind, start, stop, founder, values, diagnostic in result:
            stats["windows"] += 1
            for key in ("certified", "dp", "fixing_dps"):
                stats[key] += int(diagnostic[key])
            stats["max_frontier"] = max(stats["max_frontier"], diagnostic["frontier"])
            stats["fallbacks"] += int(diagnostic["fallback"])
            stats["beam_truncations"] += int(diagnostic.get("beam_truncations", 0))
            stats["beam_fallbacks"] += int(diagnostic.get("beam_fallback", False))
            if values is not None:
                stats["changed"] += 1
                proposals.append(RefinementProposal(kind, start, stop, founder, np.asarray(values, np.int64)))
    return tuple(proposals), stats


def _oriented_proposals(proposals, blocks, reverse):
    result = {}
    for item in proposals:
        kind = MOVE_FINE if item.kind == "fine" else MOVE_COARSE
        if reverse:
            result.setdefault(blocks - item.stop, []).append(
                (kind, blocks - item.start, item.founder, item.rows[::-1].copy())
            )
        else:
            result.setdefault(item.start, []).append((kind, item.stop, item.founder, item.rows))
    return result


def _pass(rows, arrays, forward, backward, props, reverse, convention, lex, guarded, F0, team):
    fwd, bwd, fcnt, bcnt, fscale, bscale = arrays
    with team.scope():
        if reverse:
            entering = messages.boundary_costs(forward.window(len(rows)), forward.offsets)
            stored = fwd.reversed(entering)
            orientation, stored_cnt = backward, stored.counts
            stored_scale = np.ascontiguousarray(fscale[::-1])
            oriented = np.ascontiguousarray(rows[:, ::-1])
            start_fwd = start_fcnt = None
        else:
            orientation, stored, stored_cnt, stored_scale = forward, bwd, bcnt, bscale
            oriented = rows.copy()
            start_fwd, start_fcnt = fwd, fcnt
    counters, gains, pairs = accept_pass(
        orientation,
        oriented,
        stored,
        stored_cnt,
        stored_scale,
        props,
        orientation.window(len(rows)),
        lex,
        team,
        start_fwd=start_fwd,
        start_fcnt=start_fcnt,
        scale=abs(F0),
        reverse=convention,
        optimistic=guarded and not lex,
    )
    out = np.ascontiguousarray(oriented[:, ::-1]) if reverse else oriented
    return out, counters, gains, pairs


def _weak_junctions(gains, pairs, reverse, blocks, margin):
    records = []
    for p in range(1, blocks):
        if gains[p] > -margin:
            records.append((blocks - p if reverse else p, int(pairs[p, 0]), int(pairs[p, 1]), float(gains[p])))
    records.sort()
    merged = []
    for junction, a, b, gain in records:
        if merged and merged[-1][2] == (a, b) and merged[-1][1] == junction - 1:
            merged[-1] = (merged[-1][0], junction, (a, b), max(merged[-1][3], gain))
        else:
            merged.append((junction, junction, (a, b), gain))
    return tuple(merged)


def sweep(
    problem: RefinementProblem,
    parameters: SweepParameters,
    control: SweepControl,
    workspace: SweepWorkspaceView,
    team: RefinementExecution,
) -> RefinementSweep:
    """Apply one complete fixed-K transition and return its auditable scientific state.

    problem owns the immutable arrays; parameters own scientific caps; control has
    independent direction, convention and starting message mode. workspace supplies
    evidence windows and a declared message capability; team supplies scoped kernel
    execution. Only private workspace scratch and the copied rows mutate. Raises
    InvariantError if accepted gains fail exact rescoring, or WindowWorkLimit if the
    template account exhausts (all partial results must then be discarded).
    """
    rows = np.ascontiguousarray(problem.rows, np.int64).copy()
    guarded = problem.context is not None
    context = None if problem.context is None else (problem.context.groups, problem.context.paths)
    solver = WindowSolver(
        problem.emissions,
        context,
        problem.penalty,
        parameters.window_work_coefficient,
        source_start=problem.source_start,
    )
    forward, backward = Orientation(workspace, False), Orientation(workspace, True)
    reverse = control.direction == "rev"
    lex = guarded and control.lexicographic
    with workspace.windows.lease(len(rows)):
        initial = _scores(rows, workspace, team)
        F0 = panel_objective(initial)
        arrays = _messages(rows, forward, workspace, lex, team)
        proposals, statistics = _proposals(
            rows, arrays[0], arrays[1], solver, parameters, control.convention.reverse, team
        )
        props = _oriented_proposals(proposals, rows.shape[1], reverse)
        attempts = 0
        while True:
            out, counters, gains, pairs = _pass(
                rows, arrays, forward, backward, props, reverse, control.convention.reverse, lex, guarded, F0, team
            )
            if not counters["lex_abort"]:
                break
            attempts += 1
            arrays = _messages(rows, forward, workspace, True, team)
            lex = True
        counters["lex_abort"] = attempts
        trace = counters.pop("decision_trace")
        accepted = int(counters["fine"] + counters["coarse"] + counters["exchange"])
        scores = _scores(out, workspace, team) if accepted else initial
        F = panel_objective(scores)
        accounted = float(counters["fine_gain"] + counters["coarse_gain"] + counters["exchange_gain"])
        deviation = F - (F0 + accounted)
        if abs(deviation) > EXACT_GAIN_RELATIVE * max(1.0, abs(F)) + EXACT_GAIN_ABSOLUTE * accepted:
            raise InvariantError(f"refinement exact gain {F - F0}, accounted {accounted}, deviation {deviation}")
        # Explicit zeroes identify excluded polish move families at this seam.
        counters.update(
            bridge=0,
            bridge_gain=0.0,
            escape=0,
            escape_gain=0.0,
            escape_offers=0,
            escape_windows=0,
            escape_beam_cut=0,
            escape_beam_binding=0,
            long=0,
            long_gain=0.0,
            long_offers=0,
            long_windows=0,
            long_beam_cut=0,
            long_beam_binding=0,
            long_candidates=0,
        )
        domains, work = solver.work.snapshot()
    return RefinementSweep(
        out,
        scores,
        F0,
        F,
        F - F0,
        accounted,
        deviation,
        accepted,
        accepted == 0,
        bool(lex and counters["lex_items"] > 0),
        AcceptanceCounters(**counters),
        proposals,
        ProposalCounters(**statistics),
        work,
        domains,
        gains,
        pairs,
        tuple(
            RefinementWeakJunction(*record)
            for record in _weak_junctions(gains, pairs, reverse, rows.shape[1], problem.penalty)
        ),
        RefinementDecisionTrace(trace[:, [0, 1, 2, 3, 4, 5, 6, 9]].astype(np.int64), trace[:, 7], trace[:, 8]),
    )
