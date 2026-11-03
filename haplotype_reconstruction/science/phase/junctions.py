"""Read-supported junction pinning of the final two phase solves (FP10)."""

from dataclasses import dataclass, replace

import numpy as np

from ...contracts.roles import record_field
from .diagnostics import DecisionTrace, PhaseTrace
from .parameters import PhaseParameters


@dataclass(frozen=True)
class JunctionInputs:
    """Borrow only the final labels, sources, emissions, pedigree and pin parameters."""

    X: np.ndarray = record_field("state", units="1", dtype="int64", stored=False)
    V: np.ndarray = record_field("state", units="1", dtype="int8", stored=False)
    E: np.ndarray = record_field("evidence", units="nats", dtype="float64", stored=False)
    par: np.ndarray = record_field("call", units="1", dtype="int64", stored=False)
    config: PhaseParameters = record_field("state", units="1", dtype="PhaseParameters", stored=False)
    audit: DecisionTrace | None = record_field("diagnostic", units="1", dtype="object", stored=False, default=None)

    @property
    def N(self) -> int:
        return self.X.shape[0]

    @property
    def B(self) -> int:
        return self.X.shape[2]


def junction_candidates(model: JunctionInputs, *, trace: bool = False) -> list[tuple[int, int, int, int]] | PhaseTrace:
    """(samples, edge bin j, a, b), strongest first (ties: later edge, then larger labels first): bin edges where
    free or not-a-copy tracks of at least ``junction_min_samples`` distinct samples switch between labels a and b
    (a copied homologue's switch is its parent's or a crossover, so it is not counted)."""
    if trace:
        audit = DecisionTrace(extended=True)
        value = junction_candidates(replace(model, audit=audit))
        return PhaseTrace(value, tuple(audit.rows))
    X, V, par = model.X, model.V, model.par
    who: dict[tuple[int, int, int], set[int]] = {}
    for n in range(model.N):
        for k in range(2):
            x = X[n, k]
            for j in (int(value) + 1 for value in np.flatnonzero(x[1:] != x[:-1])):
                if par[n, k] >= 0 and V[n, k, j] < 2 and V[n, k, j - 1] < 2:
                    continue
                a, b = int(x[j - 1]), int(x[j])
                who.setdefault((int(j), min(a, b), max(a, b)), set()).add(n)
    if model.audit is not None and model.audit.extended:
        for samples in who.values():
            model.audit.difference(
                "junction_count",
                len(samples),
                model.config.junction_min_samples,
                len(samples) >= model.config.junction_min_samples,
            )
    return sorted(
        ((len(s), j, a, b) for (j, a, b), s in who.items() if len(s) >= model.config.junction_min_samples),
        reverse=True,
    )


def withheld_joins(
    model: JunctionInputs, candidates: list[tuple[int, int, int, int]], *, trace: bool = False
) -> dict[int, list[int]] | PhaseTrace:
    """{sample: sorted bins}: at each candidate junction (j, a, b), the samples whose two homologues carry a and b
    on both sides of edge j and whose relative phase across j is not pinned. Their phase across j depends on
    whether the two rows continue there, and a count of relatives is no evidence of that (a relative without
    reads on one flank pins nothing). A sample is pinned only when a linked homologue copies the same parental
    homologue on both sides of j and the parent's own reads pin that homologue's content against every pair
    alternative (``homologue_pinned``); the sample's own pair then fixes its other homologue. Children are not
    used: a child's copy rules out the alternative only through a meiosis switch at exactly edge j, whose odds
    depend on the interval (none at a protected reset) and on uncertainty the children share."""
    if trace:
        audit = DecisionTrace(extended=True)
        value = withheld_joins(replace(model, audit=audit), candidates)
        return PhaseTrace(value, tuple(audit.rows))
    X, V, par = model.X, model.V, model.par
    out: dict[int, set[int]] = {}
    for _, j, a, b in candidates:
        pair = {a, b}
        for n in range(model.N):
            if {int(X[n, 0, j - 1]), int(X[n, 1, j - 1])} != pair or {int(X[n, 0, j]), int(X[n, 1, j])} != pair:
                continue
            pinned = False
            for k in range(2):
                p, h = int(par[n, k]), int(V[n, k, j])
                if p >= 0 and h < 2 and int(V[n, k, j - 1]) == h and homologue_pinned(model, p, h, j, a, b):
                    pinned = True
                    break
            if not pinned:
                out.setdefault(n, set()).add(int(j))
    return {n: sorted(v) for n, v in out.items()}


def homologue_pinned(model: JunctionInputs, p: int, h: int, j: int, a: int, b: int) -> bool:
    """Whether homologue h of sample p has known content across edge j from p's own emissions, against the
    exchange of labels a and b (``_flanks_pinned``: the other junction label on h, with any partner). A flank
    where it carries neither label is unaffected by the junction; a carrier of both labels is never pinned (its
    pair is symmetric under the exchange)."""
    X = model.X
    swap = {a: b, b: a}
    for t in (j - 1, j):
        if {int(X[p, h, t]), int(X[p, 1 - h, t])} == {a, b}:
            return False
    return _flanks_pinned(model, p, h, j, (swap, swap), unaffected_passes=True)


def _flanks_pinned(
    model: JunctionInputs,
    p: int,
    h: int,
    j: int,
    alternatives: tuple[dict[int, int], dict[int, int]],
    unaffected_passes: bool,
) -> bool:
    """Both flanks of edge j: p's pair (x, y), with x on homologue h at the flank's first bin, beats every pair
    that puts the alternative label z = alternatives[flank][x] on homologue h, whatever the partner (per bin, the
    best partner label), over up to ``junction_pin_bins`` bins while p's labels stay put, by at least
    ``junction_pin_nats``. The partner is free because a pair alternative can fit the same unphased reads (two
    rows exchanged together): holding it fixed would mistake that ambiguity for evidence. A flank with no
    alternative to x (x not in the map, or mapped to itmodel) passes when ``unaffected_passes`` and fails
    otherwise."""
    X, E, B = model.X, model.E, model.B
    window = model.config.junction_pin_bins
    flanks = ((j - 1, max(-1, j - 1 - window), -1), (j, min(B, j + window), 1))
    for (start, stop, step), alternative in zip(flanks, alternatives, strict=False):
        x, y = int(X[p, h, start]), int(X[p, 1 - h, start])
        z = alternative.get(x, x)
        if z == x:
            if unaffected_passes:
                continue
            return False
        llr = 0.0
        for t in range(start, stop, step):
            if int(X[p, h, t]) != x or int(X[p, 1 - h, t]) != y:
                break
            llr += E[p, x, y, t] - E[p, z, :, t].max()
        if model.audit is not None:
            model.audit.difference("pin", llr, model.config.junction_pin_nats, llr >= model.config.junction_pin_nats)
        if llr < model.config.junction_pin_nats:
            return False
    return True
