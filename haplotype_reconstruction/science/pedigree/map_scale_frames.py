"""Build the per-sib transmission frames of the map-scale estimate from released paintings (rev 5, DESIGN_MAPSCALE).

For each relatedness family of at least two members and each painting component, everything about the parents comes
from the OTHER carriers (members, plus the family's sequenced slot samples): a bin is a frame of target member i when
the others show four parental classes with a unique 2 + 2 split (map_scale_kernels.sib_frames); the frame's emission
is the painting's own pair emission of the target for the four full-sib configurations (power 1), with the misfit
column the mean over all packed state pairs. A relationship guard removes members whose evidence is not full-sib
like before the final frames are built:

- pair test: the full-sib IBD-count chain of a member pair must beat the half-sib and parent-offspring chains by
  ``guard_nats`` (each at its own best rate and error, summed over contigs); a member is kept when it passes with at
  least half of the others it was tested with (and at least one);
- target test (iterative): a target whose homologue tracks may leave its parents' classes (entry e_in, exit e_out)
  improves on the full-sib chain at the family's full-sib (r, delta) by more than ``guard_nats`` with a posterior
  non-full-sib frame share of at least ``target_share``: the largest such member of each family is removed (as
  target and carrier) and the frames are rebuilt, while families keep more than two members. The parental homologue
  switches follow the same Haldane process in both models, so crossovers cancel: the test does not select on a sib's
  crossover count.

Method: docs/methods/pedigree.md#map-scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..painting import api as painting_hmm
from .map_scale_kernels import (
    frame_loglik,
    pair_alternative_loglik,
    pair_fullsib_loglik,
    pair_observations,
    sib_frames,
    switch_candidates,
    target_loglik,
    target_posterior,
)
from .parameters import MapScaleParameters

__all__ = ["ComponentInput", "Families", "Frames", "component_input", "families_of", "relevant_samples", "guard",
           "build_frames", "argmax_refined", "packed_emissions", "file_frames"]

# painting.TrackStatus values (contracts.painting).
_NAMED, _POOLED, _LOCAL = 2, 3, 6
# The four full-sib configurations of the 25 track-symbol pairs (5 s1 + s2: s1 = h0, s2 = 2 + h1), then misfit.
_FULL_SIB_COLUMNS = (2, 3, 7, 8, 25)
_CHUNK = 4000  # frames per block of the 25-symbol emissions (execution grain only)


@dataclass(frozen=True)
class ComponentInput:
    """One painting component restricted to the relevant samples R (members and slots of the families).

    centers float64 [B] bin centres (bp); cls / key int32 [R, 2, B] firm (named or pooled) classes and their bin keys,
    -1 elsewhere; keys int32 [R, 2, B] bin-local keys (firm or local-allele equivalence), -1 when missing; cand1 /
    cand2 int32 [R, 2, B] switch-interval flank keys; packed float64 [R, S (S + 1) / 2, B] the painting's binned log
    emissions (upper triangle); n_states S."""

    contig: int
    centers: np.ndarray
    cls: np.ndarray
    key: np.ndarray
    keys: np.ndarray
    cand1: np.ndarray
    cand2: np.ndarray
    packed: np.ndarray
    n_states: int


@dataclass(frozen=True)
class Families:
    """Relatedness families of at least two members, in local row indices (ascending), with their sequenced slot rows
    (slot order); a slot carries parental classes only while it is not an active member."""

    members: Mapping[int, tuple[int, ...]]
    slots: Mapping[int, tuple[int, ...]]


@dataclass
class Frames:
    """Frame sequences (one per target and linked parental chain within a component): contig, family, target row,
    positions (bp) and emissions em [n, 5]."""

    contig: list[int] = field(default_factory=list)
    family: list[int] = field(default_factory=list)
    target: list[int] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    emissions: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.positions)

    def stacked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        offsets = np.zeros(len(self.positions) + 1, np.int64)
        offsets[1:] = np.cumsum([len(p) for p in self.positions])
        pos = np.ascontiguousarray(np.concatenate(self.positions), dtype=np.float64)
        em = np.ascontiguousarray(np.concatenate(self.emissions), dtype=np.float64)
        return offsets, pos, em

    def subset(self, keep: np.ndarray) -> "Frames":
        index = np.flatnonzero(keep)
        return Frames([self.contig[k] for k in index], [self.family[k] for k in index],
                      [self.target[k] for k in index], [self.positions[k] for k in index],
                      [self.emissions[k] for k in index])


def families_of(family: np.ndarray, parents: np.ndarray, relevant: np.ndarray) -> Families:
    """Families of >= 2 members from the relatedness call (``family`` per sample, ``parents`` [F, 2] slot samples or
    -1), in rows of ``relevant`` (ascending sample indices, members and slots included)."""
    local = {int(s): k for k, s in enumerate(relevant)}
    groups: dict[int, list[int]] = {}
    for sample, f in enumerate(np.asarray(family)):
        groups.setdefault(int(f), []).append(sample)
    members, slots = {}, {}
    for f, samples in sorted(groups.items()):
        if len(samples) < 2:
            continue
        members[f] = tuple(sorted(local[s] for s in samples))
        slots[f] = tuple(local[int(x)] for x in parents[f] if int(x) >= 0 and int(x) in local)
    return Families(members, slots)


def relevant_samples(family: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Ascending sample indices of every member of a family of >= 2 and of those families' sequenced slots."""
    family = np.asarray(family)
    counts = np.bincount(family, minlength=len(parents)) if len(family) else np.zeros(0, np.int64)
    keep = set(np.flatnonzero(counts[family] >= 2).tolist())
    for f in np.flatnonzero(counts >= 2):
        keep.update(int(x) for x in parents[f] if int(x) >= 0)
    return np.asarray(sorted(keep), dtype=np.int64)


def component_input(contig: int, centers, local_keys, eligible, classes, status, n_classes: int, n_states: int,
                    packed, rows: np.ndarray, *, packed_rows: np.ndarray | None = None) -> ComponentInput:
    """Restrict one painting component to ``rows`` (relevant samples, ascending) and derive its track keys;
    ``packed_rows``: the rows of ``packed`` holding those samples (default ``rows``: the painting's full cache)."""
    status = np.array(np.asarray(status)[rows], dtype=np.uint8, order="C")
    classes = np.array(np.asarray(classes)[rows], dtype=np.int32, order="C")
    eligible = np.array(np.asarray(eligible)[rows], dtype=np.bool_, order="C")
    table = np.array(local_keys, dtype=np.int32, order="C")
    B = status.shape[2]
    b_idx = np.broadcast_to(np.arange(B)[None, None, :], status.shape)
    looked = table[b_idx, np.clip(classes, 0, table.shape[1] - 1)]
    stage1 = (status == _NAMED) | (status == _POOLED)
    # firm classes and their keys (sib frames)
    firm = stage1 & (classes >= 0) & (classes < n_classes) & eligible[:, None, None] & (looked < n_classes)
    cls = np.where(firm, classes, -1).astype(np.int32)
    key = np.where(firm, looked, -1).astype(np.int32)
    # bin-local keys of every track (pair observations): firm, or the local-allele equivalence key
    keys = np.full(status.shape, -1, np.int32)
    keys[stage1] = looked[stage1]
    loc = status == _LOCAL
    keys[loc] = classes[loc]
    keys[keys >= n_classes] = -1
    keys[~eligible] = -1
    cand1, cand2 = switch_candidates(status, classes, table, int(n_classes), eligible)
    return ComponentInput(contig, np.array(centers, dtype=np.float64, order="C"), np.ascontiguousarray(cls),
                          np.ascontiguousarray(key), np.ascontiguousarray(keys), cand1, cand2,
                          np.array(np.asarray(packed)[rows if packed_rows is None else packed_rows],
                                   dtype=np.float64, order="C"), int(n_states))


def packed_emissions(
    component, painting_parameters, robustness_epsilon, evidence, rows: np.ndarray, site_correlation: float
) -> np.ndarray:
    """Packed binned log emissions [len(rows), S (S + 1) / 2, B] of ``rows`` at ``site_correlation``, recomputed from
    the contig evidence as painting computes them (binned emissions over the component's selected sites, divided by
    the design effect), when the painting's own cache is absent or at another strength (prepare.path_correlation)."""
    states, bins = component.states, component.bins
    S = int(states.n_states)
    indices = np.searchsorted(evidence.positions, states.positions)
    if np.any(indices >= len(evidence.positions)) or np.any(evidence.positions[np.minimum(indices, len(evidence.positions) - 1)] != states.positions):
        raise ValueError("painted sites are missing from the evidence record")
    sites = indices.astype(np.int64)
    alt = states.alt_probability()
    pair = _pair_index(np.arange(S)[:, None], np.arange(S)[None, :], S)
    upper = np.triu_indices(S)
    out = np.empty((len(rows), S * (S + 1) // 2, bins.n_bins))
    for k, row in enumerate(rows):
        gl, observed = painting_hmm.gather_sites(evidence.likelihoods, evidence.observed, sites,
                                                 slice(int(row), int(row) + 1))
        em = painting_hmm.binned_emissions(gl, observed, alt, states.background_frequency, bins.starts, bins.stops,
                                           robustness_epsilon=robustness_epsilon,
                                           log_floor=painting_parameters.emission_log_floor)
        em /= painting_hmm.design_effect(
            gl, observed, bins.starts, site_correlation,
            painting_parameters.reference_snps_per_bin // painting_parameters.snps_per_bin,
        )[:, None, :]
        full = em.reshape(1, S, S, -1)[0]
        out[k, pair[upper]] = full[upper]
    return out


def file_frames(frames: Frames, maps, free: set[int]) -> Frames:
    """Frames of the map-file contigs inside their files' knots, with positions on the file map as pseudo-bp
    (cM x 1e6), so that the chain's rate in cM/Mb reads the ratio of the painted crossover rate to the file map's."""
    out = Frames()
    for c, f, t_, pos, em in zip(frames.contig, frames.family, frames.target, frames.positions, frames.emissions,
                                 strict=True):
        if c in free:
            continue
        chromosome = maps[c]
        inside = (pos >= chromosome.positions_bp[0]) & (pos <= chromosome.positions_bp[-1])
        if inside.sum() < 2:
            continue
        out.contig.append(c)
        out.family.append(f)
        out.target.append(t_)
        out.positions.append(np.asarray(chromosome.cumulative_morgans(pos[inside]), dtype=np.float64) * 100.0 * 1e6)
        out.emissions.append(em[inside])
    return out


def argmax_refined(table: np.ndarray, grid: np.ndarray) -> tuple[float, int, bool]:
    """(estimate, nuisance flat index, at a grid edge) of a [R, ...] log-likelihood table: the best cell, refined by
    the parabola through its log-r neighbours at the best nuisance (none at an edge or for a non-concave triple)."""
    t2 = table.reshape(table.shape[0], -1)
    ri, ni = np.unravel_index(int(np.argmax(t2)), t2.shape)
    if ri == 0 or ri == len(grid) - 1:
        return float(grid[ri]), int(ni), True
    y0, y1, y2 = t2[ri - 1, ni], t2[ri, ni], t2[ri + 1, ni]
    den = y0 - 2 * y1 + y2
    shift = 0.0 if den >= 0 else 0.5 * (y0 - y2) / den
    lg = np.log(grid)
    return float(np.exp(lg[ri] + shift * (lg[1] - lg[0]))), int(ni), False


def _slots_of(families: Families, f: int, active: set[int]) -> np.ndarray:
    return np.asarray([s for s in families.slots[f] if s not in active], dtype=np.int64)


def _pair_index(s1, s2, S):
    lo, hi = np.minimum(s1, s2), np.maximum(s1, s2)
    return lo * S - lo * (lo - 1) // 2 + (hi - lo)


def _emission_columns(packed: np.ndarray, rows: np.ndarray, fb: np.ndarray, flab: np.ndarray, S: int, want25: bool):
    """Per frame: em5 [n, 5] (the four full-sib configurations as ratios to their maximum, misfit = mean over all
    packed pairs) and, when want25, em25 [n, 26] (the 25 track-symbol pairs including 'other class', + misfit)."""
    n = len(rows)
    em4 = np.empty((n, 4))
    for k, (u, v) in enumerate(((0, 2), (0, 3), (1, 2), (1, 3))):
        em4[:, k] = packed[rows, _pair_index(flab[:, u], flab[:, v], S), fb]
    mx = em4.max(axis=1, keepdims=True)
    full = packed[rows, :, fb]
    other = np.exp(np.minimum(full - mx, 300.0)).mean(axis=1)
    em5 = np.concatenate([np.exp(em4 - mx), other[:, None]], axis=1)
    if not want25:
        return em5, None
    em25 = np.empty((n, 26))
    em25[:, 25] = other
    states = np.arange(S)
    out_mask = np.ones((n, S), bool)
    for j in range(4):
        out_mask[np.arange(n), flab[:, j]] = False
    nout = out_mask.sum(axis=1)
    pair = _pair_index(states[:, None], states[None, :], S)
    for c0 in range(0, n, _CHUNK):
        sl = slice(c0, min(n, c0 + _CHUNK))
        mat = np.exp(np.minimum(full[sl][:, pair] - mx[sl][:, :, None], 300.0))
        om = out_mask[sl]
        no = np.maximum(nout[sl], 1)
        lab = flab[sl]
        rr = np.arange(mat.shape[0])
        for s1 in range(5):
            for s2 in range(5):
                if s1 < 4 and s2 < 4:
                    em25[sl, 5 * s1 + s2] = mat[rr, lab[:, s1], lab[:, s2]]
                elif s1 < 4:
                    em25[sl, 5 * s1 + s2] = (mat[rr, lab[:, s1], :] * om).sum(axis=1) / no
                elif s2 < 4:
                    em25[sl, 5 * s1 + s2] = (mat[rr, lab[:, s2], :] * om).sum(axis=1) / no
                else:
                    em25[sl, 24] = ((mat * om[:, :, None] * om[:, None, :]).sum(axis=(1, 2))) / (no * no)
        zero = np.flatnonzero(nout[sl] == 0) + c0
        if len(zero):
            for s1 in range(5):
                em25[zero, 5 * s1 + 4] = 0.0
                em25[zero, 5 * 4 + s1] = 0.0
    return em5, em25


def build_frames(components: Sequence[ComponentInput], families: Families, active: Mapping[int, set[int]],
                 want25: bool) -> tuple[Frames, list[np.ndarray]]:
    """Frame sequences of the active members (and their 25-symbol emissions when ``want25``), components in order."""
    frames = Frames()
    em25s: list[np.ndarray] = []
    # families in the order of their smallest active member (the painting's sample order)
    family_order = sorted((f for f in families.members if len(active.get(f, ())) >= 2), key=lambda f: min(active[f]))
    for comp in components:
        for f in family_order:
            mem = sorted(active[f])
            members = np.asarray(mem, dtype=np.int64)
            slots = _slots_of(families, f, set(mem))
            ft, fc, fb, flab = sib_frames(comp.cls, comp.key, members, slots, comp.centers)
            if not len(ft):
                continue
            order = np.lexsort((fb, fc))
            ft, fc, fb, flab = ft[order], fc[order], fb[order], flab[order]
            em5, em25 = _emission_columns(comp.packed, members[ft], fb, flab, comp.n_states, want25)
            starts = np.r_[0, np.flatnonzero(np.diff(fc)) + 1, len(fc)]
            for a, z in zip(starts[:-1], starts[1:]):
                if z - a < 2:
                    continue
                frames.contig.append(comp.contig)
                frames.family.append(int(f))
                frames.target.append(int(members[ft[a]]))
                frames.positions.append(comp.centers[fb[a:z]])
                frames.emissions.append(em5[a:z])
                if want25:
                    em25s.append(em25[a:z])
    return frames, em25s


@dataclass(frozen=True)
class GuardResult:
    """Members kept per family, and the guard's log: pair-test exclusions and target-test removals."""

    active: dict[int, set[int]]
    excluded_pair: tuple[int, ...]
    excluded_target: tuple[int, ...]
    target_log: tuple[tuple[int, int, int, float, float], ...]  # (iteration, family, row, LR, share)
    pair_tests: dict[int, tuple[int, int]]  # row -> (passed, tested)


def _pair_test(components: Sequence[ComponentInput], families: Families, p: MapScaleParameters):
    """Summed full-sib, half-sib and parent-offspring pair tables over components; the per-member pass counts."""
    r_coarse = np.asarray(p.coarse_rate_grid(), dtype=np.float64)
    eps = np.asarray(p.pair_error_rates, dtype=np.float64)
    tables: dict[tuple[int, int], list[np.ndarray]] = {}
    for comp in components:
        B = len(comp.centers)
        for f, mem in families.members.items():
            members = np.asarray(mem, dtype=np.int64)
            slots = _slots_of(families, f, set(mem))
            pos, obs, lens = pair_observations(comp.keys, comp.cand1, comp.cand2, members, slots, comp.centers)
            sel = []
            q = 0
            for a in range(len(members)):
                for c in range(a + 1, len(members)):
                    if lens[q] >= 2:
                        sel.append(((int(members[a]), int(members[c])), q * B, int(lens[q])))
                    q += 1
            if not sel:
                continue
            offsets = np.zeros(len(sel) + 1, np.int64)
            offsets[1:] = np.cumsum([n for _, _, n in sel])
            P_ = np.ascontiguousarray(np.concatenate([pos[st:st + n] for _, st, n in sel]))
            O_ = np.ascontiguousarray(np.concatenate([obs[st:st + n] for _, st, n in sel]).astype(np.int64))
            fs = pair_fullsib_loglik(offsets, P_, O_, r_coarse, eps)
            hs, po = pair_alternative_loglik(offsets, P_, O_, r_coarse, eps)
            for k, (pair, _, _) in enumerate(sel):
                acc = tables.setdefault(pair, [np.zeros(fs.shape[1:]), np.zeros(hs.shape[1:]), np.zeros(po.shape[1:])])
                acc[0] += fs[k]
                acc[1] += hs[k]
                acc[2] += po[k]
    lr = {pair: float(t[0].max() - max(t[1].max(), t[2].max())) for pair, t in tables.items()}
    excluded, counts = set(), {}
    for f, mem in families.members.items():
        for x in mem:
            tests = []
            for y in mem:
                if y == x:
                    continue
                v = lr.get((x, y), lr.get((y, x)))
                tests.append(None if v is None else v >= p.guard_nats)
            passed = sum(1 for t in tests if t)
            tested = sum(1 for t in tests if t is not None)
            if tested == 0 or passed < max(1, int(np.ceil(0.5 * tested - 1e-9))):
                excluded.add(x)
            counts[x] = (passed, tested)
    return excluded, counts


def guard(components: Sequence[ComponentInput], families: Families, p: MapScaleParameters) -> GuardResult:
    """The relationship guard: pair test, then the iterative target test (module docstring)."""
    excluded_pair, counts = _pair_test(components, families, p)
    active = {f: set(x for x in mem if x not in excluded_pair) for f, mem in families.members.items()}
    r_coarse = np.asarray(p.coarse_rate_grid(), dtype=np.float64)
    deltas = np.asarray(p.misfit_weights, dtype=np.float64)
    ein = np.asarray(p.target_entry_rates, dtype=np.float64)
    eout = np.asarray(p.target_exit_rates, dtype=np.float64)
    removed, log = [], []
    for iteration in range(p.target_iterations):
        frames, em25s = build_frames(components, families, active, True)
        if not len(frames):
            break
        offsets = np.zeros(len(frames) + 1, np.int64)
        offsets[1:] = np.cumsum([len(x) for x in frames.positions])
        pos = np.ascontiguousarray(np.concatenate(frames.positions))
        em25 = np.ascontiguousarray(np.concatenate(em25s))
        em5 = np.ascontiguousarray(em25[:, list(_FULL_SIB_COLUMNS)])
        ll4 = frame_loglik(offsets, pos, em5, r_coarse, deltas)
        famk = np.asarray(frames.family)
        names = np.asarray(frames.target)
        changed = False
        for f in sorted(set(famk.tolist())):
            mf = famk == f
            rf, di, _ = argmax_refined(ll4[mf].sum(axis=0), r_coarse)
            idx = np.flatnonzero(mf)
            sub_off = np.zeros(len(idx) + 1, np.int64)
            sub_off[1:] = np.cumsum([offsets[k + 1] - offsets[k] for k in idx])
            sel = np.concatenate([np.arange(offsets[k], offsets[k + 1]) for k in idx])
            rl = target_loglik(sub_off, np.ascontiguousarray(pos[sel]), np.ascontiguousarray(em25[sel]), float(rf),
                               float(deltas[di]), ein, eout)
            tn = names[idx]
            best = (None, -np.inf, 0.0)
            for nm in sorted(set(tn.tolist())):
                m = tn == nm
                tab = rl[m].sum(axis=0)
                lr = float(tab.max() - tab[0, 0])
                ii, jj = np.unravel_index(int(np.argmax(tab)), tab.shape)
                share = 0.0
                if lr > p.guard_nats:
                    sel_t = np.concatenate([np.arange(offsets[k], offsets[k + 1]) for k in idx[m]])
                    so = np.zeros(int(m.sum()) + 1, np.int64)
                    so[1:] = np.cumsum([offsets[k + 1] - offsets[k] for k in idx[m]])
                    share = float(np.mean(target_posterior(so, np.ascontiguousarray(pos[sel_t]),
                                                           np.ascontiguousarray(em25[sel_t]), float(rf),
                                                           float(deltas[di]), float(ein[ii]), float(eout[jj]))))
                if lr > best[1] and share >= p.target_share:
                    best = (nm, lr, share)
            if best[0] is not None and best[1] > p.guard_nats and len(active[f]) > 2:
                active[f].discard(best[0])
                removed.append(int(best[0]))
                log.append((iteration, int(f), int(best[0]), round(best[1], 1), round(best[2], 4)))
                changed = True
        if not changed:
            break
    return GuardResult(active, tuple(sorted(excluded_pair)), tuple(sorted(removed)), tuple(log), counts)
