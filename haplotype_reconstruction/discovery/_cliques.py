"""Exact compiled pivoted Bron--Kerbosch traversal on packed adjacency rows.

The first pass counts actual maximal cliques, stopping only at limit+1. Thus an
oversized component is omitted whole, independently of labels/traversal order.
Only an admitted component is materialized, in a second identical traversal.
No search-work or wall-time cutoff is a scientific decision.
"""
from contextlib import nullcontext
import numpy as np
from numba import njit


# The block's 512 MiB base/workspace already covers small hierarchy/clique scratch.
# Reserve additional storage only above 1 MiB, and parallelize components only
# above 4096 estimated traversal steps. These are execution thresholds only:
# admission, identity, size classes and omissions never consult them.
_INLINE_BYTES = 1 << 20
_INLINE_WORK = 4096


def _workspace(nbytes, label):
    if nbytes <= _INLINE_BYTES:
        return nullcontext()
    from ..execution import reserve_memory
    return reserve_memory(nbytes, label=label)


def _component_map(function, items, sizes, work, label):
    """Keep tiny tasks inline; schedule substantial independent work on live slots."""
    results = [None] * len(items)
    pending = []
    for position, item in enumerate(items):
        if work[position] <= _INLINE_WORK and sizes[position] <= _INLINE_BYTES:
            results[position] = function(item)
        else:
            pending.append(position)
    if not pending:
        return results
    from ..execution.budget import current_budget
    budget = current_budget()
    if len(pending) == 1 or budget.threads() <= 1:
        for position in pending:
            with _workspace(sizes[position], label):
                results[position] = function(items[position])
    else:
        # item_bytes accounts for simultaneously live scratch/decoded outputs;
        # callers sort these substantial tasks largest first.
        values = budget.thread_map(function, [items[i] for i in pending],
            item_bytes=[sizes[i] for i in pending], label=label)
        for position, value in zip(pending, values):
            results[position] = value
    return results


@njit(cache=True, inline='always')
def _count(word):
    count = 0
    while word:
        word &= word - np.uint64(1)
        count += 1
    return count


@njit(cache=True, inline='always')
def _low_index(word):
    # Binary search for the trailing set bit (word is nonzero).
    offset = 0
    if (word & np.uint64(0xffffffff)) == 0:
        offset += 32; word >>= np.uint64(32)
    if (word & np.uint64(0xffff)) == 0:
        offset += 16; word >>= np.uint64(16)
    if (word & np.uint64(0xff)) == 0:
        offset += 8; word >>= np.uint64(8)
    if (word & np.uint64(0xf)) == 0:
        offset += 4; word >>= np.uint64(4)
    if (word & np.uint64(3)) == 0:
        offset += 2; word >>= np.uint64(2)
    if (word & np.uint64(1)) == 0:
        offset += 1
    return offset


@njit(cache=True, nogil=True)
def _walk(adjacent, n, limit, output, histogram=None, allowed=None, memberships=None):
    words = adjacent.shape[1]
    candidates = np.zeros((n+1, words), dtype=np.uint64)
    excluded = np.zeros_like(candidates)
    todo = np.zeros_like(candidates)
    chosen = np.empty(n, dtype=np.int64)
    lifted = np.zeros((n+1, 0 if memberships is None else memberships.shape[1]), dtype=np.uint64)
    for v in range(n):
        candidates[0, v//64] |= np.uint64(1) << np.uint64(v%64)
    depth = 0
    enter = True
    emitted = 0
    retained = 0
    work = 0
    while depth >= 0:
        if enter:
            work += 1
            any_p = False
            any_x = False
            for w in range(words):
                any_p = any_p or candidates[depth,w] != 0
                any_x = any_x or excluded[depth,w] != 0
            if not any_p:
                if not any_x:
                    if emitted == limit:
                        return emitted+1, work
                    if histogram is not None:
                        histogram[depth] += 1
                    if len(output) and (allowed is None or allowed[depth]):
                        if memberships is not None:
                            for w in range(lifted.shape[1]):
                                output[retained,w] = lifted[depth,w]
                        else:
                            for w in range(words):
                                output[retained,w] = np.uint64(0)
                            for i in range(depth):
                                v = chosen[i]
                                output[retained,v//64] |= np.uint64(1) << np.uint64(v%64)
                        retained += 1
                    emitted += 1
                depth -= 1
                enter = False
                continue
            best = -1
            pivot = -1
            for w in range(words):
                union = candidates[depth,w] | excluded[depth,w]
                while union:
                    bit = _low_index(union)
                    v = w*64+bit
                    union &= union-np.uint64(1)
                    score = 0
                    for q in range(words):
                        score += _count(candidates[depth,q] & adjacent[v,q])
                    if score > best:
                        best = score; pivot = v
            for w in range(words):
                todo[depth,w] = candidates[depth,w] & ~adjacent[pivot,w]
        v = -1
        for w in range(words):
            if todo[depth,w]:
                bit = _low_index(todo[depth,w])
                v = w*64+bit
                todo[depth,w] &= todo[depth,w]-np.uint64(1)
                break
        if v < 0:
            depth -= 1
            enter = False
            continue
        for w in range(words):
            candidates[depth+1,w] = candidates[depth,w] & adjacent[v,w]
            excluded[depth+1,w] = excluded[depth,w] & adjacent[v,w]
        if memberships is not None:
            for w in range(lifted.shape[1]):
                lifted[depth+1,w] = lifted[depth,w] | memberships[v,w]
        chosen[depth] = v
        mask = np.uint64(1) << np.uint64(v%64)
        candidates[depth,v//64] &= ~mask
        excluded[depth,v//64] |= mask
        depth += 1
        enter = True
    return emitted, work


def maximal_cliques(vertices, adjacent, limit):
    """Complete clique list or None iff actual output exceeds limit; exact work count."""
    vertices = tuple(sorted(vertices))
    n = len(vertices)
    packed = _packed_graph(vertices, adjacent)
    count, work = _walk(packed, n, int(limit), np.empty((0,packed.shape[1]),dtype=np.uint64))
    if count > limit:
        return None, int(work)
    output = np.empty((count,packed.shape[1]),dtype=np.uint64)
    emitted, second_work = _walk(packed,n,int(limit),output)
    assert emitted == count
    # Decode by set bits, with O(total emitted clique size) Python work.
    result=[]
    for row in output:
        members=[]
        for w,value in enumerate(row):
            value=int(value)
            while value:
                bit=value & -value
                members.append(vertices[64*w+bit.bit_length()-1])
                value ^= bit
        result.append(tuple(members))
    return result, int(work+second_work)


def screened_components(components, adjacent, limit, cap, omissions=None):
    """Exact ordinary-size classes, with independent component work on the live budget."""
    from .residual_births import _clique_plan
    from .search import diagnostic_phase
    prepared, pending = [], []
    with diagnostic_phase('clique_counting'):
        for vertices in components:
            vertices = tuple(sorted(vertices))
            plan, _, estimate = _clique_plan(vertices, adjacent)
            if plan[0] == 'complete':
                histogram = np.zeros(len(vertices)+1, dtype=np.int64)
                histogram[len(vertices)] = 1
                prepared.append([vertices, None, histogram, 1, 1, estimate])
            else:
                prepared.append([vertices, _packed_graph(vertices, adjacent), None, None, None, estimate])
                pending.append(len(prepared)-1)
        pending.sort(key=lambda index: prepared[index][5], reverse=True)

        def count_one(index):
            vertices, packed = prepared[index][:2]
            histogram = np.zeros(len(vertices)+1, dtype=np.int64)
            count, calls = _walk(packed, len(vertices), int(limit),
                                np.empty((0, packed.shape[1]), dtype=np.uint64), histogram)
            return histogram, int(count), int(calls)

        sizes = [64*(len(prepared[i][0])+1)*prepared[i][1].shape[1] for i in pending]
        counted = _component_map(count_one, pending, sizes,
            [prepared[i][5] for i in pending], 'residual clique counting')
        for index, result in zip(pending, counted):
            prepared[index][2:5] = result
    counts, admitted, work, enumerated = {}, [], 0, True
    for vertices, packed, histogram, count, calls, _ in prepared:
        work += calls
        if count > limit:
            enumerated = False
            if omissions is not None:
                omissions.append(dict(cause='residual_clique_enumeration_tie_class_at_budget',
                    kind='tie_class_at_budget', count=1,
                    scope=f'component with {len(vertices)} records; more than {limit} maximal cliques',
                    count_kind='exact'))
            continue
        admitted.append((vertices, packed, histogram, count))
        for size in range(2, len(histogram)):
            if histogram[size]:
                counts[size] = counts.get(size, 0)+int(histogram[size])
    with diagnostic_phase('lifting_budgeting'):
        accepted, capped = _admitted_sizes(counts, int(cap), omissions)
        outputs, pending, sizes = {}, [], []
        for index, (vertices, packed, histogram, count) in enumerate(admitted):
            retained = sum(int(histogram[size]) for size in accepted if size < len(histogram))
            if not retained:
                continue
            if count == 1:
                outputs[index] = [vertices]
                work += 1
            else:
                pending.append(index)
        # Once counted, actual retained work determines materialisation order.
        pending.sort(key=lambda index: admitted[index][3], reverse=True)
        for index in pending:
            vertices, packed, histogram, _ = admitted[index]
            retained = sum(int(histogram[size]) for size in accepted if size < len(histogram))
            members = sum(size*int(histogram[size]) for size in accepted if size < len(histogram))
            sizes.append(64*(len(vertices)+1)*packed.shape[1] + 8*retained*packed.shape[1]
                         + 64*retained + 44*members)

        def materialise_one(index):
            vertices, packed, histogram, _ = admitted[index]
            retained = sum(int(histogram[size]) for size in accepted if size < len(histogram))
            allowed = np.array([size in accepted for size in range(len(histogram))], dtype=np.bool_)
            output = np.empty((retained, packed.shape[1]), dtype=np.uint64)
            _, calls = _walk(packed, len(vertices), int(limit), output, None, allowed)
            groups = [tuple(vertices[local] for local in _decode_members(row)) for row in output]
            return groups, int(calls)

        materialised = _component_map(materialise_one, pending, sizes,
            [len(admitted[i][0])*admitted[i][3] for i in pending], 'residual admitted groups')
        for index, (groups, calls) in zip(pending, materialised):
            outputs[index] = groups
            work += calls
    groups = [group for index in sorted(outputs) for group in outputs[index]]
    return groups, capped, enumerated, int(work), sum(counts.values())


def _packed_graph(vertices, adjacent):
    slots = {vertex: index for index, vertex in enumerate(vertices)}
    packed = np.zeros((len(vertices), (len(vertices)+63)//64), dtype=np.uint64)
    # Python integers accumulate a whole row; NumPy scalar operations per edge
    # dominate small dense components despite their tiny actual clique output.
    mask = (1 << 64)-1
    for index, vertex in enumerate(vertices):
        row = 0
        for other in adjacent[vertex]:
            local = slots.get(other)
            if local is not None:
                row |= 1 << local
        for word in range(packed.shape[1]):
            packed[index, word] = row & mask
            row >>= 64
    return packed


def _has_prime(plan):
    return plan[0] == 'prime' or any(_has_prime(child) for child in plan[2])


def _count_plan(plan, adjacent, limit):
    """Exact sum/product counts; only prime factors need traversal, never tuples."""
    kind, vertices, children = plan
    if kind == 'complete':
        return 1, 1
    if children:
        total, work = (0 if kind == 'union' else 1), 1
        for child in children:
            remaining = limit-total if kind == 'union' else limit//total
            count, calls = _count_plan(child, adjacent, remaining)
            work += calls
            if count > remaining:
                return limit+1, work
            total = total+count if kind == 'union' else total*count
        return total, work
    packed = _packed_graph(vertices, adjacent)
    return _walk(packed, len(vertices), limit,
                 np.empty((0, packed.shape[1]), dtype=np.uint64))


def _count_components(components, adjacent, limit):
    """Return actual component admission counts in input order; expensive work first."""
    from .residual_births import _clique_plan
    prepared, pending = [], []
    for vertices in components:
        vertices = tuple(sorted(vertices))
        plan, bound, estimate = _clique_plan(vertices, adjacent)
        # With no prime factor, the recursively summed/multiplied count is exact.
        # The bound of a prime factor is NEVER a component-admission criterion.
        count = None if _has_prime(plan) else min(bound, limit+1)
        # Complete components and exact factored overflows need no adjacency
        # packing/traversal. Avoid quadratic Python packing for this common case.
        packed = _packed_graph(vertices, adjacent) if count is None or 1 < count <= limit else None
        prepared.append([vertices, packed, count, 1, plan, estimate])
        if count is None:
            pending.append(len(prepared)-1)
    pending.sort(key=lambda index: prepared[index][5], reverse=True)

    def count_one(index):
        return _count_plan(prepared[index][4], adjacent, limit)

    sizes = [64*(len(prepared[i][0])+1)*prepared[i][1].shape[1] for i in pending]
    results = _component_map(count_one, pending, sizes,
        [prepared[i][5] for i in pending], 'residual clique counting')
    for index, (count, work) in zip(pending, results):
        prepared[index][2:4] = [int(count), int(work)]
    return [item[:4] for item in prepared]


@njit(cache=True, inline='always')
def _mix_word(value):
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xbf58476d1ce4e5b9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94d049bb133111eb)
    return value ^ (value >> np.uint64(31))


@njit(cache=True, nogil=True)
def _unique_lifted(rows, records, table_size):
    """Compact exact identities in place and count original-record size classes.

    Hash collisions compare every packed word. Destinations never exceed source
    positions, so compaction cannot overwrite an unread row. No Python groups.
    """
    slots = np.full(table_size, -1, dtype=np.int64)
    sizes = np.empty(len(rows), dtype=np.int64)
    histogram = np.zeros(records+1, dtype=np.int64)
    unique = 0
    for source in range(len(rows)):
        digest = np.uint64(0x9e3779b97f4a7c15)
        for word in range(rows.shape[1]):
            digest = _mix_word(digest ^ rows[source, word])
        slot = np.int64(digest & np.uint64(table_size-1))
        duplicate = False
        while slots[slot] >= 0:
            existing = slots[slot]
            equal = True
            for word in range(rows.shape[1]):
                if rows[source, word] != rows[existing, word]:
                    equal = False
                    break
            if equal:
                duplicate = True
                break
            slot = (slot+1) & (table_size-1)
        if duplicate:
            continue
        size = 0
        for word in range(rows.shape[1]):
            value = rows[source, word]
            rows[unique, word] = value
            size += _count(value)
        slots[slot] = unique
        sizes[unique] = size
        histogram[size] += 1
        unique += 1
    return unique, sizes, histogram


def _admitted_sizes(counts, cap, omissions):
    """The existing whole-class policy applied to exact unique-group counts."""
    sizes = sorted(counts, reverse=True)
    accepted, before, capped = set(), 0, False
    for index, size in enumerate(sizes):
        after = before+counts[size]
        if after <= cap:
            accepted.add(size)
            before = after
            continue
        tie_omitted = (before < cap or before == 0) and after > 2*cap
        if before < cap and after <= 2*cap:
            accepted.add(size)
        if omissions is not None:
            if tie_omitted:
                omissions.append(dict(cause='residual_group_tie_class_at_budget',
                    kind='tie_class_at_budget', count=counts[size],
                    scope='residual alternative size classes', count_kind='exact'))
            tail = sum(counts[q] for q in sizes[index+1:]) + (counts[size] if before >= cap and not tie_omitted else 0)
            if tail:
                omissions.append(dict(cause='residual_group_ranked_truncation',
                    kind='ranked_truncation', count=tail,
                    scope='residual alternative size classes', count_kind='exact'))
        capped = tie_omitted
        break
    return accepted, capped


def _decode_members(row):
    members = []
    for word, value in enumerate(row):
        value = int(value)
        while value:
            bit = value & -value
            members.append(64*word+bit.bit_length()-1)
            value ^= bit
    return tuple(members)


def lift_groups(groups, memberships):
    """Small hierarchy/core families, outside the alternative-group budget."""
    return sorted({_decode_members(np.bitwise_or.reduce(memberships[list(group)], axis=0))
                   for group in groups})


def screened_lifted_components(components, adjacent, limit, cap, memberships, records, omissions=None):
    """Count by lifted size and identity before materializing any alternative tuple.

    Component admission still counts ANCHOR cliques, before any lifting or dedup.
    Storage is bounded by actual admitted counts: packed original-record rows,
    a collision-checked index, and integer size counts. Small serial scratch uses
    the block workspace; large storage has an additional counted reservation;
    no work, size, hash or memory heuristic silently drops a scientific group.
    """
    from .search import diagnostic_phase
    with diagnostic_phase('clique_counting'):
        counted = _count_components(components, adjacent, int(limit))
    prepared, total, work, enumerated = [], 0, 0, True
    for vertices, packed, count, calls in counted:
        work += calls
        if count > limit:
            enumerated = False
            if omissions is not None:
                omissions.append(dict(cause='residual_clique_enumeration_tie_class_at_budget',
                    kind='tie_class_at_budget', count=1,
                    scope=f'component with {len(vertices)} records; more than {limit} maximal cliques',
                    count_kind='exact'))
        else:
            prepared.append((vertices, packed, count, total))
            total += count
    if not total:
        return [], False, enumerated, int(work), 0
    words = memberships.shape[1]
    table_size = 1 << (2*total-1).bit_length()
    # Raw rows + hash slots + sizes + selection mask/indices + histogram/mask.
    nbytes = 8*total*words + 8*table_size + 17*total + 9*(records+1)
    with diagnostic_phase('lifting_budgeting'), _workspace(nbytes, 'residual lifted identities'):
        rows = np.empty((total, words), dtype=np.uint64)
        pending = []
        for index, (vertices, packed, count, start) in enumerate(prepared):
            if count == 1:
                # One maximal clique means the entire component is complete.
                rows[start] = np.bitwise_or.reduce(memberships[list(vertices)], axis=0)
                work += 1
            else:
                pending.append(index)
        pending.sort(key=lambda index: prepared[index][2], reverse=True)

        def fill_one(index):
            vertices, packed, count, start = prepared[index]
            local = np.ascontiguousarray(memberships[list(vertices)])
            emitted, calls = _walk(packed, len(vertices), int(limit), rows[start:start+count],
                                   None, None, local)
            assert emitted == count
            return int(calls)

        sizes = [(64*prepared[i][1].shape[1]+16*words)*(len(prepared[i][0])+1)
                 for i in pending]
        work += sum(_component_map(fill_one, pending, sizes,
            [len(prepared[i][0])*prepared[i][2] for i in pending], 'residual lifted traversal'))
        unique, sizes, histogram = _unique_lifted(rows, int(records), table_size)
        counts = {size: int(count) for size, count in enumerate(histogram) if count}
        accepted, capped = _admitted_sizes(counts, int(cap), omissions)
        allowed = np.zeros(records+1, dtype=np.bool_)
        for size in accepted:
            allowed[size] = True
        selected = np.flatnonzero(allowed[sizes[:unique]])
        decoded_bytes = sum(counts[size]*(64+44*size) for size in accepted)
        with _workspace(decoded_bytes, 'residual admitted groups'):
            result = sorted(_decode_members(rows[index]) for index in selected)
    return result, capped, enumerated, int(work), int(unique)
