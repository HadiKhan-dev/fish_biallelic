"""Polynomial residual edge-clique covers on packed adjacency rows.

Every edge seeds a maximal greedy clique in hard-call content order. Deduplication
and per-component size-class budgets follow construction, after lifting when needed.
This covers every compatible pair before budgeting, not every larger clique.
"""
from contextlib import nullcontext
import numpy as np
from numba import njit
from .conventions import rev


_INLINE_BYTES = 1 << 20


def _workspace(nbytes, label):
    if nbytes <= _INLINE_BYTES:
        return nullcontext()
    from ..execution import reserve_memory
    return reserve_memory(nbytes, label=label)


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
def _edge_cover(adjacent, edges):
    """One packed maximal clique per edge; vertices already in content order.

    E seeds, at most A additions each, ceil(A/64) words per intersection.
    All edges seed a group, including already covered edges: this retains more
    larger groups without introducing a Cartesian or recursive search.
    """
    n, words = adjacent.shape
    output = np.zeros((edges, words), dtype=np.uint64)
    candidates = np.empty(words, dtype=np.uint64)
    emitted, work = 0, 0
    for first in range(n):
        for second in range(first + 1, n):
            if not (adjacent[first, second//64] & (np.uint64(1) << np.uint64(second%64))):
                continue
            row = output[emitted]
            row[first//64] |= np.uint64(1) << np.uint64(first%64)
            row[second//64] |= np.uint64(1) << np.uint64(second%64)
            for word in range(words):
                candidates[word] = adjacent[first, word] & adjacent[second, word]
            work += words
            for word in range(words):
                while candidates[word]:
                    vertex = 64*word + _low_index(candidates[word])
                    row[word] |= np.uint64(1) << np.uint64(vertex%64)
                    for other in range(words):
                        candidates[other] &= adjacent[vertex, other]
                    work += words
            emitted += 1
    return output, work


def edge_clique_cover(vertices, adjacent, content_keys):
    """Packed groups and their local-to-record axis; no component admission cap.

    Keys are hard-call rows, never record IDs. Equal nonisolated descriptors are
    true twins and all enter the same maximal extensions, so their relative order
    cannot change membership. Isolated vertices do not seed alternatives.
    """
    vertices = tuple(sorted(vertices, key=content_keys.__getitem__, reverse=rev("d13")))
    n = len(vertices)
    packed = _packed_graph(vertices, adjacent)
    edges = sum(len(adjacent[vertex]) for vertex in vertices)//2
    if edges == n*(n-1)//2:
        # Avoid E identical maximal extensions in the common complete component.
        row = np.zeros((1, packed.shape[1]), dtype=np.uint64)
        for vertex in range(n):
            row[0, vertex//64] |= np.uint64(1) << np.uint64(vertex%64)
        return vertices, row, n
    with _workspace(16*edges*packed.shape[1], 'residual edge cover'):
        rows, work = _edge_cover(packed, edges)
        rows = np.unique(rows, axis=0)
    return vertices, rows, int(work)


@njit(cache=True, nogil=True)
def _lift_cover(groups, memberships):
    output = np.zeros((len(groups), memberships.shape[1]), dtype=np.uint64)
    for group in range(len(groups)):
        for word in range(groups.shape[1]):
            remaining = groups[group, word]
            while remaining:
                vertex = word*64 + _low_index(remaining)
                remaining &= remaining - np.uint64(1)
                for target in range(output.shape[1]):
                    output[group, target] |= memberships[vertex, target]
    return output


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
                    scope='residual component alternative size classes', count_kind='exact'))
            tail = sum(counts[q] for q in sizes[index+1:]) + (counts[size] if before >= cap and not tie_omitted else 0)
            if tail:
                omissions.append(dict(cause='residual_group_ranked_truncation',
                    kind='ranked_truncation', count=tail,
                    scope='residual component alternative size classes', count_kind='exact'))
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


def screened_components(components, adjacent, content_keys, cap, memberships=None,
                        records=None, omissions=None):
    """Cover, lift and screen whole size classes independently per component.

    Only admitted groups are decoded to Python tuples. Lifted packed storage is
    polynomial in edge count and original record count, independent of the number
    of maximal cliques. A crowded component cannot evict another component's
    alternatives. Hierarchy/core groups are handled outside these budgets.
    """
    from .search import diagnostic_phase
    prepared, work = [], 0
    with diagnostic_phase('clique_counting'):
        for component in components:
            vertices, rows, calls = edge_clique_cover(component, adjacent, content_keys)
            prepared.append((vertices, rows))
            work += calls
    total = sum(len(rows) for _, rows in prepared)
    if not total:
        return [], False, int(work), 0
    records = len(content_keys) if records is None else int(records)
    words = (records+63)//64
    table_size = 1 << (2*total-1).bit_length()
    # Concatenated rows, one temporary lifted component, exact hash slots/sizes,
    # selection arrays and histogram; output Python tuples reserved separately.
    nbytes = 16*total*words + 8*table_size + 17*total + 9*(records+1)
    with diagnostic_phase('lifting_budgeting'), _workspace(nbytes, 'residual cover identities'):
        rows = np.zeros((total, words), dtype=np.uint64)
        start, capped, decoded_bytes = 0, False, 0
        selections = []
        for vertices, local in prepared:
            if memberships is None:
                member_bits = np.zeros((len(vertices), words), dtype=np.uint64)
                for index, vertex in enumerate(vertices):
                    member_bits[index, vertex//64] |= np.uint64(1) << np.uint64(vertex%64)
            else:
                member_bits = np.ascontiguousarray(memberships[list(vertices)])
            component_rows = rows[start:start+len(local)]
            component_rows[:] = _lift_cover(local, member_bits)
            unique, sizes, histogram = _unique_lifted(component_rows, records, table_size)
            counts = {size: int(count) for size, count in enumerate(histogram) if size >= 2 and count}
            accepted, component_capped = _admitted_sizes(counts, int(cap), omissions)
            capped |= component_capped
            allowed = np.zeros(records+1, dtype=np.bool_)
            for size in accepted:
                allowed[size] = True
            selections.append(start + np.flatnonzero(allowed[sizes[:unique]]))
            decoded_bytes += sum(counts[size]*(64+44*size) for size in accepted)
            start += len(local)
        with _workspace(decoded_bytes, 'residual admitted groups'):
            result = sorted({_decode_members(rows[index])
                             for selected in selections for index in selected})
        # Component compaction leaves only copies of original groups in its tail.
        # Count the exact global union for diagnostics, including groups screened
        # out above; overlapping nearest-anchor memberships can duplicate groups
        # across disconnected anchor components. This count does not govern admission.
        unique, _, _ = _unique_lifted(rows, records, table_size)
    return result, capped, int(work), int(unique)
