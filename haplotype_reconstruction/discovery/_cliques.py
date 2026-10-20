"""Exact compiled pivoted Bron--Kerbosch traversal on packed adjacency rows.

The first pass counts actual maximal cliques, stopping only at limit+1. Thus an
oversized component is omitted whole, independently of labels/traversal order.
Only an admitted component is materialized, in a second identical traversal.
No search-work or wall-time cutoff is a scientific decision.
"""
import numpy as np
from numba import njit


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
def _walk(adjacent, n, limit, output, histogram=None, allowed=None):
    words = adjacent.shape[1]
    candidates = np.zeros((n+1, words), dtype=np.uint64)
    excluded = np.zeros_like(candidates)
    todo = np.zeros_like(candidates)
    chosen = np.empty(n, dtype=np.int64)
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
    slots = {v:i for i,v in enumerate(vertices)}
    n = len(vertices)
    packed = np.zeros((n, (n+63)//64), dtype=np.uint64)
    for i,v in enumerate(vertices):
        for u in adjacent[v]:
            if u in slots:
                j=slots[u]
                packed[i,j//64] |= np.uint64(1) << np.uint64(j%64)
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
    """Exactly capped_by_size(all admitted components), without allocating discarded cliques.

    This specialization is for ordinary record-count classes, with no lifted
    identity/deduplication. Components are disjoint, so no cross-component
    clique duplicate is possible. Counts and the existing omission records are
    exact; all tied members of every admitted size class are materialized.
    """
    prepared=[]
    counts={}
    work=0
    enumerated=True
    for vertices in components:
        vertices=tuple(sorted(vertices));n=len(vertices)
        slots={v:i for i,v in enumerate(vertices)}
        packed=np.zeros((n,(n+63)//64),dtype=np.uint64)
        for i,v in enumerate(vertices):
            for u in adjacent[v]:
                if u in slots:
                    j=slots[u];packed[i,j//64]|=np.uint64(1)<<np.uint64(j%64)
        histogram=np.zeros(n+1,dtype=np.int64)
        count,calls=_walk(packed,n,int(limit),np.empty((0,packed.shape[1]),dtype=np.uint64),histogram)
        work+=calls
        if count>limit:
            enumerated=False
            if omissions is not None:
                omissions.append(dict(cause='residual_clique_enumeration_tie_class_at_budget',
                    kind='tie_class_at_budget',count=1,
                    scope=f'component with {n} records; more than {limit} maximal cliques',count_kind='exact'))
            continue
        prepared.append((vertices,packed,histogram))
        for size in range(2,len(histogram)):
            if histogram[size]:counts[size]=counts.get(size,0)+int(histogram[size])
    sizes=sorted(counts,reverse=True)
    total=sum(counts.values());before=0;accepted=set();capped=False
    for i,size in enumerate(sizes):
        after=before+counts[size]
        if after<=cap:
            accepted.add(size);before=after;continue
        tie_omitted=(before<cap or before==0) and after>2*cap
        if before<cap and after<=2*cap:accepted.add(size)
        if omissions is not None:
            if tie_omitted:
                omissions.append(dict(cause='residual_group_tie_class_at_budget',kind='tie_class_at_budget',
                    count=counts[size],scope='residual alternative size classes',count_kind='exact'))
            tail=sum(counts[q] for q in sizes[i+1:])+(counts[size] if before>=cap and not tie_omitted else 0)
            if tail:
                omissions.append(dict(cause='residual_group_ranked_truncation',kind='ranked_truncation',
                    count=tail,scope='residual alternative size classes',count_kind='exact'))
        capped=tie_omitted
        break
    result=[]
    for vertices,packed,histogram in prepared:
        count=sum(int(histogram[size]) for size in accepted if size<len(histogram))
        if not count:continue
        allowed=np.array([i in accepted for i in range(len(histogram))],dtype=np.bool_)
        output=np.empty((count,packed.shape[1]),dtype=np.uint64)
        _,calls=_walk(packed,len(vertices),int(limit),output,None,allowed)
        work+=calls
        for row in output:
            members=[]
            for w,value in enumerate(row):
                value=int(value)
                while value:
                    bit=value & -value;members.append(vertices[64*w+bit.bit_length()-1]);value^=bit
            result.append(tuple(members))
    return result,capped,enumerated,int(work),total
