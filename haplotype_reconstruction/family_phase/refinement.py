"""Deterministic equitable refinement of discrete pedigree/data classes.

Only smaller split pieces enter the work queue; each incidence is visited at
most logarithmically many times. Blocks remain implicit sets, without an N-round
whole-graph rescan. Tokens encode existing class/slot/pedigree split history,
never sample identity, storage position or likelihood floats.
"""
from __future__ import annotations
import hashlib
import heapq
import numpy as np


def refine(initial, incoming, outgoing):
    """Return canonical refined colours and work diagnostics, O((N+E) log N)."""
    n = len(initial)
    count = int(max(initial)) + 1 if n else 0
    members = [set() for _ in range(count)]
    colours = np.asarray(initial, dtype=np.int64).copy()
    for node, colour in enumerate(colours):
        members[int(colour)].add(node)
    origins = list(range(count))
    tokens = [hashlib.sha256(b'initial' + k.to_bytes(8, 'big')).digest() for k in range(count)]
    queue = [(token, block) for block, token in enumerate(tokens)]
    heapq.heapify(queue)
    visited, splitters, splits = 0, 0, 0
    if count == n:
        queue.clear()
    while queue:
        splitter_token, splitter = heapq.heappop(queue)
        # The splitter can itself be split below. Its original members define
        # every relation in this iteration, so snapshot the set before updating.
        snapshot = tuple(members[splitter])
        relations = ({}, {}, {}, {})
        for node in snapshot:
            for direction, adjacency in enumerate((incoming[node], outgoing[node])):
                for slot, other in adjacency:
                    counts = relations[2 * direction + int(slot)]
                    counts[other] = counts.get(other, 0) + 1
                    visited += 1
        splitters += 1
        for relation, counts in enumerate(relations):
            affected = {}
            for node, value in counts.items():
                block = int(colours[node])
                if len(members[block]) > 1:
                    affected.setdefault(block, {}).setdefault(value, set()).add(node)
            # Disjoint affected blocks commute. Their tokens and the priority
            # queue, not the iteration order here, choose subsequent splitters.
            for block, buckets in affected.items():
                old = members[block]
                touched = sum(map(len, buckets.values()))
                zero_size = len(old) - touched
                sizes = {value: len(nodes) for value, nodes in buckets.items()}
                if zero_size:
                    sizes[0] = zero_size
                if len(sizes) <= 1:
                    continue
                keep = min(sizes, key=lambda value: (-sizes[value], value))
                parent_token = tokens[block]
                if keep == 0:
                    for nodes in buckets.values():
                        old.difference_update(nodes)
                    pieces = buckets.items()
                else:
                    # If the zero bucket is not largest, len(old)<=2*touched;
                    # materialising it is therefore charged to visited nodes.
                    if zero_size:
                        zero = old.copy()
                        for nodes in buckets.values():
                            zero.difference_update(nodes)
                        buckets[0] = zero
                    members[block] = buckets[keep]
                    pieces = ((value, nodes) for value, nodes in buckets.items() if value != keep)
                for value, nodes in pieces:
                    new = len(members)
                    members.append(nodes)
                    origins.append(origins[block])
                    token = hashlib.sha256(b'split' + parent_token + splitter_token
                                           + relation.to_bytes(1, 'big') + value.to_bytes(8, 'big')).digest()
                    tokens.append(token)
                    for node in nodes:
                        colours[node] = new
                    heapq.heappush(queue, (token, new))
                    splits += 1
    # Preserve initial discrete-data order. Only newly distinguished members of
    # an initial class can acquire a different order from synchronous refinement.
    order = sorted(range(len(members)), key=lambda block: (origins[block], tokens[block]))
    result = np.empty(n, dtype=np.int64)
    for rank, block in enumerate(order):
        for node in members[block]:
            result[node] = rank
    return result, dict(splitters=splitters, splits=splits, incidence_visits=visited,
                        initial_classes=count, refined_classes=len(members))
