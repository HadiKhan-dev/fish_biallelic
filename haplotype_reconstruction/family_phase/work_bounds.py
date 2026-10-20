"""Fixed complete-neighbourhood effort bound; never silently truncate factors."""
import os


class FamilyWorkBound(Exception):
    def __init__(self, bound):
        self.bound = int(bound)
        super().__init__(f'complete family exceeds {bound} sample units')


def configured_bound(name="HAPLOTYPES_FAMILY_WORK_BOUND"):
    # Reference remains default until the standard-data binding census is reviewed.
    bound = int(os.environ.get(name, '0'))
    if bound < 0:
        raise ValueError(name + ' must be nonnegative (0 = reference)')
    return bound


class BoundedSet(set):
    def __init__(self, values=(), bound=0):
        super().__init__()
        self.bound = bound
        self.update(values)

    def add(self, value):
        super().add(int(value))
        if self.bound and len(self) > self.bound:
            raise FamilyWorkBound(self.bound)

    def update(self, values):
        for value in values:
            self.add(value)


def witness_family(fm, sample, bound=0):
    """Original mutable family and complete read factors, with lazy overflow.

    Each group has one distinct child. Bounding group sets therefore detects
    certain sample overflow early, before materializing a large sibling set.
    The final bound also includes pinned parental rows and shared root factors.
    """
    g = fm.graph
    def children(parent):
        return fm.child_groups[fm.child_offsets[parent]:fm.child_offsets[parent + 1]]
    def parents(sample):
        return g.group_parent[g.node_group[sample]]
    members = BoundedSet((sample,), bound)
    members.update(p for p in parents(sample) if p >= 0 and not fm.roots[p])
    roots = BoundedSet((p for x in members for p in parents(x) if p >= 0 and fm.roots[p]), bound)
    members.update(g.group_child[x] for p in roots for x in children(p))
    joint_members = tuple(members)
    members.update(g.group_child[x] for p in joint_members for x in children(p))
    groups = tuple(sorted(g.node_group[x] for x in members if g.node_group[x] >= 0))
    joint = BoundedSet(roots, bound)
    joint.update(p for p in joint_members if fm.child_offsets[p] < fm.child_offsets[p + 1])
    read_groups = BoundedSet(groups, bound)
    read_groups.update(x for p in members for x in children(p))
    root_terms = BoundedSet((p for x in groups for p in g.group_parent[x] if p >= 0 and fm.roots[p]), bound)
    read_groups.update(x for p in root_terms for x in children(p))
    read_groups.update(x for p in joint for x in children(p))
    read_groups.update(g.node_group[p] for p in joint if not fm.roots[p])
    rows = BoundedSet(root_terms, bound)
    rows.update(members)
    rows.update(joint)
    for group in read_groups:
        rows.add(g.group_child[group])
        rows.update(p for p in g.group_parent[group] if p >= 0)
    return groups, tuple(sorted(roots)), tuple(sorted(joint)), tuple(sorted(rows))
