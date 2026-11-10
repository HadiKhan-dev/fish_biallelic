"""Founder refinement: fixed-K fits, count decisions and component polish.

Numerical kernels use flat scalar/array arguments because Numba cannot lower the
frozen Python boundary records. Hot recurrences stay whole to expose update order
and avoid temporary arrays. Parallel kernels write disjoint sample slices; sample
reductions retain their declared order independently of team size.
"""
