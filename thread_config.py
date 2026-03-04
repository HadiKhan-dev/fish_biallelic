"""
Thread configuration — import before numpy or numba.

Ensures all internal threading libraries (BLAS, OpenMP, Numba) default
to single-threaded mode. This prevents worker processes spawned by
multiprocess.Pool from each creating their own thread pools, which
causes massive oversubscription on multi-core machines.

All parallelism should be controlled at the process level via Pool.

Uses setdefault so that advanced users can override via environment
variables if they know what they are doing.
"""
import os

# BLAS / OpenMP / MKL
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Numba
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")