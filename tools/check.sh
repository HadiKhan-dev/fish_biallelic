#!/bin/bash
# The checks run on every change (STYLE_40 §9): lint, types at public boundaries, import layers and the fast tier.
# Frozen reviews: set HR_ORACLE_BASE to the absolute 3.5 tree; include the package root __init__.py.
# Usage: tools/check.sh [pytest args]. Runs in the approved development environment (bio40-dev); ordinary tests single-threaded; threads-marked tests use four threads without xdist.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${HR_PY:-python}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-1}
export NUMBA_DISABLE_INTEL_SVML=1 NUMBA_THREADING_LAYER=omp OMP_WAIT_POLICY=PASSIVE
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$PWD/.numba_cache} PYTHONDONTWRITEBYTECODE=1
status=0
echo "== ruff"
$PY -m ruff check haplotype_reconstruction tests || status=1
echo "== format"
$PY -m ruff format --check haplotype_reconstruction tests || status=1
echo "== docs"
$PY tools/check_docs.py || status=1
echo "== mypy"
$PY -m mypy --no-error-summary || status=1
# Preserve user selection while partitioning the two CPU contracts. Last user -m
# wins, as in pytest; the default fast tier still excludes stage/slow tests.
selection="not stage and not slow"
args=()
thread_args=()
while (($#)); do
    case "$1" in
        -m) selection="$2"; shift 2 ;;
        -m?*) selection="${1#-m}"; shift ;;
        --markexpr=*) selection="${1#*=}"; shift ;;
        --markexpr) selection="$2"; shift 2 ;;
        -n|--numprocesses) args+=("$1" "$2"); shift 2 ;;
        -n?*|--numprocesses=*) args+=("$1"); shift ;;
        --dist|--maxprocesses) args+=("$1" "$2"); shift 2 ;;
        --dist=*|--maxprocesses=*) args+=("$1"); shift ;;
        *) args+=("$1"); thread_args+=("$1"); shift ;;
    esac
done
echo "== pytest (one thread)"
NUMBA_NUM_THREADS=1 $PY -m pytest -q -p no:cacheprovider "${args[@]}" -m "($selection) and not threads" || status=1
echo "== pytest (four threads, no xdist)"
NUMBA_NUM_THREADS=4 $PY -m pytest -q -p no:cacheprovider "${thread_args[@]}" -m "($selection) and threads"
threads_status=$?
# Scoped checks legitimately select no threads-marked tests (pytest exit 5).
if ((threads_status != 0 && threads_status != 5)); then status=1; fi
exit $status
