# Tested environment and runtime variables

Generated from package metadata, `requirements-tested.txt` and runtime source reads. Regenerate with `python docs/generate_docs.py`.

Packaging requires Python `>=3.14`. Reference capture: Linux x86_64, CPython 3.14.8; scientific reference captured 2026-11-01.

The reference platform is Linux x86_64 with CPU execution. These are tested runtime constraints, not a complete transitive lockfile or proof for every later dependency version. No GPU is required.

| Distribution | Tested version |
|---|---|
| `numpy` | `2.5.4` |
| `numba` | `0.68.0` |
| `llvmlite` | `0.50.0` |
| `scipy` | `1.18.1` |
| `pandas` | `3.0.6` |
| `cyvcf2` | `0.34.0` |
| `blosc2` | `4.14.1` |
| `matplotlib` | `3.11.2` |
| `threadpoolctl` | `3.7.0` |
| `hdbscan` | `0.8.44` |

Install a supplied wheel or source distribution with these constraints; see [installation](../user/installation.md). Build tools and optional developer/test tools are separate from runtime dependencies.

## Environment reads

Science reads no environment variables. The application establishes the numerical policy before numerical imports. Runtime reads below choose process limits, cache paths or cache identity. They do not select a scientific model. Names are listed even when a caller never supplies them.

| Variable | Runtime use | Readers |
|---|---|---|
| `JOB_ID` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `LSB_JOBID` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `LSB_JOBINDEX` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `NUMBA_*` | Inherited code-generation controls can change compiled code; the compilation cache keys their values. NUMBA_CACHE_DIR, NUMBA_NUM_THREADS and NUMBA_THREADING_LAYER are excluded from that key. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `NUMBA_CACHE_DIR` | Explicit/inherited private compilation cache directory. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py), [runtime.py](../../haplotype_reconstruction/engine/runtime.py) |
| `NUMBA_NUM_THREADS` | Inherited runtime team capacity. | [runtime.py](../../haplotype_reconstruction/engine/runtime.py) |
| `PBS_ARRAYID` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `PBS_ARRAY_INDEX` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `PBS_JOBID` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `SGE_ROOT` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `SGE_TASK_ID` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `SLURM_ARRAY_TASK_ID` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `SLURM_JOB_ID` | Optional external execution-scope identifier for private cache naming; no scientific effect. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `SLURM_MEM_PER_NODE` | Optional memory-limit input in MiB, subordinate to validated process limits. | [runtime.py](../../haplotype_reconstruction/engine/runtime.py) |
| `SLURM_TMPDIR` | Optional temporary cache directory when a run identifier is present. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |
| `XDG_CACHE_HOME` | Persistent user cache base when supplied. | [kernels.py](../../haplotype_reconstruction/engine/kernels.py) |

The runtime sets numerical-library thread limits and fixes `NUMBA_DISABLE_INTEL_SVML`, `NUMBA_THREADING_LAYER` and `OMP_WAIT_POLICY`. These assignments are not additional user-controlled scientific options. Tie reversal and census use explicit settings described in [ties](../developer/ties.md).

## Fixed numerical-library thread limits

The runtime sets these variables to `1` before numerical imports and worker startup; they do not select a scientific model or compiler option.

| Variable | Runtime action |
|---|---|
| `OMP_NUM_THREADS` | Set to `1` by `engine/runtime.py`. |
| `OPENBLAS_NUM_THREADS` | Set to `1` by `engine/runtime.py`. |
| `MKL_NUM_THREADS` | Set to `1` by `engine/runtime.py`. |
| `VECLIB_MAXIMUM_THREADS` | Set to `1` by `engine/runtime.py`. |
| `NUMEXPR_NUM_THREADS` | Set to `1` by `engine/runtime.py`. |
