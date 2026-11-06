# Manuscript

The manuscript describes release 4.4.0 (commit `867f163`, 6 November 2026). Methods follow the
release's documentation and code. Historical 4.1/4.2 accuracy and cost comparisons retain their
measured versions; the 4.4 validation scope is recorded in `data/provenance.json`.

- `manuscript.pdf` and `supplement.pdf`: built documents.
- `metadata.yaml`, `sections/*.md`: main text, in pandoc Markdown and reading order.
- `supplement/metadata.yaml`, `supplement/S*.md`: Supplementary Information. S1–S6 give finer rules
  and defaults; `S7_tables.md` is generated, not edited by hand.
- `references.bib`: bibliography. Check publication details before submission.
- `figures/`: Figures 1–2 from `tools/make_figures.py`, and results Figures 3–13 from
  `tools/make_results_figures.py`. Captions and definitions are in `figures/CAPTIONS.md`.
- `data/`: compact CSV/JSON summaries. Results plots read only these files; they never open runs
  or invoke the pipeline. Source labels are portable, without host names or site paths.
- `tools/make_tables.py`: generates Supplementary Tables S1 and S2 from the repository's cost
  register and numerical ledger. Both documentation-based generators accept `--docs-dir PATH`.
- `style.css`: HTML/PDF style; `tools/svg_to_png.lua`: figure handling for the Word build.

## Figures and summaries

Regenerate Figures 3–13 from the retained summaries (matplotlib and NumPy required):

```sh
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/paper-matplotlib"
nice -n 19 python -P paper/tools/make_results_figures.py
```

To extract the 4.4 depth series and check the retained matrix evidence, supply completed inputs:

```sh
nice -n 19 python -P paper/tools/figure_data.py \
  --depth-run 1=/path/to/depth1 --depth-run 1.5=/path/to/depth1p5 \
  --depth-run 2=/path/to/depth2 --depth-run 5=/path/to/depth5 \
  --depth-baseline43 2=/path/to/baseline43_depth2 \
  --depth-baseline43 5=/path/to/baseline43_depth5 \
  --depth-comparison43 /path/to/d2_43_vs_44_final.json \
  --matrix44 /path/to/matrix44 --realdata44 \
  --processor 'PREFIX_A=Sapphire Rapids:112' \
  --processor 'PREFIX_B=Ice Lake:76' --out paper/data
```

`--depth-run DEPTH=PATH` is repeatable. Each directory contains `config.toml`, `RESULT.txt`,
`run.time.log`, `run.log` and `run/{evaluation,run_records}`. The repeatable
`--depth-baseline43 DEPTH=PATH` accepts the old run wrapper or a directory containing its
`evaluation/`; all saved evaluation CSV/JSON values are compared exactly, except the checkpoint
root. Runtime records, when retained, also produce `depth_baselines.csv`, with resumed attempts
kept separate. `depth_series.csv` stores counts, denominators, GNU run-command wall/RSS,
sampled pressure, recorded limits, admissions and hardware. GNU wall time includes command
overhead outside the runtime timer. Initialisation and evaluation are separate commands.

`--matrix44` checks all saved scores and completed command receipts against `baseline43/` and the
unchanged `data/simulation_accuracy.csv`; it updates provenance without regenerating old data.
This checks only retained aggregate scalar fields, transitively through the archived 4.3-to-4.2
comparisons. The deleted raw 4.3 matrix outputs do not support a new full-product identity claim.
Optional `--realdata44` directly compares nine selected released-product families within 4.4,
including streamed, sample-aligned VCF records, and writes `robustness44.csv`. It excludes
numerical diagnostics from the founder-count and phase-summary families. The raw exact real-data
comparison reports also contain diagnostic and ordering differences; their overall failure flags
are not released-product or biological-truth verdicts. `--depth-comparison43` records the saved
exact depth-2 full-output comparison's scope and counts. These modes preserve legacy CSVs.

The repeatable `--processor PREFIX=NAME:CORES` argument is required in every extraction mode.
Replace the prefix placeholders with receipt host prefixes and specify full-node core counts.
There is no built-in site mapping; prefixes are used only to interpret receipts and are never
written into summaries. Use the existing analysis Python environment. Extraction reads published
CSV/JSON tables, streamed VCF records, configs, logs and receipts; it does not run the pipeline,
evaluation or any matrix writing tool. It may take several minutes.

The legacy extraction interface remains:

```sh
nice -n 19 python -P paper/tools/figure_data.py \
  --matrix42 /path/to/matrix42 --matrix41 /path/to/matrix41 \
  --probes /path/to/probes --node-logs /path/to/node-queue/logs \
  --processor 'PREFIX_A=Sapphire Rapids:112' \
  --processor 'PREFIX_B=Ice Lake:76' \
  --mosaic-chromosome chr1 --out paper/data
```

The original `--matrix42` and `--matrix41` run directories were deleted in the 4 November cleanup.
Their retained compact summaries still reproduce Figures 3–12, but full legacy re-extraction is
no longer possible from those locations. Do not regenerate them from incomplete substitutes.
`--costs-only` refreshes paired costs if its original inputs are available. The results plotter
accepts `--data-dir` and `--output-dir` for isolated checks. SVG identifiers and PDF timestamps
are fixed; identical inputs give deterministic figures.

## Build

```sh
paper/build.sh
```

Needs pandoc 3 and Chromium/Chrome on `PATH` (or `CHROME`). The build regenerates all figures
and the two documentation tables, then writes HTML, PDF, DOCX and LaTeX into `paper/build/`,
copying the PDFs to `paper/`. LaTeX output references SVGs and needs conversion for a LaTeX build.
For a standalone paper copy, pass `--docs-dir /path/to/repository/docs` to `build.sh`; the default
is its sibling `docs/` in a normal repository checkout. The output stays in the paper directory.

On the campaign host, load the recorded modules in a login shell:

```sh
module load ceuadmin/pandoc/3.6.2 ceuadmin/chromium/132.0.6798.0
export PATH="/path/to/bio40-dev/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
nice -n 19 paper/build.sh
```

Put the analysis environment's `bin` first on `PATH`, since the build calls `python3`. On a long
workspace path, Chromium can exceed the Unix-socket path limit in the inherited temporary
directory. Create `paper/build/tmp` and export `TMPDIR=build/tmp` before invoking the build;
`build.sh` changes into `paper/`, so these temporary files stay inside the paper copy.

## Placeholders

Remaining red-box placeholders concern co-authors and affiliations, the precise TroMau species
and sampling/sequencing description, public data and template accessions, and acknowledgements.
The pending 4.3 comparison placeholder was resolved using the retained archived score evidence.
All new numerical results trace to the completed evaluations and receipts recorded in
`data/provenance.json`; the manuscript distinguishes these from historical measurements.

List remaining placeholders with:

```sh
rg -n 'TO FILL' paper/metadata.yaml paper/sections paper/supplement
```
