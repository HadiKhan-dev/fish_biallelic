# Manuscript

The manuscript describing release 4.0 of the pipeline, written from scratch from the 4.0 methods
documentation (`docs/methods/`) and code. It replaces the earlier, pre-4.0 draft, which is kept
outside source control (`/manuscript/`, ignored).

- [manuscript.pdf](manuscript.pdf) and [supplement.pdf](supplement.pdf): the built documents.
- `metadata.yaml`, `sections/*.md`: the main text, in pandoc Markdown, in reading order.
- `supplement/metadata.yaml`, `supplement/*.md`: the Supplementary Information.
- `references.bib`: the bibliography. Each entry was written by hand; check DOIs and pages against
  the journal before submission.
- `figures/`: SVG figures, written by `tools/make_figures.py` (standard library only). Figure 1 is
  the stage graph of `haplotype_reconstruction/stages/graph.py`; Figure 2 is parsed from the
  "Kernel ladders" table of `docs/developer/costs.md`, so rebuilding it after that table changes
  updates the figure.
- `style.css`: page and caption style for the HTML and PDF builds.

## Build

```sh
paper/build.sh
```

Needs pandoc 3 (citations use its built-in citeproc) and, for the PDFs, a Chromium or Chrome
binary on `PATH` or in `CHROME`. The PDF is Chromium's print of the HTML, so no LaTeX is needed.
The build also writes `paper/build/manuscript.{html,docx,tex}` and the same for the supplement;
`paper/build/` is not tracked. The LaTeX output references the SVG figures, which a LaTeX
toolchain must convert (for example with the `svg` package or `rsvg-convert`).

## Placeholders

Text in red boxes, written as `[TO FILL: ...]{.todo}` or in a `::: todo` block, marks content that
cannot be written from the repository alone:

- accuracy on simulated crosses, and the TroMau results, which need the scored outputs of the 4.0
  validation matrix (they live on project storage, not in the repository);
- wall time and memory of whole-genome runs;
- data description details (sample counts, depth, species name, the genetic map used),
  co-authors, affiliations, data accessions and acknowledgements.

Every number that is in the text without a placeholder comes from the repository: the cost and
kernel-ladder tables (`docs/developer/costs.md`), the 4.0 changelog (`CHANGELOG.md`), the numerical
ledger (`docs/developer/ties.md`) and the generated parameter reference
(`docs/reference/parameters.md`).

List the remaining placeholders with:

```sh
grep -n "TO FILL" paper/metadata.yaml paper/sections/*.md paper/supplement/*.md
```
