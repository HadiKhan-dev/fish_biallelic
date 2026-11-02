#!/bin/bash
# Build the manuscript and its supplement from the Markdown sources.
#
#   paper/build.sh            # HTML, PDF, DOCX and LaTeX into paper/build/, PDFs copied to paper/
#
# Needs pandoc (3.x, with its built-in citeproc) and, for the PDFs, a Chromium or Chrome
# binary (set CHROME=/path/to/chrome, or have chromium / google-chrome on PATH). The PDF is
# Chromium's print of the HTML, so no LaTeX installation is needed. The .tex output is for
# journal submission; it references the SVG figures, which a LaTeX build must convert.
set -euo pipefail
cd "$(dirname "$0")"

command -v pandoc >/dev/null || { echo "build.sh: pandoc not found" >&2; exit 1; }
mkdir -p build

python3 tools/make_figures.py

common=(--from markdown+tex_math_dollars+pipe_tables+grid_tables+footnotes+implicit_figures
        --citeproc --bibliography references.bib --resource-path .:figures)

main=(metadata.yaml sections/*.md)
supp=(supplement/metadata.yaml supplement/*.md)

for doc in manuscript supplement; do
    if [[ $doc == manuscript ]]; then inputs=("${main[@]}"); else inputs=("${supp[@]}"); fi
    pandoc "${common[@]}" --standalone --mathml --embed-resources --css style.css \
        --number-sections --toc=false -o "build/$doc.html" "${inputs[@]}"
    pandoc "${common[@]}" --standalone --number-sections -o "build/$doc.docx" "${inputs[@]}"
    pandoc "${common[@]}" --standalone --number-sections -o "build/$doc.tex" "${inputs[@]}"
done

chrome=${CHROME:-}
if [[ -z $chrome ]]; then
    for candidate in chromium chromium-browser google-chrome google-chrome-stable \
                     /opt/pw-browsers/chromium-*/chrome-linux64/chrome; do
        if command -v "$candidate" >/dev/null 2>&1; then chrome=$(command -v "$candidate"); break; fi
    done
fi
if [[ -z $chrome ]]; then
    echo "build.sh: no Chromium/Chrome found; HTML, DOCX and LaTeX built, PDFs skipped" >&2
    exit 0
fi
for doc in manuscript supplement; do
    "$chrome" --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
        --print-to-pdf="build/$doc.pdf" "file://$PWD/build/$doc.html" 2>/dev/null
    cp "build/$doc.pdf" "$doc.pdf"
done
echo "built: paper/manuscript.pdf paper/supplement.pdf (and paper/build/*.html|docx|tex)"
