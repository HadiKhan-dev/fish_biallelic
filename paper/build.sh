#!/bin/bash
# Build the manuscript and its supplement from the Markdown sources.
#
#   paper/build.sh            # HTML, PDF, DOCX and LaTeX into paper/build/, PDFs copied to paper/
#
# Needs pandoc (3.x, with its built-in citeproc) and, for the PDFs and the Word figures, a
# Chromium or Chrome binary (set CHROME=/path/to/chrome, or have chromium / google-chrome on
# PATH). The PDF is Chromium's print of the HTML, so no LaTeX installation is needed. The .tex
# output is for journal submission; it references the SVG figures, which a LaTeX build must convert.
set -euo pipefail
cd "$(dirname "$0")"

command -v pandoc >/dev/null || { echo "build.sh: pandoc not found" >&2; exit 1; }
mkdir -p build/png

python3 tools/make_figures.py "$@"
python3 tools/make_tables.py "$@"
python3 tools/make_results_figures.py

chrome=${CHROME:-}
if [[ -z $chrome ]]; then
    for candidate in chromium chromium-browser chrome google-chrome google-chrome-stable \
                     /opt/pw-browsers/chromium-*/chrome-linux64/chrome; do
        if command -v "$candidate" >/dev/null 2>&1; then chrome=$(command -v "$candidate"); break; fi
    done
fi
headless=(--headless --no-sandbox --disable-gpu --hide-scrollbars)

word_figures=()
if [[ -n $chrome ]]; then
    for svg in figures/*.svg; do
        size=$(python3 -c "import re,sys; s=open(sys.argv[1]).read(); print(*re.search(r'width=\"(\d+)\" height=\"(\d+)\"', s).groups(), sep=',')" "$svg")
        "$chrome" "${headless[@]}" --force-device-scale-factor=3 --window-size="$size" \
            --screenshot="build/png/$(basename "$svg" .svg).png" "file://$PWD/$svg" 2>/dev/null
    done
    word_figures=(--lua-filter tools/svg_to_png.lua)
fi

common=(--from markdown+tex_math_dollars+pipe_tables+grid_tables+footnotes+implicit_figures
        --citeproc --bibliography references.bib --resource-path .:figures)
main=(metadata.yaml sections/*.md)
supp=(supplement/metadata.yaml supplement/S*.md)

for doc in manuscript supplement; do
    if [[ $doc == manuscript ]]; then inputs=("${main[@]}"); else inputs=("${supp[@]}"); fi
    # The main text is numbered by pandoc; the supplement's sections carry their own S numbers.
    numbering=(); [[ $doc == manuscript ]] && numbering=(--number-sections)
    pandoc "${common[@]}" --standalone --mathml --embed-resources --css style.css \
        "${numbering[@]}" -o "build/$doc.html" "${inputs[@]}"
    pandoc "${common[@]}" "${word_figures[@]}" --standalone "${numbering[@]}" -o "build/$doc.docx" "${inputs[@]}"
    pandoc "${common[@]}" --standalone "${numbering[@]}" -o "build/$doc.tex" "${inputs[@]}"
done

if [[ -z $chrome ]]; then
    echo "build.sh: no Chromium/Chrome found; HTML, DOCX and LaTeX built, PDFs skipped" >&2
    exit 0
fi
for doc in manuscript supplement; do
    "$chrome" "${headless[@]}" --no-pdf-header-footer \
        --print-to-pdf="build/$doc.pdf" "file://$PWD/build/$doc.html" 2>/dev/null
    cp "build/$doc.pdf" "$doc.pdf"
done
echo "built: paper/manuscript.pdf paper/supplement.pdf (and paper/build/*.html|docx|tex)"
