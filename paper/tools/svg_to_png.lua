-- For the Word build: use the PNG renders of the SVG figures (Word cannot show pandoc's SVG fallback).
function Image(el)
  el.src = el.src:gsub("^figures/(.*)%.svg$", "build/png/%1.png")
  return el
end
