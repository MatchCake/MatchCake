-- Number every display-math block when rendering to PDF.
--
-- Pandoc renders Markdown `$$ ... $$` as unnumbered display math (`\[ ... \]`). This filter
-- wraps each display block in a LaTeX `equation` environment so it is auto-numbered. The inner
-- `aligned` accommodates both single-line content (matrices, boxed results) and multi-line content
-- with `\\`, yielding exactly one equation number per block.
function Math(elem)
  if elem.mathtype == "DisplayMath" then
    return pandoc.RawInline(
      "latex",
      "\\begin{equation}\\begin{aligned}" .. elem.text .. "\\end{aligned}\\end{equation}"
    )
  end
  return elem
end
