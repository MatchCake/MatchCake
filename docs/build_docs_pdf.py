"""Compile the Markdown documents in ``docs/`` to PDF via pandoc + xelatex.

Run without arguments to build every top-level ``docs/*.md`` into ``docs/_pdf/``:

    python docs/build_docs_pdf.py

Build a single file, or override the output location:

    python docs/build_docs_pdf.py docs/swap_injection_theory.md -o out.pdf
    python docs/build_docs_pdf.py --out-dir sphinx/build/html/_pdf

SVG figures placed in ``docs/figures/`` are converted to PDF with inkscape and their
``figures/<name>.svg`` references are patched in before the pandoc call; documents without
figures need neither the directory nor inkscape.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = DOCS_DIR / "_pdf"
# Lua filter that wraps each display-math block in a numbered LaTeX equation environment.
EQUATION_FILTER = DOCS_DIR / "_pandoc" / "number_equations.lua"


def convert_svgs_to_pdf(figures_dir: Path, out_dir: Path) -> dict[str, Path]:
    """Convert every SVG in ``figures_dir`` to a PDF in ``out_dir`` and return the name map."""
    mapping = {}
    if not figures_dir.is_dir():
        return mapping
    for svg in figures_dir.glob("*.svg"):
        pdf = out_dir / (svg.stem + ".pdf")
        subprocess.run(
            ["inkscape", "--export-type=pdf", f"--export-filename={pdf}", str(svg)],
            check=True,
        )
        mapping[svg.name] = pdf
    return mapping


def build_pdf(md_path: Path, output: Path | None = None) -> Path:
    """Compile a single Markdown file to PDF and return the output path."""
    md_path = md_path.resolve()
    docs_dir = md_path.parent
    figures_dir = docs_dir / "figures"
    # Resolve to an absolute path: pandoc runs with cwd=docs_dir, so a relative output (e.g. an
    # --out-dir given from the repo root) would otherwise be written under docs/.
    out_pdf = (output or md_path.with_suffix(".pdf")).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Convert SVG figures to PDF and patch their references in the markdown.
        svg_to_pdf = convert_svgs_to_pdf(figures_dir, tmp_dir)
        content = md_path.read_text(encoding="utf-8")
        for svg_name, pdf_path in svg_to_pdf.items():
            content = content.replace(f"figures/{svg_name}", str(pdf_path).replace("\\", "/"))

        patched_md = tmp_dir / md_path.name
        patched_md.write_text(content, encoding="utf-8")

        command = [
            "pandoc",
            str(patched_md),
            "--pdf-engine=xelatex",
            "--from=markdown",
            "-o",
            str(out_pdf),
            "-V",
            "geometry:margin=2.5cm",
            "-V",
            "fontsize=11pt",
            "-V",
            "colorlinks=true",
        ]
        if EQUATION_FILTER.is_file():
            command += ["--lua-filter", str(EQUATION_FILTER)]
        subprocess.run(command, check=True, cwd=docs_dir)

    print(f"PDF written to: {out_pdf}")
    return out_pdf


def build_all(docs_dir: Path, out_dir: Path) -> int:
    """Compile every top-level ``*.md`` in ``docs_dir`` to ``out_dir``; return a process exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        print(f"No Markdown files found in {docs_dir}", file=sys.stderr)
        return 1

    failures = []
    for md in md_files:
        try:
            build_pdf(md, out_dir / (md.stem + ".pdf"))
        except (subprocess.CalledProcessError, OSError) as error:
            failures.append((md.name, error))
            print(f"FAILED to build {md.name}: {error}", file=sys.stderr)

    print(f"Built {len(md_files) - len(failures)}/{len(md_files)} PDFs into {out_dir}")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PDF(s) from the docs Markdown files.")
    parser.add_argument(
        "md_file",
        type=Path,
        nargs="?",
        default=None,
        help="A single Markdown file to build. If omitted, every docs/*.md is built.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PDF for the single-file mode.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the build-all mode (default: docs/_pdf).",
    )
    args = parser.parse_args()

    if args.md_file is not None:
        build_pdf(args.md_file, args.output)
    else:
        sys.exit(build_all(DOCS_DIR, args.out_dir))


if __name__ == "__main__":
    main()
