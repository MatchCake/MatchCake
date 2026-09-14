import importlib.util
import pathlib

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / ".github" / "scripts" / "check_math_rendering.py"

_SPEC = importlib.util.spec_from_file_location("check_math_rendering", SCRIPT_PATH)
check_math_rendering = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_math_rendering)

MATHJAX_SCRIPT = '<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
INLINE_MATH = '<span class="math notranslate nohighlight">\\(\\Lambda_0\\)</span>'
DISPLAY_MATH = '<div class="math notranslate nohighlight">\\[\\Lambda = R^\\top \\Lambda_0 R\\]</div>'


class TestCheckMathRendering:
    """
    Guard the check that no built documentation page shows the LaTeX source of a formula.

    Sphinx loads MathJax only on the pages whose own document holds math, so a page that merely
    quotes math written elsewhere, such as a ``toctree`` entry repeating a math-bearing section
    title, keeps the markup without the script that turns it into a formula. The check walks a
    built HTML tree and reports every page in that state.
    """

    @staticmethod
    def write_page(root: pathlib.Path, name: str, body: str) -> pathlib.Path:
        """
        Write one HTML page into a build tree.

        :param root: Path to the root of the build tree.
        :type root: pathlib.Path
        :param name: Name of the page, relative to the root.
        :type name: str
        :param body: Content of the body of the page.
        :type body: str
        :return: Path of the page written.
        :rtype: pathlib.Path
        """
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"<html><body>{body}</body></html>", encoding="utf-8")
        return path

    def test_read_page_finds_inline_math(self):
        math_fragments, _ = check_math_rendering.read_page(INLINE_MATH)
        assert math_fragments == ["\\(\\Lambda_0\\)"]

    def test_read_page_finds_display_math(self):
        math_fragments, _ = check_math_rendering.read_page(DISPLAY_MATH)
        assert math_fragments == ["\\[\\Lambda = R^\\top \\Lambda_0 R\\]"]

    def test_read_page_finds_math_carrying_extra_classes(self):
        math_fragments, _ = check_math_rendering.read_page(
            '<div class="math notranslate nohighlight amsmath">\\[x\\]</div>'
        )
        assert math_fragments == ["\\[x\\]"]

    def test_read_page_accepts_a_single_quoted_class(self):
        math_fragments, _ = check_math_rendering.read_page("<span class='math'>\\(x\\)</span>")
        assert math_fragments == ["\\(x\\)"]

    def test_read_page_decodes_character_references(self):
        math_fragments, _ = check_math_rendering.read_page('<div class="math">\\[a &lt; b\\]</div>')
        assert math_fragments == ["\\[a < b\\]"]

    def test_read_page_closes_a_math_element_on_its_own_end_tag(self):
        math_fragments, _ = check_math_rendering.read_page(
            '<div class="math notranslate nohighlight">'
            '<span class="eqno">(1)<a class="headerlink" href="#equation-one">¶</a></span>'
            "\\[x\\]</div><p>after</p>"
        )
        assert math_fragments == ["(1)¶\\[x\\]"]

    def test_read_page_keeps_a_math_element_open_across_a_nested_element_of_the_same_name(self):
        math_fragments, _ = check_math_rendering.read_page(
            '<span class="math notranslate nohighlight">\\(<span>a</span>b\\)</span><span>after</span>'
        )
        assert math_fragments == ["\\(ab\\)"]

    def test_read_page_keeps_a_math_element_the_page_leaves_open(self):
        math_fragments, _ = check_math_rendering.read_page('<span class="math">\\(\\Lambda_0\\)')
        assert math_fragments == ["\\(\\Lambda_0\\)"]

    def test_read_page_ignores_the_myst_ignore_wrapper(self):
        math_fragments, _ = check_math_rendering.read_page(
            '<section class="tex2jax_ignore mathjax_ignore"><p>no math here</p></section>'
        )
        assert math_fragments == []

    def test_read_page_ignores_a_class_that_only_contains_the_math_word(self):
        math_fragments, _ = check_math_rendering.read_page('<span class="mathematics">\\(x\\)</span>')
        assert math_fragments == []

    def test_read_page_ignores_an_element_without_a_class(self):
        math_fragments, _ = check_math_rendering.read_page("<span>\\(x\\)</span>")
        assert math_fragments == []

    def test_read_page_collects_script_sources(self):
        _, script_sources = check_math_rendering.read_page(
            f'{MATHJAX_SCRIPT}<script src="static/js/theme.js"></script><script>var a = 1;</script>'
        )
        assert script_sources == [
            "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
            "static/js/theme.js",
        ]

    @pytest.mark.parametrize(
        "script_sources, expected",
        [
            (["https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"], True),
            (["static/mathjax/tex-chtml.js"], True),
            (["static/MathJax.js"], True),
            (["static/js/theme.js", "static/doctools.js"], False),
            (["None"], False),
            ([], False),
        ],
    )
    def test_loads_mathjax_recognizes_the_loader(self, script_sources, expected):
        assert check_math_rendering.loads_mathjax(script_sources) is expected

    def test_list_pages_finds_nested_pages_and_ignores_other_files(self, tmp_path):
        self.write_page(tmp_path, "index.html", "")
        self.write_page(tmp_path, "docs/theory.html", "")
        (tmp_path / "searchindex.js").write_text("", encoding="utf-8")
        (tmp_path / "sources").mkdir()
        (tmp_path / "sources" / "theory.md.txt").write_text("", encoding="utf-8")
        assert check_math_rendering.list_pages(tmp_path) == [
            pathlib.Path("docs/theory.html"),
            pathlib.Path("index.html"),
        ]

    def test_list_pages_finds_a_page_whose_suffix_is_capitalised(self, tmp_path):
        self.write_page(tmp_path, "THEORY.HTML", "")
        assert check_math_rendering.list_pages(tmp_path) == [pathlib.Path("THEORY.HTML")]

    def test_list_pages_ignores_a_directory_named_like_a_page(self, tmp_path):
        (tmp_path / "theory.html").mkdir()
        assert check_math_rendering.list_pages(tmp_path) == []

    def test_list_pages_returns_nothing_for_a_missing_directory(self, tmp_path):
        assert check_math_rendering.list_pages(tmp_path / "absent") == []

    def test_scan_pages_replaces_bytes_that_cannot_be_decoded(self, tmp_path):
        (tmp_path / "theory.html").write_bytes(b'<span class="math">\\(\xff\\)</span>')
        pages = check_math_rendering.list_pages(tmp_path)
        _, math_fragments, _ = check_math_rendering.scan_pages(tmp_path, pages)[0]
        assert math_fragments == ["\\(\ufffd\\)"]

    def test_scan_pages_reports_math_and_mathjax_per_page(self, tmp_path):
        self.write_page(tmp_path, "theory.html", INLINE_MATH)
        self.write_page(tmp_path, "readme.html", f"<p>text</p>{MATHJAX_SCRIPT}")
        pages = check_math_rendering.list_pages(tmp_path)
        assert check_math_rendering.scan_pages(tmp_path, pages) == [
            (pathlib.Path("readme.html"), [], True),
            (pathlib.Path("theory.html"), ["\\(\\Lambda_0\\)"], False),
        ]

    def test_find_pages_missing_mathjax_reports_math_without_the_loader(self):
        scanned_pages = [
            (pathlib.Path("theory.html"), ["\\(\\Lambda_0\\)"], False),
            (pathlib.Path("docs/theory.html"), ["\\(\\Lambda_0\\)"], True),
            (pathlib.Path("readme.html"), [], False),
        ]
        assert check_math_rendering.find_pages_missing_mathjax(scanned_pages) == [
            (pathlib.Path("theory.html"), ["\\(\\Lambda_0\\)"])
        ]

    def test_main_accepts_a_build_whose_math_is_rendered(self, tmp_path, monkeypatch, capsys):
        self.write_page(tmp_path, "theory.html", f"{INLINE_MATH}{MATHJAX_SCRIPT}")
        self.write_page(tmp_path, "readme.html", "<p>text</p>")
        monkeypatch.setattr("sys.argv", ["check_math_rendering.py", "--build-dir", str(tmp_path)])
        assert check_math_rendering.main() == 0
        assert capsys.readouterr().out == ""

    def test_main_reports_a_page_whose_math_is_left_unrendered(self, tmp_path, monkeypatch, capsys):
        self.write_page(tmp_path, "docs/theory.html", f"{INLINE_MATH}{MATHJAX_SCRIPT}")
        self.write_page(tmp_path, "theory.html", INLINE_MATH)
        monkeypatch.setattr("sys.argv", ["check_math_rendering.py", "--build-dir", str(tmp_path)])
        assert check_math_rendering.main() == 1
        output = capsys.readouterr().out
        assert output.count("::error::") == 1
        assert "theory.html holds 1 math element(s), starting with \\(\\Lambda_0\\)" in output
        assert "docs/theory.html" not in output

    @pytest.mark.parametrize(
        "math_fragment, expected",
        [
            ("\\[R_{\\mu\\nu} &= \\frac{1}{4} \\text{Tr}\\left[U c_\\mu\\right]\\]", True),
            ("\\[R_{\\mu\\nu} = \\frac{1}{4} \\text{Tr}\\left[U c_\\mu\\right]\\]", False),
            ("\\[\\begin{aligned} R_{\\mu\\nu} &= \\frac{1}{4} T \\end{aligned}\\]", False),
            ("\\[\\begin{split}U = M \\\\ = \\begin{bmatrix} 1 & 0 \\end{bmatrix}\\end{split}\\]", False),
            ("\\[\\Lambda_0 = \\begin{pmatrix} 0 & 1 \\\\ -1 & 0 \\end{pmatrix}.\\]", False),
            ("\\[\\begin{aligned} a &= b \\end{aligned} & \\begin{aligned} c &= d \\end{aligned}\\]", True),
            ("\\[\\text{a \\& b}\\]", False),
            ("\\[a = 1 \\\\& b = 2\\]", True),
            ("\\[\\begin {aligned} a &= b \\end {aligned}\\]", False),
            ("\\(\\Lambda_0\\)", False),
        ],
    )
    def test_has_unaligned_ampersand_only_flags_a_marker_outside_an_environment(self, math_fragment, expected):
        assert check_math_rendering.has_unaligned_ampersand(math_fragment) is expected

    def test_find_unaligned_ampersands_reports_the_page_and_the_formula(self):
        scanned_pages = [
            (pathlib.Path("matchcake.html"), ["\\[R &= T\\]", "\\(R\\)"], True),
            (pathlib.Path("theory.html"), ["\\[\\begin{pmatrix} 0 & 1 \\end{pmatrix}\\]"], True),
        ]
        assert check_math_rendering.find_unaligned_ampersands(scanned_pages) == [
            (pathlib.Path("matchcake.html"), "\\[R &= T\\]")
        ]

    def test_main_reports_a_formula_whose_alignment_marker_is_misplaced(self, tmp_path, monkeypatch, capsys):
        self.write_page(tmp_path, "matchcake.html", f'<div class="math">\\[R &amp;= T\\]</div>{MATHJAX_SCRIPT}')
        monkeypatch.setattr("sys.argv", ["check_math_rendering.py", "--build-dir", str(tmp_path)])
        assert check_math_rendering.main() == 1
        output = capsys.readouterr().out
        assert output.count("::error::") == 1
        assert "Misplaced &" in output
        assert "\\[R &= T\\]" in output

    def test_main_reports_a_page_that_cannot_be_read(self, tmp_path, monkeypatch, capsys):
        self.write_page(tmp_path, "theory.html", f"{INLINE_MATH}{MATHJAX_SCRIPT}")
        (tmp_path / "broken.html").symlink_to(tmp_path / "absent.html")
        monkeypatch.setattr("sys.argv", ["check_math_rendering.py", "--build-dir", str(tmp_path)])
        assert check_math_rendering.main() == 1
        assert "could not be read" in capsys.readouterr().out

    def test_main_reports_a_build_that_holds_no_math_at_all(self, tmp_path, monkeypatch, capsys):
        self.write_page(tmp_path, "readme.html", "<p>text</p>")
        monkeypatch.setattr("sys.argv", ["check_math_rendering.py", "--build-dir", str(tmp_path)])
        assert check_math_rendering.main() == 1
        assert "holds any math" in capsys.readouterr().out
