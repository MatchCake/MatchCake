import argparse
import html.parser
import pathlib
import re
import sys
from collections.abc import Iterable

MATH_ELEMENT_TAGS = ("span", "div")
MATH_CLASS = "math"
MATHJAX_SOURCE_PATTERN = re.compile("mathjax", re.IGNORECASE)
PAGE_SUFFIX = ".html"
ENVIRONMENT_PATTERN = re.compile(r"\\begin\s*\{(?P<name>[^{}]*)\}.*?\\end\s*\{(?P=name)\}", re.DOTALL)
ESCAPED_TOKEN_PATTERN = re.compile(r"\\\\|\\&")


class MathMarkupParser(html.parser.HTMLParser):
    """
    Collect the math elements and the script sources of one built documentation page.

    ``sphinx.ext.mathjax`` renders inline math as a ``span`` and display math as a ``div``, both
    carrying the ``math`` class and holding the LaTeX source of the formula wrapped in the
    delimiters MathJax looks for. That source only becomes a formula once MathJax runs on the page,
    and Sphinx loads MathJax only on the pages whose own document holds math. A page that quotes
    math it did not write, such as a ``toctree`` entry repeating a math-bearing section title,
    therefore ends up with the markup but without the script that acts on it.

    Character references are converted while parsing, so the collected math holds the LaTeX source
    as the reader sees it rather than its escaped form.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.math_fragments: list[str] = []
        self.script_sources: list[str] = []
        self._open_math_tag: str | None = None
        self._open_math_depth = 0
        self._math_text: list[str] = []

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        """
        Record a script source, and open a math element or track nesting inside the open one.

        :param tag: Name of the element that starts.
        :type tag: str
        :param attributes: Attributes of the element, as ``(name, value)`` pairs.
        :type attributes: list[tuple[str, str | None]]
        :return: Nothing.
        :rtype: None
        """
        values = dict(attributes)
        source = values.get("src")
        if tag == "script" and source is not None:
            self.script_sources.append(source)
        if self._open_math_tag is not None:
            if tag == self._open_math_tag:
                self._open_math_depth += 1
            return
        if tag in MATH_ELEMENT_TAGS and MATH_CLASS in (values.get("class") or "").split():
            self._open_math_tag = tag
            self._open_math_depth = 1
            self._math_text = []

    def handle_endtag(self, tag: str) -> None:
        """
        Close the open math element once its own end tag is reached.

        :param tag: Name of the element that ends.
        :type tag: str
        :return: Nothing.
        :rtype: None
        """
        if self._open_math_tag is None or tag != self._open_math_tag:
            return
        self._open_math_depth -= 1
        if self._open_math_depth == 0:
            self._close_math_element()

    def handle_data(self, data: str) -> None:
        """
        Keep the text held by the open math element.

        :param data: Text read between two tags.
        :type data: str
        :return: Nothing.
        :rtype: None
        """
        if self._open_math_tag is not None:
            self._math_text.append(data)

    def close(self) -> None:
        """
        Finish parsing, keeping a math element that the page leaves open.

        ``html.parser.HTMLParser`` does not close the elements a truncated page leaves open, so
        without this the math of such a page would be read as no math at all and the page would
        pass the check it is meant to fail.

        :return: Nothing.
        :rtype: None
        """
        super().close()
        self._close_math_element()

    def _close_math_element(self) -> None:
        """
        Keep the text of the open math element, if any, and forget it.

        :return: Nothing.
        :rtype: None
        """
        if self._open_math_tag is None:
            return
        self.math_fragments.append(" ".join("".join(self._math_text).split()))
        self._open_math_tag = None
        self._open_math_depth = 0
        self._math_text = []


def list_pages(root: pathlib.Path) -> list[pathlib.Path]:
    """
    List the HTML pages of a built documentation tree.

    The suffix is matched whatever its case, so that a page a build names in capitals is looked at
    rather than silently left out of the check.

    :param root: Path to the root of the built documentation tree.
    :type root: pathlib.Path
    :return: The pages found, relative to the root, in a stable order.
    :rtype: list[pathlib.Path]
    """
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root) for path in root.rglob("*") if path.suffix.lower() == PAGE_SUFFIX and not path.is_dir()
    )


def read_page(text: str) -> tuple[list[str], list[str]]:
    """
    Read the math elements and the script sources out of the HTML of one page.

    :param text: Content of a built HTML page.
    :type text: str
    :return: The text of every math element, and the source of every script the page loads.
    :rtype: tuple[list[str], list[str]]
    """
    parser = MathMarkupParser()
    parser.feed(text)
    parser.close()
    return parser.math_fragments, parser.script_sources


def loads_mathjax(script_sources: Iterable[str]) -> bool:
    """
    Tell whether one of the scripts a page loads is MathJax.

    The MathJax loader is recognized by its name rather than by the exact URL, so that the check
    keeps working whether it is served from a content delivery network or from the static folder
    of the build.

    :param script_sources: Source of every script the page loads.
    :type script_sources: Iterable[str]
    :return: Whether the page loads MathJax.
    :rtype: bool
    """
    return any(MATHJAX_SOURCE_PATTERN.search(source) is not None for source in script_sources)


def scan_pages(root: pathlib.Path, pages: Iterable[pathlib.Path]) -> list[tuple[pathlib.Path, list[str], bool]]:
    """
    Read every page of a built documentation tree.

    A byte a page cannot be decoded from is replaced rather than raised over, since the markup the
    check looks at is plain ASCII and a page is worth looking at even when part of its text is not
    readable.

    :param root: Path to the root of the built documentation tree.
    :type root: pathlib.Path
    :param pages: Pages to read, relative to the root.
    :type pages: Iterable[pathlib.Path]
    :return: The page, the text of the math elements it holds and whether it loads MathJax, for
        every page read.
    :rtype: list[tuple[pathlib.Path, list[str], bool]]
    """
    scanned_pages = []
    for page in pages:
        text = (root / page).read_text(encoding="utf-8", errors="replace")
        math_fragments, script_sources = read_page(text)
        scanned_pages.append((page, math_fragments, loads_mathjax(script_sources)))
    return scanned_pages


def find_pages_missing_mathjax(
    scanned_pages: Iterable[tuple[pathlib.Path, list[str], bool]],
) -> list[tuple[pathlib.Path, list[str]]]:
    """
    Find the pages that hold math markup without loading the script that renders it.

    :param scanned_pages: Pages read by :func:`scan_pages`.
    :type scanned_pages: Iterable[tuple[pathlib.Path, list[str], bool]]
    :return: The page and the text of the math elements it holds, for every such page.
    :rtype: list[tuple[pathlib.Path, list[str]]]
    """
    return [
        (page, math_fragments)
        for page, math_fragments, page_loads_mathjax in scanned_pages
        if math_fragments and not page_loads_mathjax
    ]


def has_unaligned_ampersand(math_fragment: str) -> bool:
    """
    Tell whether a formula uses an alignment marker outside of an alignment environment.

    An ``&`` only means "align here" inside an environment such as ``aligned``, ``split`` or
    ``pmatrix``; anywhere else MathJax stops with ``Misplaced &`` and prints the LaTeX source of the
    formula in its place. Sphinx wraps a formula in ``split`` on its own only when the formula holds
    a ``\\\\`` row separator, so a one-line formula written with a leftover ``&=`` reaches MathJax
    bare.

    Escaped tokens are removed first, in one left to right pass, so that the second backslash of a
    ``\\\\`` row separator is not mistaken for the backslash of an escaped ``\\&``. Environments are
    then removed innermost first, and whatever ``&`` is left belongs to none of them.

    This reads the source rather than parsing it, so it recognizes the one failure it is named for
    and not every formula MathJax could refuse. It does not model the argument of a macro that takes
    alignment of its own, such as ``\\substack``, nor an ``&`` inside a TeX comment, nor an
    environment nested inside another of the same name, and it reports those as alignment markers
    left outside an environment. Reword the formula, or teach this function about the construct.

    :param math_fragment: LaTeX source of one formula, as the page hands it to MathJax.
    :type math_fragment: str
    :return: Whether an alignment marker is left outside of every environment.
    :rtype: bool
    """
    body = ESCAPED_TOKEN_PATTERN.sub("", math_fragment)
    while True:
        stripped = ENVIRONMENT_PATTERN.sub("", body)
        if stripped == body:
            return "&" in body
        body = stripped


def find_unaligned_ampersands(
    scanned_pages: Iterable[tuple[pathlib.Path, list[str], bool]],
) -> list[tuple[pathlib.Path, str]]:
    """
    Find the formulas that MathJax refuses over an alignment marker it cannot place.

    :param scanned_pages: Pages read by :func:`scan_pages`.
    :type scanned_pages: Iterable[tuple[pathlib.Path, list[str], bool]]
    :return: The page and the LaTeX source of every such formula.
    :rtype: list[tuple[pathlib.Path, str]]
    """
    return [
        (page, math_fragment)
        for page, math_fragments, _ in scanned_pages
        for math_fragment in math_fragments
        if has_unaligned_ampersand(math_fragment)
    ]


def main() -> int:
    """
    Report every formula of a built documentation tree that the reader does not see as a formula.

    Two ways of ending up with LaTeX source on the page are looked for: a page holding math without
    loading MathJax at all, and a formula MathJax refuses over an alignment marker it cannot place.
    Each is printed as a GitHub Actions error annotation. A build in which no page holds any math is
    reported as an error too, since the check would otherwise pass without having looked at
    anything, which is what a wrong or stale build directory looks like. A page that cannot be read
    at all is reported the same way rather than raised over, so that the reason the check could not
    run reaches the reader of the log.

    :return: The exit status, 1 when the check found something to report and 0 otherwise.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Check that every formula of the built documentation reaches the reader as a formula."
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=pathlib.Path("sphinx") / "build" / "html",
        help="Path to the root of the built HTML documentation. Defaults to sphinx/build/html.",
    )
    arguments = parser.parse_args()

    try:
        scanned_pages = scan_pages(arguments.build_dir, list_pages(arguments.build_dir))
    except OSError as error:
        print(f"::error::A page of {arguments.build_dir} could not be read: {error}")
        return 1

    if not any(math_fragments for _, math_fragments, _ in scanned_pages):
        print(
            f"::error::No page of {arguments.build_dir} holds any math, so this check looked at "
            f"nothing. Point --build-dir at a freshly built HTML documentation tree."
        )
        return 1

    pages_missing_mathjax = find_pages_missing_mathjax(scanned_pages)
    for page, math_fragments in pages_missing_mathjax:
        print(
            f"::error::{page} holds {len(math_fragments)} math element(s), starting with "
            f"{math_fragments[0]}, but loads no MathJax, so the reader sees the LaTeX source "
            f"instead of the formula."
        )

    unaligned_ampersands = find_unaligned_ampersands(scanned_pages)
    for page, math_fragment in unaligned_ampersands:
        print(
            f"::error::{page} holds a formula whose alignment marker sits outside of an alignment "
            f"environment, which MathJax refuses with Misplaced &, printing the LaTeX source in "
            f"place of the formula: {math_fragment}. Drop the & or wrap the formula in "
            f"\\begin{{aligned}} ... \\end{{aligned}}."
        )

    return 1 if pages_missing_mathjax or unaligned_ampersands else 0


if __name__ == "__main__":
    sys.exit(main())
