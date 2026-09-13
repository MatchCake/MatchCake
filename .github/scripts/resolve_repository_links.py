import argparse
import pathlib
import re
import subprocess
import sys
import urllib.parse
from collections.abc import Iterable

REPOSITORY_LINK_PATTERN = re.compile(
    r"""https://github\.com/MatchCake/MatchCake/(?:blob|tree)/[^/]+/([^)\s"'>?#`|]+)"""
)
SOURCE_FILE_PATTERNS = ("*.md", "*.rst")
TRAILING_PUNCTUATION = ".,;:*"


def find_repository_links(text: str) -> list[tuple[int, str]]:
    """
    Find every link into this repository contained in a documentation source.

    The branch a link names is ignored, since a link is resolved against the checkout rather than
    over HTTP. Only the path it points to is returned, percent-decoded and stripped of any anchor,
    query string and trailing sentence or markup punctuation. A branch whose own name contains a
    slash is not supported, since nothing in the URL marks where the branch ends and the path
    begins.

    :param text: Content of a documentation source.
    :type text: str
    :return: The one-based line number and the linked path of every link found.
    :rtype: list[tuple[int, str]]
    """
    links = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for target in REPOSITORY_LINK_PATTERN.findall(line):
            links.append((line_number, urllib.parse.unquote(target.rstrip(TRAILING_PUNCTUATION))))
    return links


def list_source_files(root: pathlib.Path) -> list[pathlib.Path]:
    """
    List the tracked documentation sources of the repository.

    :param root: Path to the root of the repository.
    :type root: pathlib.Path
    :return: The tracked Markdown and reStructuredText files, relative to the root.
    :rtype: list[pathlib.Path]
    """
    completed_process = subprocess.run(
        ["git", "ls-files", *SOURCE_FILE_PATTERNS],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [pathlib.Path(line) for line in completed_process.stdout.splitlines()]


def find_broken_links(root: pathlib.Path, source_files: Iterable[pathlib.Path]) -> list[tuple[pathlib.Path, int, str]]:
    """
    Find the links into this repository that do not resolve to a file in the checkout.

    :param root: Path to the root of the repository.
    :type root: pathlib.Path
    :param source_files: Documentation sources to scan, relative to the root.
    :type source_files: Iterable[pathlib.Path]
    :return: The source file, the one-based line number and the linked path of every broken link.
    :rtype: list[tuple[pathlib.Path, int, str]]
    """
    broken_links = []
    for source_file in source_files:
        text = (root / source_file).read_text(encoding="utf-8")
        for line_number, target in find_repository_links(text):
            if not (root / target).exists():
                broken_links.append((source_file, line_number, target))
    return broken_links


def main() -> int:
    """
    Report every link into this repository that does not resolve to a file in the checkout.

    Each broken link is printed as a GitHub Actions error annotation on the line that carries it.

    :return: The exit status, 1 when at least one link is broken and 0 otherwise.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Check that the links into this repository point to a file that exists in the checkout."
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Path to the root of the repository. Defaults to the current working directory.",
    )
    arguments = parser.parse_args()

    broken_links = find_broken_links(arguments.root, list_source_files(arguments.root))
    for source_file, line_number, target in broken_links:
        print(
            f"::error file={source_file},line={line_number}::{target} is linked from {source_file} "
            f"but does not exist in the repository."
        )
    return 1 if broken_links else 0


if __name__ == "__main__":
    sys.exit(main())
