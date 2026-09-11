import json
import re
from pathlib import Path
from typing import List, Tuple

import pytest

TUTORIALS_DIR = Path(__file__).resolve().parents[1] / "tutorials"
NOTEBOOK_PATHS = sorted(TUTORIALS_DIR.glob("*.ipynb"))
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


class TestTutorialsStructure:
    """
    Guard the heading structure that the Sphinx sidebar relies on.

    ``myst_nb`` renders each tutorial notebook as a single document, and Sphinx only nests a
    document's sections underneath it in a ``toctree`` when that document has one
    top-level section.
    """

    @staticmethod
    def read_headings(notebook_path: Path) -> List[Tuple[int, str]]:
        """
        Collect the ATX headings of the markdown cells of a notebook, in document order.

        :param notebook_path: Path of the notebook to read.
        :return: The ``(level, title)`` pair of every heading found.
        :rtype: List[Tuple[int, str]]
        """
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        headings = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            source = cell["source"]
            lines = source if isinstance(source, list) else source.splitlines()
            is_inside_fence = False
            for line in lines:
                if line.lstrip().startswith("```"):
                    is_inside_fence = not is_inside_fence
                    continue
                if is_inside_fence:
                    continue
                match = HEADING_PATTERN.match(line)
                if match is not None:
                    headings.append((len(match.group(1)), match.group(2).strip()))
        return headings

    def test_notebooks_are_discovered(self) -> None:
        assert NOTEBOOK_PATHS, f"No tutorial notebook found in {TUTORIALS_DIR}"

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS, ids=lambda path: path.name)
    def test_notebook_starts_with_a_title(self, notebook_path: Path) -> None:
        headings = self.read_headings(notebook_path)
        assert headings, f"{notebook_path.name} has no markdown heading to act as a page title"
        assert headings[0][0] == 1, (
            f"{notebook_path.name} starts with a level-{headings[0][0]} heading "
            f"{headings[0][1]!r}; the first heading must be the level-1 page title"
        )

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS, ids=lambda path: path.name)
    def test_notebook_has_a_single_top_level_heading(self, notebook_path: Path) -> None:
        top_level_titles = [title for level, title in self.read_headings(notebook_path) if level == 1]
        assert len(top_level_titles) == 1, (
            f"{notebook_path.name} has {len(top_level_titles)} level-1 headings "
            f"({top_level_titles}); a tutorial must have exactly one so that Sphinx nests its "
            f"sections under it in the sidebar instead of flattening them"
        )

    @pytest.mark.parametrize("notebook_path", NOTEBOOK_PATHS, ids=lambda path: path.name)
    def test_notebook_heading_levels_do_not_skip(self, notebook_path: Path) -> None:
        headings = self.read_headings(notebook_path)
        for (previous_level, previous_title), (level, title) in zip(headings, headings[1:]):
            assert level <= previous_level + 1, (
                f"{notebook_path.name} jumps from level {previous_level} ({previous_title!r}) "
                f"to level {level} ({title!r}); heading levels must increase one at a time"
            )
