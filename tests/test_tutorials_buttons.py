import json
import re
from pathlib import Path
from typing import Dict, List

import pytest

TUTORIALS_DIR = Path(__file__).resolve().parents[1] / "tutorials"
NOTEBOOK_PATHS = sorted(TUTORIALS_DIR.glob("*.ipynb"))
NOTEBOOK_NAMES_WITH_A_BUTTON_TABLE = {
    "iris_classification.ipynb",
    "matchcake_basics.ipynb",
    "nystroem_kernel_approximation.ipynb",
}
BUTTON_NOTEBOOK_PATHS = [path for path in NOTEBOOK_PATHS if path.name in NOTEBOOK_NAMES_WITH_A_BUTTON_TABLE]
BUTTON_TABLE_PATTERN = re.compile(r'<table class="nt-notebook-buttons".*?</table>', re.DOTALL)
ANCHOR_PATTERN = re.compile(r'<a\s[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
TAG_PATTERN = re.compile(r"<[^>]+>")
BUTTON_URL_TEMPLATES = {
    "Documentation": "https://MatchCake.github.io/MatchCake/",
    "Run in Google Colab": "https://colab.research.google.com/github/MatchCake/MatchCake/blob/main/tutorials/{name}",
    "View source on GitHub": "https://github.com/MatchCake/MatchCake/blob/main/tutorials/{name}",
    "Download notebook": "https://github.com/MatchCake/MatchCake/blob/main/tutorials/{name}?raw=true",
}


class TestTutorialsButtons:
    """
    Guard the button table that the tutorial notebooks carry above their first code cell.

    The table is raw HTML repeated in every notebook that has one, so a notebook added by copying
    another one inherits its links and silently sends the reader to the wrong tutorial. Nothing else
    in the continuous integration sees these links: Sphinx renders the table into a single docutils
    raw node, whose content the linkcheck builder never parses, and
    ``.github/scripts/resolve_repository_links.py`` only reports a link whose target is missing from
    the checkout, which a link to another existing tutorial is not.
    """

    @staticmethod
    def read_markdown_source(notebook_path: Path) -> str:
        """
        Join the source of the markdown cells of a notebook.

        :param notebook_path: Path of the notebook to read.
        :return: The concatenated source of every markdown cell, in document order.
        :rtype: str
        """
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        sources: List[str] = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            source = cell["source"]
            sources.append("".join(source) if isinstance(source, list) else source)
        return "\n".join(sources)

    @staticmethod
    def find_button_tables(markdown_source: str) -> List[str]:
        """
        Extract the button tables from the markdown source of a notebook.

        Every table found is returned, so that a notebook holding more than one has all of them
        checked rather than only the first.

        :param markdown_source: Concatenated source of the markdown cells of a notebook.
        :return: The raw HTML of every button table, in document order.
        :rtype: List[str]
        """
        return BUTTON_TABLE_PATTERN.findall(markdown_source)

    @staticmethod
    def find_buttons(button_table: str) -> Dict[str, str]:
        """
        Map the label of every button of the table to the address it points to.

        :param button_table: Raw HTML of the button table.
        :return: The link target of every button, keyed by its visible label.
        :rtype: Dict[str, str]
        """
        buttons = {}
        for url, content in ANCHOR_PATTERN.findall(button_table):
            buttons[TAG_PATTERN.sub("", content).strip()] = url
        return buttons

    def test_notebooks_are_discovered(self) -> None:
        assert NOTEBOOK_PATHS, f"No tutorial notebook found in {TUTORIALS_DIR}"

    def test_the_notebooks_carrying_a_button_table_are_the_expected_ones(self) -> None:
        carrying_a_button_table = {
            path.name for path in NOTEBOOK_PATHS if self.find_button_tables(self.read_markdown_source(path))
        }
        assert carrying_a_button_table == NOTEBOOK_NAMES_WITH_A_BUTTON_TABLE, (
            f"{sorted(carrying_a_button_table)} carry a button table but "
            f"{sorted(NOTEBOOK_NAMES_WITH_A_BUTTON_TABLE)} were expected; add a tutorial to "
            f"NOTEBOOK_NAMES_WITH_A_BUTTON_TABLE when you give it a button table, so that its links "
            f"are checked too"
        )

    @pytest.mark.parametrize("notebook_path", BUTTON_NOTEBOOK_PATHS, ids=lambda path: path.name)
    def test_button_table_holds_every_expected_button(self, notebook_path: Path) -> None:
        for button_table in self.find_button_tables(self.read_markdown_source(notebook_path)):
            buttons = self.find_buttons(button_table)
            assert set(buttons) == set(BUTTON_URL_TEMPLATES), (
                f"{notebook_path.name} has the buttons {sorted(buttons)}; {sorted(BUTTON_URL_TEMPLATES)} were expected"
            )

    @pytest.mark.parametrize("notebook_path", BUTTON_NOTEBOOK_PATHS, ids=lambda path: path.name)
    def test_every_button_points_to_the_notebook_that_carries_it(self, notebook_path: Path) -> None:
        mismatches = []
        for button_table in self.find_button_tables(self.read_markdown_source(notebook_path)):
            buttons = self.find_buttons(button_table)
            for label, template in BUTTON_URL_TEMPLATES.items():
                expected_url = template.format(name=notebook_path.name)
                if buttons.get(label) != expected_url:
                    mismatches.append(f"{label!r} points to {buttons.get(label)} instead of {expected_url}")
        assert not mismatches, (
            f"{notebook_path.name} carries button addresses that are not its own: "
            f"{'; '.join(mismatches)}. A button table copied from another tutorial must have every "
            f"address rewritten to the notebook that carries it"
        )
