import ast
import importlib.util
import pathlib
import re
import subprocess

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPOSITORY_ROOT / ".github" / "scripts" / "resolve_repository_links.py"
SPHINX_CONF_PATH = REPOSITORY_ROOT / "sphinx" / "source" / "conf.py"

_SPEC = importlib.util.spec_from_file_location("resolve_repository_links", SCRIPT_PATH)
resolve_repository_links = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(resolve_repository_links)


class TestResolveRepositoryLinks:
    @staticmethod
    def make_repository(root, sources):
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        for name, text in sources.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        subprocess.run(["git", "add", *sources], cwd=root, check=True, capture_output=True)

    def test_find_repository_links_reports_line_number_and_path(self):
        text = "intro\n[guidelines](https://github.com/MatchCake/MatchCake/blob/dev/CONTRIBUTING.md)\n"
        assert resolve_repository_links.find_repository_links(text) == [(2, "CONTRIBUTING.md")]

    def test_find_repository_links_accepts_a_tree_link(self):
        text = "[sources](https://github.com/MatchCake/MatchCake/tree/main/src/matchcake)"
        assert resolve_repository_links.find_repository_links(text) == [(1, "src/matchcake")]

    def test_find_repository_links_strips_anchor_and_query(self):
        text = (
            "[a](https://github.com/MatchCake/MatchCake/blob/dev/AGENTS.md#naming-conventions)\n"
            '<img src="https://github.com/MatchCake/MatchCake/blob/main/images/logo/Logo.svg?raw=true" />\n'
        )
        assert resolve_repository_links.find_repository_links(text) == [
            (1, "AGENTS.md"),
            (2, "images/logo/Logo.svg"),
        ]

    def test_find_repository_links_stops_at_an_escaped_quotation_mark(self):
        text = (
            '    "    <a href=\\"https://github.com/MatchCake/MatchCake/blob/main/tutorials/'
            'matchcake_basics.ipynb\\"><img src=\\"https://github.com/MatchCake/MatchCake/blob/main/'
            'images/logo/Logo.svg?raw=true\\" />Download notebook</a>\\n",'
        )
        assert resolve_repository_links.find_repository_links(text) == [
            (1, "tutorials/matchcake_basics.ipynb"),
            (1, "images/logo/Logo.svg"),
        ]

    def test_find_repository_links_strips_trailing_punctuation(self):
        text = (
            "See https://github.com/MatchCake/MatchCake/blob/dev/LICENSE, "
            "or https://github.com/MatchCake/MatchCake/blob/dev/README.md."
        )
        assert resolve_repository_links.find_repository_links(text) == [(1, "LICENSE"), (1, "README.md")]

    def test_find_repository_links_handles_a_backticked_link(self):
        text = "See `https://github.com/MatchCake/MatchCake/blob/dev/CONTRIBUTING.md` for details."
        assert resolve_repository_links.find_repository_links(text) == [(1, "CONTRIBUTING.md")]

    def test_find_repository_links_handles_emphasis_and_table_delimiters(self):
        text = (
            "**https://github.com/MatchCake/MatchCake/blob/dev/LICENSE**\n"
            "|https://github.com/MatchCake/MatchCake/blob/dev/README.md|\n"
        )
        assert resolve_repository_links.find_repository_links(text) == [(1, "LICENSE"), (2, "README.md")]

    def test_find_repository_links_percent_decodes_the_path(self):
        text = "[x](https://github.com/MatchCake/MatchCake/blob/dev/docs/some%20file.md)"
        assert resolve_repository_links.find_repository_links(text) == [(1, "docs/some file.md")]

    def test_find_repository_links_ignores_a_link_without_a_path(self):
        text = "[branch](https://github.com/MatchCake/MatchCake/tree/dev)"
        assert resolve_repository_links.find_repository_links(text) == []

    def test_linkcheck_ignore_covers_exactly_what_the_script_checks(self):
        module = ast.parse(SPHINX_CONF_PATH.read_text(encoding="utf-8"))
        patterns = next(
            ast.literal_eval(node.value)
            for node in ast.walk(module)
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "linkcheck_ignore" for t in node.targets)
        )
        is_ignored = [re.compile(pattern).match for pattern in patterns]
        urls = [
            "https://github.com/MatchCake/MatchCake/blob/dev/CONTRIBUTING.md",
            "https://github.com/MatchCake/MatchCake/tree/dev/src/matchcake",
            "https://github.com/MatchCake/MatchCake/blob/main/images/logo/Logo.svg?raw=true",
            "https://github.com/MatchCake/MatchCake/tree/dev",
            "https://github.com/MatchCake/MatchCake/blob/dev",
            "https://github.com/MatchCake/MatchCake/issues/new/choose",
            "https://github.com/pennylane/pennylane/blob/master/README.md",
        ]
        for url in urls:
            skipped_by_linkcheck = any(match(url) for match in is_ignored)
            checked_by_script = bool(resolve_repository_links.find_repository_links(url))
            assert skipped_by_linkcheck == checked_by_script, f"{url} is checked by neither or by both"

    def test_find_repository_links_ignores_foreign_urls(self):
        text = (
            "[other repository](https://github.com/pennylane/pennylane/blob/master/README.md)\n"
            "[issues](https://github.com/MatchCake/MatchCake/issues/new/choose)\n"
        )
        assert resolve_repository_links.find_repository_links(text) == []

    def test_list_source_files_lists_tracked_documentation(self, tmp_path):
        self.make_repository(
            tmp_path,
            {
                "README.md": "",
                "docs/theory.rst": "",
                "tutorials/demonstration.ipynb": "",
                "src/matchcake.py": "",
                "pyproject.toml": "",
            },
        )
        source_files = resolve_repository_links.list_source_files(tmp_path)
        assert set(source_files) == {
            pathlib.Path("README.md"),
            pathlib.Path("docs/theory.rst"),
            pathlib.Path("tutorials/demonstration.ipynb"),
        }

    def test_find_broken_links_reports_a_missing_target(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "[typo](https://github.com/MatchCake/MatchCake/blob/dev/CONTRIBUTNG.md)", encoding="utf-8"
        )
        broken_links = resolve_repository_links.find_broken_links(tmp_path, [pathlib.Path("README.md")])
        assert broken_links == [(pathlib.Path("README.md"), 1, "CONTRIBUTNG.md")]

    def test_find_broken_links_accepts_an_existing_file_and_directory(self, tmp_path):
        (tmp_path / "LICENSE").write_text("", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "README.md").write_text(
            "[license](https://github.com/MatchCake/MatchCake/blob/dev/LICENSE)\n"
            "[sources](https://github.com/MatchCake/MatchCake/tree/dev/src)\n",
            encoding="utf-8",
        )
        assert resolve_repository_links.find_broken_links(tmp_path, [pathlib.Path("README.md")]) == []

    def test_main_annotates_a_broken_link_and_fails(self, tmp_path, monkeypatch, capsys):
        self.make_repository(
            tmp_path, {"README.md": "\n[typo](https://github.com/MatchCake/MatchCake/blob/dev/NOPE.md)"}
        )
        monkeypatch.setattr("sys.argv", ["resolve_repository_links.py", "--root", str(tmp_path)])
        assert resolve_repository_links.main() == 1
        assert "::error file=README.md,line=2::NOPE.md is linked from README.md" in capsys.readouterr().out

    def test_main_passes_on_this_repository(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["resolve_repository_links.py", "--root", str(REPOSITORY_ROOT)])
        assert resolve_repository_links.main() == 0
        assert capsys.readouterr().out == ""
