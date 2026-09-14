import importlib.util
import os
import pathlib
import types
import unittest.mock
from collections.abc import Callable

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SPHINX_CONF_PATH = REPOSITORY_ROOT / "sphinx" / "source" / "conf.py"
MYST_HANDLER_PRIORITY = 500
MULTIVERSION_HANDLER_PRIORITY = 500


def load_sphinx_conf(multiversion_sourcedir: str | None = None) -> types.ModuleType:
    """
    Run the Sphinx configuration as a module and hand back what it defined.

    :param multiversion_sourcedir: Value of ``SPHINX_MULTIVERSION_SOURCEDIR`` the configuration is
        read under, or ``None`` to read it as a plain build does, outside of a multiversion run.
    :type multiversion_sourcedir: str | None
    :return: The module the configuration file defines.
    :rtype: types.ModuleType
    """
    environment = dict(os.environ)
    environment.pop("SPHINX_MULTIVERSION_SOURCEDIR", None)
    if multiversion_sourcedir is not None:
        environment["SPHINX_MULTIVERSION_SOURCEDIR"] = multiversion_sourcedir
    spec = importlib.util.spec_from_file_location("matchcake_sphinx_conf", SPHINX_CONF_PATH)
    module = importlib.util.module_from_spec(spec)
    with unittest.mock.patch.dict(os.environ, environment, clear=True):
        spec.loader.exec_module(module)
    return module


sphinx_conf = load_sphinx_conf()


class _RecordingApp:
    """
    Stand-in for the Sphinx application object that the ``setup`` of ``conf.py`` receives.

    Sphinx only comes with the documentation dependency group, which the test job does not install,
    so ``setup`` is driven with this recorder rather than with a real application.
    """

    def __init__(self) -> None:
        self.html_assets_policy: str | None = None
        self.connections: list[tuple[str, Callable, int]] = []
        self.config = types.SimpleNamespace(mathjax3_config=None)

    def set_html_assets_policy(self, policy: str) -> None:
        """
        Record the asset policy asked for.

        :param policy: Policy deciding on which pages the assets of an extension are included.
        :type policy: str
        :return: Nothing.
        :rtype: None
        """
        self.html_assets_policy = policy

    def connect(self, event: str, handler: Callable, priority: int = MYST_HANDLER_PRIORITY) -> None:
        """
        Record a handler connected to an event.

        :param event: Name of the event the handler is connected to.
        :type event: str
        :param handler: Handler called when the event is emitted.
        :type handler: Callable
        :param priority: Rank of the handler among the handlers of that event, lowest called first.
        :type priority: int
        :return: Nothing.
        :rtype: None
        """
        self.connections.append((event, handler, priority))


class TestSphinxConf:
    """
    Guard the MathJax and the multiversion wiring of the Sphinx configuration.

    ``sphinx.ext.mathjax`` loads MathJax only on the pages whose own document holds math, so a page
    that merely quotes math written elsewhere, such as the ``toctree`` of ``theory.rst`` repeating
    a math-bearing section title, shows the LaTeX source of that title. The configuration asks for
    the assets of every extension on every page to avoid that, and drops the MathJax configuration
    ``myst_parser`` generates, which this theme renders as a link to a file that does not exist.

    ``sphinx_multiversion`` builds every published version with the configuration of the ref that
    triggered the run, and hands that configuration only the sources of the ref being built. The
    configuration reads the ref being built out of the environment so that the package documented,
    the version a page is labelled with, and the order of the version selector all follow the ref
    rather than the branch that happened to trigger the run.
    """

    @staticmethod
    def run_setup() -> _RecordingApp:
        """
        Run the ``setup`` of the Sphinx configuration against a recording application.

        :return: The application the configuration was applied to.
        :rtype: _RecordingApp
        """
        app = _RecordingApp()
        sphinx_conf.setup(app)
        return app

    @staticmethod
    def make_config(metadata: dict | None, current_version: str) -> types.SimpleNamespace:
        """
        Build the configuration object ``sphinx_multiversion`` leaves behind on ``config-inited``.

        :param metadata: Description of every ref of the run, or ``None`` outside a multiversion run.
        :type metadata: dict | None
        :param current_version: Name of the ref being built.
        :type current_version: str
        :return: A stand-in for the configuration of the build.
        :rtype: types.SimpleNamespace
        """
        return types.SimpleNamespace(
            smv_metadata=metadata,
            smv_current_version=current_version,
            version="0.9.9",
            release="",
        )

    @staticmethod
    def make_versions(branches: list[str], tags: list[str]) -> types.SimpleNamespace:
        """
        Build the version listing ``sphinx_multiversion`` puts in the context of every page.

        :param branches: Names of the branches the run published.
        :type branches: list[str]
        :param tags: Names of the tags the run published.
        :type tags: list[str]
        :return: A stand-in for the version listing of the run.
        :rtype: types.SimpleNamespace
        """
        return types.SimpleNamespace(
            branches=[types.SimpleNamespace(name=name) for name in branches],
            tags=[types.SimpleNamespace(name=name) for name in tags],
        )

    def test_setup_asks_for_the_assets_of_every_extension_on_every_page(self):
        assert self.run_setup().html_assets_policy == "always"

    def test_setup_connects_every_handler_of_the_configuration(self):
        connected_handlers = {(event, handler) for event, handler, _ in self.run_setup().connections}
        assert connected_handlers == {
            ("builder-inited", sphinx_conf.drop_mathjax3_config),
            ("config-inited", sphinx_conf.name_version_after_ref),
            ("source-read", sphinx_conf.github_math_to_myst),
            ("autodoc-skip-member", sphinx_conf.skip),
            ("html-page-context", sphinx_conf.change_pathto),
            ("html-page-context", sphinx_conf.order_versions),
            ("build-finished", sphinx_conf.move_private_folders),
        }

    def test_setup_drops_the_generated_mathjax_configuration_after_myst_writes_it(self):
        priorities = [
            priority
            for event, handler, priority in self.run_setup().connections
            if handler is sphinx_conf.drop_mathjax3_config
        ]
        assert priorities == [MYST_HANDLER_PRIORITY + 300]

    def test_setup_runs_the_multiversion_handlers_after_the_ones_of_the_extension(self):
        priorities = {
            handler: priority
            for _, handler, priority in self.run_setup().connections
            if handler in (sphinx_conf.name_version_after_ref, sphinx_conf.order_versions)
        }
        assert set(priorities) == {sphinx_conf.name_version_after_ref, sphinx_conf.order_versions}
        assert all(priority > MULTIVERSION_HANDLER_PRIORITY for priority in priorities.values())

    def test_drop_mathjax3_config_clears_the_generated_configuration(self):
        app = _RecordingApp()
        app.config.mathjax3_config = {"options": {"processHtmlClass": "math"}}
        sphinx_conf.drop_mathjax3_config(app)
        assert app.config.mathjax3_config is None

    def test_the_documented_package_is_read_next_to_the_sources_of_the_ref_being_built(self, tmp_path):
        sourcedir = tmp_path / "checkout" / "sphinx" / "source"
        sourcedir.mkdir(parents=True)
        module = load_sphinx_conf(multiversion_sourcedir=str(sourcedir))
        assert pathlib.Path(module.basedir) == tmp_path / "checkout" / "src"

    def test_the_documented_package_is_read_next_to_the_configuration_outside_a_multiversion_run(self):
        assert pathlib.Path(load_sphinx_conf().basedir) == REPOSITORY_ROOT / "src"

    def test_version_sort_key_orders_a_version_by_its_numbers(self):
        assert sphinx_conf._version_sort_key("0.10.0") > sphinx_conf._version_sort_key("0.2.0")

    def test_name_version_after_ref_labels_a_tag_with_its_own_name(self):
        config = self.make_config({"0.2.3": {"source": "tags"}}, "0.2.3")
        sphinx_conf.name_version_after_ref(_RecordingApp(), config)
        assert (config.version, config.release) == ("0.2.3", "0.2.3")

    def test_name_version_after_ref_leaves_a_branch_with_the_version_of_its_package(self):
        config = self.make_config({"dev": {"source": "heads"}}, "dev")
        sphinx_conf.name_version_after_ref(_RecordingApp(), config)
        assert (config.version, config.release) == ("0.9.9", "")

    def test_name_version_after_ref_leaves_a_build_that_is_not_a_multiversion_one_alone(self):
        config = self.make_config(None, "")
        sphinx_conf.name_version_after_ref(_RecordingApp(), config)
        assert (config.version, config.release) == ("0.9.9", "")

    def test_order_versions_lists_the_branches_then_the_tags_from_the_newest(self):
        context = {"versions": self.make_versions(["dev", "main"], ["0.2.3", "0.10.0", "0.3.0"])}
        sphinx_conf.order_versions(_RecordingApp(), "index", "page.html", context, None)
        names = [version.name for version in context["ordered_versions"]]
        assert names == ["dev", "main", "0.10.0", "0.3.0", "0.2.3"]

    def test_order_versions_leaves_a_build_that_is_not_a_multiversion_one_alone(self):
        context = {}
        sphinx_conf.order_versions(_RecordingApp(), "index", "page.html", context, None)
        assert context == {}
