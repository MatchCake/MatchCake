import importlib.util
import pathlib
import types
from collections.abc import Callable

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SPHINX_CONF_PATH = REPOSITORY_ROOT / "sphinx" / "source" / "conf.py"
MYST_HANDLER_PRIORITY = 500

_SPEC = importlib.util.spec_from_file_location("matchcake_sphinx_conf", SPHINX_CONF_PATH)
sphinx_conf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sphinx_conf)


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
    Guard the MathJax wiring of the Sphinx configuration.

    ``sphinx.ext.mathjax`` loads MathJax only on the pages whose own document holds math, so a page
    that merely quotes math written elsewhere, such as the ``toctree`` of ``theory.rst`` repeating
    a math-bearing section title, shows the LaTeX source of that title. The configuration asks for
    the assets of every extension on every page to avoid that, and drops the MathJax configuration
    ``myst_parser`` generates, which this theme renders as a link to a file that does not exist.
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

    def test_setup_asks_for_the_assets_of_every_extension_on_every_page(self):
        assert self.run_setup().html_assets_policy == "always"

    def test_setup_connects_every_handler_of_the_configuration(self):
        connected_handlers = {(event, handler) for event, handler, _ in self.run_setup().connections}
        assert connected_handlers == {
            ("builder-inited", sphinx_conf.drop_mathjax3_config),
            ("source-read", sphinx_conf.github_math_to_myst),
            ("autodoc-skip-member", sphinx_conf.skip),
            ("html-page-context", sphinx_conf.change_pathto),
            ("build-finished", sphinx_conf.move_private_folders),
        }

    def test_setup_drops_the_generated_mathjax_configuration_after_myst_writes_it(self):
        priorities = [
            priority
            for event, handler, priority in self.run_setup().connections
            if handler is sphinx_conf.drop_mathjax3_config
        ]
        assert priorities == [MYST_HANDLER_PRIORITY + 300]

    def test_drop_mathjax3_config_clears_the_generated_configuration(self):
        app = _RecordingApp()
        app.config.mathjax3_config = {"options": {"processHtmlClass": "math"}}
        sphinx_conf.drop_mathjax3_config(app)
        assert app.config.mathjax3_config is None
