import pytest
from .configs import RUN_SLOW_TESTS

RUN_SLOW_ARG_NAME = "run_slow"


@pytest.hookimpl()
def pytest_sessionstart(session):
    pass


# @pytest.hookimpl()
# def pytest_sessionfinish(session, exitstatus):
#     reporter = session.config.pluginmanager.get_plugin('terminalreporter')
#     print('passed amount:', len(reporter.stats['passed']))


def pytest_addoption(parser):
    r"""
    Add options to the pytest command.

    See: https://jwodder.github.io/kbits/posts/pytest-mark-off/.

    :param parser: The pytest parser.
    :return: None
    """
    parser.addoption(
        f"--{RUN_SLOW_ARG_NAME}",
        action="store_true",
        default=RUN_SLOW_TESTS,
        help="Run slow tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption(f"--{RUN_SLOW_ARG_NAME}"):
        # if RUN_SLOW_TESTS:
        #     print("Running slow tests")
        pass
    else:
        # print("Skipping slow tests")
        skipper = pytest.mark.skip(reason=f"Only run when '--{RUN_SLOW_ARG_NAME}=True' is given")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skipper)
