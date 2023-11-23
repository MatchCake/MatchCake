import pytest
import os
import json


@pytest.hookimpl()
def pytest_sessionstart(session):
    pass


# @pytest.hookimpl()
# def pytest_sessionfinish(session, exitstatus):
#     reporter = session.config.pluginmanager.get_plugin('terminalreporter')
#     print('passed amount:', len(reporter.stats['passed']))
