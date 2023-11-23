import json
import os

import pytest
import pythonbasictools as pbt
from pytest_jsonreport.plugin import JSONReport

from tests import configs

if __name__ == '__main__':
    sys_args_dict = pbt.cmds.get_cmd_kwargs(
        {
            1                        : os.path.join(os.getcwd(), "tests"),
            "N_RANDOM_TESTS_PER_CASE": configs.N_RANDOM_TESTS_PER_CASE,
        }
    )
    configs.N_RANDOM_TESTS_PER_CASE = sys_args_dict["N_RANDOM_TESTS_PER_CASE"]
    plugin = JSONReport()
    pytest.main([sys_args_dict[1]], plugins=[plugin])
    json_path = os.path.join(os.getcwd(), "tests", "tmp", "tests_report.json")
    plugin.save_report(json_path)
    json_data = json.load(open(json_path))
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
