import json
import os

import pytest
import pythonbasictools as pbt
from pytest_jsonreport.plugin import JSONReport
from pytest_cov.plugin import CovPlugin

from tests import configs
from tests.conftest import RUN_SLOW_ARG_NAME

if __name__ == '__main__':
    sys_args_dict = pbt.cmds.get_cmd_kwargs(
        {
            1                        : os.path.join(os.getcwd(), "tests"),
            "N_RANDOM_TESTS_PER_CASE": configs.N_RANDOM_TESTS_PER_CASE,
            "save_report"            : "True",
            "cov"                    : True,
            RUN_SLOW_ARG_NAME        : "True",
        }
    )
    configs.N_RANDOM_TESTS_PER_CASE = sys_args_dict["N_RANDOM_TESTS_PER_CASE"]
    configs.RUN_SLOW_TESTS = 't' in str(sys_args_dict[RUN_SLOW_ARG_NAME]).lower()
    json_plugin = JSONReport()
    pytest.main([sys_args_dict[1], ], plugins=[json_plugin])
    json_path = os.path.join(os.getcwd(), "tests", "tmp", f"tests_report_rn{configs.N_RANDOM_TESTS_PER_CASE}.json")
    save_report = 't' in str(sys_args_dict["save_report"]).lower()
    if save_report:
        json_plugin.save_report(json_path)
        json_data = json.load(open(json_path))
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
