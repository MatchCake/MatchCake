import json
import os

import pytest
from pytest_jsonreport.plugin import JSONReport
import argparse

from tests import configs
from tests.conftest import RUN_SLOW_ARG_NAME


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="tests", help="Path to the tests directory")
    parser.add_argument("--N_RANDOM_TESTS_PER_CASE", type=int, default=configs.N_RANDOM_TESTS_PER_CASE,
                        help="Number of random tests per test case")
    parser.add_argument("--save_report", type=str, default="True", help="Save the report to a file")
    parser.add_argument("--cov", type=str, default="src", help="Coverage")
    parser.add_argument("--cov-report", type=str, default="xml", help="Coverage report")
    parser.add_argument("--durations", type=int, default=10, help="Number of durations")
    parser.add_argument(f"--{RUN_SLOW_ARG_NAME}", type=str, default="True", help="Run slow tests")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    configs.N_RANDOM_TESTS_PER_CASE = args.N_RANDOM_TESTS_PER_CASE
    configs.RUN_SLOW_TESTS = 't' in str(getattr(args, RUN_SLOW_ARG_NAME)).lower()
    json_plugin = JSONReport()
    pytest_main_args_names = ["cov", "cov-report", "durations"]
    pytest_main_args = [f"--cov={args.cov}", f"--cov-report={args.cov_report}", f"--durations={args.durations}"]
    pytest.main([args.path, *pytest_main_args], plugins=[json_plugin])
    json_path = os.path.join(os.getcwd(), "tests", "tmp", f"tests_report_rn{configs.N_RANDOM_TESTS_PER_CASE}.json")
    save_report = 't' in str(args.save_report).lower()
    if save_report:
        json_plugin.save_report(json_path)
        json_data = json.load(open(json_path))
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
