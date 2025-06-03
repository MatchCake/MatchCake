import argparse
import json
import os

import pytest
from pytest_jsonreport.plugin import JSONReport

from tests import configs
from tests.conftest import RUN_SLOW_ARG_NAME


def get_args_parser():
    parser = argparse.ArgumentParser(description="Tests Runner")
    parser.add_argument(
        "--tests_folder",
        type=str,
        default=os.path.join(os.getcwd(), "tests"),
        help="Path to the folder containing tests.",
    )
    parser.add_argument(
        "--N_RANDOM_TESTS_PER_CASE",
        type=int,
        default=configs.N_RANDOM_TESTS_PER_CASE,
        help="Number of random tests to run per test case.",
    )
    parser.add_argument(
        "--save_report",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to save the report in JSON format.",
    )
    parser.add_argument(
        "--cov",
        type=str,
        default="src",
        help="Path to the source code for coverage.",
    )
    parser.add_argument(
        "--cov-report",
        type=str,
        default="xml:tests/.tmp/coverage.xml",
        help="Format of the coverage report.",
    )
    parser.add_argument(
        f"--{RUN_SLOW_ARG_NAME}",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to run slow tests.",
    )
    parser.add_argument(
        "--durations",
        type=int,
        default=10,
        help="Number of slowest test durations to report.",
    )
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    configs.N_RANDOM_TESTS_PER_CASE = args.N_RANDOM_TESTS_PER_CASE
    configs.RUN_SLOW_TESTS = args.run_slow
    json_plugin = JSONReport()
    pytest_main_args = [
        args.tests_folder,
        f"--cov={args.cov}",
        f"--cov-report={args.cov_report}",
        f"--cov-report=term-missing",
        f"--durations={args.durations}",
    ]
    pytest.main(pytest_main_args, plugins=[json_plugin])
    json_path = os.path.join(args.tests_folder, ".tmp", f"tests_report_rn{configs.N_RANDOM_TESTS_PER_CASE}.json")
    if args.save_report:
        json_plugin.save_report(json_path)
        json_data = json.load(open(json_path))
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
    return 0


if __name__ == '__main__':
    exit(main())
