# Introduction

Thank you for your interest in contributing to our project! We welcome contributions from the community and appreciate
your efforts to help improve our codebase. This document provides guidelines and instructions for contributing to our
project.

- [Getting help and reporting problems](#getting-help-and-reporting-problems)
- [Installation](#installation)
- [Coding conventions](#coding-conventions)
- [Running the checks that CI enforces](#running-the-checks-that-ci-enforces)
- [Building the documentation locally](#building-the-documentation-locally)
- [Guidelines](#guidelines)
- [Code of Conduct](#code-of-conduct)


# Getting help and reporting problems

- To report a bug or to request a feature,
  [open an issue](https://github.com/MatchCake/MatchCake/issues/new/choose). Please search the
  [existing issues](https://github.com/MatchCake/MatchCake/issues) first to make sure it has not already been reported.
  The bug report template asks for a minimal reproducer, the traceback, and information about your Python version and
  platform.
- To ask a question about how to use MatchCake, or to discuss an idea before you write code, open a thread in
  [GitHub Discussions](https://github.com/MatchCake/MatchCake/discussions).


## Installation

To set up the development environment, please follow these steps:

1. Fork the repository:
   ```bash
    gh repo fork MatchCake/MatchCake
   ```
2. Clone your forked repository:
   ```bash
    git clone <your-fork-url>
    ```
3. Navigate to the project directory:
4. Initialize the Virtual Environment using [uv](https://docs.astral.sh/uv/):
    ```bash
     uv sync --dev --extra <extra-option>
    ```
    - Extra options: See the [README.md](README.md) for details on extra options for installation. If you are
      unsure, use `--extra cpu`, which installs the CPU build of PyTorch and is what CI uses.
5. Set up Pre-commit hooks:
    ```bash
     uv run pre-commit install
    ```

MatchCake supports Python 3.11 through 3.14. Every command below is prefixed with `uv run` so that it executes inside
the project environment; there is no need to activate the virtual environment yourself.


# Coding conventions

The conventions that reviewers apply are written down in [`AGENTS.md`](AGENTS.md) at the root of the repository.
Please read it before writing code. In short, it covers:

- **Files and comments**: no file-level or module-level docstrings, and no separator comments between methods.
- **Naming**: `snake_case` variables, `UPPER_SNAKE_CASE` module-level constants, and descriptive names over
  abbreviations.
- **Type hints and docstrings**: full type annotations on every public parameter and return value, with Sphinx style
  docstrings (`:param name:`, `:return:`, `:rtype:`).
- **Tensors and backends**: a function taking a `TensorLike` returns the same backend and dtype as its primary input
  (via `convert_and_cast_like`), and supports arbitrary leading batch dimensions.
- **Testing**: one `Test*` class per file, no module-level `test_*` functions, a `tests/` layout that mirrors the
  package layout, tolerance constants taken from `tests/configs.py`, and tests that are safe under `pytest-xdist`.
- **Class method ordering**: static methods, class methods, `__init__`, dunder methods, public methods, protected
  methods, private methods, then properties.

How matchgate operations, single particle transition matrices and fermionic operators are named is documented
separately in [`naming_conventions.md`](naming_conventions.md).


# Running the checks that CI enforces

Every pull request runs [`.github/workflows/tests.yml`](.github/workflows/tests.yml) on Ubuntu and on Windows, across Python
3.11, 3.12, 3.13 and 3.14. The sections below show how to run the same checks on your machine before you open the pull
request.

## Pre-commit hooks

```bash
uv run pre-commit run --all-files
```

The hooks are declared in [`.pre-commit-config.yaml`](.pre-commit-config.yaml). They check trailing whitespace, end
of file newlines, YAML and TOML syntax, merge conflict markers and accidentally committed private keys; they keep
`uv.lock` and `requirements.txt` in sync with `pyproject.toml`; and they run the Ruff linter and formatter. Several
hooks rewrite files instead of only reporting problems, so if a run fails, inspect the changes, stage them and commit
again.

## Linting and formatting

```bash
uv run ruff format --check src tests
uv run ruff check src tests
```

These are the two commands CI runs, and a failure in either one fails the build. To apply the fixes rather than only
reporting them:

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
```

Ruff is configured in `pyproject.toml`: a line length of 120, double quotes, and the `E`, `F` and `I` rule sets
(pycodestyle errors, Pyflakes and import sorting).

## Type checking

```bash
uv run mypy src tests
```

Type checking is reported in CI but does not currently block the build, because a number of errors predate the check.
Please do not add new ones in the code you touch. Type checking will be enforce in the future.

## Test suite and coverage

```bash
uv run pytest --session-timeout=600
```

The default options live under `[tool.pytest.ini_options]` in `pyproject.toml`, so this single command already collects
everything under `tests/`, runs it in parallel (`-n=auto`), measures branch coverage of `src/`, and prints the lines
that are not covered.

While you iterate, it is often useful to narrow the run:

```bash
uv run pytest tests/test_utils/test_jordan_wigner.py     # a single file
uv run pytest -k "pfaffian"                              # tests matching an expression
uv run pytest -n=0 -x --pdb                              # serial, stop at the first failure, drop into the debugger
```

New and modified code must reach **at least 98% coverage**. To apply that gate locally exactly as CI does:

```bash
uv run pytest --cov-fail-under=98 --session-timeout=600
```

On a pull request opened from a fork, that flag is passed to pytest directly. On a pull request opened from a branch of
this repository, a bot posts a coverage report on the pull request instead and applies the same 98% threshold to the
overall coverage, to the new lines and to the modified lines.

## Notebook tests

The notebooks in `tutorials/` are executed in CI, so a change to a public API can break them even when the test suite
passes:

```bash
uv run pytest --nbmake tutorials -n=auto --nbmake-kernel=python3 --nbmake-timeout=600
```

Each notebook is executed as it is stored in the repository, with a 10 minute timeout.

## Package build

```bash
uv run python -m build --sdist --wheel --no-isolation --outdir dist/ .
uv run twine check dist/*
```

This is the last step of the Ubuntu job on Python 3.11, and it catches packaging metadata problems.


# Building the documentation locally

The documentation is built with Sphinx from `sphinx/source`:

```bash
uv run make -C sphinx clean
uv run make -C sphinx html
```

The result lands in `sphinx/build/html`; open `sphinx/build/html/index.html` in your browser. If you add a new module
or subpackage, regenerate the API stubs before building:

```bash
uv run sphinx-apidoc -f -o sphinx/source src/matchcake
```

Notebooks are not executed during the documentation build (`nb_execution_mode = "off"` in `sphinx/source/conf.py`), so
the outputs rendered on the site are the ones stored in the notebook files. Use the nbmake command above to check that
the notebooks still run.

The published site is built by [`.github/workflows/docs.yml`](.github/workflows/docs.yml) with `sphinx-multiversion`, which
builds the `main` and `dev` branches from the remote. It therefore never reflects uncommitted work, so use
`uv run make -C sphinx html` when you are developing.


# Guidelines

1. When you whish to contribute, please create a new fork of the repository and create a new branch for your changes.
2. Make your changes and commit them with clear and descriptive commit messages.
3. Run the checks described in [Running the checks that CI enforces](#running-the-checks-that-ci-enforces) and make
   sure they pass locally.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository on the `dev` branch, describing the changes you have made and why they
   are necessary, and complete the checklist in the pull request template.
6. Our team will review your pull request and provide feedback. We may ask for changes or improvements before merging
   your contribution.
7. Once your contribution is approved, it will be merged into the main repository on the `dev` branch and will
   eventually be included in the next release.
8. Please ensure that your code follows the project's coding standards and guidelines, and that it includes appropriate
   tests and documentation.


## Code of Conduct

We expect all contributors to adhere to our code of conduct, which can be found in the [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) file. This code of conduct outlines our expectations for behavior and communication within the community, and we take it seriously to ensure a welcoming and inclusive environment for all contributors.
