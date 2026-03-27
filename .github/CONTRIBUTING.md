# Introduction

Thank you for your interest in contributing to our project! We welcome contributions from the community and appreciate your efforts to help improve our codebase. This document provides guidelines and instructions for contributing to our project.


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
    - Extra options: See the [README.md](README.md) for details on extra options for installation.
5. Set up Pre-commit hooks:
    ```bash
     uv run pre-commit install
    ```

# Guidelines

1. When you whish to contribute, please create a new fork of the repository and create a new branch for your changes.
2. Make your changes and commit them with clear and descriptive commit messages.
3. Push your changes to your forked repository.
4. Create a pull request to the main repository on the `dev` branch, describing the changes you have made and why they are necessary.
5. Our team will review your pull request and provide feedback. We may ask for changes or improvements before merging your contribution.
6. Once your contribution is approved, it will be merged into the main repository on the `dev` branch and will eventually be included in the next release.
7. Please ensure that your code follows the project's coding standards and guidelines, and that it includes appropriate tests and documentation.


## Code of Conduct

We expect all contributors to adhere to our code of conduct, which can be found in the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) file. This code of conduct outlines our expectations for behavior and communication within the community, and we take it seriously to ensure a welcoming and inclusive environment for all contributors.
