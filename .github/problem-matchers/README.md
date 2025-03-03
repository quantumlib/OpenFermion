# Problem Matchers

GitHub [Problem
Matchers](https://github.com/actions/toolkit/blob/main/docs/problem-matchers.md)
are a mechanism that enable workflow steps to scan the outputs of GitHub
Actions for regex patterns and automatically write annotations in the workflow
summary page. Using Problem Matchers allows information to be displayed more
prominently in the GitHub user interface.

This directory contains Problem Matchers used by the GitHub Actions workflows
in the [`workflows`](./workflows) subdirectory.

## Original sources

The following problem matcher JSON files found in this directory were copied
from the [Home Assistant](https://github.com/home-assistant/core) project on
GitHub. The Home Assistant project is licensed under the Apache 2.0 open-source
license. The version of the files at the time they were copied was 2025.1.2.

- [`pylint.json`](https://github.com/home-assistant/core/blob/dev/.github/workflows/matchers/pylint.json)
- [`yamllint.json`](https://github.com/home-assistant/core/blob/dev/.github/workflows/matchers/yamllint.json)

The following problem matcher JSON file came from the
[actionlint](https://github.com/rhysd/actionlint/blob/v1.7.7/docs/usage.md)
documentation (copied on 2025-02-12, version 1.7.7):

- [`actionlint.json`](https://raw.githubusercontent.com/rhysd/actionlint/main/.github/actionlint-matcher.json)

The following problem matcher JSON file came from the
[hadolint-action](https://github.com/hadolint/hadolint-action) repository
(copied on 2025-02-17, version 3.1.0):

- [`hadolint.json`](https://github.com/hadolint/hadolint-action/blob/master/problem-matcher.json)

The Shellcheck problem matcher JSON file came from the
[shellcheck-problem-matchers](uhttps://github.com/lumaxis/shellcheck-problem-matchers)
repository (copied 2025-02-26, version v2.1.0).

- [`shellcheck-tty.json`](https://github.com/lumaxis/shellcheck-problem-matchers/blob/main/.github/shellcheck-tty.json)
