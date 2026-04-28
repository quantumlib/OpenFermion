# How to contribute

Thank you for your interest in contributing to this project! We look forward to working with you.
Here are some guidelines to get you started.

## Before you begin

### Summary

*   Read and sign the [Contributor License Agreement (CLA)](https://cla.developers.google.com/).
*   Read the [code of conduct](CODE_OF_CONDUCT.md).
*   Follow the [contribution process](#development-process).

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a [Contributor License
Agreement](https://cla.developers.google.com/about) (CLA). You (or your employer) retain the
copyright to your contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. If you or your current employer have already signed the Google
CLA (even if it was for a different project), you probably don't need to do it again. Visit
<https://cla.developers.google.com/> to see your current agreements or to sign a new one.

Please note that only original work from you and other people who have signed the CLA can be
incorporated into the project. By signing the Contributor License Agreement, you agree that your
contributions are an original work of authorship.

### Review our community guidelines

In the interest of fostering an open and welcoming environment, contributors and maintainers pledge
to make participation in our project and our community a harassment-free experience for everyone.
Our community aspires to treat everyone equally, and to value all contributions. Please review our
[code of conduct](CODE_OF_CONDUCT.md) for more information.

## Code base conventions

OpenFermion is an open-source Python package for compiling and analyzing quantum algorithms to
simulate fermionic systems, including quantum chemistry. Among other features, it includes data
structures and tools for obtaining and manipulating representations of fermionic and qubit
Hamiltonians.

### Main subdirectories

*   `check/`: contains scripts for testing

*   `docker/`: contains a Docker configuration

*   `docs/`: contains OpenFermion documentation

*   `src/`: contains the main code

*   `dev_tools/`: contains programs and configuration files used during development

The legacy subdirectories `cloud_library/` and `rtd_docs/` should be ignored.

### Coding conventions

#### File naming conventions

*   Regular file names should be in all lower case, using underscores as word separators as needed.
    The names should indicate the purpose of the code in the file, while also be kept as short as
    possible without compromising understandability.

*   Test files are usually named after the file they test but with a name ending in `_test.py`. For
    example, `something.py` would have tests in a file named `something_test.py`.

#### File structure conventions

*   Files must end with a final newline, unless they are special files that do not normally have
    ending newlines.

#### Code formatting conventions

This project follows Google coding conventions, with a few changes that are mostly defined in
configuration files at the top level of the source tree and in `dev_tools/conf/`. The following
files configure various tools to the project conventions:

*   `dev_tools/conf/.pylintrc`: Python code

*   `.editorconfig`: basic code editor configuration

*   `.hadolint.yaml`: Dockerfiles

*   `.jsonlintrc.yaml`: JSON files

*   `.markdownlintrc`: Markdown files

*   `.yamllint.yaml`: YAML files

#### Code comment conventions

Every source code file longer than 2 lines must begin with a header comment with the copyright and
license. We use the Apache 2.0 license. License headers are necessary in Python, Bash/shell, and
other programming language files, as well as configuration files in YAML, TOML, ini, and other
config file formats. They are not necessary in Markdown or plain text files.

For comments in other parts of the files, follow these guidelines:

*   _Write clear and concise comments_: Comments should explain the "why", not the "what". The
    comments should explain the intent, trade-offs, and reasoning behind the implementation.

*   _Comment sparingly_: Well-written code should be self-documenting where possible. It's not
    necessary to add comments for code fragments that can reasonably be assumed to be
    self-explanatory.

### Python docstrings and documentation conventions

This project uses [Google-style docstrings](
http://google.github.io/styleguide/pyguide.html#381-docstrings) with a Markdown flavor and support
for LaTeX. Docstrings use triple double quotes, and the first line should be a concise one-line
summary of the function or object.

Here is an example docstring:

```python
def some_function(a: int, b: str) -> float:
    r"""One-line summary of method.

    Additional information about the method, perhaps with LaTeX equations:

        $$
        M = \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}
        $$

    Notice that this docstring is an r-string, since the LaTeX has backslashes. We can also include
    example code:

        print(openfermion.FermionOperator('1^ 2'))

    You can also do inline LaTeX like $y = x^2$ and inline code like `openfermion.count_qubits(op)`.

    Finally, there should be the standard Python docstring sections for arguments, return values,
    and other information, as shown in the remaining lines below.

    Args:
        a: The first argument.
        b: Another argument.

    Returns:
        An important value.

    Raises:
        ValueError: The value of `a` wasn't quite right.
    """
```

#### Git commit conventions

*   Use `git commit` to commit changes to files as you work. Each commit should encompass a
    subportion of your work that is conceptually related.

*   Each commit must have a title and a description.

## Development process

All submissions, including submissions by project members, require review. We use the tools provided
by GitHub for [pull requests](https://docs.github.com/articles/about-pull-requests) for this
purpose. The preferred manner for submitting pull requests is to
[fork](https://docs.github.com/articles/fork-a-repo) the repository, create a new [git
branch](https://docs.github.com/articles/about-branches) in this fork to do your work, and when
ready, create a pull request from your branch to the main project repository.

### Repository forks

1.  Fork the OpenFermion repository on GitHub. Forking creates a new GitHub repo at the location
    `https://github.com/USERNAME/OpenFermion`, where `USERNAME` is your GitHub user name.

2.  Clone (using `git clone`) or otherwise download your forked repository to your local computer,
    so that you have a local copy where you can do your development work using your preferred editor
    and development tools.

3.  Check out the `main` branch and create a new [git branch](
    https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell) from `main`:

    ```shell
    git checkout main -b YOUR_BRANCH_NAME
    ```

    where `YOUR_BRANCH_NAME` is the name of your new branch.

### `git` configuration

The following command will set up large refactoring revisions to be ignored by `git blame`:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

### Pre-commit git hooks (optional)

The project includes a `.pre-commit-config.yaml` file for [pre-commit](https://pre-commit.com), an
open-source utility that configures functions to run when triggered by certain git operations such
as `git commit`. This can help you meet project conventions, at the cost of introducing small delays
in `git commit` and `git push` operations. If you want to use `pre-commit`, you can install and
configure it like this:

```shell
pip install pre-commit
pre-commit install -t pre-commit -t pre-push -t commit-msg
```

Next, run it once after installation to initialize it:

```shell
pre-commit run
```

After that, `pre-commit` will run automatically when triggered by specific git operations.

### Python setup

1.  Create a Python virtual environment. To use Python's built-in `venv` package, run:

    ```shell
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip
    ```

2.  Use the following command to install Python dependencies into the virtual environment:

    ```shell
    pip install -r dev_tools/requirements/envs/dev.env.txt
    ```

Please refer to the section _Developer install_ of the [installation instructions](docs/install.md)
for information about how to set up a local copy of the software for development.

### Type annotation conventions

Code should have [type annotations](https://www.python.org/dev/peps/pep-0484/). We use
[mypy](http://mypy-lang.org/) to check that type annotations are correct, and the following script
to run it:

```shell
check/mypy
```

### Linting and formatting

Code should meet common style standards for Python and be free of error-prone constructs. We use
[Pylint](https://www.pylint.org/) to check for code lint and [Black](https://github.com/psf/black)
for formatting code.

*   To check that code is formatted properly after editing Python files:

    ```shell
    check/format-incremental
    ```

*   To run the linter:

    ```shell
    check/pylint-changed-files
    ```

### Testing and test coverage

When new functions, classes, and files are introduced, they should also have corresponding tests.
Existing tests must continue to pass (or be updated) when changes are introduced. When writing
tests, follow these general principles:

*   _Isolate tests_: Tests must be independent and must not rely on the state of other tests. Use
    setup and teardown functions to create a clean environment for each test run.

*   _Cover edge cases_: Test for invalid inputs, null values, empty arrays, zero values, and
    off-by-one errors.

*   _Mock dependencies_: In unit tests, external dependencies (e.g., databases, network services,
    file system) must be mocked to ensure the test is isolated to the unit under test.

*   _Use asserts intelligently_: Test assertions should be specific. Instead of just asserting
    `true`, assert that a specific value equals an expected value. Provide meaningful failure
    messages.

We use [pytest](https://docs.pytest.org) to run our tests and
[pytest-cov](https://pytest-cov.readthedocs.io) to compute coverage.

*   While developing, periodically check that changes do not break anything. For fast checks, use
    `pytest -c dev_tools/conf/pytest.ini PATH`, where `PATH` is a directory or pytest file to test.

*   After finishing a task, run `check/pytest` to test all of the OpenFermion code.

We don't require 100% coverage, but coverage should be very high, and any uncovered code must be
annotated with `# pragma: no cover`. To ignore coverage of a single line, place `# pragma: no cover`
at the end of the line. To ignore coverage for an entire block, start the block with a `# pragma: no
cover` comment on its own line. Note, however, that these annotations should be rare.

### Final checks

After a task is finished, run each of the following to make sure everything passes all the tests:

*   `check/format-incremental`
*   `check/pylint`
*   `check/mypy`
*   `check/pytest`
*   `check/pytest-and-incremental-coverage`

### Pull requests and code reviews

1.  If your local copy has drifted out of sync with the `main` branch of the main OpenFermion repo,
    you may need to merge the latest changes into your branch. To do this, first update your local
    `main` and then merge your local `main` into your branch:

    ```shell
    # Track the upstream repo (if your local repo hasn't):
    git remote add upstream https://github.com/quantumlib/OpenFermion.git

    # Update your local main.
    git fetch upstream
    git checkout main
    git merge upstream/main
    # Merge local main into your branch.
    git checkout YOUR_BRANCH_NAME
    git merge main
    ```

    If git reports conflicts during one or both of these merge processes, you may need to [resolve
    the merge conflicts]( https://docs.github.com/articles/about-merge-conflicts) before continuing.

1.  Finally, push your changes to your fork of the OpenFermion repo on GitHub:

    ```shell
    git push origin YOUR_BRANCH_NAME
    ```

1.  Now when you navigate to the OpenFermion repository on GitHub
    (https://github.com/quantumlib/OpenFermion), you should see the option to create a new [pull
    request](https://help.github.com/articles/about-pull-requests/) from your forked repository.
    Alternatively, you can create the pull request by navigating to the "Pull requests" tab near the
    top of the page, and selecting the appropriate branches.

1.  A reviewer from the OpenFermion team will comment on your code and may ask for changes. You can
    perform the necessary changes locally, commit them to your branch as usual, and then push
    changes to your fork on GitHub following the same process as above. When you do that, GitHub
    will update the code in the pull request automatically.
