# How to contribute

We'd love to accept your patches and contributions to this project. We do have
some guidelines to follow, covered in this document, but don't worry about (or
expect to) get everything right the first time! Create a pull request and we'll
nudge you in the right direction. Please also note that we have a [code of
conduct](CODE_OF_CONDUCT.md) to make OpenFermion an open and welcoming
environment.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a [Contributor License
Agreement](https://cla.developers.google.com/about) (CLA). You (or your
employer) retain the copyright to your contribution; the CLA simply gives us
permission to use and redistribute your contributions as part of this project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult [GitHub
Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

The preferred approach for submitting pull requests is for developers to fork
the OpenFermion [GitHub repository](https://github.com/quantumlib/OpenFermion)
and then use a [git
branch](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
from the fork to create a pull request to the main OpenFermion repo.

### Development environment setup

Please refer to the section _Developer install_ of the [installation
instructions](docs/install.md) for information about how to set up a local copy
of the software for development.

### Tests and test coverage

Existing tests must continue to pass (or be updated) when changes are
introduced, and code should be covered by tests. We use
[pytest](https://docs.pytest.org) to run our tests and
[pytest-cov](https://pytest-cov.readthedocs.io) to compute coverage. We use the
scripts [`./check/pytest`](./check/pytest) and
[`./check/pytest-and-incremental-coverage`](./check/pytest-and-incremental-coverage)
to run these programs with custom configurations for this project.

We don't require 100% coverage, but any uncovered code must be annotated with
`# pragma: no cover`. To ignore coverage of a single line, place `# pragma: no
cover` at the end of the line. To ignore coverage for an entire block, start
the block with a `# pragma: no cover` comment on its own line.

### Lint

Code should meet common style standards for Python and be free of error-prone
constructs. We use [Pylint](https://www.pylint.org/) to check for code lint,
and the script [`./check/pylint`](./check/pylint) to run it. When Pylint
produces a false positive, it can be silenced with annotations. For example,
the annotation `# pylint: disable=unused-import` would silence a warning about
an unused import.

### Types

Code should have [type annotations](https://www.python.org/dev/peps/pep-0484/).
We use [mypy](http://mypy-lang.org/) to check that type annotations are
correct, and the script [`./check/mypy`](./check/mypy) to run it. When type
checking produces a false positive, it can be silenced with annotations such as
`# type: ignore`.
