# Install

Installing OpenFermion requires pip. Make sure that you are using an up-to-date
version of it. For information about getting started beyond what is provided
below please see our
[Intro to OpenFermion](./tutorials/intro_to_openfermion.ipynb) tutorial.

Currently, OpenFermion is only tested on Mac and Linux for the reason that both
electronic structure plugins are only compatible with Mac and Linux. However,
for those who would like to use Windows, or for anyone having other difficulties
with installing OpenFermion or its plugins, we have provided a Docker image and
usage instructions in the
[docker folder](https://github.com/quantumlib/OpenFermion/tree/master/docker).
The Docker image provides a virtual environment with OpenFermion and select
plugins pre-installed. The Docker installation should run on any operating
system.

You might also want to explore the alpha release of the
[OpenFermion Cloud Library](https://github.com/quantumlib/OpenFermion/tree/master/cloud_library)
where users can share and download precomputed molecular benchmark files.

Check out other [projects and papers](docs/projects.md) using OpenFermion for
inspiration, and let us know if you've been using OpenFermion!

## Developer install

To install the latest version of OpenFermion (in development mode):

```bash
git clone https://github.com/quantumlib/OpenFermion
cd OpenFermion
python3 -m pip install -e .
```

## Library install

To install the latest PyPI release as a library (in user mode):

```bash
python3 -m pip install --user openfermion
```

## Build the docs

### Narrative docs

The guides and tutorials are built from the `docs/` directory. Preview Markdown
files directly in GitHub. Notebooks can be loaded, viewed, and executed in Colab
by passing the GitHub location in the URL, for example:
<a href="https://colab.research.google.com/github/quantumlib/OpenFermion/blob/master/docs/tutorials/intro_to_openfermion.ipynb"
class="external">https://colab.research.google.com/github/quantumlib/OpenFermion/blob/master/docs/tutorials/intro_to_openfermion.ipynb</a>

See the
<a href="https://www.tensorflow.org/community/contribute/docs_style" class="external">TensorFlow docs style guide</a>
for Markdown and MathJax usage.

### API reference

The API reference is generated from docstrings in the OpenFermion package using
the latest *stable* release in PyPI. See the
<a href="https://www.tensorflow.org/community/contribute/docs_ref" class="external">API reference style guide</a>
for examples of docstrings and testable code snippets.

To build the Markdown files from your local repo (to preview or for archive),
install the `tensorflow-docs` package to use the
<a href="https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/api_generator" class="external">API docs generator</a>
library:

```
python3 -m pip install -U --user git+https://github.com/tensorflow/docs
```

Run the OpenFermion docs build script to generate the Markdown files. These can
be previewed in GitHub:

```
python3 dev_tools/docs/build_api_docs.py --output_dir=docs/api_docs
```
